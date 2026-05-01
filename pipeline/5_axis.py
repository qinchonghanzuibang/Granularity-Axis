#!/usr/bin/env python3
"""
Compute the Granularity Axis from per-role vectors.

Adapts the methodology from Lu et al. (2026) "The Assistant Axis" to find
a direction in activation space that captures social granularity
(micro/individual ↔ macro/institutional).

Two methods are computed and compared:

  Method 1 — Contrast Vector (primary):
    granularity_axis = mean(macro_vectors) - mean(micro_vectors)
    Points FROM micro TOWARD macro.
    Analogous to the Assistant Axis formulation: contrast = mean(default) - mean(roles)

  Method 2 — PCA (validation):
    Compute PCA on all role vectors. PC1 should align with the granularity axis
    if the micro-macro dimension is the main axis of variation, as the
    assistant/default dimension is for persona space.

Both vectors are saved. Cosine similarity is computed to verify alignment.

Usage:
    python pipeline/5_axis.py \
        --vectors_dir outputs/qwen3-8b/vectors \
        --output_dir outputs/qwen3-8b/axis

    # Specify which levels count as micro vs macro
    python pipeline/5_axis.py \
        --vectors_dir outputs/qwen3-8b/vectors \
        --output_dir outputs/qwen3-8b/axis \
        --micro_levels 1 2 \
        --macro_levels 4 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.pca import compute_pca, MeanScaler
from lib.axis import cosine_similarity_per_layer


def load_vector(vector_file: Path) -> dict:
    """Load vector data from .pt file."""
    return torch.load(vector_file, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(
        description="Compute granularity axis from per-role vectors"
    )
    parser.add_argument("--vectors_dir", type=str, required=True,
                        help="Directory with vector .pt files from step 4")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for axis files")
    parser.add_argument("--micro_levels", type=int, nargs="+", default=[1, 2],
                        help="Granularity levels to treat as 'micro' (default: 1 2)")
    parser.add_argument("--macro_levels", type=int, nargs="+", default=[4, 5],
                        help="Granularity levels to treat as 'macro' (default: 4 5)")
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Layer for PCA analysis (default: middle layer)")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load all vectors and sort by level
    # ============================================================
    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files")

    micro_vectors = []
    macro_vectors = []
    meso_vectors = []
    default_vectors = []
    all_role_vectors = []
    all_role_labels = []
    all_role_levels = []

    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector(vec_file)
        vector = data["vector"]  # (n_layers, hidden_dim)
        role = data.get("role", vec_file.stem)
        level = data.get("level", -1)

        if "default" in role or level == 0:
            default_vectors.append(vector)
            continue

        all_role_vectors.append(vector)
        all_role_labels.append(role)
        all_role_levels.append(level)

        if level in args.micro_levels:
            micro_vectors.append(vector)
        elif level in args.macro_levels:
            macro_vectors.append(vector)
        else:
            meso_vectors.append(vector)

    print(f"\nVector breakdown:")
    print(f"  Micro (levels {args.micro_levels}): {len(micro_vectors)}")
    print(f"  Meso (middle levels):               {len(meso_vectors)}")
    print(f"  Macro (levels {args.macro_levels}): {len(macro_vectors)}")
    print(f"  Default:                             {len(default_vectors)}")
    print(f"  Total role vectors:                  {len(all_role_vectors)}")

    if not micro_vectors:
        print("Error: No micro-level vectors found")
        sys.exit(1)
    if not macro_vectors:
        print("Error: No macro-level vectors found")
        sys.exit(1)

    # ============================================================
    # 2. Method 1: Contrast Vector (Granularity Axis)
    # ============================================================
    print("\n" + "=" * 60)
    print("METHOD 1: Contrast Vector")
    print("  granularity_axis = mean(macro) - mean(micro)")
    print("=" * 60)

    micro_stacked = torch.stack(micro_vectors).float()   # (n_micro, n_layers, hidden_dim)
    macro_stacked = torch.stack(macro_vectors).float()    # (n_macro, n_layers, hidden_dim)

    micro_mean = micro_stacked.mean(dim=0)  # (n_layers, hidden_dim)
    macro_mean = macro_stacked.mean(dim=0)  # (n_layers, hidden_dim)

    # Axis points FROM micro TOWARD macro
    granularity_axis = macro_mean - micro_mean  # (n_layers, hidden_dim)

    n_layers = granularity_axis.shape[0]
    target_layer = args.target_layer if args.target_layer else n_layers // 2

    print(f"\nAxis shape: {granularity_axis.shape}")
    norms = granularity_axis.norm(dim=1)
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Max norm:  {norms.max():.4f} (layer {norms.argmax().item()})")
    print(f"  Target layer ({target_layer}) norm: {norms[target_layer]:.4f}")

    # Save contrast-vector axis
    axis_path = output_dir / "granularity_axis.pt"
    torch.save({
        "axis": granularity_axis,
        "method": "contrast_vector",
        "micro_levels": args.micro_levels,
        "macro_levels": args.macro_levels,
        "n_micro": len(micro_vectors),
        "n_macro": len(macro_vectors),
    }, axis_path)
    print(f"\nSaved contrast-vector axis to: {axis_path}")

    # ============================================================
    # 3. Method 2: PCA on Role Vectors
    # ============================================================
    print("\n" + "=" * 60)
    print("METHOD 2: PCA on Role Vectors")
    print(f"  Using layer {target_layer} (middle layer)")
    print("=" * 60)

    role_vectors_stacked = torch.stack(all_role_vectors).float()  # (n_roles, n_layers, hidden_dim)

    # Use lib PCA implementation with L2 mean scaler
    scaler = MeanScaler()
    pca_result, variance_explained, n_components, pca, fitted_scaler = compute_pca(
        role_vectors_stacked,
        layer=target_layer,
        scaler=scaler,
        verbose=True,
    )

    # PC1 direction
    pc1_vector_np = pca.components_[0]  # (hidden_dim,)
    pc1_vector = torch.tensor(pc1_vector_np, dtype=torch.float32)

    # Calibrate PC1 direction: ensure it points micro → macro
    # Check mean projection of micro vs macro on PC1
    levels_array = np.array(all_role_levels)
    micro_mask = np.isin(levels_array, args.micro_levels)
    macro_mask = np.isin(levels_array, args.macro_levels)

    micro_pc1_mean = pca_result[micro_mask, 0].mean()
    macro_pc1_mean = pca_result[macro_mask, 0].mean()

    if macro_pc1_mean < micro_pc1_mean:
        print(f"\nFlipping PC1: macro mean ({macro_pc1_mean:.4f}) < micro mean ({micro_pc1_mean:.4f})")
        pc1_vector = -pc1_vector
        pca_result[:, 0] = -pca_result[:, 0]
        micro_pc1_mean = -micro_pc1_mean
        macro_pc1_mean = -macro_pc1_mean

    print(f"\nPC1 direction (micro→macro):")
    print(f"  Micro mean projection: {micro_pc1_mean:.4f}")
    print(f"  Macro mean projection: {macro_pc1_mean:.4f}")
    print(f"  Separation:            {macro_pc1_mean - micro_pc1_mean:.4f}")

    # Save PCA results
    pca_path = output_dir / "pca_results.pt"
    torch.save({
        "pc1_vector": pc1_vector,  # (hidden_dim,) at target_layer
        "pca_result": pca_result,   # (n_roles, n_components)
        "variance_explained": variance_explained,
        "role_labels": all_role_labels,
        "role_levels": all_role_levels,
        "target_layer": target_layer,
        "scaler_state": fitted_scaler.state_dict() if fitted_scaler else None,
    }, pca_path)
    print(f"Saved PCA results to: {pca_path}")

    # ============================================================
    # 4. Compare Methods: Cosine Similarity
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON: Contrast Vector vs PCA PC1")
    print("=" * 60)

    # For cosine similarity, we need both vectors at the same layer
    # The contrast vector is per-layer, PC1 is at a single layer
    contrast_at_layer = granularity_axis[target_layer]  # (hidden_dim,)
    contrast_norm = contrast_at_layer / (contrast_at_layer.norm() + 1e-8)
    pc1_norm = pc1_vector / (pc1_vector.norm() + 1e-8)

    cos_sim = float(contrast_norm @ pc1_norm)
    print(f"\n  Cosine similarity at layer {target_layer}: {cos_sim:.4f}")

    if abs(cos_sim) > 0.6:
        print("  ✓ High alignment — both methods agree on the granularity direction")
    elif abs(cos_sim) > 0.3:
        print("  ~ Moderate alignment — methods partially agree")
    else:
        print("  ✗ Low alignment — methods disagree; check data quality")

    # Also compute per-layer cosine similarity if we have a full axis
    # Build a "full PC1 axis" by computing PCA at each layer
    print("\nPer-layer cosine similarity (contrast vs PC1):")
    cos_sims_by_layer = []
    for layer_idx in range(n_layers):
        layer_vecs = role_vectors_stacked[:, layer_idx, :].numpy()
        layer_mean = layer_vecs.mean(axis=0)
        centered = layer_vecs - layer_mean
        
        from sklearn.decomposition import PCA as SklearnPCA
        pca_layer = SklearnPCA(n_components=1)
        pca_layer.fit(centered)
        pc1_at_layer = torch.tensor(pca_layer.components_[0], dtype=torch.float32)
        
        contrast_at_l = granularity_axis[layer_idx]
        c_norm = contrast_at_l / (contrast_at_l.norm() + 1e-8)
        p_norm = pc1_at_layer / (pc1_at_layer.norm() + 1e-8)
        sim = float(c_norm @ p_norm)
        cos_sims_by_layer.append(abs(sim))  # abs because PC1 direction is arbitrary

    cos_sims_array = np.array(cos_sims_by_layer)
    print(f"  Mean |cosine similarity|: {cos_sims_array.mean():.4f}")
    print(f"  Max  |cosine similarity|: {cos_sims_array.max():.4f} (layer {cos_sims_array.argmax()})")
    print(f"  At target layer {target_layer}:  {cos_sims_array[target_layer]:.4f}")

    # ============================================================
    # 5. Per-level analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-LEVEL ANALYSIS")
    print(f"  Projection onto granularity axis at layer {target_layer}")
    print("=" * 60)

    axis_at_layer = granularity_axis[target_layer]
    axis_normalized = axis_at_layer / (axis_at_layer.norm() + 1e-8)

    level_projections = {1: [], 2: [], 3: [], 4: [], 5: []}
    for vec, level in zip(all_role_vectors, all_role_levels):
        proj = float(vec[target_layer].float() @ axis_normalized)
        if level in level_projections:
            level_projections[level].append(proj)

    level_names = {
        1: "Individual (Micro)",
        2: "Group",
        3: "Organization (Meso)",
        4: "Institution",
        5: "Nation (Macro)",
    }

    print(f"\n{'Level':<10} {'Name':<25} {'Mean Proj':>10} {'Std':>8} {'Count':>6}")
    print("-" * 65)
    for level in sorted(level_projections.keys()):
        projs = level_projections[level]
        if projs:
            mean_proj = np.mean(projs)
            std_proj = np.std(projs)
            print(f"  {level:<8} {level_names.get(level, '?'):<25} {mean_proj:>10.4f} {std_proj:>8.4f} {len(projs):>6}")

    # Check monotonicity
    means = [np.mean(level_projections[l]) for l in sorted(level_projections.keys()) if level_projections[l]]
    is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
    print(f"\n  Monotonic ordering (1→5): {'✓ Yes' if is_monotonic else '✗ No'}")

    # ============================================================
    # 6. Save single-layer steering vector (convenience)
    # ============================================================
    steering_vector = granularity_axis[target_layer]
    steering_vector = steering_vector / steering_vector.norm()

    steering_path = output_dir / "granularity_vector.pt"
    torch.save(steering_vector, steering_path)
    print(f"\nSaved single-layer steering vector (layer {target_layer}) to: {steering_path}")

    print("\n" + "=" * 60)
    print("DONE — Granularity Axis computed successfully")
    print(f"  Full axis:       {axis_path}")
    print(f"  PCA results:     {pca_path}")
    print(f"  Steering vector: {steering_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

