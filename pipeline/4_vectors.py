#!/usr/bin/env python3
"""
Compute per-role vectors from activations and scores.

Follows the assistant-axis methodology (Lu et al. 2026):
  - For regular roles: mean of activations where score=3 (fully role-playing)
  - For default role: mean of ALL activations (no score filtering)
  - If no scores available: uses all activations (useful if judge step is skipped)

Also stores the granularity level metadata in each vector file for step 5.

Usage:
    python pipeline/4_vectors.py \
        --activations_dir outputs/qwen3-8b/activations \
        --scores_dir outputs/qwen3-8b/scores \
        --output_dir outputs/qwen3-8b/vectors

    # Without judge scores (use all activations)
    python pipeline/4_vectors.py \
        --activations_dir outputs/qwen3-8b/activations \
        --output_dir outputs/qwen3-8b/vectors \
        --no_scores
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent


def load_scores(scores_file: Path) -> dict:
    """Load scores from JSON file."""
    with open(scores_file, 'r') as f:
        return json.load(f)


def load_activations(activations_file: Path) -> dict:
    """Load activations from .pt file."""
    return torch.load(activations_file, map_location="cpu", weights_only=False)


def load_companion_metadata(metadata_file: Path | None) -> dict:
    if metadata_file is None or not metadata_file.exists():
        return {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_role_metadata(roles_dir: Path, role_name: str, companion_metadata: dict | None = None) -> dict:
    """Load level metadata from role JSON file."""
    role_file = roles_dir / f"{role_name}.json"
    companion = (companion_metadata or {}).get(role_name, {})
    if role_file.exists():
        with open(role_file, 'r') as f:
            data = json.load(f)
        return {
            "level": data.get("level", -1),
            "level_name": data.get("level_name", "Unknown"),
            "entity_name": data.get("entity_name", role_name),
            "domain": data.get("domain", companion.get("domain", "unknown")),
            "role_type": data.get("role_type", companion.get("role_type", "unknown")),
            "family": data.get("family", companion.get("family")),
            "ladder_position": data.get("ladder_position", companion.get("ladder_position")),
        }
    return {
        "level": companion.get("level", -1),
        "level_name": companion.get("level_name", "Unknown"),
        "entity_name": companion.get("entity_name", role_name),
        "domain": companion.get("domain", "unknown"),
        "role_type": companion.get("role_type", "unknown"),
        "family": companion.get("family"),
        "ladder_position": companion.get("ladder_position"),
    }


def compute_filtered_vector(
    activations: dict,
    scores: dict,
    min_count: int,
    min_score: int,
    score_mode: str,
) -> tuple[torch.Tensor, int, int]:
    """Compute mean vector from activations satisfying the score threshold."""
    filtered_acts = []
    total_acts = 0
    for key, act in activations.items():
        if key not in scores:
            continue
        total_acts += 1
        score = scores[key]
        keep = score == min_score if score_mode == "exact" else score >= min_score
        if keep:
            filtered_acts.append(act)

    if len(filtered_acts) < min_count:
        comparator = "==" if score_mode == "exact" else ">="
        raise ValueError(f"Only {len(filtered_acts)} samples with score {comparator} {min_score}, need {min_count}")

    stacked = torch.stack(filtered_acts)  # (n_samples, n_layers, hidden_dim)
    return stacked.mean(dim=0), len(filtered_acts), total_acts  # (n_layers, hidden_dim)


def compute_mean_vector(activations: dict) -> tuple[torch.Tensor, int]:
    """Compute mean vector from all activations (no filtering)."""
    all_acts = list(activations.values())
    stacked = torch.stack(all_acts)
    return stacked.mean(dim=0), len(all_acts)


def main():
    parser = argparse.ArgumentParser(description="Compute per-role vectors")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, default=None,
                        help="Directory with score JSON files (optional)")
    parser.add_argument("--roles_dir", type=str,
                        default=str(PROJECT_DIR / "data" / "roles" / "instructions"))
    parser.add_argument("--metadata_file", type=str,
                        default=str(PROJECT_DIR / "data" / "role_metadata.json"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_count", type=int, default=30,
                        help="Minimum score=3 samples required")
    parser.add_argument("--min_score", type=int, default=3,
                        help="Minimum judge score to keep when score filtering is enabled")
    parser.add_argument("--score_mode", choices=["at_least", "exact"], default="at_least",
                        help="How to apply --min_score")
    parser.add_argument("--no_scores", action="store_true",
                        help="Skip score filtering, use all activations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir) if args.scores_dir else None
    roles_dir = Path(args.roles_dir)
    metadata_file = Path(args.metadata_file) if args.metadata_file else None
    companion_metadata = load_companion_metadata(metadata_file)

    activation_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} activation files")

    use_scores = not args.no_scores and scores_dir is not None and scores_dir.exists()
    if use_scores:
        print(f"Using judge scores from: {scores_dir}")
    else:
        print("No score filtering — using all activations per role")

    successful = 0
    skipped = 0
    failed = 0
    summary_rows = []

    for act_file in tqdm(activation_files, desc="Computing vectors"):
        role = act_file.stem
        output_file = output_dir / f"{role}.pt"

        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        activations = load_activations(act_file)
        if not activations:
            print(f"Warning: No activations for {role}")
            failed += 1
            continue

        # Get role metadata (level info)
        metadata = load_role_metadata(roles_dir, role, companion_metadata=companion_metadata)

        try:
            if "default" in role:
                # Default: always use all activations
                vector, retained_count = compute_mean_vector(activations)
                vector_type = "mean"
                total_count = retained_count
            elif use_scores:
                # Regular roles with scores: filter by score=3
                scores_file = scores_dir / f"{role}.json"
                if not scores_file.exists():
                    print(f"Warning: No scores file for {role}, using all activations")
                    vector, retained_count = compute_mean_vector(activations)
                    vector_type = "mean_no_scores"
                    total_count = retained_count
                else:
                    scores = load_scores(scores_file)
                    vector, retained_count, total_count = compute_filtered_vector(
                        activations=activations,
                        scores=scores,
                        min_count=args.min_count,
                        min_score=args.min_score,
                        score_mode=args.score_mode,
                    )
                    vector_type = "filtered_scores"
            else:
                # No score filtering
                vector, retained_count = compute_mean_vector(activations)
                vector_type = "mean_no_scores"
                total_count = retained_count

            # Save vector with metadata
            save_data = {
                "vector": vector,
                "type": vector_type,
                "role": role,
                "level": metadata["level"],
                "level_name": metadata["level_name"],
                "entity_name": metadata["entity_name"],
                "domain": metadata["domain"],
                "role_type": metadata["role_type"],
                "family": metadata["family"],
                "ladder_position": metadata["ladder_position"],
                "retained_count": retained_count,
                "total_count": total_count,
                "score_threshold": args.min_score if use_scores and role != "default" else None,
                "score_mode": args.score_mode if use_scores and role != "default" else None,
            }
            torch.save(save_data, output_file)
            successful += 1
            summary_rows.append({
                "role": role,
                "vector_type": vector_type,
                "retained_count": retained_count,
                "total_count": total_count,
                "level": metadata["level"],
                "domain": metadata["domain"],
                "role_type": metadata["role_type"],
            })

        except ValueError as e:
            print(f"Warning: {role}: {e}")
            failed += 1

    summary_file = output_dir / "vector_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()

