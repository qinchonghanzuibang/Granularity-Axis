#!/usr/bin/env python3
"""
Representation robustness suite for the Granularity Axis.

This script reuses existing vector / activation artifacts to produce the main
supplementary representation analyses:
  - layer-wise stability
  - endpoint-ablation robustness
  - held-out prompt / question / role splits
  - default assistant placement
  - prompt-template sensitivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.pca import MeanScaler, compute_pca

from analysis_utils import (
    build_role_vectors_from_activations,
    compute_contrast_axis,
    deterministic_half_split,
    ensure_dir,
    even_odd_split,
    filter_vector_records,
    is_monotonic_non_decreasing,
    load_activation_dicts,
    load_vectors,
    normalize_vector,
    pearson_correlation,
    project_records,
    save_csv,
    save_json,
    spearman_correlation,
    split_role_records_by_level,
    stack_role_vectors,
    summarize_level_projections,
)


def calibrate_pc1_direction(
    pca_result: np.ndarray,
    pca,
    levels: Sequence[int],
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
) -> torch.Tensor:
    """Flip PC1 so it points from micro to macro."""
    levels_array = np.asarray(levels)
    micro_mask = np.isin(levels_array, np.asarray(micro_levels))
    macro_mask = np.isin(levels_array, np.asarray(macro_levels))
    pc1_vector = torch.tensor(pca.components_[0], dtype=torch.float32)
    micro_mean = float(pca_result[micro_mask, 0].mean())
    macro_mean = float(pca_result[macro_mask, 0].mean())

    if macro_mean < micro_mean:
        pc1_vector = -pc1_vector
    return pc1_vector


def evaluate_axis(
    train_records: Sequence[dict],
    eval_records: Sequence[dict],
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
    target_layer: int,
) -> tuple[dict, list[dict]]:
    """Build an axis from train_records and evaluate it on eval_records."""
    axis = compute_contrast_axis(train_records, micro_levels=micro_levels, macro_levels=macro_levels)
    eval_role_records = filter_vector_records(eval_records, include_default=False)
    stacked = stack_role_vectors(eval_role_records)
    levels = [int(record["level"]) for record in eval_role_records]

    pca_result, variance_explained, _, pca, _ = compute_pca(
        stacked,
        layer=target_layer,
        scaler=MeanScaler(),
        verbose=False,
    )
    pc1 = calibrate_pc1_direction(
        pca_result=pca_result,
        pca=pca,
        levels=levels,
        micro_levels=micro_levels,
        macro_levels=macro_levels,
    )

    axis_layer = normalize_vector(axis[target_layer])
    pc1_layer = normalize_vector(pc1)
    projections = project_records(eval_role_records, axis=axis, layer=target_layer)
    level_summary = summarize_level_projections(projections)
    monotonic = is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary])

    micro_values = [row["projection"] for row in projections if row["level"] in set(micro_levels)]
    macro_values = [row["projection"] for row in projections if row["level"] in set(macro_levels)]
    summary = {
        "target_layer": target_layer,
        "pc1_explained_variance": float(variance_explained[0]),
        "pc1_cumulative_variance_5": float(np.cumsum(variance_explained[:5])[-1]),
        "contrast_pc1_cosine": float(axis_layer @ pc1_layer),
        "projection_level_spearman": spearman_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "projection_level_pearson": pearson_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "monotonic_ordering": monotonic,
        "micro_mean_projection": float(np.mean(micro_values)),
        "macro_mean_projection": float(np.mean(macro_values)),
        "macro_micro_gap": float(np.mean(macro_values) - np.mean(micro_values)),
        "role_count": len(eval_role_records),
    }
    return summary, level_summary


def compute_layerwise_metrics(
    role_records: Sequence[dict],
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
) -> tuple[list[dict], list[dict]]:
    """Compute layer-wise stability metrics and per-level projection trends."""
    axis = compute_contrast_axis(role_records, micro_levels=micro_levels, macro_levels=macro_levels)
    stacked = stack_role_vectors(role_records)
    levels = [int(record["level"]) for record in role_records]
    layer_rows = []
    trend_rows = []

    for layer in range(axis.shape[0]):
        pca_result, variance_explained, _, pca, _ = compute_pca(
            stacked,
            layer=layer,
            scaler=MeanScaler(),
            verbose=False,
        )
        pc1 = calibrate_pc1_direction(
            pca_result=pca_result,
            pca=pca,
            levels=levels,
            micro_levels=micro_levels,
            macro_levels=macro_levels,
        )
        projections = project_records(role_records, axis=axis, layer=layer)
        level_summary = summarize_level_projections(projections)
        monotonic = is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary])

        layer_rows.append(
            {
                "layer": layer,
                "axis_norm": float(axis[layer].norm()),
                "pc1_explained_variance": float(variance_explained[0]),
                "contrast_pc1_cosine": float(normalize_vector(axis[layer]) @ normalize_vector(pc1)),
                "projection_level_spearman": spearman_correlation(
                    [row["projection"] for row in projections],
                    [row["level"] for row in projections],
                ),
                "projection_level_pearson": pearson_correlation(
                    [row["projection"] for row in projections],
                    [row["level"] for row in projections],
                ),
                "monotonic_ordering": monotonic,
            }
        )
        for level_row in level_summary:
            trend_rows.append(
                {
                    "layer": layer,
                    "level": level_row["level"],
                    "mean_projection": level_row["mean_projection"],
                    "std_projection": level_row["std_projection"],
                    "count": level_row["count"],
                }
            )

    return layer_rows, trend_rows


def compute_endpoint_ablation(
    role_records: Sequence[dict],
    target_layer: int,
) -> list[dict]:
    """Compare alternative endpoint definitions for the granularity axis."""
    definitions = [
        {"name": "main_L45_minus_L12", "micro_levels": [1, 2], "macro_levels": [4, 5]},
        {"name": "extreme_L5_minus_L1", "micro_levels": [1], "macro_levels": [5]},
        {"name": "macro_heavy_L45_minus_L1", "micro_levels": [1], "macro_levels": [4, 5]},
        {"name": "micro_heavy_L5_minus_L12", "micro_levels": [1, 2], "macro_levels": [5]},
    ]

    rows = []
    for definition in definitions:
        summary, level_summary = evaluate_axis(
            train_records=role_records,
            eval_records=role_records,
            micro_levels=definition["micro_levels"],
            macro_levels=definition["macro_levels"],
            target_layer=target_layer,
        )
        rows.append(
            {
                "axis_definition": definition["name"],
                "micro_levels": definition["micro_levels"],
                "macro_levels": definition["macro_levels"],
                **summary,
                "level_summary": level_summary,
            }
        )
    return rows


def compute_heldout_splits(
    role_records: Sequence[dict],
    activations_dir: Path | None,
    roles_dir: Path,
    metadata_file: Path | None,
    scores_dir: Path | None,
    min_score: int,
    score_mode: str,
    target_layer: int,
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
    role_holdout_fraction: float,
    seed: int,
) -> list[dict]:
    rows = []

    train_roles, heldout_roles = split_role_records_by_level(
        role_records,
        holdout_fraction=role_holdout_fraction,
        seed=seed,
    )
    summary, _ = evaluate_axis(
        train_records=train_roles,
        eval_records=heldout_roles,
        micro_levels=micro_levels,
        macro_levels=macro_levels,
        target_layer=target_layer,
    )
    rows.append(
        {
            "split_type": "role_holdout",
            "train_spec": f"{len(train_roles)} roles",
            "eval_spec": f"{len(heldout_roles)} roles",
            **summary,
        }
    )

    if activations_dir is None or not activations_dir.exists():
        return rows

    activation_payloads = load_activation_dicts(
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
    )
    sample_role = next(iter(activation_payloads.values()))
    sample_keys = list(sample_role["activations"].keys())
    prompt_indices = sorted({int(k.split("_p", 1)[1].split("_q", 1)[0]) for k in sample_keys})
    question_indices = sorted({int(k.rsplit("_q", 1)[1]) for k in sample_keys})

    prompt_train, prompt_eval = deterministic_half_split(prompt_indices)
    question_train, question_eval = even_odd_split(question_indices)

    prompt_train_vectors = filter_vector_records(
        build_role_vectors_from_activations(
            activation_payloads,
            scores_dir=scores_dir,
            prompt_indices=prompt_train,
            min_score=min_score,
            score_mode=score_mode,
        ),
        include_default=False,
    )
    prompt_eval_vectors = filter_vector_records(
        build_role_vectors_from_activations(
            activation_payloads,
            scores_dir=scores_dir,
            prompt_indices=prompt_eval,
            min_score=min_score,
            score_mode=score_mode,
        ),
        include_default=False,
    )
    summary, _ = evaluate_axis(
        train_records=prompt_train_vectors,
        eval_records=prompt_eval_vectors,
        micro_levels=micro_levels,
        macro_levels=macro_levels,
        target_layer=target_layer,
    )
    rows.append(
        {
            "split_type": "prompt_holdout",
            "train_spec": ",".join(map(str, prompt_train)),
            "eval_spec": ",".join(map(str, prompt_eval)),
            **summary,
        }
    )

    question_train_vectors = filter_vector_records(
        build_role_vectors_from_activations(
            activation_payloads,
            scores_dir=scores_dir,
            question_indices=question_train,
            min_score=min_score,
            score_mode=score_mode,
        ),
        include_default=False,
    )
    question_eval_vectors = filter_vector_records(
        build_role_vectors_from_activations(
            activation_payloads,
            scores_dir=scores_dir,
            question_indices=question_eval,
            min_score=min_score,
            score_mode=score_mode,
        ),
        include_default=False,
    )
    summary, _ = evaluate_axis(
        train_records=question_train_vectors,
        eval_records=question_eval_vectors,
        micro_levels=micro_levels,
        macro_levels=macro_levels,
        target_layer=target_layer,
    )
    rows.append(
        {
            "split_type": "question_holdout",
            "train_spec": f"{len(question_train)} questions",
            "eval_spec": f"{len(question_eval)} questions",
            **summary,
        }
    )

    return rows


def compute_prompt_template_sensitivity(
    activations_dir: Path | None,
    roles_dir: Path,
    metadata_file: Path | None,
    scores_dir: Path | None,
    min_score: int,
    score_mode: str,
    target_layer: int,
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
) -> list[dict]:
    if activations_dir is None or not activations_dir.exists():
        return []

    activation_payloads = load_activation_dicts(
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
    )
    sample_role = next(iter(activation_payloads.values()))
    prompt_indices = sorted(
        {int(key.split("_p", 1)[1].split("_q", 1)[0]) for key in sample_role["activations"].keys()}
    )
    rows = []
    for prompt_idx in prompt_indices:
        prompt_vectors = filter_vector_records(
            build_role_vectors_from_activations(
                activation_payloads,
                scores_dir=scores_dir,
                prompt_indices=[prompt_idx],
                min_score=min_score,
                score_mode=score_mode,
            ),
            include_default=False,
        )
        summary, _ = evaluate_axis(
            train_records=prompt_vectors,
            eval_records=prompt_vectors,
            micro_levels=micro_levels,
            macro_levels=macro_levels,
            target_layer=target_layer,
        )
        rows.append({"prompt_index": prompt_idx, **summary})
    return rows


def compute_default_assistant_placement(
    all_records: Sequence[dict],
    role_records: Sequence[dict],
    micro_levels: Sequence[int],
    macro_levels: Sequence[int],
    target_layer: int,
) -> dict:
    default_record = next((record for record in all_records if record.get("is_default")), None)
    if default_record is None:
        return {}

    axis = compute_contrast_axis(role_records, micro_levels=micro_levels, macro_levels=macro_levels)
    axis_layer = normalize_vector(axis[target_layer])
    default_projection = float(default_record["vector"][target_layer].float() @ axis_layer)

    level_centroids = {}
    level_distances = {}
    for level in sorted({int(record["level"]) for record in role_records}):
        level_vectors = torch.stack(
            [record["vector"][target_layer].float() for record in role_records if int(record["level"]) == level],
            dim=0,
        )
        centroid = level_vectors.mean(dim=0)
        level_centroids[level] = centroid
        level_distances[level] = float(torch.norm(default_record["vector"][target_layer].float() - centroid))

    nearest_level = min(level_distances, key=level_distances.get)
    level_projection_rows = summarize_level_projections(project_records(role_records, axis=axis, layer=target_layer))
    return {
        "target_layer": target_layer,
        "default_projection": default_projection,
        "default_nearest_level": int(nearest_level),
        "default_level_distances": level_distances,
        "level_projection_summary": level_projection_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation robustness suite for Granularity Axis")
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--roles_dir", type=str, default=str(Path(__file__).parent.parent / "data" / "roles" / "instructions"))
    parser.add_argument("--metadata_file", type=str, default=str(Path(__file__).parent.parent / "data" / "role_metadata.json"))
    parser.add_argument("--activations_dir", type=str, default=None)
    parser.add_argument("--scores_dir", type=str, default=None)
    parser.add_argument("--min_score", type=int, default=3)
    parser.add_argument("--score_mode", choices=["at_least", "exact"], default="at_least")
    parser.add_argument("--target_layer", type=int, default=18)
    parser.add_argument("--micro_levels", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--macro_levels", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--role_holdout_fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    roles_dir = Path(args.roles_dir)
    metadata_file = Path(args.metadata_file) if args.metadata_file else None
    activations_dir = Path(args.activations_dir) if args.activations_dir else None
    scores_dir = Path(args.scores_dir) if args.scores_dir else None

    all_records = load_vectors(
        vectors_dir=vectors_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
    )
    role_records = filter_vector_records(all_records, include_default=False)

    layer_rows, layer_trends = compute_layerwise_metrics(
        role_records=role_records,
        micro_levels=args.micro_levels,
        macro_levels=args.macro_levels,
    )
    endpoint_rows = compute_endpoint_ablation(role_records=role_records, target_layer=args.target_layer)
    heldout_rows = compute_heldout_splits(
        role_records=role_records,
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
        scores_dir=scores_dir,
        min_score=args.min_score,
        score_mode=args.score_mode,
        target_layer=args.target_layer,
        micro_levels=args.micro_levels,
        macro_levels=args.macro_levels,
        role_holdout_fraction=args.role_holdout_fraction,
        seed=args.seed,
    )
    default_placement = compute_default_assistant_placement(
        all_records=all_records,
        role_records=role_records,
        micro_levels=args.micro_levels,
        macro_levels=args.macro_levels,
        target_layer=args.target_layer,
    )
    prompt_rows = compute_prompt_template_sensitivity(
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
        scores_dir=scores_dir,
        min_score=args.min_score,
        score_mode=args.score_mode,
        target_layer=args.target_layer,
        micro_levels=args.micro_levels,
        macro_levels=args.macro_levels,
    )

    save_csv(output_dir / "layerwise_metrics.csv", layer_rows)
    save_json(output_dir / "layerwise_metrics.json", layer_rows)
    save_csv(output_dir / "layerwise_level_trends.csv", layer_trends)
    save_json(output_dir / "endpoint_ablation.json", endpoint_rows)
    save_csv(
        output_dir / "endpoint_ablation.csv",
        [{key: value for key, value in row.items() if key != "level_summary"} for row in endpoint_rows],
    )
    save_json(output_dir / "heldout_robustness.json", heldout_rows)
    save_csv(output_dir / "heldout_robustness.csv", heldout_rows)
    save_json(output_dir / "default_assistant_placement.json", default_placement)
    if prompt_rows:
        save_json(output_dir / "prompt_sensitivity.json", prompt_rows)
        save_csv(output_dir / "prompt_sensitivity.csv", prompt_rows)

    print(f"Saved representation suite outputs to: {output_dir}")


if __name__ == "__main__":
    main()
