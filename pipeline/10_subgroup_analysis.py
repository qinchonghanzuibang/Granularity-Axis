#!/usr/bin/env python3
"""
Subgroup and confound-control analysis for the Granularity Axis.

Focuses on:
  - within-domain ladders / family progressions
  - generic vs title-heavy role subsets
  - score-filtering ablations
  - optional domain-level subgroup summaries
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.pca import MeanScaler, compute_pca

from analysis_utils import (
    build_role_vectors_from_activations,
    compute_contrast_axis,
    ensure_dir,
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
    stack_role_vectors,
    summarize_level_projections,
)


def evaluate_subset(records, micro_levels, macro_levels, target_layer):
    axis = compute_contrast_axis(records, micro_levels=micro_levels, macro_levels=macro_levels)
    stacked = stack_role_vectors(records)
    levels = [int(record["level"]) for record in records]
    pca_result, variance_explained, _, pca, _ = compute_pca(
        stacked,
        layer=target_layer,
        scaler=MeanScaler(),
        verbose=False,
    )
    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    levels_array = np.asarray(levels)
    micro_mask = np.isin(levels_array, np.asarray(micro_levels))
    macro_mask = np.isin(levels_array, np.asarray(macro_levels))
    if float(pca_result[macro_mask, 0].mean()) < float(pca_result[micro_mask, 0].mean()):
        pc1 = -pc1

    projections = project_records(records, axis=axis, layer=target_layer)
    level_summary = summarize_level_projections(projections)
    return {
        "pc1_explained_variance": float(variance_explained[0]),
        "contrast_pc1_cosine": float(normalize_vector(axis[target_layer]) @ normalize_vector(pc1)),
        "projection_level_spearman": spearman_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "projection_level_pearson": pearson_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "monotonic_ordering": is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary]),
        "role_count": len(records),
        "level_summary": level_summary,
    }


def family_rows(role_records, global_axis, target_layer):
    rows = []
    families = sorted({record.get("family") for record in role_records if record.get("family") not in {None, "unknown", "default"}})
    for family in families:
        family_records = [record for record in role_records if record.get("family") == family]
        if len({record["level"] for record in family_records}) < 4:
            continue
        projections = project_records(family_records, axis=global_axis, layer=target_layer)
        level_summary = summarize_level_projections(projections)
        rows.append(
            {
                "family": family,
                "role_count": len(family_records),
                "projection_level_spearman": spearman_correlation(
                    [row["projection"] for row in projections],
                    [row["level"] for row in projections],
                ),
                "monotonic_ordering": is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary]),
                "level_summary": level_summary,
            }
        )
    return rows


def level_count_map(records):
    counts = Counter(int(record["level"]) for record in records)
    return {str(level): counts[level] for level in sorted(counts)}


def projection_subset_summary(records, global_axis, target_layer):
    projections = project_records(records, axis=global_axis, layer=target_layer)
    level_summary = summarize_level_projections(projections)
    return {
        "projection_level_spearman": spearman_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "projection_level_pearson": pearson_correlation(
            [row["projection"] for row in projections],
            [row["level"] for row in projections],
        ),
        "monotonic_ordering": is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary]),
        "role_count": len(records),
        "levels_present": sorted({int(record["level"]) for record in records}),
        "level_count_by_level": level_count_map(records),
        "level_summary": level_summary,
    }


def role_type_rows(role_records, global_axis, target_layer, role_type_key: str = "role_type_bucket"):
    rows = []
    role_types = sorted(
        {
            record.get(role_type_key, "unknown")
            for record in role_records
            if record.get(role_type_key, "unknown") not in {"default", "unknown"}
        }
    )
    for role_type in role_types:
        subset = [record for record in role_records if record.get(role_type_key) == role_type]
        if len(subset) < 5 or len({record["level"] for record in subset}) < 2:
            continue
        row = {"role_type": role_type, **projection_subset_summary(subset, global_axis=global_axis, target_layer=target_layer)}
        if role_type_key == "role_type_bucket":
            row["raw_role_type_breakdown"] = {
                key: value
                for key, value in sorted(Counter(record.get("role_type", "unknown") for record in subset).items())
            }
        rows.append(row)
    return rows


def domain_rows(role_records, global_axis, target_layer):
    rows = []
    domains = sorted({record.get("domain", "unknown") for record in role_records})
    for domain in domains:
        subset = [record for record in role_records if record.get("domain") == domain]
        if len(subset) < 4 or len({record["level"] for record in subset}) < 3:
            continue
        projections = project_records(subset, axis=global_axis, layer=target_layer)
        level_summary = summarize_level_projections(projections)
        rows.append(
            {
                "domain": domain,
                "role_count": len(subset),
                "projection_level_spearman": spearman_correlation(
                    [row["projection"] for row in projections],
                    [row["level"] for row in projections],
                ),
                "monotonic_ordering": is_monotonic_non_decreasing([row["mean_projection"] for row in level_summary]),
                "level_summary": level_summary,
            }
        )
    return rows


def flatten_for_csv(row: dict, drop_keys: set[str] | None = None) -> dict:
    flat = {}
    for key, value in row.items():
        if drop_keys and key in drop_keys:
            continue
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flat[key] = value
    return flat


def score_filter_rows(
    activations_dir: Path | None,
    roles_dir: Path,
    metadata_file: Path | None,
    scores_dir: Path | None,
    micro_levels,
    macro_levels,
    target_layer,
):
    if activations_dir is None or scores_dir is None or not activations_dir.exists() or not scores_dir.exists():
        return []

    activation_payloads = load_activation_dicts(
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
    )
    configurations = [
        {"name": "all_responses", "min_score": 0, "score_mode": "at_least", "use_scores": False},
        {"name": "score_ge_2", "min_score": 2, "score_mode": "at_least", "use_scores": True},
        {"name": "score_ge_3", "min_score": 3, "score_mode": "at_least", "use_scores": True},
        {"name": "score_eq_3", "min_score": 3, "score_mode": "exact", "use_scores": True},
    ]

    rows = []
    for config in configurations:
        vectors = build_role_vectors_from_activations(
            activation_payloads=activation_payloads,
            scores_dir=scores_dir if config["use_scores"] else None,
            min_score=config["min_score"],
            score_mode=config["score_mode"],
        )
        role_vectors = filter_vector_records(vectors, include_default=False)
        if len(role_vectors) < 5:
            continue
        retained = sum(int(record.get("retained_count", 0)) for record in vectors)
        total = sum(int(record.get("total_count", 0)) for record in vectors)
        rows.append(
            {
                "filter_condition": config["name"],
                "retained_samples": retained,
                "total_samples": total,
                **evaluate_subset(role_vectors, micro_levels, macro_levels, target_layer),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Subgroup and control analyses for granularity experiments")
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--roles_dir", type=str, default=str(Path(__file__).parent.parent / "data" / "roles" / "instructions"))
    parser.add_argument("--metadata_file", type=str, default=str(Path(__file__).parent.parent / "data" / "role_metadata.json"))
    parser.add_argument("--activations_dir", type=str, default=None)
    parser.add_argument("--scores_dir", type=str, default=None)
    parser.add_argument("--micro_levels", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--macro_levels", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--target_layer", type=int, default=18)
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    roles_dir = Path(args.roles_dir)
    metadata_file = Path(args.metadata_file) if args.metadata_file else None
    activations_dir = Path(args.activations_dir) if args.activations_dir else None
    scores_dir = Path(args.scores_dir) if args.scores_dir else None
    ensure_dir(output_dir)

    all_records = load_vectors(vectors_dir=vectors_dir, roles_dir=roles_dir, metadata_file=metadata_file)
    role_records = filter_vector_records(all_records, include_default=False)
    global_axis = compute_contrast_axis(role_records, micro_levels=args.micro_levels, macro_levels=args.macro_levels)

    family_summary = family_rows(role_records, global_axis=global_axis, target_layer=args.target_layer)
    role_type_summary = role_type_rows(
        role_records,
        global_axis=global_axis,
        target_layer=args.target_layer,
        role_type_key="role_type_bucket",
    )
    role_type_detailed_summary = role_type_rows(
        role_records,
        global_axis=global_axis,
        target_layer=args.target_layer,
        role_type_key="role_type",
    )
    domain_summary = domain_rows(role_records, global_axis=global_axis, target_layer=args.target_layer)
    score_summary = score_filter_rows(
        activations_dir=activations_dir,
        roles_dir=roles_dir,
        metadata_file=metadata_file,
        scores_dir=scores_dir,
        micro_levels=args.micro_levels,
        macro_levels=args.macro_levels,
        target_layer=args.target_layer,
    )

    save_json(output_dir / "family_ladders.json", family_summary)
    save_csv(
        output_dir / "family_ladders.csv",
        [{key: value for key, value in row.items() if key != "level_summary"} for row in family_summary],
    )
    save_json(output_dir / "role_type_comparison.json", role_type_summary)
    save_csv(
        output_dir / "role_type_comparison.csv",
        [flatten_for_csv(row, drop_keys={"level_summary"}) for row in role_type_summary],
    )
    save_json(output_dir / "role_type_detailed_comparison.json", role_type_detailed_summary)
    save_csv(
        output_dir / "role_type_detailed_comparison.csv",
        [flatten_for_csv(row, drop_keys={"level_summary"}) for row in role_type_detailed_summary],
    )
    save_json(output_dir / "domain_subgroups.json", domain_summary)
    save_csv(
        output_dir / "domain_subgroups.csv",
        [{key: value for key, value in row.items() if key != "level_summary"} for row in domain_summary],
    )
    if score_summary:
        save_json(output_dir / "score_filtering_ablation.json", score_summary)
        save_csv(
            output_dir / "score_filtering_ablation.csv",
            [{key: value for key, value in row.items() if key != "level_summary"} for row in score_summary],
        )

    print(f"Saved subgroup analyses to: {output_dir}")


if __name__ == "__main__":
    main()
