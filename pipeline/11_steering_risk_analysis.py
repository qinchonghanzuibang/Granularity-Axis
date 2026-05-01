#!/usr/bin/env python3
"""
Summarize steering asymmetry, default placement, and truncation-focused reruns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_utils import load_json, save_csv, save_json


def load_named_paths(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        name, raw_path = item.split("=", 1)
        mapping[name] = Path(raw_path)
    return mapping


def load_summary_map(paths: dict[str, Path]) -> dict[str, list[dict]]:
    return {name: load_json(path) for name, path in paths.items()}


def index_summary(rows: list[dict]) -> tuple[dict, dict[float, dict]]:
    baseline = None
    granularity_rows: dict[float, dict] = {}
    for row in rows:
        if row.get("direction") == "baseline":
            baseline = row
        elif row.get("direction") == "granularity":
            granularity_rows[float(row.get("coefficient", 0.0))] = row
    if baseline is None:
        raise ValueError("Missing baseline row in summary")
    return baseline, granularity_rows


def build_delta_rows(summary_name: str, rows: list[dict], metrics: list[str]) -> list[dict]:
    baseline, granularity_rows = index_summary(rows)
    output = []
    for coeff in sorted(granularity_rows):
        row = granularity_rows[coeff]
        item = {"summary": summary_name, "coefficient": coeff}
        for metric in metrics:
            base_value = baseline.get(metric)
            value = row.get(metric)
            item[f"{metric}_baseline"] = base_value
            item[f"{metric}_value"] = value
            item[f"{metric}_delta"] = None if base_value is None or value is None else float(value) - float(base_value)
        output.append(item)
    return output


def summarize_asymmetry(summary_name: str, rows: list[dict], metric: str) -> dict:
    baseline, granularity_rows = index_summary(rows)
    negative_coeffs = sorted([coeff for coeff in granularity_rows if coeff < 0])
    positive_coeffs = sorted([coeff for coeff in granularity_rows if coeff > 0])
    if not negative_coeffs or not positive_coeffs:
        raise ValueError("Need both positive and negative coefficients for asymmetry analysis")

    neg_coeff = negative_coeffs[0]
    pos_coeff = positive_coeffs[-1]
    baseline_value = float(baseline[metric])
    neg_value = float(granularity_rows[neg_coeff][metric])
    pos_value = float(granularity_rows[pos_coeff][metric])
    neg_delta = neg_value - baseline_value
    pos_delta = pos_value - baseline_value

    return {
        "summary": summary_name,
        "metric": metric,
        "baseline_value": baseline_value,
        "negative_coeff": neg_coeff,
        "negative_value": neg_value,
        "negative_delta": neg_delta,
        "positive_coeff": pos_coeff,
        "positive_value": pos_value,
        "positive_delta": pos_delta,
        "absolute_negative_delta": abs(neg_delta),
        "absolute_positive_delta": abs(pos_delta),
        "positive_minus_negative_abs_gap": abs(pos_delta) - abs(neg_delta),
        "macro_push_stronger": abs(pos_delta) > abs(neg_delta),
    }


def compare_summaries(
    name_a: str,
    rows_a: list[dict],
    name_b: str,
    rows_b: list[dict],
    metrics: list[str],
) -> dict:
    baseline_a, granularity_a = index_summary(rows_a)
    baseline_b, granularity_b = index_summary(rows_b)
    shared_coeffs = sorted(set(granularity_a) & set(granularity_b))
    result = {
        "comparison": f"{name_a}_vs_{name_b}",
        "shared_coefficients": shared_coeffs,
        "baseline": {},
        "granularity": [],
    }
    for metric in metrics:
        a_val = baseline_a.get(metric)
        b_val = baseline_b.get(metric)
        result["baseline"][metric] = {
            name_a: a_val,
            name_b: b_val,
            "delta": None if a_val is None or b_val is None else float(b_val) - float(a_val),
        }
    for coeff in shared_coeffs:
        row = {"coefficient": coeff}
        for metric in metrics:
            a_val = granularity_a[coeff].get(metric)
            b_val = granularity_b[coeff].get(metric)
            row[metric] = {
                name_a: a_val,
                name_b: b_val,
                "delta": None if a_val is None or b_val is None else float(b_val) - float(a_val),
            }
        result["granularity"].append(row)
    return result


def write_markdown(
    output_path: Path,
    default_payload: dict,
    asymmetry_rows: list[dict],
    truncation_comparisons: list[dict],
) -> None:
    lines = [
        "# Steering Risk Summary",
        "",
        "## Default Placement",
        f"- Default projection at target layer: {default_payload['default_projection']:.4f}",
        f"- Nearest level: {default_payload['default_nearest_level']}",
        "",
        "## Asymmetry",
    ]
    for row in asymmetry_rows:
        lines.append(
            f"- `{row['summary']}` on `{row['metric']}`: baseline={row['baseline_value']:.4f}, "
            f"{row['negative_coeff']} -> {row['negative_delta']:+.4f}, "
            f"{row['positive_coeff']} -> {row['positive_delta']:+.4f}, "
            f"macro_push_stronger={row['macro_push_stronger']}"
        )
    if truncation_comparisons:
        lines.extend(["", "## Truncation Comparisons"])
        for item in truncation_comparisons:
            lines.append(f"- `{item['comparison']}`")
            for metric, payload in item["baseline"].items():
                delta = payload["delta"]
                if delta is not None:
                    lines.append(f"  {metric}: baseline delta={delta:+.4f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize truncation and asymmetry steering risks")
    parser.add_argument("--default_placement", type=str, required=True)
    parser.add_argument("--judge_summary", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--text_summary", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--compare_pair", action="append", default=[], help="NAME_A=NAME_B")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_payload = load_json(Path(args.default_placement))
    judge_summaries = load_summary_map(load_named_paths(args.judge_summary))
    text_summaries = load_summary_map(load_named_paths(args.text_summary))

    judge_metrics = [
        "granularity_overall",
        "temporal_scope",
        "collectivity",
        "abstraction",
        "decision_logic",
        "degeneration",
    ]
    text_metrics = [
        "policy_term_ratio",
        "first_person_ratio",
        "hedge_rate",
        "degeneration_heuristic",
        "generated_token_count",
        "hit_max_new_tokens",
        "ended_with_eos",
        "ends_with_terminal_punctuation",
    ]

    judge_delta_rows: list[dict] = []
    judge_asymmetry_rows: list[dict] = []
    for name, rows in judge_summaries.items():
        judge_delta_rows.extend(build_delta_rows(name, rows, judge_metrics))
        judge_asymmetry_rows.append(summarize_asymmetry(name, rows, metric="granularity_overall"))

    text_delta_rows: list[dict] = []
    text_asymmetry_rows: list[dict] = []
    for name, rows in text_summaries.items():
        text_delta_rows.extend(build_delta_rows(name, rows, text_metrics))
        if any(row.get("direction") == "granularity" for row in rows):
            text_asymmetry_rows.append(summarize_asymmetry(name, rows, metric="policy_term_ratio"))

    comparisons = []
    for pair in args.compare_pair:
        if "=" not in pair:
            raise ValueError(f"Expected NAME_A=NAME_B, got: {pair}")
        left, right = pair.split("=", 1)
        if left in judge_summaries and right in judge_summaries:
            comparisons.append(compare_summaries(left, judge_summaries[left], right, judge_summaries[right], judge_metrics))
        if left in text_summaries and right in text_summaries:
            comparisons.append(compare_summaries(left, text_summaries[left], right, text_summaries[right], text_metrics))

    summary_payload = {
        "default_placement": default_payload,
        "judge_asymmetry": judge_asymmetry_rows,
        "text_asymmetry": text_asymmetry_rows,
        "comparisons": comparisons,
    }

    save_csv(output_dir / "judge_baseline_centered_deltas.csv", judge_delta_rows)
    save_csv(output_dir / "text_baseline_centered_deltas.csv", text_delta_rows)
    save_json(output_dir / "steering_risk_summary.json", summary_payload)
    write_markdown(output_dir / "steering_risk_summary.md", default_payload, judge_asymmetry_rows, comparisons)


if __name__ == "__main__":
    main()
