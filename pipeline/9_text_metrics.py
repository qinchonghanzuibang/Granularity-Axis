#!/usr/bin/env python3
"""
Compute lightweight lexical and structural metrics for steering outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analysis_utils import ensure_dir

TOKEN_RE = re.compile(r"\b\w+\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "ours"}
HEDGES = {
    "maybe", "perhaps", "might", "could", "seems", "seem", "probably",
    "possibly", "i think", "i guess", "i don't know", "sort of", "kind of",
}
POLICY_TERMS = {
    "government", "policy", "policies", "regulation", "regulatory", "institution",
    "institutions", "international", "system", "systems", "ministry", "state",
    "public", "governance", "framework", "reform", "subsidy", "tax", "central",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def max_ngram_repeat(tokens: list[str], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(counts.values()) if counts else 0


def count_phrase_occurrences(text: str, phrases: set[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(phrase) for phrase in phrases)


def measure_text(row: dict) -> dict:
    text = row["response"]
    lowered = text.lower()
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    token_count = len(tokens)
    sentences = [segment.strip() for segment in SENTENCE_RE.findall(text) if segment.strip()]
    sentence_lengths = [len(TOKEN_RE.findall(sentence)) for sentence in sentences] or [0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numbered_lines = sum(1 for line in lines if re.match(r"^(\d+[\).\:]|-|\*)\s+", line))
    first_person_count = sum(token in FIRST_PERSON for token in tokens)
    policy_count = sum(token in POLICY_TERMS for token in tokens)
    hedge_count = count_phrase_occurrences(lowered, HEDGES)
    repeat_3 = max_ngram_repeat(tokens, 3)
    stripped = text.rstrip()
    ends_with_terminal_punctuation = 1 if stripped.endswith((".", "!", "?", "\"", "'", "”", "’", ".)", "?)", "!)")) else 0

    return {
        **row,
        "token_count": token_count,
        "sentence_count": len(sentences),
        "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths),
        "first_person_ratio": first_person_count / token_count if token_count else 0.0,
        "policy_term_ratio": policy_count / token_count if token_count else 0.0,
        "hedge_rate": hedge_count / max(1, len(sentences)),
        "numbered_line_count": numbered_lines,
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "max_repeat_3gram": repeat_3,
        "degeneration_heuristic": 1 if repeat_3 >= 6 or distinct_n(tokens, 2) < 0.2 else 0,
        "ends_with_terminal_punctuation": ends_with_terminal_punctuation,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["direction"], row["layer"], row["coefficient"], row["decoding"])].append(row)

    numeric_fields = [
        "token_count",
        "sentence_count",
        "avg_sentence_length",
        "first_person_ratio",
        "policy_term_ratio",
        "hedge_rate",
        "numbered_line_count",
        "distinct_1",
        "distinct_2",
        "max_repeat_3gram",
        "degeneration_heuristic",
        "generated_token_count",
        "hit_max_new_tokens",
        "ended_with_eos",
        "ends_with_terminal_punctuation",
    ]
    summary_rows = []
    for (direction, layer, coefficient, decoding), bucket in grouped.items():
        summary = {
            "direction": direction,
            "layer": layer,
            "coefficient": coefficient,
            "decoding": decoding,
            "count": len(bucket),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in bucket if field in item and item[field] is not None]
            summary[field] = sum(values) / len(values) if values else None
        summary_rows.append(summary)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical and structural metrics for steering outputs")
    parser.add_argument("--responses_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    responses_file = Path(args.responses_file)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    rows = load_jsonl(responses_file)
    measured = [measure_text(row) for row in rows]
    summary = aggregate(measured) if measured else []

    with open(output_dir / "steering_text_metrics.jsonl", "w", encoding="utf-8") as handle:
        for row in measured:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "steering_text_metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved lexical metrics to: {output_dir}")


if __name__ == "__main__":
    main()
