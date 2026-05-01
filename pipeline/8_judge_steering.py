#!/usr/bin/env python3
"""
Judge steering outputs with a structured granularity rubric.

This script scores steering generations along several social-granularity
dimensions and writes both per-record JSONL scores and aggregate summaries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from lib.judge import RateLimiter, call_judge_batch
import openai

from analysis_utils import ensure_dir, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    clean = base_url.rstrip("/")
    if not clean.endswith("/v1"):
        clean += "/v1"
    return clean


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_record_key(row: dict) -> str:
    return "|".join(
        [
            str(row.get("prompt_id")),
            str(row.get("direction")),
            str(row.get("layer")),
            f"{float(row.get('coefficient', 0.0)):+.4f}",
            str(row.get("decoding")),
        ]
    )


def build_prompt(row: dict, rubric: dict) -> str:
    dimension_lines = []
    for dim in rubric["dimensions"]:
        dimension_lines.append(
            f'- "{dim["id"]}": {dim["label"]}. '
            f'Score 1-5 where 1="{dim["low_anchor"]}" and 5="{dim["high_anchor"]}".'
        )

    extra_lines = []
    for field in rubric["extra_fields"]:
        if field["id"] == "degeneration":
            extra_lines.append('- "degeneration": 0 or 1. Use 1 if the response is clearly repetitive, incoherent, or degraded.')
        else:
            extra_lines.append(
                f'- "{field["id"]}": 1-5. {field["description"]}'
            )

    return (
        "You are evaluating how socially micro-vs-macro a response is.\n\n"
        "Rate the response on the following dimensions:\n"
        + "\n".join(dimension_lines)
        + "\n"
        + "\n".join(extra_lines)
        + "\n\n"
        "Prompt:\n"
        f"[QUESTION START]\n{row['prompt']}\n[QUESTION END]\n\n"
        "Model response:\n"
        f"[ANSWER START]\n{row['response']}\n[ANSWER END]\n\n"
        "Return ONLY valid JSON with numeric fields for all requested ids. "
        "Do not include markdown fences or extra commentary."
    )


def parse_json_response(text: str, rubric: dict) -> dict | None:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        candidate = match.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    expected_fields = [dim["id"] for dim in rubric["dimensions"]] + [
        field["id"] for field in rubric["extra_fields"]
    ]
    parsed = {}
    for field in expected_fields:
        if field not in payload:
            return None
        parsed[field] = payload[field]
    return parsed


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["direction"], row["layer"], row["coefficient"], row["decoding"])].append(row)

    summary_rows = []
    numeric_fields = [
        key
        for key in rows[0].keys()
        if key
        not in {
            "record_key",
            "prompt_id",
            "prompt_domain",
            "prompt",
            "response",
            "system_prompt",
            "system_prompt_mode",
            "direction",
            "layer",
            "coefficient",
            "decoding",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
        }
    ]
    for (direction, layer, coefficient, decoding), bucket in grouped.items():
        row = {
            "direction": direction,
            "layer": layer,
            "coefficient": coefficient,
            "decoding": decoding,
            "count": len(bucket),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in bucket if item.get(field) is not None]
            row[field] = sum(values) / len(values) if values else None
        summary_rows.append(row)
    return summary_rows


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Judge steering outputs with a granularity rubric")
    parser.add_argument("--responses_file", type=str, required=True, help="Steering sweep JSONL file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--rubric_file",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "steering_judge_rubric.json"),
    )
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--requests_per_second", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base_url = normalize_base_url(args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_HOST"))
    if not args.dry_run and not api_key:
        raise SystemExit("No API key found. Set OPENAI_API_KEY/API_KEY or pass --api_key.")

    responses_file = Path(args.responses_file)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    rubric = load_json(Path(args.rubric_file))
    rows = load_jsonl(responses_file)

    scored_path = output_dir / "steering_judge_scores.jsonl"
    existing_rows = load_jsonl(scored_path) if scored_path.exists() else []
    existing_by_key = {row["record_key"]: row for row in existing_rows}

    pending_rows = [row for row in rows if make_record_key(row) not in existing_by_key]
    logger.info("Loaded %s steering rows (%s pending for judging)", len(rows), len(pending_rows))

    if args.dry_run:
        if pending_rows:
            sample_prompt = build_prompt(pending_rows[0], rubric)
            logger.info("Sample judge prompt:\n%s", sample_prompt)
        return

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    rate_limiter = RateLimiter(args.requests_per_second)

    prompts = [build_prompt(row, rubric) for row in pending_rows]
    if prompts:
        responses = await call_judge_batch(
            client=client,
            prompts=prompts,
            model=args.judge_model,
            max_tokens=args.max_tokens,
            rate_limiter=rate_limiter,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
    else:
        responses = []

    new_rows = []
    for row, response_text in tqdm(list(zip(pending_rows, responses)), desc="Parsing judge outputs"):
        parsed = parse_json_response(response_text, rubric)
        if parsed is None:
            logger.warning("Could not parse judge response for %s", make_record_key(row))
            continue
        merged = dict(row)
        merged.update(parsed)
        merged["record_key"] = make_record_key(row)
        new_rows.append(merged)

    final_rows = list(existing_by_key.values()) + new_rows
    with open(scored_path, "w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = aggregate_rows(final_rows) if final_rows else []
    with open(output_dir / "steering_judge_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %s judged rows to %s", len(final_rows), scored_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
