#!/usr/bin/env python3
"""
Score role responses using a judge LLM.

Follows the role-adherence judge scoring methodology (Lu et al. 2026):
  Score how well model responses adhere to their assigned roles (0-3 scale).

Usage:
    OPENAI_API_KEY=... python pipeline/3_judge.py \
        --responses_dir outputs/qwen3-8b/responses \
        --output_dir outputs/qwen3-8b/scores

    # Dry run to preview
    python pipeline/3_judge.py \
        --responses_dir outputs/qwen3-8b/responses \
        --output_dir outputs/qwen3-8b/scores \
        --dry_run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)

from lib.judge import RateLimiter, call_judge_batch, parse_judge_score
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> list:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


async def process_role(
    role: str,
    responses: list,
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
    temperature: float,
) -> dict:
    """Process a single role and return scores."""
    prompts = []
    keys = []

    for resp in responses:
        prompt_idx = resp["prompt_index"]
        question_idx = resp["question_index"]
        question = resp["question"]
        label = resp["label"]

        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        key = f"{label}_p{prompt_idx}_q{question_idx}"
        if key in existing_scores:
            continue

        judge_prompt = eval_prompt_template.format(
            question=question, answer=assistant_response
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {}

    logger.info(f"Scoring {len(prompts)} new responses for {role}...")
    responses_text = await call_judge_batch(
        client=client, prompts=prompts, model=judge_model,
        max_tokens=max_tokens, rate_limiter=rate_limiter, batch_size=batch_size, temperature=temperature
    )

    scores = {}
    for key, response_text in zip(keys, responses_text):
        if response_text:
            score = parse_judge_score(response_text)
            if score is not None:
                scores[key] = score

    return scores


async def main_async():
    parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--roles_dir", type=str,
                        default=str(PROJECT_DIR / "data" / "roles" / "instructions"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--requests_per_second", type=int, default=100)
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base_url = normalize_base_url(args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_HOST"))
    if not args.dry_run and not api_key:
        logger.error("No API key found. Set OPENAI_API_KEY/API_KEY or pass --api_key.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)

    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    if args.dry_run:
        total = 0
        for rf in response_files:
            role = rf.stem
            role_file = roles_dir / f"{role}.json"
            if not role_file.exists():
                continue
            eval_prompt = load_role_eval_prompt(role_file)
            if not eval_prompt:
                continue
            responses = load_responses(rf)
            total += len(responses)
            logger.info(f"  {role}: {len(responses)} responses")
        logger.info(f"\nTotal responses to score: {total}")
        return

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    rate_limiter = RateLimiter(args.requests_per_second)

    successful = 0
    for response_file in tqdm(response_files, desc="Scoring roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.json"

        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            continue

        responses = load_responses(response_file)
        if not responses:
            continue

        try:
            new_scores = await process_role(
                role=role, responses=responses,
                eval_prompt_template=eval_prompt_template,
                client=client, rate_limiter=rate_limiter,
                judge_model=args.judge_model, max_tokens=args.max_tokens,
                batch_size=args.batch_size, existing_scores=existing_scores,
                temperature=args.temperature,
            )
            all_scores = {**existing_scores, **new_scores}
            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)
            logger.info(f"Saved {len(all_scores)} scores for {role}")
            successful += 1
        except Exception as e:
            logger.error(f"Error processing {role}: {e}")

    logger.info(f"Done! {successful} roles scored.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

