#!/usr/bin/env python3
"""
Extract activations from response JSONL files.

Follows the exact methodology from Lu et al. (2026):
  - For each role's responses, extract mean response activations at all layers
  - Saves per-role .pt files with activation tensors

Usage:
    python pipeline/2_activations.py \
        --model /path/to/Qwen3-8B \
        --responses_dir outputs/qwen3-8b/responses \
        --output_dir outputs/qwen3-8b/activations \
        --batch_size 8
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import jsonlines
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.internals import ProbingModel, ConversationEncoder, ActivationExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_responses(responses_file: Path) -> list:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def pool_span_activations(
    span_activations: torch.Tensor,
    pooling: str,
    k_tokens: int,
) -> Optional[torch.Tensor]:
    """
    Pool activations inside one assistant span.

    Args:
        span_activations: Tensor of shape (num_layers, span_length, hidden_size)
    """
    span_length = span_activations.size(1)
    if span_length == 0:
        return None

    if pooling == "mean":
        return span_activations.mean(dim=1)
    if pooling == "last":
        return span_activations[:, -1, :]
    if pooling == "first":
        return span_activations[:, 0, :]
    if pooling == "first_k":
        return span_activations[:, : min(k_tokens, span_length), :].mean(dim=1)
    if pooling == "last_k":
        return span_activations[:, -min(k_tokens, span_length):, :].mean(dim=1)

    raise ValueError(f"Unsupported pooling mode: {pooling}")


def extract_assistant_span_activations(
    batch_activations: torch.Tensor,
    batch_spans: list,
    batch_metadata: dict,
    pooling: str,
    k_tokens: int,
) -> List[Optional[torch.Tensor]]:
    """
    Convert batched token activations into one pooled representation per conversation.
    """
    total_conversations = batch_metadata["total_conversations"]
    spans_by_conversation = {idx: [] for idx in range(total_conversations)}
    for span in batch_spans:
        if span.get("role") == "assistant":
            spans_by_conversation[span["conversation_id"]].append(span)

    pooled_conversations: List[Optional[torch.Tensor]] = []
    for conv_id in range(total_conversations):
        assistant_spans = sorted(spans_by_conversation.get(conv_id, []), key=lambda item: item["turn"])
        pooled_turns = []
        actual_length = batch_metadata["truncated_lengths"][conv_id]

        for span in assistant_spans:
            start_idx = span["start"]
            end_idx = min(span["end"], actual_length)
            if start_idx >= actual_length or start_idx >= end_idx:
                continue

            span_acts = batch_activations[:, conv_id, start_idx:end_idx, :]
            pooled = pool_span_activations(span_acts, pooling=pooling, k_tokens=k_tokens)
            if pooled is not None:
                pooled_turns.append(pooled)

        if not pooled_turns:
            pooled_conversations.append(None)
        elif len(pooled_turns) == 1:
            pooled_conversations.append(pooled_turns[0].cpu())
        elif pooling in {"last", "last_k"}:
            pooled_conversations.append(pooled_turns[-1].cpu())
        elif pooling in {"first", "first_k"}:
            pooled_conversations.append(pooled_turns[0].cpu())
        else:
            pooled_conversations.append(torch.stack(pooled_turns, dim=0).mean(dim=0).cpu())

    return pooled_conversations


def extract_activations_batch(
    pm: ProbingModel,
    conversations: list,
    layers: List[int],
    batch_size: int = 8,
    max_length: int = 2048,
    enable_thinking: bool = False,
    pooling: str = "mean",
    k_tokens: int = 16,
) -> List[Optional[torch.Tensor]]:
    """Extract mean response activations for a batch of conversations."""
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)

    chat_kwargs = {}
    if 'qwen' in pm.model_name.lower():
        chat_kwargs['enable_thinking'] = enable_thinking

    all_activations = []
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        batch_activations, batch_metadata = extractor.batch_conversations(
            batch_conversations,
            layer=layers,
            max_length=max_length,
            **chat_kwargs,
        )

        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(
            batch_conversations, **chat_kwargs
        )

        conv_activations_list = extract_assistant_span_activations(
            batch_activations=batch_activations,
            batch_spans=batch_spans,
            batch_metadata=batch_metadata,
            pooling=pooling,
            k_tokens=k_tokens,
        )

        for conv_acts in conv_activations_list:
            if conv_acts is None or conv_acts.numel() == 0:
                all_activations.append(None)
            else:
                all_activations.append(conv_acts)

        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def process_role(
    pm: ProbingModel,
    role_file: Path,
    output_dir: Path,
    layers: List[int],
    batch_size: int,
    max_length: int,
    enable_thinking: bool = False,
    pooling: str = "mean",
    k_tokens: int = 16,
) -> bool:
    """Process a single role file and save activations."""
    role = role_file.stem
    output_file = output_dir / f"{role}.pt"

    responses = load_responses(role_file)
    if not responses:
        return False

    conversations = []
    metadata = []
    for resp in responses:
        conversations.append(resp["conversation"])
        metadata.append({
            "prompt_index": resp["prompt_index"],
            "question_index": resp["question_index"],
            "label": resp["label"],
        })

    logger.info(f"Processing {role}: {len(conversations)} conversations")

    activations_list = extract_activations_batch(
        pm=pm,
        conversations=conversations,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
        enable_thinking=enable_thinking,
        pooling=pooling,
        k_tokens=k_tokens,
    )

    activations_dict = {}
    for i, (act, meta) in enumerate(zip(activations_list, metadata)):
        if act is not None:
            key = f"{meta['label']}_p{meta['prompt_index']}_q{meta['question_index']}"
            activations_dict[key] = act

    if activations_dict:
        torch.save(activations_dict, output_file)
        logger.info(f"Saved {len(activations_dict)} activations for {role}")

    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract activations from responses")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to extract: 'all' or comma-separated (e.g. '8,16,24')")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--pooling", choices=["mean", "last", "first", "first_k", "last_k"],
                        default="mean", help="Pooling strategy over assistant response tokens")
    parser.add_argument("--k_tokens", type=int, default=16,
                        help="Token count used by first_k / last_k pooling")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--thinking", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False, help="Enable thinking mode for Qwen3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = Path(args.responses_dir)

    # Load model
    logger.info(f"Loading model: {args.model}")
    pm = ProbingModel(args.model)

    # Determine layers
    n_layers = len(pm.get_layers())
    logger.info(f"Model has {n_layers} layers")

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger.info(f"Extracting {len(layers)} layers")
    logger.info(f"Pooling mode: {args.pooling} (k_tokens={args.k_tokens})")

    # Get response files
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    # Filter out existing
    role_files = []
    for f in response_files:
        output_file = output_dir / f"{f.stem}.pt"
        if output_file.exists():
            logger.info(f"Skipping {f.stem} (already exists)")
            continue
        role_files.append(f)

    logger.info(f"Processing {len(role_files)} roles")

    for role_file in tqdm(role_files, desc="Processing roles"):
        process_role(pm, role_file, output_dir, layers, args.batch_size,
                     args.max_length, args.thinking, args.pooling, args.k_tokens)

    config_file = output_dir / "activation_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        import json
        json.dump(
            {
                "model": args.model,
                "layers": layers,
                "pooling": args.pooling,
                "k_tokens": args.k_tokens,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "thinking": args.thinking,
            },
            f,
            indent=2,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()

