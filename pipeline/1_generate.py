#!/usr/bin/env python3
"""
Generate model responses for all granularity roles using vLLM batch inference.

Follows the exact methodology from Lu et al. (2026) "The Assistant Axis":
  - For each role, generate responses to extraction questions
  - 5 system prompt variants × N questions per role
  - Saves one JSONL file per role

Usage:
    python pipeline/1_generate.py \
        --model /path/to/Qwen3-8B \
        --output_dir outputs/qwen3-8b/responses

    # Test with specific roles
    python pipeline/1_generate.py \
        --model /path/to/Qwen3-8B \
        --output_dir outputs/qwen3-8b/responses \
        --roles angry_protester default
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_DIR = Path(__file__).parent.parent  # pipeline -> granularity

from lib.generation import RoleResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate role responses for granularity experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name or path')
    parser.add_argument('--roles_dir', type=str,
                        default=str(PROJECT_DIR / "data" / "roles" / "instructions"),
                        help='Directory containing role JSON files')
    parser.add_argument('--questions_file', type=str,
                        default=str(PROJECT_DIR / "data" / "extraction_questions.jsonl"),
                        help='Path to questions JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for response JSONL files')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='Maximum model context length')
    parser.add_argument('--tensor_parallel_size', type=int, default=None,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                        help='GPU memory utilization')
    parser.add_argument('--question_count', type=int, default=240,
                        help='Number of questions per role (default: 240, same as assistant-axis)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=1536,
                        help='Maximum tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling')
    parser.add_argument('--roles', nargs='+',
                        help='Specific role slugs to process (default: all)')

    args = parser.parse_args()

    # Validate paths
    roles_dir = Path(args.roles_dir)
    if not roles_dir.exists():
        logger.error(f"Roles directory not found: {roles_dir}")
        logger.error("Run pipeline/0_prepare_roles.py first!")
        sys.exit(1)

    questions_file = Path(args.questions_file)
    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count available roles
    role_count = len(list(roles_dir.glob("*.json")))
    logger.info(f"Found {role_count} role files in {roles_dir}")

    # Detect GPUs
    import torch
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        total_gpus = len([x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()])
    else:
        total_gpus = torch.cuda.device_count()

    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus
    logger.info(f"Using {tensor_parallel_size} GPU(s)")

    # Infer short name for {model_name} placeholder
    model_lower = args.model.lower()
    if "qwen" in model_lower:
        short_name = "Qwen"
    elif "llama" in model_lower:
        short_name = "Llama"
    elif "gemma" in model_lower:
        short_name = "Gemma"
    else:
        short_name = Path(args.model).name.split("-")[0]

    # Create generator
    generator = RoleResponseGenerator(
        model_name=args.model,
        roles_dir=args.roles_dir,
        output_dir=args.output_dir,
        questions_file=args.questions_file,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        question_count=args.question_count,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        short_name=short_name,
    )

    # Process all roles
    generator.process_all_roles(
        skip_existing=True,
        roles=args.roles,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()

