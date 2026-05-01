#!/usr/bin/env python3
"""
Steering experiment harness for the Granularity Axis.

Supports coefficient sweeps, layer sweeps, random and assistant-axis baselines,
greedy vs sampled decoding, and prompt files for larger steering studies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.steering import ActivationSteering

PROJECT_DIR = Path(__file__).parent.parent
from analysis_utils import ensure_dir, load_json, load_prompt_texts, load_vectors, normalize_vector


def load_axis(axis_file: str) -> torch.Tensor:
    """Load a saved axis tensor."""
    data = torch.load(axis_file, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "axis" in data:
        return data["axis"].float()
    if isinstance(data, torch.Tensor):
        return data.float()
    raise ValueError(f"Unexpected axis format in {axis_file}")


def load_default_system_prompt(roles_dir: Path) -> str:
    role_file = roles_dir / "default.json"
    if not role_file.exists():
        return "You are an AI assistant."
    data = load_json(role_file)
    for entry in data.get("instruction", []):
        text = entry.get("pos", "").strip()
        if text:
            return text
    return "You are an AI assistant."


def compute_assistant_axis(vectors_dir: Path, roles_dir: Path, metadata_file: Path | None) -> torch.Tensor:
    records = load_vectors(vectors_dir=vectors_dir, roles_dir=roles_dir, metadata_file=metadata_file)
    default_vectors = [record["vector"].float() for record in records if record.get("is_default")]
    role_vectors = [record["vector"].float() for record in records if not record.get("is_default")]
    if not default_vectors or not role_vectors:
        raise ValueError("Need both default and non-default vectors to compute the assistant baseline axis")
    return torch.stack(default_vectors, dim=0).mean(dim=0) - torch.stack(role_vectors, dim=0).mean(dim=0)


def build_random_axis(reference_axis: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_axis = torch.randn(reference_axis.shape, generator=generator)
    if random_axis.ndim == 2:
        random_axis = torch.stack([normalize_vector(layer) for layer in random_axis], dim=0)
    else:
        random_axis = normalize_vector(random_axis)
    return random_axis.float()


def parse_extra_axis(items: list[str] | None) -> dict[str, str]:
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH for --extra_axis, got: {item}")
        name, path = item.split("=", 1)
        result[name] = path
    return result


def parse_prompt_ids(items: list[str] | None) -> set[int] | None:
    if not items:
        return None

    prompt_ids: set[int] = set()
    for item in items:
        for chunk in str(item).split(","):
            value = chunk.strip()
            if value:
                prompt_ids.add(int(value))
    return prompt_ids or None


def build_conversation(prompt: str, system_prompt: str | None = None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict:
    conversation = build_conversation(prompt=prompt, system_prompt=system_prompt)

    chat_kwargs = {}
    if "qwen" in tokenizer.name_or_path.lower():
        chat_kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if decoding == "sample":
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    generated_ids = outputs[0][input_len:]
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = set()
    if isinstance(eos_token_id, (list, tuple)):
        eos_ids.update(int(item) for item in eos_token_id)
    elif eos_token_id is not None:
        eos_ids.add(int(eos_token_id))
    elif tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    ended_with_eos = bool(generated_ids.numel() and eos_ids and int(generated_ids[-1]) in eos_ids)
    generated_token_count = int(generated_ids.shape[0])
    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_token_count": generated_token_count,
        "hit_max_new_tokens": generated_token_count >= max_new_tokens and not ended_with_eos,
        "ended_with_eos": ended_with_eos,
    }


def load_prompt_rows(
    prompt: str | None,
    prompts_file: Path,
    max_prompts: int | None,
    prompt_ids: set[int] | None = None,
) -> list[dict]:
    if prompt:
        return [{"id": 0, "domain": "custom", "prompt": prompt}]
    rows = load_prompt_texts(prompts_file)
    if prompt_ids is not None:
        rows = [row for row in rows if int(row.get("id", -1)) in prompt_ids]
    if max_prompts is not None:
        rows = rows[:max_prompts]
    return rows


def get_system_prompt(mode: str, roles_dir: Path, custom_prompt: str | None) -> str | None:
    if mode == "none":
        return None
    if mode == "custom":
        if not custom_prompt:
            raise ValueError("--custom_system_prompt is required when --system_prompt_mode custom")
        return custom_prompt
    if mode == "default_assistant":
        return load_default_system_prompt(roles_dir)
    raise ValueError(f"Unsupported system prompt mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Steering experiment harness for Granularity Axis")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--axis_file", type=str, required=True, help="Path to granularity_axis.pt")
    parser.add_argument("--vectors_dir", type=str, default=None, help="Vectors dir for assistant-axis baseline")
    parser.add_argument("--roles_dir", type=str, default=str(PROJECT_DIR / "data" / "roles" / "instructions"))
    parser.add_argument("--metadata_file", type=str, default=str(PROJECT_DIR / "data" / "role_metadata.json"))
    parser.add_argument("--layers", type=int, nargs="+", default=[18], help="Layers to steer at")
    parser.add_argument("--coeff", type=float, default=4.0, help="Fallback steering magnitude if --coeffs is omitted")
    parser.add_argument("--coeffs", type=float, nargs="+", default=None, help="Explicit coefficient sweep")
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["granularity"],
        choices=["granularity", "assistant", "random"],
        help="Built-in steering directions to evaluate",
    )
    parser.add_argument("--extra_axis", action="append", default=None, help="Additional custom axes as NAME=PATH")
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to test")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=str(PROJECT_DIR / "data" / "steering_prompts.jsonl"),
        help="Prompt set for steering sweeps",
    )
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument(
        "--prompt_ids",
        nargs="+",
        default=None,
        help="Optional prompt ids to keep, e.g. --prompt_ids 0 3 7 or --prompt_ids 0,3,7",
    )
    parser.add_argument("--system_prompt_mode", choices=["none", "default_assistant", "custom"], default="none")
    parser.add_argument("--custom_system_prompt", type=str, default=None)
    parser.add_argument("--decoding", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, required=True, help="Save sweep results to JSONL")
    args = parser.parse_args()

    coeffs = args.coeffs if args.coeffs is not None else [-args.coeff, 0.0, args.coeff]
    coeffs = [float(value) for value in coeffs]
    prompt_ids = parse_prompt_ids(args.prompt_ids)
    roles_dir = Path(args.roles_dir)
    metadata_file = Path(args.metadata_file) if args.metadata_file else None

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        if args.decoding == "greedy":
            model.generation_config.temperature = None
            model.generation_config.top_k = None
            model.generation_config.top_p = None

    print(f"Loading axis: {args.axis_file}")
    reference_axis = load_axis(args.axis_file)
    print(f"  Axis shape: {reference_axis.shape}")

    direction_map = {"granularity": reference_axis}
    if "assistant" in args.directions:
        if not args.vectors_dir:
            raise ValueError("--vectors_dir is required when using the assistant baseline direction")
        direction_map["assistant"] = compute_assistant_axis(
            vectors_dir=Path(args.vectors_dir),
            roles_dir=roles_dir,
            metadata_file=metadata_file,
        )
    if "random" in args.directions:
        direction_map["random"] = build_random_axis(reference_axis, seed=args.seed)

    for name, path in parse_extra_axis(args.extra_axis).items():
        direction_map[name] = load_axis(path)

    built_in = set(args.directions)
    directions_to_run = list(dict.fromkeys(list(built_in) + list(parse_extra_axis(args.extra_axis).keys())))

    prompts = load_prompt_rows(
        prompt=args.prompt,
        prompts_file=Path(args.prompts_file),
        max_prompts=args.max_prompts,
        prompt_ids=prompt_ids,
    )
    if not prompts:
        raise ValueError("No prompts selected. Check --prompt_ids / --max_prompts / --prompts_file.")
    system_prompt = get_system_prompt(
        mode=args.system_prompt_mode,
        roles_dir=roles_dir,
        custom_prompt=args.custom_system_prompt,
    )

    records = []
    ensure_dir(Path(args.output_file).parent)

    for prompt_row in prompts:
        prompt_id = prompt_row.get("id")
        prompt_domain = prompt_row.get("domain", "unknown")
        prompt_text = prompt_row["prompt"]

        baseline_result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            decoding=args.decoding,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_domain": prompt_domain,
                "prompt": prompt_text,
                "system_prompt_mode": args.system_prompt_mode,
                "system_prompt": system_prompt,
                "direction": "baseline",
                "layer": None,
                "coefficient": 0.0,
                "decoding": args.decoding,
                "temperature": args.temperature if args.decoding == "sample" else None,
                "top_p": args.top_p if args.decoding == "sample" else None,
                "top_k": args.top_k if args.decoding == "sample" else None,
                "max_new_tokens": args.max_new_tokens,
                "response": baseline_result["text"],
                "generated_token_count": baseline_result["generated_token_count"],
                "hit_max_new_tokens": baseline_result["hit_max_new_tokens"],
                "ended_with_eos": baseline_result["ended_with_eos"],
            }
        )

        print(f"\nPrompt {prompt_id}: {prompt_text}")
        print(f"  Baseline: {baseline_result['text'][:160].replace(chr(10), ' ')}")

        for direction_name in directions_to_run:
            direction_axis = direction_map[direction_name].float()
            for layer in args.layers:
                if direction_axis.ndim == 1:
                    steering_vector = normalize_vector(direction_axis)
                else:
                    steering_vector = normalize_vector(direction_axis[layer])

                for coeff in coeffs:
                    if coeff == 0.0:
                        continue

                    with ActivationSteering(
                        model=model,
                        steering_vectors=[steering_vector],
                        coefficients=[coeff],
                        layer_indices=[layer],
                        intervention_type="addition",
                        positions="all",
                    ):
                        result = generate(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_text,
                            system_prompt=system_prompt,
                            max_new_tokens=args.max_new_tokens,
                            decoding=args.decoding,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )

                    print(
                        f"  {direction_name}@L{layer} coeff={coeff:+.2f}: "
                        f"{result['text'][:120].replace(chr(10), ' ')}"
                    )
                    records.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_domain": prompt_domain,
                            "prompt": prompt_text,
                            "system_prompt_mode": args.system_prompt_mode,
                            "system_prompt": system_prompt,
                            "direction": direction_name,
                            "layer": layer,
                            "coefficient": coeff,
                            "decoding": args.decoding,
                            "temperature": args.temperature if args.decoding == "sample" else None,
                            "top_p": args.top_p if args.decoding == "sample" else None,
                            "top_k": args.top_k if args.decoding == "sample" else None,
                            "max_new_tokens": args.max_new_tokens,
                            "response": result["text"],
                            "generated_token_count": result["generated_token_count"],
                            "hit_max_new_tokens": result["hit_max_new_tokens"],
                            "ended_with_eos": result["ended_with_eos"],
                        }
                    )

    with open(args.output_file, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("STEERING SWEEP COMPLETE")
    print("=" * 70)
    print(f"Model:         {args.model}")
    print(f"Prompts:       {len(prompts)}")
    print(f"Directions:    {sorted(direction_map.keys())}")
    print(f"Layers:        {args.layers}")
    print(f"Coefficients:  {coeffs}")
    print(f"Decoding:      {args.decoding}")
    print(f"Output file:   {args.output_file}")
    print(f"Records:       {len(records)}")


if __name__ == "__main__":
    main()
