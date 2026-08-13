#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — Dataset Preparation
A Project by Osama

Downloads instruction datasets from HuggingFace, formats them into the
Qwen chat template, splits into train/validation sets, and saves them
to disk with full statistics.

Usage:
    python prepare_dataset.py --dataset tatsu-lab/alpaca --max_samples 10000
    python prepare_dataset.py --dataset teknium/OpenHermes-2.5 --max_samples 50000
    python prepare_dataset.py --dataset custom_data.jsonl --format alpaca
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

# Type aliases
Record = dict[str, Any]

# Qwen 2.5 chat template format
QWEN_CHAT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as helpfully as possible while being safe. "
    "If you don't know the answer, say so honestly."
)

# Known dataset format mappings
DATASET_FORMATS: dict[str, str] = {
    "tatsu-lab/alpaca": "alpaca",
    "StanfordSchnell/AlpacaDatasetCleaned": "alpaca",
    "teknium/OpenHermes-2.5": "openhermes",
    "HuggingFaceH4/ultrachat_200k": "sharegpt",
    "tatsu-lab/alpaca_cleaned": "alpaca",
    "vicgalle/alpaca-gpt4": "alpaca",
    "allenai/dolma": "dolma",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare instruction datasets for NexusAI fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_dataset.py --dataset tatsu-lab/alpaca --max_samples 10000
  python prepare_dataset.py --dataset teknium/OpenHermes-2.5 --output_dir data/hermes
  python prepare_dataset.py --dataset my_data.jsonl --format alpaca --local_file
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="HuggingFace dataset name or path to local JSONL file (default: tatsu-lab/alpaca)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "openhermes", "sharegpt", "auto"],
        default="auto",
        help="Dataset format. 'auto' detects from dataset name (default: auto)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum number of samples to keep (default: 50000, -1 for all)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for processed datasets (default: data/)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"System prompt for all samples (default: '{DEFAULT_SYSTEM_PROMPT}')",
    )
    parser.add_argument(
        "--min_instruction_length",
        type=int,
        default=3,
        help="Minimum instruction character length to keep (default: 3)",
    )
    parser.add_argument(
        "--min_response_length",
        type=int,
        default=5,
        help="Minimum response character length to keep (default: 5)",
    )
    parser.add_argument(
        "--local_file",
        action="store_true",
        help="Treat --dataset as a local file path instead of HuggingFace name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def detect_format(dataset_name: str) -> str:
    """Auto-detect the dataset format from the name."""
    for name, fmt in DATASET_FORMATS.items():
        if name.lower() in dataset_name.lower():
            return fmt
    # Try heuristic matching
    if "alpaca" in dataset_name.lower():
        return "alpaca"
    if "hermes" in dataset_name.lower():
        return "openhermes"
    if "sharegpt" in dataset_name.lower() or "ultrachat" in dataset_name.lower():
        return "sharegpt"
    return "alpaca"  # Default fallback


def load_local_jsonl(path: str) -> list[Record]:
    """Load records from a local JSONL file."""
    records: list[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    warnings.warn(f"Skipping malformed JSON line: {e}")
    return records


def format_alpaca(
    record: Record,
    system_prompt: str,
) -> tuple[str, str, str] | None:
    """Format a record from Alpaca-style datasets.

    Expected fields: instruction, input (optional), output
    Returns (system, instruction, response) or None if invalid.
    """
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    response = record.get("output", "").strip()

    if not instruction or not response:
        return None

    # Combine instruction and input if input is present
    if input_text:
        instruction = f"{instruction}\n\n### Input:\n{input_text}"

    return (system_prompt, instruction, response)


def format_openhermes(
    record: Record,
    system_prompt: str,
) -> tuple[str, str, str] | None:
    """Format a record from OpenHermes-style datasets.

    Expected fields: conversations (list of {role, content})
    Returns (system, instruction, response) or None if invalid.
    """
    conversations = record.get("conversations", [])
    if not conversations:
        return None

    system = system_prompt
    instruction_parts: list[str] = []
    response_parts: list[str] = []

    for turn in conversations:
        role = turn.get("role", "").lower()
        content = turn.get("content", "").strip()
        if not content:
            continue

        if role == "system":
            system = content
        elif role == "user" or role == "human":
            instruction_parts.append(content)
        elif role == "assistant" or role == "gpt":
            response_parts.append(content)

    if not instruction_parts or not response_parts:
        return None

    # Use the last user-assistant pair
    instruction = instruction_parts[-1]
    response = response_parts[-1]

    return (system, instruction, response)


def format_sharegpt(
    record: Record,
    system_prompt: str,
) -> tuple[str, str, str] | None:
    """Format a record from ShareGPT-style datasets.

    Expected fields: conversations with human/gpt roles
    Returns (system, instruction, response) or None if invalid.
    """
    # ShareGPT format can vary; handle common variants
    conv = record.get("conversations", [])
    if not conv:
        return None

    system = system_prompt
    instruction = ""
    response = ""

    for turn in conv:
        role = turn.get("from", turn.get("role", "")).lower()
        value = turn.get("value", turn.get("content", "")).strip()

        if not value:
            continue

        if role == "system":
            system = value
        elif role in ("human", "user"):
            instruction = value
        elif role in ("gpt", "assistant", "bot"):
            response = value

    if not instruction or not response:
        return None

    return (system, instruction, response)


FORMATTERS = {
    "alpaca": format_alpaca,
    "openhermes": format_openhermes,
    "sharegpt": format_sharegpt,
}


def apply_chat_template(
    system: str,
    instruction: str,
    response: str,
    template: str = QWEN_CHAT_TEMPLATE,
) -> str:
    """Apply the chat template to produce the full training text."""
    return template.format(
        system=system,
        instruction=instruction,
        response=response,
    )


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def process_dataset(
    raw_data: Dataset | list[Record],
    fmt: str,
    system_prompt: str,
    min_instruction_length: int,
    min_response_length: int,
    max_samples: int,
) -> list[Record]:
    """Process raw data into formatted records."""
    formatter = FORMATTERS.get(fmt, format_alpaca)
    processed: list[Record] = []
    skipped = 0

    iterable = raw_data if isinstance(raw_data, list) else raw_data
    total = min(len(iterable), max_samples) if max_samples > 0 else len(iterable)

    for record in tqdm(iterable, total=total, desc="Processing records"):
        if max_samples > 0 and len(processed) >= max_samples:
            break

        # Convert Dataset row to dict if needed
        if hasattr(record, "items"):
            record = dict(record)

        result = formatter(record, system_prompt)
        if result is None:
            skipped += 1
            continue

        system, instruction, response = result

        # Clean texts
        instruction = clean_text(instruction)
        response = clean_text(response)

        # Filter by length
        if len(instruction) < min_instruction_length:
            skipped += 1
            continue
        if len(response) < min_response_length:
            skipped += 1
            continue

        full_text = apply_chat_template(system, instruction, response)

        processed.append(
            {
                "system": system,
                "instruction": instruction,
                "response": response,
                "text": full_text,
            }
        )

    print(f"  Skipped {skipped} invalid records")
    return processed


def compute_statistics(records: list[Record]) -> dict[str, Any]:
    """Compute dataset statistics."""
    import statistics

    if not records:
        return {"error": "No records to analyze"}

    instruction_lengths = [len(r["instruction"]) for r in records]
    response_lengths = [len(r["response"]) for r in records]
    text_lengths = [len(r["text"]) for r in records]

    # Rough token estimate (1 token ≈ 4 chars for English)
    text_tokens = [l // 4 for l in text_lengths]

    return {
        "num_samples": len(records),
        "instruction_length": {
            "min": min(instruction_lengths),
            "max": max(instruction_lengths),
            "mean": round(statistics.mean(instruction_lengths), 1),
            "median": round(statistics.median(instruction_lengths), 1),
        },
        "response_length": {
            "min": min(response_lengths),
            "max": max(response_lengths),
            "mean": round(statistics.mean(response_lengths), 1),
            "median": round(statistics.median(response_lengths), 1),
        },
        "estimated_tokens": {
            "min": min(text_tokens),
            "max": max(text_tokens),
            "mean": round(statistics.mean(text_tokens), 1),
            "median": round(statistics.median(text_tokens), 1),
            "total": sum(text_tokens),
        },
    }


def print_statistics(stats: dict[str, Any], label: str = "Dataset") -> None:
    """Print formatted dataset statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {label} Statistics")
    print(f"{'=' * 60}")
    print(f"  Samples:             {stats.get('num_samples', 'N/A')}")
    print(f"  Est. total tokens:   {stats.get('estimated_tokens', {}).get('total', 'N/A'):,}")
    print()
    print(f"  Instruction Length:")
    il = stats.get("instruction_length", {})
    print(f"    Min:  {il.get('min', 'N/A'):>8,} chars")
    print(f"    Max:  {il.get('max', 'N/A'):>8,} chars")
    print(f"    Mean: {il.get('mean', 'N/A'):>8.1f} chars")
    print(f"    Med:  {il.get('median', 'N/A'):>8.1f} chars")
    print()
    print(f"  Response Length:")
    rl = stats.get("response_length", {})
    print(f"    Min:  {rl.get('min', 'N/A'):>8,} chars")
    print(f"    Max:  {rl.get('max', 'N/A'):>8,} chars")
    print(f"    Mean: {rl.get('mean', 'N/A'):>8.1f} chars")
    print(f"    Med:  {rl.get('median', 'N/A'):>8.1f} chars")
    print()
    print(f"  Estimated Token Count:")
    et = stats.get("estimated_tokens", {})
    print(f"    Min:  {et.get('min', 'N/A'):>8,} tokens")
    print(f"    Max:  {et.get('max', 'N/A'):>8,} tokens")
    print(f"    Mean: {et.get('mean', 'N/A'):>8.1f} tokens")
    print(f"    Med:  {et.get('median', 'N/A'):>8.1f} tokens")
    print(f"{'=' * 60}\n")


def save_dataset(
    records: list[Record],
    output_path: str,
) -> None:
    """Save processed records to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI Dataset Preparation")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    # Determine format
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(args.dataset)
    print(f"  Dataset:  {args.dataset}")
    print(f"  Format:   {fmt}")
    print(f"  Max samples: {args.max_samples if args.max_samples > 0 else 'All'}")
    print()

    # Load dataset
    if args.local_file:
        print(f"  Loading local file: {args.dataset}")
        if not os.path.exists(args.dataset):
            print(f"  ERROR: File not found: {args.dataset}")
            return 1
        raw_data = load_local_jsonl(args.dataset)
        print(f"  Loaded {len(raw_data)} records")
    else:
        print(f"  Downloading from HuggingFace: {args.dataset}")
        try:
            raw_data = load_dataset(args.dataset, split="train", trust_remote_code=True)
            print(f"  Downloaded {len(raw_data)} records")
        except Exception as e:
            print(f"  ERROR: Failed to load dataset: {e}")
            print(f"  Hint: Check the dataset name or use --local_file for local data")
            return 1

    if not raw_data:
        print("  ERROR: No data loaded!")
        return 1

    # Process records
    print("\n  Processing records...")
    processed = process_dataset(
        raw_data,
        fmt=fmt,
        system_prompt=args.system_prompt,
        min_instruction_length=args.min_instruction_length,
        min_response_length=args.min_response_length,
        max_samples=args.max_samples,
    )

    if not processed:
        print("  ERROR: No valid records after processing!")
        return 1

    # Compute and print statistics
    all_stats = compute_statistics(processed)
    print_statistics(all_stats, "Full Dataset")

    # Split into train/val
    import random
    random.seed(args.seed)
    random.shuffle(processed)

    val_count = int(len(processed) * args.val_split)
    train_count = len(processed) - val_count
    val_records = processed[:val_count]
    train_records = processed[val_count:]

    print(f"  Split: {train_count} train / {val_count} validation")

    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(output_dir / "train.jsonl")
    val_path = str(output_dir / "val.jsonl")
    stats_path = str(output_dir / "dataset_stats.json")

    print("\n  Saving datasets...")
    save_dataset(train_records, train_path)
    save_dataset(val_records, val_path)

    # Save statistics
    train_stats = compute_statistics(train_records)
    val_stats = compute_statistics(val_records)
    stats_out = {
        "source_dataset": args.dataset,
        "format": fmt,
        "total_samples": len(processed),
        "train_samples": train_count,
        "val_samples": val_count,
        "val_split": args.val_split,
        "train_statistics": train_stats,
        "val_statistics": val_stats,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"  Saved statistics to {stats_path}")

    print_statistics(train_stats, "Training Set")
    print_statistics(val_stats, "Validation Set")

    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print(f"\n  Next step: Fine-tune the model")
    print(f"    python finetune_qlora.py --config finetune_config.yaml")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
