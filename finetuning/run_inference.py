#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — Inference Testing
A Project by Osama

Interactive and batch inference testing for fine-tuned models.
Supports comparing outputs between base and fine-tuned models,
measuring latency, and logging conversations.

Usage:
    # Interactive chat mode
    python run_inference.py --model_path outputs/merged

    # Batch inference from a file
    python run_inference.py --model_path outputs/merged --input_file prompts.txt --output_file results.jsonl

    # Compare base vs fine-tuned
    python run_inference.py --model_path outputs/merged --compare_with_base

    # Use with LoRA adapter
    python run_inference.py --adapter_path outputs/qlora-adapters/final_adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexusai.inference")

SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as helpfully as possible while being safe."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned NexusAI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a full merged model",
    )
    model_group.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (base model auto-detected)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Base model name (for adapter loading, default: Qwen/Qwen2.5-7B)",
    )

    # Inference parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")

    # Loading
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load in 4-bit")
    parser.add_argument("--use_flash_attention", action="store_true", default=False, help="Use Flash Attention 2")

    # Modes
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File with prompts (one per line) for batch mode",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for batch results (JSONL format)",
    )
    parser.add_argument(
        "--compare_with_base",
        action="store_true",
        help="Compare outputs between base and fine-tuned models",
    )
    parser.add_argument(
        "--num_benchmark_runs",
        type=int,
        default=5,
        help="Number of benchmark runs for latency measurement",
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        default=True,
        help="Save conversation logs to logs/ directory",
    )

    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    """Load model and tokenizer based on arguments. Returns (model, tokenizer, model_label)."""
    if args.model_path:
        model_path = args.model_path
        model_label = f"merged ({model_path})"
        logger.info(f"Loading merged model: {model_path}")

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        }

        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        if args.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    elif args.adapter_path:
        model_label = f"LoRA ({args.adapter_path})"
        logger.info(f"Loading base model: {args.base_model}")

        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        }

        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)

        logger.info(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        raise ValueError("Provide --model_path or --adapter_path")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, model_label


def format_prompt(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    """Format an instruction into the Qwen chat template."""
    return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


def generate_response(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    """Generate a response and measure latency."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    end_time = time.perf_counter()
    latency = end_time - start_time

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_generated = generated_ids.shape[0]
    tokens_per_sec = num_generated / latency if latency > 0 else 0

    return {
        "instruction": instruction,
        "response": response,
        "latency_seconds": round(latency, 3),
        "tokens_generated": num_generated,
        "tokens_per_second": round(tokens_per_sec, 2),
    }


def interactive_chat(args: argparse.Namespace, model: Any, tokenizer: Any, model_label: str) -> None:
    """Run an interactive chat loop."""
    print(f"\n{'=' * 60}")
    print(f"  NexusAI Interactive Chat")
    print(f"  Model: {model_label}")
    print(f"  Type 'quit' or 'exit' to stop")
    print(f"  Type 'benchmark' to run latency benchmark")
    print(f"{'=' * 60}\n")

    conversation_history: list[dict[str, str]] = []

    while True:
        try:
            instruction = input("\n\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not instruction:
            continue
        if instruction.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if instruction.lower() == "benchmark":
            run_benchmark(model, tokenizer, args.num_benchmark_runs, args)
            continue

        if instruction.lower() == "clear":
            conversation_history.clear()
            print("\n  [Conversation history cleared]")
            continue

        print("\n\033[1mAssistant:\033[0m", end=" ", flush=True)

        result = generate_response(
            model, tokenizer, instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_seq_length=args.max_seq_length,
        )

        print(result["response"])
        print(f"  \033[90m[{result['latency_seconds']:.2f}s | "
              f"{result['tokens_generated']} tokens | "
              f"{result['tokens_per_second']:.1f} tok/s]\033[0m")

        conversation_history.append(result)

    # Save logs
    if args.save_logs and conversation_history:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/chat_{timestamp}.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in conversation_history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n  Conversation saved to {log_path}")


def batch_inference(args: argparse.Namespace, model: Any, tokenizer: Any, model_label: str) -> None:
    """Run batch inference from a file of prompts."""
    if not args.input_file:
        logger.error("--input_file required for batch mode")
        return

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Read prompts
    with open(args.input_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(prompts)} prompts from {args.input_file}")

    results: list[dict[str, Any]] = []
    total_latency = 0.0
    total_tokens = 0

    for prompt in tqdm(prompts, desc="Generating"):
        result = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_seq_length=args.max_seq_length,
        )
        results.append(result)
        total_latency += result["latency_seconds"]
        total_tokens += result["tokens_generated"]

    # Print summary
    avg_latency = total_latency / len(results) if results else 0
    avg_tok_sec = total_tokens / total_latency if total_latency > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  BATCH INFERENCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Prompts processed:  {len(results)}")
    print(f"  Total latency:      {total_latency:.2f}s")
    print(f"  Avg latency:        {avg_latency:.2f}s")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Avg tokens/sec:     {avg_tok_sec:.2f}")
    print(f"{'=' * 60}")

    # Save results
    output_path = args.output_file or f"logs/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


def compare_models(args: argparse.Namespace, tokenizer: Any) -> None:
    """Compare outputs between base and fine-tuned models."""
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to check if a string is a palindrome.",
        "What are the main differences between TCP and UDP?",
        "Summarize the key principles of machine learning.",
        "Write a haiku about programming.",
    ]

    import gc

    # Load fine-tuned model
    logger.info("Loading fine-tuned model...")
    ft_model, _, ft_label = load_model(args)

    # Load base model
    logger.info("Loading base model for comparison...")
    del ft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if args.load_in_4bit:
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
    base_model.eval()

    # Re-load fine-tuned model
    ft_model, _, _ = load_model(args)

    print(f"\n{'=' * 70}")
    print(f"  BASE vs FINE-TUNED COMPARISON")
    print(f"{'=' * 70}")

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'─' * 70}")
        print(f"  Prompt {i + 1}: {prompt}")
        print(f"{'─' * 70}")

        # Base model
        base_result = generate_response(
            base_model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\n  \033[90m[BASE MODEL] ({base_result['latency_seconds']:.2f}s)\033[0m:")
        print(f"  {base_result['response'][:300]}..." if len(base_result['response']) > 300 else f"  {base_result['response']}")

        # Fine-tuned model
        ft_result = generate_response(
            ft_model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\n  \033[92m[FINE-TUNED] ({ft_result['latency_seconds']:.2f}s)\033[0m:")
        print(f"  {ft_result['response'][:300]}..." if len(ft_result['response']) > 300 else f"  {ft_result['response']}")

    print(f"\n{'=' * 70}\n")

    # Cleanup
    del base_model, ft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(
    model: Any,
    tokenizer: Any,
    num_runs: int,
    args: argparse.Namespace,
) -> None:
    """Run a latency benchmark."""
    bench_prompt = "Write a detailed explanation of how neural networks learn through backpropagation. Include the key mathematical concepts."

    print(f"\n  Running {num_runs} benchmark generation(s)...")
    print('  Prompt: "' + bench_prompt[:80] + '..."\n')

    latencies: list[float] = []
    tok_rates: list[float] = []

    for i in range(num_runs):
        result = generate_response(
            model, tokenizer, bench_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_seq_length=args.max_seq_length,
        )
        latencies.append(result["latency_seconds"])
        tok_rates.append(result["tokens_per_second"])
        print(f"    Run {i + 1}/{num_runs}: {result['latency_seconds']:.2f}s | "
              f"{result['tokens_generated']} tokens | {result['tokens_per_second']:.1f} tok/s")

    avg_latency = sum(latencies) / len(latencies)
    avg_tok_rate = sum(tok_rates) / len(tok_rates)

    print(f"\n  \033[1mBenchmark Results:\033[0m")
    print(f"    Avg latency:     {avg_latency:.2f}s")
    print(f"    Avg tokens/sec:  {avg_tok_rate:.1f}")
    print(f"    Min latency:     {min(latencies):.2f}s")
    print(f"    Max latency:     {max(latencies):.2f}s")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI Inference Testing")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    if args.compare_with_base:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        compare_models(args, tokenizer)
        return 0

    # Load model
    model, tokenizer, model_label = load_model(args)

    logger.info(f"Model: {model_label}")
    logger.info(f"Device: {model.device}")
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"VRAM allocated: {vram_used:.2f} GB")

    # Choose mode
    if args.input_file:
        batch_inference(args, model, tokenizer, model_label)
    else:
        interactive_chat(args, model, tokenizer, model_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())
