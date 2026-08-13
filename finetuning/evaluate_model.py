#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — Model Evaluation
A Project by Osama

Evaluates a fine-tuned model (base + LoRA adapters) against a held-out test set.
Computes loss, perplexity, and generates sample outputs. Can also compare
the fine-tuned model against the base model.

Usage:
    python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter
    python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter --compare_with_base
    python evaluate_model.py --model_path outputs/merged --no_adapter --generate_samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexusai.evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned NexusAI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model loading
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (from finetune_qlora.py)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a full merged model (skip adapter loading)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Base model name (used when --adapter_path is provided, default: Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help="Evaluate the base model without any adapter",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization for evaluation (default: True)",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Load model in full precision (requires more VRAM)",
    )

    # Data
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="data/val.jsonl",
        help="Path to evaluation dataset (default: data/val.jsonl)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for evaluation (default: 2048)",
    )

    # Generation-based evaluation
    parser.add_argument(
        "--generate_samples",
        type=int,
        default=5,
        help="Number of sample outputs to generate (default: 5)",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=512,
        help="Maximum generation length in tokens (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (default: 0.9)",
    )

    # Comparison
    parser.add_argument(
        "--compare_with_base",
        action="store_true",
        help="Compare fine-tuned model outputs against the base model",
    )

    # Output
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON (default: <output_dir>/eval_results.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results (default: outputs/evaluation)",
    )

    return parser.parse_args()


def load_model_for_eval(args: argparse.Namespace):
    """Load the model and tokenizer based on arguments."""
    if args.no_adapter or args.no_4bit:
        # Load base or merged model in full precision
        model_name = args.model_path or args.base_model
        logger.info(f"Loading model: {model_name} (full precision)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.model_path:
        # Load a merged model in 4-bit
        logger.info(f"Loading merged model: {args.model_path} (4-bit)")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.adapter_path:
        # Load base + LoRA adapter
        logger.info(f"Loading base model: {args.base_model} (4-bit)")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        raise ValueError("Provide --adapter_path, --model_path, or --no_adapter")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_seq_length: int = 2048,
    stride: int = 512,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute perplexity on a list of texts using sliding window."""
    total_loss = 0.0
    total_tokens = 0
    nlls: list[float] = []

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        ).to(device)

        input_ids = encodings.input_ids[0]
        seq_len = input_ids.size(0)

        if seq_len < 2:
            continue

        nll = 0.0
        count = 0

        for i in range(0, seq_len - 1, stride):
            begin_loc = max(i, 0)
            end_loc = min(i + max_seq_length, seq_len)
            target_len = end_loc - begin_loc

            if target_len <= 1:
                continue

            input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0)
            target_chunk = input_chunk.clone()
            target_chunk[:, :-1] = -100  # Ignore all but last token

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss.item()

            nll += neg_log_likelihood * target_len
            count += target_len

        if count > 0:
            avg_nll = nll / count
            nlls.append(avg_nll)
            total_loss += nll
            total_tokens += count

    if not nlls:
        return {"perplexity": float("inf"), "avg_loss": float("inf"), "num_samples": 0}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    mean_ppl = math.exp(sum(nlls) / len(nlls))

    return {
        "perplexity": round(perplexity, 4),
        "mean_perplexity": round(mean_ppl, 4),
        "avg_loss": round(avg_loss, 6),
        "num_samples": len(nlls),
        "median_perplexity": round(math.exp(sorted(nlls)[len(nlls) // 2]), 4),
    }


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[dict[str, str]],
    max_gen_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> list[dict[str, str]]:
    """Generate responses for a list of prompts."""
    results: list[dict[str, str]] = []

    for prompt_data in tqdm(prompts, desc="Generating responses"):
        instruction = prompt_data.get("instruction", "")

        # Build the chat-formatted prompt
        chat_prompt = f"""<|im_start|>system
You are a helpful, respectful, and honest assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "instruction": instruction,
            "expected_response": prompt_data.get("response", "N/A"),
            "generated_response": response,
        })

    return results


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """Simple word-overlap similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def main() -> int:
    """Main evaluation pipeline."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI Model Evaluation")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    # Verify GPU
    if not torch.cuda.is_available():
        logger.error("GPU required for evaluation. See gpu_requirements.md for options.")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model identifier
    model_label = args.adapter_path or args.model_path or args.base_model
    logger.info(f"Evaluating model: {model_label}")

    # ---- Load fine-tuned model ----
    model, tokenizer = load_model_for_eval(args)

    # ---- Load evaluation dataset ----
    eval_path: str = args.eval_dataset
    if not os.path.exists(eval_path):
        logger.error(f"Evaluation dataset not found: {eval_path}")
        logger.error("Run prepare_dataset.py first to create it.")
        return 1

    eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    max_samples = min(args.max_eval_samples, len(eval_dataset))
    eval_dataset = eval_dataset.select(range(max_samples))
    logger.info(f"Loaded {max_samples} evaluation samples")

    # ---- Compute perplexity ----
    logger.info("Computing perplexity on evaluation set...")
    eval_texts = [record["text"] for record in eval_dataset]
    ppl_results = compute_perplexity(
        model, tokenizer, eval_texts,
        max_seq_length=args.max_seq_length,
    )
    logger.info(f"  Perplexity: {ppl_results['perplexity']}")
    logger.info(f"  Median PPL: {ppl_results['median_perplexity']}")
    logger.info(f"  Avg Loss:   {ppl_results['avg_loss']}")

    # ---- Generate sample outputs ----
    logger.info(f"Generating {args.generate_samples} sample outputs...")
    sample_prompts = [eval_dataset[i] for i in range(min(args.generate_samples, len(eval_dataset)))]
    generated = generate_responses(
        model, tokenizer, sample_prompts,
        max_gen_length=args.max_gen_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Compute similarity scores for generated outputs
    for gen in generated:
        gen["jaccard_similarity"] = round(
            compute_text_similarity(gen["expected_response"], gen["generated_response"]), 4
        )

    # ---- Optionally compare with base model ----
    base_ppl_results: dict[str, Any] | None = None
    base_generated: list[dict[str, str]] | None = None

    if args.compare_with_base and not args.no_adapter:
        logger.info("\nLoading base model for comparison...")
        import gc

        # Free fine-tuned model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Load base model
        tokenizer_base = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer_base.pad_token is None:
            tokenizer_base.pad_token = tokenizer_base.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.eval()

        # Compute base perplexity
        logger.info("Computing base model perplexity...")
        base_ppl_results = compute_perplexity(
            base_model, tokenizer_base, eval_texts,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f"  Base Perplexity: {base_ppl_results['perplexity']}")

        # Generate base model outputs
        logger.info("Generating base model samples for comparison...")
        base_generated = generate_responses(
            base_model, tokenizer_base, sample_prompts,
            max_gen_length=args.max_gen_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Compile results ----
    results: dict[str, Any] = {
        "model_path": model_label,
        "eval_dataset": eval_path,
        "num_eval_samples": max_samples,
        "finetuned_model": {
            "perplexity": ppl_results["perplexity"],
            "median_perplexity": ppl_results["median_perplexity"],
            "avg_loss": ppl_results["avg_loss"],
        },
        "generated_samples": generated,
    }

    if base_ppl_results:
        results["base_model"] = {
            "perplexity": base_ppl_results["perplexity"],
            "median_perplexity": base_ppl_results["median_perplexity"],
            "avg_loss": base_ppl_results["avg_loss"],
        }
        results["comparison"] = {
            "ppl_improvement": round(
                base_ppl_results["perplexity"] - ppl_results["perplexity"], 4
            ),
            "ppl_improvement_pct": round(
                100 * (1 - ppl_results["perplexity"] / base_ppl_results["perplexity"]), 2
            ),
        }
        if base_generated:
            results["base_generated_samples"] = base_generated

    # ---- Save results ----
    output_path = args.output_json or os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {output_path}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model:       {model_label}")
    print(f"  Eval Samples: {max_samples}")
    print()
    print(f"  Fine-tuned Model:")
    print(f"    Perplexity: {ppl_results['perplexity']}")
    print(f"    Median PPL: {ppl_results['median_perplexity']}")
    print(f"    Avg Loss:   {ppl_results['avg_loss']}")

    if base_ppl_results:
        print()
        print(f"  Base Model (Comparison):")
        print(f"    Perplexity: {base_ppl_results['perplexity']}")
        print(f"    Median PPL: {base_ppl_results['median_perplexity']}")
        print()
        improvement = results["comparison"]["ppl_improvement"]
        pct = results["comparison"]["ppl_improvement_pct"]
        direction = "lower is better" if improvement > 0 else "higher (worse)"
        print(f"  PPL Change:  {improvement:+.4f} ({direction})")
        print(f"  PPL Δ%:      {pct:+.2f}%")

    print()
    print(f"  Sample Generated Outputs:")
    for i, gen in enumerate(generated[:3]):
        print(f"\n  --- Sample {i + 1} ---")
        print(f"  Instruction: {gen['instruction'][:100]}...")
        print(f"  Generated:   {gen['generated_response'][:200]}...")
        print(f"  Similarity:  {gen['jaccard_similarity']:.4f}")

    print("\n" + "=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
