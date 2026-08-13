#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — DPO (Direct Preference Optimization)
A Project by Osama

Advanced fine-tuning script using DPO for preference alignment.
DPO directly optimizes the language model to prefer chosen responses
over rejected ones without needing an explicit reward model.

Key concepts:
  - Reference model: Frozen copy of the (optionally SFT'd) model
  - Beta parameter: Controls the deviation from the reference model
  - Preference pairs: (prompt, chosen_response, rejected_response)

Usage:
    python advanced_finetune_dpo.py --config finetune_config.yaml --sft_adapter_path outputs/qlora-adapters/final_adapter
    python advanced_finetune_dpo.py --dataset_path data/dpo_pairs.jsonl --beta 0.1 --num_train_epochs 1
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("nexusai.dpo")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Configure logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logging.basicConfig(level=level, handlers=[ch, fh], force=True)
    logger.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NexusAI DPO Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DPO after SFT
  python advanced_finetune_dpo.py --sft_adapter_path outputs/qlora-adapters/final_adapter

  # DPO with custom preference dataset
  python advanced_finetune_dpo.py --dataset_path data/dpo_pairs.jsonl --beta 0.1

  # DPO from a HuggingFace preference dataset
  python advanced_finetune_dpo.py --hf_dataset Anthropic/hh-rlhf --max_samples 10000
        """,
    )

    # Config
    parser.add_argument("--config", type=str, default="finetune_config.yaml", help="YAML config file")

    # Model
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B", help="Base model")
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default=None,
        help="Path to SFT LoRA adapter (the SFT'd model becomes the DPO reference)",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Separate reference model path (default: same as base model or SFT adapter)",
    )

    # Data
    parser.add_argument("--dataset_path", type=str, default=None, help="Local DPO dataset (JSONL)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace DPO dataset name")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max training samples")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")

    # DPO parameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (controls deviation from reference)")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard"],
        help="DPO loss type (default: sigmoid)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for DPO loss",
    )
    parser.add_argument(
        "--use_score_scaling",
        action="store_true",
        help="Enable score scaling for DPO loss",
    )
    parser.add_argument(
        "--use_query_fusion",
        action="store_true",
        help="Enable query fusion for efficiency",
    )

    # LoRA for DPO
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--no_lora", action="store_true", help="Train without LoRA (full fine-tuning)")

    # Training
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate (DPO typically uses lower LR)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/dpo-adapters", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="nexusai-dpo", help="W&B project name")
    parser.add_argument("--wandb_disabled", action="store_true", help="Disable W&B")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=100, help="Save every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every N steps")

    # Eval / reward
    parser.add_argument(
        "--eval_during_training",
        action="store_true",
        default=True,
        help="Evaluate during training",
    )
    parser.add_argument(
        "--generate_during_eval",
        action="store_true",
        default=False,
        help="Generate samples during evaluation (slow but informative)",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=100,
        help="Number of eval samples",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading for DPO
# ---------------------------------------------------------------------------
DPO_KNOWN_DATASETS: dict[str, dict[str, str]] = {
    "Anthropic/hh-rlhf": {
        "chosen": "chosen",
        "rejected": "rejected",
        "prompt": None,  # Extract from chosen
    },
    "openai/summarize_from_feedback": {
        "chosen": "chosen",
        "rejected": "rejected",
        "prompt": "info",
    },
    "HuggingFaceH4/ultrafeedback_binarized": {
        "chosen": "chosen",
        "rejected": "rejected",
        "prompt": "prompt",
    },
    "argilla/ultrafeedback-binarized-preferences-cleaned": {
        "chosen": "chosen",
        "rejected": "rejected",
        "prompt": "prompt",
    },
}


def load_dpo_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    """Load and format a DPO preference dataset."""
    if args.dataset_path:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

        logger.info(f"Loading local DPO dataset: {args.dataset_path}")
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")

        # Validate required columns
        required = {"prompt", "chosen", "rejected"}
        cols = set(dataset.column_names)
        missing = required - cols
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Need: prompt, chosen, rejected")

    elif args.hf_dataset:
        logger.info(f"Loading HuggingFace dataset: {args.hf_dataset}")

        if args.hf_dataset in DPO_KNOWN_DATASETS:
            info = DPO_KNOWN_DATASETS[args.hf_dataset]
            dataset = load_dataset(args.hf_dataset, split="train_sft" if "train_sft" in str(args.hf_dataset) else "train", trust_remote_code=True)
        else:
            dataset = load_dataset(args.hf_dataset, split="train", trust_remote_code=True)
            info = {
                "chosen": "chosen",
                "rejected": "rejected",
                "prompt": "prompt",
            }

        # Standardize column names
        if info["prompt"] is None and "chosen" in dataset.column_names:
            # For hh-rlhf, prompt is the first assistant response in the rejected pair
            logger.info("Extracting prompts from rejected responses...")
            prompts = []
            for row in tqdm(dataset, desc="Extracting prompts"):
                rejected_text = row["rejected"]
                # In hh-rlhf, the format is often "Human: ... Assistant: ..."
                if "Assistant:" in rejected_text:
                    prompt_text = rejected_text.split("Assistant:")[0].strip()
                else:
                    prompt_text = rejected_text.split("\n")[0]
                prompts.append(prompt_text)
            dataset = dataset.add_column("prompt", prompts)

        # Rename columns if needed
        rename_map = {}
        if info["chosen"] != "chosen" and info["chosen"] in dataset.column_names:
            rename_map[info["chosen"]] = "chosen"
        if info["rejected"] != "rejected" and info["rejected"] in dataset.column_names:
            rename_map[info["rejected"]] = "rejected"
        if info["prompt"] and info["prompt"] != "prompt" and info["prompt"] in dataset.column_names:
            rename_map[info["prompt"]] = "prompt"

        if rename_map:
            dataset = dataset.rename_columns(rename_map)

    else:
        raise ValueError("Provide --dataset_path or --hf_dataset")

    # Filter: ensure all fields are non-empty
    def is_valid(example: dict[str, Any]) -> bool:
        return (
            bool(example.get("prompt", "").strip())
            and bool(example.get("chosen", "").strip())
            and bool(example.get("rejected", "").strip())
        )

    original_len = len(dataset)
    dataset = dataset.filter(is_valid)
    filtered = original_len - len(dataset)
    if filtered > 0:
        logger.warning(f"Filtered out {filtered} invalid samples")

    # Limit samples
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Split into train/val
    dataset = dataset.shuffle(seed=42)
    val_count = int(len(dataset) * args.val_split)
    train_dataset = dataset.skip(val_count)
    eval_dataset = dataset.take(val_count)

    logger.info(f"  Training pairs: {len(train_dataset):,}")
    logger.info(f"  Validation pairs: {len(eval_dataset):,}")

    return train_dataset, eval_dataset if val_count > 0 else None


def compute_dataset_stats(dataset: Dataset) -> None:
    """Print statistics about the DPO dataset."""
    if not dataset:
        return

    prompt_lens = [len(str(row["prompt"])) for row in dataset]
    chosen_lens = [len(str(row["chosen"])) for row in dataset]
    rejected_lens = [len(str(row["rejected"])) for row in dataset]

    logger.info(f"  Prompt lengths — min: {min(prompt_lens)}, max: {max(prompt_lens)}, "
                f"mean: {sum(prompt_lens)/len(prompt_lens):.0f} chars")
    logger.info(f"  Chosen lengths — min: {min(chosen_lens)}, max: {max(chosen_lens)}, "
                f"mean: {sum(chosen_lens)/len(chosen_lens):.0f} chars")
    logger.info(f"  Rejected lengths — min: {min(rejected_lens)}, max: {max(rejected_lens)}, "
                f"mean: {sum(rejected_lens)/len(rejected_lens):.0f} chars")


def create_sample_dpo_dataset(output_path: str, num_samples: int = 100) -> None:
    """Create a sample DPO dataset for testing."""
    import random
    random.seed(42)

    templates = [
        ("Explain {topic}.",
         "{topic} is a fascinating subject. Here's a clear explanation with key concepts and examples that make it easy to understand.",
         "I don't know what {topic} is."),
        ("Write a {type_} about {subject}.",
         "Here's a well-crafted {type_} about {subject} that demonstrates creativity and skill in writing.",
         "{subject}. {type_}. The end."),
        ("What are the benefits of {thing}?",
         "The benefits of {thing} include improved efficiency, cost savings, and better outcomes. Here's a detailed breakdown with evidence.",
         "I don't know. Maybe some benefits exist."),
    ]

    topics = ["machine learning", "blockchain", "climate change", "quantum physics", "neuroscience",
              "renewable energy", "cybersecurity", "genetics", "artificial intelligence", "space exploration"]
    types_ = ["poem", "essay", "story", "summary", "analysis"]
    subjects = ["nature", "technology", "history", "science", "art"]
    things = ["exercise", "meditation", "reading", "healthy eating", "sleep"]

    samples = []
    for _ in range(num_samples):
        t = random.choice(templates)
        prompt = t[0].format(
            topic=random.choice(topics),
            type_=random.choice(types_),
            subject=random.choice(subjects),
            thing=random.choice(things),
        )
        chosen = t[1].format(
            topic=random.choice(topics),
            type_=random.choice(types_),
            subject=random.choice(subjects),
            thing=random.choice(things),
        )
        rejected = t[2].format(
            topic=random.choice(topics),
            type_=random.choice(types_),
            subject=random.choice(subjects),
            thing=random.choice(things),
        )
        samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Created sample DPO dataset: {output_path} ({num_samples} pairs)")


# ---------------------------------------------------------------------------
# Main DPO training
# ---------------------------------------------------------------------------
def train_dpo(args: argparse.Namespace) -> dict[str, Any]:
    """Run the DPO training pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    setup_logging(log_dir)

    # GPU check
    if not torch.cuda.is_available():
        logger.error("GPU required for DPO training")
        return {"status": "error", "message": "No GPU"}

    # Clear VRAM
    torch.cuda.empty_cache()
    gc.collect()

    # Load dataset
    logger.info("Loading DPO preference dataset...")
    try:
        train_dataset, eval_dataset = load_dpo_dataset(args)
    except FileNotFoundError:
        logger.info("Creating a sample DPO dataset for testing...")
        sample_path = "data/dpo_sample.jsonl"
        create_sample_dpo_dataset(sample_path, num_samples=100)
        args.dataset_path = sample_path
        train_dataset, eval_dataset = load_dpo_dataset(args)

    compute_dataset_stats(train_dataset)

    # Determine the model to start from
    model_name = args.base_model
    if args.sft_adapter_path:
        logger.info(f"Will apply DPO on top of SFT adapter: {args.sft_adapter_path}")

    # Model quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {model_name}")
    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except torch.cuda.OutOfMemoryError:
        logger.error("OOM while loading model! Try a smaller model or use a GPU with more VRAM.")
        return {"status": "error", "message": "OOM on model load"}

    # Apply LoRA if requested
    if not args.no_lora:
        logger.info("Preparing model for k-bit training and applying LoRA...")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    else:
        logger.warning("Full fine-tuning mode (no LoRA). This requires significantly more VRAM!")

    # Load SFT adapter as the starting point if provided
    if args.sft_adapter_path and not args.no_lora:
        logger.info(f"Loading SFT adapter weights: {args.sft_adapter_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.sft_adapter_path)
            logger.info("SFT adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load SFT adapter: {e}. Starting from base model.")

    # Reference model (frozen, no LoRA)
    ref_model_name = args.reference_model or model_name
    logger.info(f"Loading reference model: {ref_model_name}")

    ref_model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": {"": 0} if torch.cuda.device_count() == 1 else "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    try:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **ref_model_kwargs)
        ref_model.eval()
    except Exception as e:
        logger.warning(f"Could not load separate reference model: {e}")
        logger.info("DPOTrainer will create the reference model internally.")
        ref_model = None

    # W&B setup
    if args.wandb_disabled:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Training arguments
    gc_on = args.gradient_checkpointing and not args.no_gradient_checkpointing

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if args.eval_during_training and eval_dataset else "no",
        eval_steps=args.eval_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=gc_on,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        seed=42,
        report_to="wandb" if not args.wandb_disabled else "none",
        remove_unused_columns=False,
    )

    if args.max_steps > 0:
        training_args.max_steps = args.max_steps

    # Print summary
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, torch.cuda.device_count())
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else (len(train_dataset) // effective_batch) * args.num_train_epochs
    )

    logger.info("\n" + "=" * 60)
    logger.info("  DPO TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Base Model:          {model_name}")
    logger.info(f"  Reference Model:     {ref_model_name}")
    logger.info(f"  SFT Adapter:         {args.sft_adapter_path or 'None'}")
    logger.info(f"  Preference Pairs:    {len(train_dataset):,}")
    logger.info(f"  Beta:                {args.beta}")
    logger.info(f"  Loss Type:           {args.loss_type}")
    logger.info(f"  Epochs:              {args.num_train_epochs}")
    logger.info(f"  Effective Batch:     {effective_batch}")
    logger.info(f"  Learning Rate:       {args.learning_rate}")
    logger.info(f"  Total Steps:         {total_steps:,}")
    logger.info(f"  LoRA:                {'r=' + str(args.lora_r) if not args.no_lora else 'None (full FT)'}")
    logger.info("=" * 60 + "\n")

    # Create DPO Trainer
    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_type=args.loss_type,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        label_smoothing=args.label_smoothing,
        use_score_scaling=args.use_score_scaling,
        use_query_fusion=args.use_query_fusion,
        generate_during_eval=args.generate_during_eval,
    )

    # Train
    start_time = time.time()
    try:
        logger.info("Starting DPO training...")
        dpo_trainer.train()
    except torch.cuda.OutOfMemoryError:
        logger.error("OUT OF MEMORY during DPO training!")
        logger.error("Try: reducing batch size, enabling gradient checkpointing, or using a smaller model.")
        return {"status": "error", "message": "OOM during training"}
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"DPO training failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    training_time = time.time() - start_time
    logger.info(f"DPO training completed in {training_time / 60:.1f} minutes")

    # Save adapter
    final_path = os.path.join(args.output_dir, "final_dpo_adapter")
    dpo_trainer.model.save_pretrained(final_path)
    dpo_trainer.tokenizer.save_pretrained(final_path)
    logger.info(f"DPO adapter saved to {final_path}")

    # Save metrics
    metrics = {
        "status": "completed",
        "training_time_minutes": round(training_time / 60, 1),
        "total_steps": dpo_trainer.state.global_step,
        "preference_pairs": len(train_dataset),
        "beta": args.beta,
        "loss_type": args.loss_type,
    }

    log_history = dpo_trainer.state.log_history
    if log_history:
        # Get the last training loss
        for entry in reversed(log_history):
            if "loss" in entry:
                metrics["final_train_loss"] = round(entry["loss"], 6)
                break
        # Get the last eval loss
        for entry in reversed(log_history):
            if "eval_loss" in entry:
                metrics["final_eval_loss"] = round(entry["eval_loss"], 6)
                break
            if "eval_rewards/chosen" in entry:
                metrics["final_chosen_reward"] = round(entry["eval_rewards/chosen"], 4)
                metrics["final_rejected_reward"] = round(entry["eval_rewards/rejected"], 4)
                metrics["reward_margin"] = round(
                    entry["eval_rewards/chosen"] - entry["eval_rewards/rejected"], 4
                )
                break

    metrics_path = os.path.join(args.output_dir, "dpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Cleanup
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return metrics


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI DPO Fine-Tuning Pipeline")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    try:
        metrics = train_dpo(args)

        if metrics.get("status") == "completed":
            print("\n" + "=" * 60)
            print("  DPO TRAINING COMPLETE!")
            print("=" * 60)
            print(f"  Steps:           {metrics.get('total_steps', 'N/A'):,}")
            print(f"  Training Time:   {metrics.get('training_time_minutes', 'N/A')} min")
            print(f"  Final Loss:      {metrics.get('final_train_loss', 'N/A')}")
            if "reward_margin" in metrics:
                print(f"  Reward Margin:   {metrics.get('reward_margin', 'N/A')}")
            print()
            print("  Next steps:")
            print("    1. Merge: python merge_and_export.py --adapter_path outputs/dpo-adapters/final_dpo_adapter")
            print("    2. Test:   python run_inference.py --model_path outputs/merged")
            print("=" * 60 + "\n")
            return 0
        else:
            logger.error(f"DPO training failed: {metrics.get('message')}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
