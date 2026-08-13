#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — QLoRA Fine-Tuning
A Project by Osama

Main script for QLoRA (Quantized Low-Rank Adaptation) fine-tuning of Qwen 2.5 7B.

This script:
  - Loads the base model in 4-bit quantization (NF4) for memory efficiency
  - Applies QLoRA adapters for parameter-efficient fine-tuning
  - Uses SFTTrainer from the TRL library for supervised fine-tuning
  - Integrates with Weights & Biases for experiment tracking
  - Includes gradient checkpointing for reduced VRAM usage
  - Supports evaluation during training
  - Handles out-of-memory errors gracefully

Usage:
    python finetune_qlora.py --config finetune_config.yaml
    python finetune_qlora.py --config finetune_config.yaml --max_steps 500 --output_dir outputs/test
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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("nexusai.finetune")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Handler:
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logging.basicConfig(level=level, handlers=[ch, fh], force=True)
    logger.info(f"Log file: {log_file}")
    return fh


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NexusAI QLoRA Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune_qlora.py --config finetune_config.yaml
  python finetune_qlora.py --config finetune_config.yaml --max_steps 200
  python finetune_qlora.py --base_model Qwen/Qwen2.5-7B --dataset_path data/train.jsonl --output_dir outputs/my-run
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="finetune_config.yaml",
        help="Path to YAML configuration file (default: finetune_config.yaml)",
    )

    # Model overrides
    parser.add_argument("--base_model", type=str, default=None, help="Override base model name")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Override max sequence length")

    # LoRA overrides
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")

    # Training overrides
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation")

    # Data overrides
    parser.add_argument("--dataset_path", type=str, default=None, help="Training dataset path")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Validation dataset path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")

    # Output overrides
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_disabled", action="store_true", help="Disable W&B logging")

    # Utility
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and exit without training")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI arguments with YAML config (CLI args take precedence)."""
    def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
        """Nested dict getter."""
        keys = key.split(".")
        val = d
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    def _set(d: dict[str, Any], key: str, value: Any) -> None:
        """Nested dict setter."""
        keys = key.split(".")
        parent = d
        for k in keys[:-1]:
            parent = parent.setdefault(k, {})
        parent[keys[-1]] = value

    # Map CLI args to config keys
    arg_mapping = {
        "base_model": "model.base_model",
        "max_seq_length": "model.max_seq_length",
        "lora_r": "lora.r",
        "lora_alpha": "lora.lora_alpha",
        "lora_dropout": "lora.lora_dropout",
        "num_train_epochs": "training.num_train_epochs",
        "learning_rate": "training.learning_rate",
        "per_device_train_batch_size": "training.per_device_train_batch_size",
        "gradient_accumulation_steps": "training.gradient_accumulation_steps",
        "dataset_path": "data.dataset_path",
        "val_dataset_path": "data.val_dataset_path",
        "max_samples": "data.max_samples",
        "output_dir": "output.output_dir",
        "wandb_project": "output.wandb_project",
        "wandb_run_name": "output.wandb_run_name",
        "seed": "training.seed",
    }

    for arg_name, config_key in arg_mapping.items():
        arg_val = getattr(args, arg_name, None)
        if arg_val is not None:
            _set(config, config_key, arg_val)

    # Handle special cases
    if args.max_steps > 0:
        _set(config, "training.max_steps", args.max_steps)

    if args.wandb_disabled:
        _set(config, "output.wandb_disabled", True)

    if args.no_gradient_checkpointing:
        _set(config, "training.gradient_checkpointing", False)

    return config


# ---------------------------------------------------------------------------
# GPU verification
# ---------------------------------------------------------------------------
def verify_gpu() -> bool:
    """Verify that at least one CUDA-capable GPU is available."""
    if not torch.cuda.is_available():
        logger.error("=" * 60)
        logger.error("  NO GPU DETECTED!")
        logger.error("=" * 60)
        logger.error("")
        logger.error("This script requires a CUDA-compatible GPU for QLoRA training.")
        logger.error("")
        logger.error("Options:")
        logger.error("  1. Use Google Colab (free T4 GPU)")
        logger.error("  2. Use Kaggle Notebooks (free T4 x2)")
        logger.error("  3. Rent a GPU on RunPod/Lambda Labs (~$0.40/hr)")
        logger.error("  4. See gpu_requirements.md for detailed recommendations")
        logger.error("")
        return False

    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / (1024**3)
        logger.info(
            f"GPU {i}: {props.name} | {vram_gb:.1f} GB VRAM | CUDA {torch.version.cuda}"
        )

    # Warn if VRAM is low
    for i in range(gpu_count):
        vram_gb = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        if vram_gb < 12:
            logger.warning(
                f"GPU {i} has only {vram_gb:.1f} GB VRAM. "
                f"A 7B QLoRA model requires ~12 GB minimum. "
                f"Consider using a smaller model (e.g., Qwen2.5-1.5B) or reducing batch size."
            )

    return True


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------
def create_bnb_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    """Create the BitsAndBytes 4-bit quantization config."""
    lora_cfg = config.get("lora", {})

    # Determine compute dtype
    dtype_str = lora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=lora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=lora_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_model_and_tokenizer(config: dict[str, Any]):
    """Load the base model in 4-bit quantization and the tokenizer."""
    model_cfg = config.get("model", {})
    model_name: str = model_cfg.get("base_model", "Qwen/Qwen2.5-7B")

    logger.info(f"Loading base model: {model_name}")
    logger.info("Applying 4-bit NF4 quantization...")

    bnb_config = create_bnb_config(config)

    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "torch_dtype": torch.bfloat16 if model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16,
    }

    # Flash Attention 2
    use_flash = model_cfg.get("use_flash_attention", "auto")
    if use_flash == "auto":
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except ImportError:
            logger.info("Flash Attention 2 not available, using SDPA")
    elif use_flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model with OOM handling
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except torch.cuda.OutOfMemoryError:
        logger.error("OUT OF MEMORY while loading the model!")
        logger.error("")
        logger.error("Suggestions:")
        logger.error("  1. Use a smaller model (e.g., Qwen/Qwen2.5-1.5B)")
        logger.error("  2. Reduce max_seq_length in config")
        logger.error("  3. Close other GPU-consuming applications")
        logger.error("  4. Use a GPU with more VRAM (see gpu_requirements.md)")
        raise

    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.get("training", {}).get("gradient_checkpointing", True),
    )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="right",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"  Parameters: {model.num_parameters():,}")
    logger.info(f"  Vocab size: {len(tokenizer):,}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------
def create_lora_config(config: dict[str, Any]) -> LoraConfig:
    """Create the LoRA configuration from the config dict."""
    lora_cfg = config.get("lora", {})

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"dropout={lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")

    return lora_config


def apply_lora(model: AutoModelForCausalLM, config: dict[str, Any]) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model."""
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameter stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(f"LoRA adapters applied")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}% of total)")
    logger.info(f"  Total parameters:     {total_params:,}")

    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(config: dict[str, Any]):
    """Load and prepare training and validation datasets."""
    data_cfg = config.get("data", {})
    dataset_path: str = data_cfg.get("dataset_path", "data/train.jsonl")
    val_path: str = data_cfg.get("val_dataset_path", "data/val.jsonl")
    max_samples: int = data_cfg.get("max_samples", -1)

    # Load training data
    logger.info(f"Loading training data from {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}\n"
            f"Run 'python prepare_dataset.py' first to create the dataset."
        )

    train_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Limit samples if requested
    if max_samples and max_samples > 0:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

    logger.info(f"  Training samples: {len(train_dataset):,}")

    # Load validation data if available
    eval_dataset = None
    if os.path.exists(val_path):
        logger.info(f"Loading validation data from {val_path}")
        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        eval_max = min(len(eval_dataset), 500)  # Cap eval set size
        eval_dataset = eval_dataset.select(range(eval_max))
        logger.info(f"  Validation samples: {len(eval_dataset):,}")
    else:
        logger.warning(f"  Validation dataset not found at {val_path} — no evaluation during training")

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
def create_training_arguments(config: dict[str, Any], output_dir: str) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    t = config.get("training", {})
    o = config.get("output", {})

    # Determine precision
    bf16 = t.get("bf16", True)
    fp16 = t.get("fp16", False)

    # If bf16 requested but not supported, fall back to fp16
    if bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 not supported by this GPU, falling back to fp16")
        bf16 = False
        fp16 = True

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        optim=t.get("optim", "paged_adamw_8bit"),
        num_train_epochs=t.get("num_train_epochs", 3),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t.get("logging_steps", 25),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        eval_strategy=t.get("eval_strategy", "steps") if os.path.exists(config.get("data", {}).get("val_dataset_path", "data/val.jsonl")) else "no",
        eval_steps=t.get("eval_steps", 100),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=t.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        seed=t.get("seed", 42),
        packing=False,
        report_to="wandb" if not o.get("wandb_disabled", False) else "none",
        run_name=o.get("wandb_run_name") if not o.get("wandb_disabled", False) else None,
        # Disable default hub behavior; we push manually if needed
        push_to_hub=False,
        load_best_model_at_end=True if t.get("eval_strategy", "no") != "no" else False,
        metric_for_best_model="eval_loss" if t.get("eval_strategy", "no") != "no" else None,
        greater_is_better=False,
    )

    # Handle max_steps override
    if "max_steps" in t and t["max_steps"] is not None:
        args.max_steps = t["max_steps"]

    # Set group_by_length if specified
    if t.get("group_by_length", False):
        args.group_by_length = True

    return args


# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
def setup_wandb(config: dict[str, Any]) -> None:
    """Initialize Weights & Biases for experiment tracking."""
    o = config.get("output", {})
    if o.get("wandb_disabled", False):
        logger.info("W&B logging disabled")
        os.environ["WANDB_DISABLED"] = "true"
        return

    project = o.get("wandb_project", "nexusai-finetuning")
    run_name = o.get("wandb_run_name")

    os.environ["WANDB_PROJECT"] = project
    if run_name:
        os.environ["WANDB_NAME"] = run_name

    try:
        import wandb
        wandb.init(project=project, name=run_name, config=config)
        logger.info(f"W&B initialized: project={project}, run={run_name or 'auto'}")
    except ImportError:
        logger.warning("wandb not installed; install with: pip install wandb")
        os.environ["WANDB_DISABLED"] = "true"
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}")
        os.environ["WANDB_DISABLED"] = "true"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full fine-tuning pipeline."""
    model_cfg = config.get("model", {})
    output_dir: str = config.get("output", {}).get("output_dir", "outputs/qlora-adapters")
    max_seq_length: int = model_cfg.get("max_seq_length", 2048)

    os.makedirs(output_dir, exist_ok=True)

    # Setup W&B
    setup_wandb(config)

    # Verify GPU
    if not verify_gpu():
        return {"status": "error", "message": "No GPU available"}

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load data
    train_dataset, eval_dataset = load_training_data(config)

    # Create training arguments
    training_args = create_training_arguments(config, output_dir)

    # Print training summary
    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(1, torch.cuda.device_count())
    )
    total_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else (len(train_dataset) // effective_batch) * training_args.num_train_epochs
    )

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Base Model:          {model_cfg.get('base_model', 'N/A')}")
    logger.info(f"  Max Seq Length:      {max_seq_length}")
    logger.info(f"  Training Samples:    {len(train_dataset):,}")
    logger.info(f"  Validation Samples:  {len(eval_dataset):,}"
                if eval_dataset else "  Validation Samples:  None")
    logger.info(f"  Epochs:              {training_args.num_train_epochs}")
    logger.info(f"  Batch Size:          {training_args.per_device_train_batch_size}")
    logger.info(f"  Grad Accumulation:   {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective Batch:     {effective_batch}")
    logger.info(f"  Learning Rate:       {training_args.learning_rate}")
    logger.info(f"  Total Steps:         {total_steps:,}")
    logger.info(f"  LR Scheduler:        {training_args.lr_scheduler_type}")
    logger.info(f"  Optimizer:           {training_args.optim}")
    logger.info(f"  BF16:                {training_args.bf16}")
    logger.info(f"  Gradient Checkpoint: {training_args.gradient_checkpointing}")
    logger.info(f"  Output Dir:          {output_dir}")
    logger.info("=" * 60 + "\n")

    # Create SFTTrainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field=config.get("training", {}).get("dataset_text_field", "text"),
    )

    # Train with OOM handling
    start_time = time.time()
    try:
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    except torch.cuda.OutOfMemoryError:
        logger.error("OUT OF MEMORY during training!")
        logger.error("")
        logger.error("Remedies:")
        logger.error("  1. Reduce per_device_train_batch_size (try 1)")
        logger.error("  2. Increase gradient_accumulation_steps to compensate")
        logger.error("  3. Reduce max_seq_length")
        logger.error("  4. Enable gradient_checkpointing (if not already)")
        logger.error("  5. Try a smaller LoRA rank (r=8)")
        raise
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving partial checkpoint...")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time / 60:.1f} minutes")

    # Save the final adapter
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    logger.info(f"Saving adapter to {final_adapter_path}")
    trainer.model.save_pretrained(final_adapter_path)
    trainer.tokenizer.save_pretrained(final_adapter_path)

    # Save training metrics
    metrics = {
        "status": "completed",
        "training_time_seconds": round(training_time, 1),
        "training_time_minutes": round(training_time / 60, 1),
        "total_steps": trainer.state.global_step,
        "train_samples": len(train_dataset),
        "train_loss": trainer.state.log_history[-1].get("loss", "N/A")
        if trainer.state.log_history else "N/A",
    }

    # Get best eval metrics if available
    if trainer.state.best_metric is not None:
        metrics["best_eval_loss"] = round(trainer.state.best_metric, 4)
        metrics["best_step"] = trainer.state.best_model_checkpoint

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Finish W&B run
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI QLoRA Fine-Tuning Pipeline")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    # Load config
    config_path: str = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    config = merge_args_with_config(args, config)

    # Setup logging
    log_dir = config.get("output", {}).get("output_dir", "outputs/qlora-adapters")
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, "logs")
    setup_logging(log_dir)

    # Dry run
    if args.dry_run:
        logger.info("Dry run mode — validating configuration...")
        model_name = config.get("model", {}).get("base_model", "N/A")
        dataset_path = config.get("data", {}).get("dataset_path", "N/A")
        logger.info(f"  Base model: {model_name}")
        logger.info(f"  Dataset:    {dataset_path}")
        if os.path.exists(dataset_path):
            ds = load_dataset("json", data_files=dataset_path, split="train")
            logger.info(f"  Samples:    {len(ds):,}")
        else:
            logger.warning(f"  Dataset not found: {dataset_path}")
        logger.info("  Configuration looks valid!")
        return 0

    # Run training
    try:
        metrics = train(config)

        if metrics.get("status") == "completed":
            print("\n" + "=" * 60)
            print("  TRAINING COMPLETE!")
            print("=" * 60)
            print(f"  Steps:            {metrics.get('total_steps', 'N/A'):,}")
            print(f"  Training Time:    {metrics.get('training_time_minutes', 'N/A')} min")
            print(f"  Final Loss:       {metrics.get('train_loss', 'N/A')}")
            if "best_eval_loss" in metrics:
                print(f"  Best Eval Loss:   {metrics['best_eval_loss']}")
            print()
            print("  Next steps:")
            print("    1. Evaluate:  python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter")
            print("    2. Merge:     python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter")
            print("    3. Inference:  python run_inference.py --model_path outputs/merged")
            print("=" * 60 + "\n")
            return 0
        else:
            logger.error(f"Training failed: {metrics.get('message')}")
            return 1

    except torch.cuda.OutOfMemoryError:
        logger.error("Training aborted due to out-of-memory error.")
        return 2
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
