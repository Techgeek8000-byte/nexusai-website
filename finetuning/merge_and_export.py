#!/usr/bin/env python3
"""
NexusAI Fine-Tuning Pipeline — Model Merging & Export
A Project by Osama

Merges LoRA adapters into the base model and exports in multiple formats:
  1. Full precision (safetensors) — for further training or deployment
  2. GGUF format — for llama.cpp / Ollama local inference
  3. 4-bit quantized (GPTQ/AWQ) — for efficient inference

Optionally uploads to HuggingFace Hub and generates a model card.

Usage:
    python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter
    python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter --export_gguf
    python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter --push_to_hub --hub_repo_id my-org/nexusai-qwen-7b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexusai.merge")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and export the model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name/path (auto-detected from adapter config if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/merged",
        help="Directory for the merged model (default: outputs/merged)",
    )

    # Export formats
    parser.add_argument(
        "--export_full",
        action="store_true",
        default=True,
        help="Export full precision merged model (default: True)",
    )
    parser.add_argument(
        "--no_full_export",
        action="store_true",
        help="Skip full precision export",
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Export in GGUF format for llama.cpp/Ollama",
    )
    parser.add_argument(
        "--gguf_quantize",
        type=str,
        default=None,
        help="GGUF quantization type (e.g., q4_k_m, q5_k_m, q8_0). Requires llama.cpp.",
    )
    parser.add_argument(
        "--export_4bit",
        action="store_true",
        help="Export 4-bit quantized model (using bitsandbytes)",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )

    # Model card
    parser.add_argument(
        "--model_card",
        action="store_true",
        default=True,
        help="Generate a model card README (default: True)",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default=None,
        help="Path to the training config YAML (for model card details)",
    )

    # Precision
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type for merged model weights (default: bfloat16)",
    )

    return parser.parse_args()


def detect_base_model(adapter_path: str) -> str:
    """Try to detect the base model from the adapter config."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path", "")
        if base_model:
            logger.info(f"Detected base model from adapter config: {base_model}")
            return base_model

    # Fallback
    logger.warning("Could not detect base model from adapter config, using default")
    return "Qwen/Qwen2.5-7B"


def load_and_merge(adapter_path: str, base_model: str, dtype: str = "bfloat16") -> tuple[Any, Any]:
    """Load the base model and merge LoRA adapters."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Loading LoRA adapters: {adapter_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and merge adapters
    logger.info("Merging LoRA adapters into base model...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    logger.info("Merge complete!")
    return model, tokenizer


def save_full_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
) -> None:
    """Save the full merged model."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving full model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Full model saved to {output_dir}")


def export_gguf(
    model_dir: str,
    output_dir: str,
    quantize: str | None = None,
) -> None:
    """Export model in GGUF format using llama.cpp's convert script."""
    try:
        from llama_cpp import Llama
        logger.info("llama-cpp-python detected")
    except ImportError:
        pass

    # Try using the transformers GGUF converter
    try:
        logger.info("Attempting GGUF export...")

        # Method 1: Use llama-cpp-python if available
        try:
            from llama_cpp import llama_cpp
            logger.info("Using llama-cpp-python for GGUF conversion")
        except ImportError:
            pass

        # Method 2: Use subprocess with llama.cpp convert_hf_to_gguf.py
        gguf_output = os.path.join(output_dir, "model.gguf")

        # Try the gguf package directly
        try:
            from gguf import GGUFWriter
            logger.info("gguf package available, attempting conversion...")

            # Use llama.cpp's conversion script if available
            import subprocess

            # Check for convert_hf_to_gguf.py
            convert_script = shutil.which("convert_hf_to_gguf.py")
            if convert_script is None:
                # Try to find it in common locations
                possible_paths = [
                    os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
                    "/opt/llama.cpp/convert_hf_to_gguf.py",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        convert_script = p
                        break

            if convert_script:
                cmd = [sys.executable, convert_script, model_dir, "--outfile", gguf_output, "--outtype", "f16"]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    logger.info(f"GGUF model saved to {gguf_output}")
                    return
                else:
                    logger.warning(f"GGUF conversion failed: {result.stderr}")
            else:
                logger.warning("convert_hf_to_gguf.py not found")
                logger.info("To export GGUF:")
                logger.info("  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
                logger.info("  2. Run: python llama.cpp/convert_hf_to_gguf.py <model_dir> --outfile model.gguf --outtype f16")

        except ImportError:
            logger.warning("gguf package not installed. Skipping GGUF export.")
            logger.info("To enable GGUF export: pip install gguf")

        # Fallback: save a placeholder
        os.makedirs(output_dir, exist_ok=True)
        placeholder = os.path.join(output_dir, "GGUF_EXPORT_INSTRUCTIONS.txt")
        with open(placeholder, "w") as f:
            f.write("GGUF Export Instructions\n")
            f.write("=" * 60 + "\n\n")
            f.write("The merged model is at: " + model_dir + "\n\n")
            f.write("To convert to GGUF:\n\n")
            f.write("  git clone https://github.com/ggerganov/llama.cpp\n")
            f.write("  cd llama.cpp\n")
            f.write("  python convert_hf_to_gguf.py " + model_dir + " --outfile model.gguf --outtype f16\n\n")
            f.write("For quantized GGUF:\n\n")
            f.write("  ./llama-quantize model.gguf model-q4_k_m.gguf q4_k_m\n\n")
            f.write("To use with Ollama:\n\n")
            f.write("  ollama create mymodel -f Modelfile  # (see Ollama docs)\n")

        logger.info(f"GGUF export instructions saved to {placeholder}")

    except Exception as e:
        logger.error(f"GGUF export failed: {e}")


def export_4bit(
    model_dir: str,
    output_dir: str,
) -> None:
    """Export a 4-bit quantized version of the merged model."""
    logger.info("Exporting 4-bit quantized model...")
    output_path = os.path.join(output_dir, "4bit-quantized")

    try:
        from transformers import BitsAndBytesConfig

        logger.info(f"Loading merged model from {model_dir}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"4-bit quantized model saved to {output_path}")

        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"4-bit export failed: {e}")
        logger.info("Note: For GGUF 4-bit, use llama.cpp quantize instead.")


def generate_model_card(
    output_dir: str,
    adapter_path: str,
    base_model: str,
    training_config_path: str | None = None,
) -> None:
    """Generate a comprehensive model card README.md."""
    # Load training config if available
    training_details = ""
    if training_config_path and os.path.exists(training_config_path):
        try:
            import yaml
            with open(training_config_path, "r") as f:
                cfg = yaml.safe_load(f)
            lora = cfg.get("lora", {})
            training = cfg.get("training", {})
            training_details = f"""
### Training Details

- **Base Model:** {cfg.get('model', {}).get('base_model', base_model)}
- **LoRA Rank (r):** {lora.get('r', 'N/A')}
- **LoRA Alpha:** {lora.get('lora_alpha', 'N/A')}
- **LoRA Dropout:** {lora.get('lora_dropout', 'N/A')}
- **Learning Rate:** {training.get('learning_rate', 'N/A')}
- **Epochs:** {training.get('num_train_epochs', 'N/A')}
- **Batch Size:** {training.get('per_device_train_batch_size', 'N/A')}
- **Gradient Accumulation:** {training.get('gradient_accumulation_steps', 'N/A')}
- **LR Scheduler:** {training.get('lr_scheduler_type', 'N/A')}
- **Precision:** BF16 ({training.get('bf16', False)})
- **Gradient Checkpointing:** {training.get('gradient_checkpointing', False)}
"""
        except Exception:
            pass

    model_card = f"""# NexusAI Fine-Tuned Model

**A Project by Osama**

## Model Description

This model was fine-tuned using the NexusAI Fine-Tuning Pipeline with QLoRA
(Quantized Low-Rank Adaptation) for parameter-efficient instruction tuning.

{training_details}
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Pipeline

Fine-tuned with the [NexusAI Fine-Tuning Pipeline](https://github.com/osama/nexusai-finetuning).

## License

This model inherits the license of the base model. Please refer to the base model's
license for usage terms and conditions.
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info(f"Model card saved to {readme_path}")


def push_to_hub(
    model_dir: str,
    repo_id: str,
    token: str | None = None,
) -> None:
    """Upload the model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        if token:
            api = HfApi(token=token)
        else:
            api = HfApi()

        logger.info(f"Uploading to HuggingFace Hub: {repo_id}")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Successfully uploaded to https://huggingface.co/{repo_id}")

    except Exception as e:
        logger.error(f"Failed to upload to Hub: {e}")
        logger.info("Make sure you're logged in: huggingface-cli login")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NexusAI Model Merge & Export")
    print("  A Project by Osama")
    print("=" * 60 + "\n")

    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        return 1

    # Determine base model
    base_model = args.base_model or detect_base_model(args.adapter_path)

    # GPU check (warn, not error — merging can work on CPU if enough RAM)
    if not torch.cuda.is_available():
        logger.warning("No GPU detected. Merging on CPU will be slow and requires ~32 GB RAM for 7B models.")

    # Load and merge
    model, tokenizer = load_and_merge(args.adapter_path, base_model, args.dtype)

    # Save full model
    if args.export_full and not args.no_full_export:
        save_full_model(model, tokenizer, args.output_dir)

    # Export GGUF
    if args.export_gguf:
        export_gguf(args.output_dir, os.path.join(args.output_dir, "gguf"), args.gguf_quantize)

    # Export 4-bit
    if args.export_4bit:
        export_4bit(args.output_dir, args.output_dir)

    # Generate model card
    if args.model_card:
        generate_model_card(args.output_dir, args.adapter_path, base_model, args.training_config)

    # Push to Hub
    if args.push_to_hub:
        repo_id = args.hub_repo_id
        if not repo_id:
            logger.error("--hub_repo_id required when --push_to_hub is set")
            return 1
        push_to_hub(args.output_dir, repo_id, args.hub_token)

    # Cleanup
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("  MERGE & EXPORT COMPLETE!")
    print("=" * 60)
    print(f"  Merged model:   {args.output_dir}")
    if args.export_gguf:
        print(f"  GGUF:           {args.output_dir}/gguf/")
    if args.export_4bit:
        print(f"  4-bit:          {args.output_dir}/4bit-quantized/")
    print()
    print("  Next steps:")
    print(f"    python run_inference.py --model_path {args.output_dir}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
