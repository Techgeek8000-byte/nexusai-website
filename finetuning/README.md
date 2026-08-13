# NexusAI Fine-Tuning Pipeline

> A comprehensive, production-ready fine-tuning pipeline for large language models.
> **A Project by Osama**

---

## Overview

NexusAI Fine-Tuning Pipeline is a complete toolkit for fine-tuning large language models using **QLoRA** (Quantized Low-Rank Adaptation) and **DPO** (Direct Preference Optimization). It is designed for Qwen 2.5 models but works with any HuggingFace-compatible LLM.

### Key Features

- **QLoRA Fine-Tuning** — Fine-tune 7B models on a single 16GB GPU with 4-bit quantization
- **DPO Alignment** — Align models with human preferences using Direct Preference Optimization
- **Multi-Format Export** — Export merged models in full precision, GGUF (for Ollama), and 4-bit
- **Comprehensive Evaluation** — Perplexity, loss metrics, and qualitative comparison against the base model
- **Interactive Inference** — Chat interface with latency benchmarking and batch processing
- **W&B Integration** — Full experiment tracking with Weights & Biases
- **Production-Ready** — Proper error handling, logging, OOM recovery, and progress tracking

### Pipeline Architecture

```
┌─────────────────────┐
│  prepare_dataset.py │  Download & format instruction data
└──────────┬──────────┘
           │ data/train.jsonl, data/val.jsonl
           ▼
┌─────────────────────┐
│  finetune_qlora.py  │  QLoRA supervised fine-tuning
└──────────┬──────────┘
           │ outputs/qlora-adapters/
           ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  evaluate_model.py  │     │ advanced_finetune_dpo.py │
│  (optional compare)  │     │ (preference alignment)   │
└──────────┬──────────┘     └────────────┬─────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────┐
│ merge_and_export.py │     │   merge_and_export.py    │
│ (full/GGUF/4-bit)   │     │  (DPO merged model)      │
└──────────┬──────────┘     └────────────┬─────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────────────────────────┐
│             run_inference.py                │
│   (interactive chat / batch / comparison)    │
└─────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware

| Component       | Minimum           | Recommended       |
|-----------------|-------------------|-------------------|
| GPU             | NVIDIA T4 (16 GB) | NVIDIA RTX 4090   |
| VRAM            | 12 GB             | 24 GB+            |
| System RAM      | 16 GB             | 32 GB+            |
| Disk Space      | 50 GB             | 100 GB+           |
| Python          | 3.10+             | 3.12+             |
| CUDA            | 11.8+             | 12.4+             |

> **No GPU?** See [gpu_requirements.md](gpu_requirements.md) for free GPU options and cloud alternatives.

### Software

- **NVIDIA Drivers** — Latest stable
- **CUDA Toolkit** — 12.x recommended
- **Python** — 3.10 or higher (3.12+ recommended)
- **Git** — For cloning models from HuggingFace
- **HuggingFace CLI** — `pip install huggingface_hub` (optional, for pushing models)

---

## Quick Start

### 1. Clone and Set Up

```bash
cd nexusai-finetuning
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Prepare Your Dataset

```bash
# Download and format the Alpaca dataset (52K instructions)
python prepare_dataset.py --dataset tatsu-lab/alpaca --max_samples 10000

# Or use a higher-quality dataset
python prepare_dataset.py --dataset teknium/OpenHermes-2.5 --max_samples 50000

# Or use your own JSONL file
python prepare_dataset.py --dataset my_instructions.jsonl --format alpaca --local_file
```

Your custom JSONL should have this format:
```jsonl
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses..."}
```

### 3. Fine-Tune with QLoRA

```bash
# Using the config file
python finetune_qlora.py --config finetune_config.yaml

# Quick test (200 steps)
python finetune_qlora.py --config finetune_config.yaml --max_steps 200 --output_dir outputs/test-run

# Override specific settings
python finetune_qlora.py --config finetune_config.yaml \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --num_train_epochs 1
```

### 4. Evaluate the Model

```bash
# Basic evaluation (perplexity + sample outputs)
python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter

# Compare fine-tuned vs base model
python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter --compare_with_base

# Evaluate a merged model
python evaluate_model.py --model_path outputs/merged --no_adapter
```

### 5. Merge and Export

```bash
# Merge adapters into full model
python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter

# Also export GGUF for Ollama
python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter --export_gguf

# Push to HuggingFace Hub
python merge_and_export.py \
  --adapter_path outputs/qlora-adapters/final_adapter \
  --push_to_hub --hub_repo_id your-username/nexusai-qwen-7b
```

### 6. Test Inference

```bash
# Interactive chat
python run_inference.py --model_path outputs/merged

# Batch inference from a file
python run_inference.py --model_path outputs/merged \
  --input_file prompts.txt --output_file results.jsonl

# Compare base vs fine-tuned side-by-side
python run_inference.py --model_path outputs/merged --compare_with_base
```

### 7. Advanced: DPO Alignment (Optional)

```bash
# Create a sample preference dataset
python advanced_finetune_dpo.py --dataset_path data/dpo_sample.jsonl --beta 0.1

# DPO with a HuggingFace preference dataset
python advanced_finetune_dpo.py \
  --hf_dataset HuggingFaceH4/ultrafeedback_binarized \
  --max_samples 10000 \
  --beta 0.1 \
  --num_train_epochs 1

# DPO on top of an SFT adapter
python advanced_finetune_dpo.py \
  --sft_adapter_path outputs/qlora-adapters/final_adapter \
  --dataset_path data/dpo_pairs.jsonl
```

---

## Configuration Options

All training parameters are controlled via `finetune_config.yaml`. Key settings:

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `Qwen/Qwen2.5-7B` | HuggingFace model identifier |
| `max_seq_length` | `2048` | Max tokens per sample (increase for long documents) |
| `dtype` | `bfloat16` | Compute precision (use `float16` for V100) |

### LoRA

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | `16` | LoRA rank. Higher = more capacity. Try 8, 16, 32, 64 |
| `lora_alpha` | `32` | Scaling factor. Usually 2× the rank |
| `lora_dropout` | `0.05` | Dropout for regularization |
| `target_modules` | 7 modules | Which layers to adapt |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `2e-4` | Peak LR for QLoRA |
| `num_train_epochs` | `3` | Full passes over the data |
| `per_device_train_batch_size` | `2` | Samples per GPU per step |
| `gradient_accumulation_steps` | `4` | Steps before weight update |
| `gradient_checkpointing` | `true` | Saves VRAM at ~20% speed cost |

> **Effective batch size** = `batch_size × gradient_accumulation × num_gpus`

---

## Project Structure

```
nexusai-finetuning/
├── setup.sh                    # Environment setup script
├── finetune_config.yaml        # Training configuration
├── prepare_dataset.py          # Dataset download & formatting
├── finetune_qlora.py           # Main QLoRA fine-tuning script
├── evaluate_model.py           # Model evaluation & comparison
├── merge_and_export.py         # Merge adapters, export formats
├── run_inference.py            # Interactive & batch inference
├── advanced_finetune_dpo.py    # DPO preference alignment
├── README.md                   # This file
├── gpu_requirements.md         # Hardware guide
├── data/                       # Training/validation datasets
│   ├── train.jsonl
│   ├── val.jsonl
│   └── dataset_stats.json
├── models/                     # Cached models
├── outputs/                    # Training outputs & merged models
│   ├── qlora-adapters/         # LoRA adapter checkpoints
│   ├── dpo-adapters/           # DPO adapter checkpoints
│   ├── merged/                 # Full merged model
│   └── evaluation/             # Evaluation results
└── logs/                       # Training & chat logs
```

---

## Troubleshooting

### Out of Memory (OOM)

**During model loading:**
- Reduce `max_seq_length` (try 1024 or 512)
- Use a smaller model (Qwen2.5-1.5B or 3B)
- Close other GPU-consuming processes

**During training:**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to compensate
- Enable `gradient_checkpointing: true`
- Reduce LoRA rank (`r: 8`)
- Add `--no_gradient_checkpointing` and reduce sequence length

**During merging:**
- Merging requires ~32 GB RAM for a 7B model
- The merged model needs ~14 GB disk space

### CUDA / GPU Issues

```bash
# Check GPU status
nvidia-smi

# Verify PyTorch sees the GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use smaller model |
| `ModuleNotFoundError: bitsandbytes` | `pip install bitsandbytes` |
| `ValueError: Tokenizer not found` | Check model path and `trust_remote_code` |
| `EOFError: Ran out of input` | Corrupted download; clear HF cache and retry |
| `Flash Attention not available` | Ignored — falls back to SDPA automatically |
| `WandB not initialized` | Set `wandb_disabled: true` in config or `--wandb_disabled` |

### Dataset Issues

```bash
# If dataset download fails, try with --trust_remote_code
# Or download manually and use --local_file

# Check your JSONL format
python -c "import json; [json.loads(l) for l in open('data/train.jsonl')]"
```

---

## Tips for Best Results

1. **Data quality matters more than quantity** — 5K high-quality examples often beats 50K noisy ones
2. **Use a diverse dataset** — Mix different instruction types and domains
3. **Monitor eval loss** — If training loss drops but eval loss rises, reduce epochs or add dropout
4. **Start with defaults** — The provided config works well for most 7B instruction-tuning tasks
5. **Use W&B** — It provides invaluable insights into training dynamics
6. **Compare against the base** — Always run `--compare_with_base` to verify improvement

---

## Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Efficient 4-bit fine-tuning
- [DPO Paper](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [Qwen2.5 Models](https://huggingface.co/Qwen) — Base models
- [TRL Library](https://huggingface.co/docs/trl/) — Training library
- [PEFT Library](https://huggingface.co/docs/peft/) — LoRA implementation
- [Weights & Biases](https://wandb.ai/) — Experiment tracking

---

## License

This pipeline is provided as-is for educational and research purposes. The fine-tuned models inherit the license of their base model.

---

*Built with care by Osama*
