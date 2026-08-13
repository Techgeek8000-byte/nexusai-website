# GPU Requirements Guide

> Hardware recommendations for the NexusAI Fine-Tuning Pipeline
> **A Project by Osama**

---

## TL;DR — What Do You Need?

| Task | Minimum GPU | Recommended GPU | Est. VRAM Used |
|------|-------------|-----------------|----------------|
| Fine-tune 1.5B (QLoRA) | 8 GB (GTX 1060) | 12 GB (T4) | ~6 GB |
| Fine-tune 3B (QLoRA) | 12 GB (T4) | 16 GB (V100) | ~10 GB |
| Fine-tune 7B (QLoRA) | 16 GB (T4) | 24 GB (RTX 3090/4090) | ~14 GB |
| Fine-tune 7B (full) | 40 GB (A100) | 80 GB (H100) | ~60 GB |
| Fine-tune 14B (QLoRA) | 24 GB (A10G) | 48 GB (A6000) | ~24 GB |
| Fine-tune 72B (QLoRA) | 80 GB (A100) | 4×80 GB (H100) | ~72 GB |
| Merge 7B model | CPU (32 GB RAM) | Any GPU | ~14 GB RAM |
| Inference 7B | 8 GB | 16 GB | ~8 GB |
| DPO 7B | 24 GB (A10G) | 2×24 GB | ~22 GB |

---

## VRAM Breakdown by Model Size (QLoRA)

### 7B Model (Qwen2.5-7B) — ~14 GB VRAM

| Component | VRAM |
|-----------|------|
| 4-bit quantized model weights | ~4.5 GB |
| LoRA adapters (r=16, 7 modules) | ~0.3 GB |
| Optimizer states (paged AdamW 8-bit) | ~1.5 GB |
| Gradients | ~0.5 GB |
| Activations (with gradient checkpointing) | ~4.0 GB |
| Activations (without gradient checkpointing) | ~12.0 GB |
| KV Cache & overhead | ~1.5 GB |
| **Total (with grad checkpointing)** | **~14 GB** |
| **Total (without grad checkpointing)** | **~24 GB** |

### 3B Model (Qwen2.5-3B) — ~7 GB VRAM

| Component | VRAM |
|-----------|------|
| 4-bit model weights | ~2.0 GB |
| LoRA adapters | ~0.15 GB |
| Optimizer states | ~0.8 GB |
| Gradients | ~0.3 GB |
| Activations (grad ckpt) | ~2.0 GB |
| KV Cache & overhead | ~1.0 GB |
| **Total** | **~7 GB** |

### 14B Model (Qwen2.5-14B) — ~26 GB VRAM

| Component | VRAM |
|-----------|------|
| 4-bit model weights | ~8.5 GB |
| LoRA adapters | ~0.6 GB |
| Optimizer states | ~3.0 GB |
| Gradients | ~1.0 GB |
| Activations (grad ckpt) | ~8.0 GB |
| KV Cache & overhead | ~2.0 GB |
| **Total** | **~26 GB** |

---

## Recommended GPUs by Budget

### Free Options

| Platform | GPU | VRAM | Notes |
|----------|-----|------|-------|
| **Google Colab** | T4 | 16 GB | Free tier, 12hr limit, may disconnect |
| **Google Colab Pro** | T4 / A100 | 16 / 40 GB | $10/mo, more stable |
| **Kaggle Notebooks** | T4 × 2 | 32 GB | Free, 30hr/week, 2 GPUs! |
| **HuggingFace Spaces** | T4 | 16 GB | Free, Docker-based |
| **Lightning AI Studios** | T4 | 16 GB | Free 15 credits/mo |

### Budget GPUs ($100–$500)

| GPU | VRAM | Price (Used) | Can Fine-Tune |
|-----|------|-------------|---------------|
| RTX 3060 12GB | 12 GB | ~$200 | 3B (QLoRA), 7B (tight) |
| RTX 3060 Ti 8GB | 8 GB | ~$200 | 1.5B–3B only |
| RTX 3070 Ti 8GB | 8 GB | ~$250 | 1.5B–3B only |
| RTX 3080 10GB | 10 GB | ~$300 | 3B (QLoRA) |
| RTX 4060 Ti 16GB | 16 GB | ~$400 | **7B (QLoRA)** |
| RTX 4070 Ti Super 16GB | 16 GB | ~$800 | **7B (QLoRA)** |

### Mid-Range GPUs ($500–$1,500)

| GPU | VRAM | Price (Used) | Can Fine-Tune |
|-----|------|-------------|---------------|
| RTX 3090 | 24 GB | ~$600 | **7B (QLoRA), 14B (tight)** |
| RTX 4090 | 24 GB | ~$1,500 | **7B (QLoRA), 14B (QLoRA)** |
| RTX A4000 | 16 GB | ~$700 | 7B (QLoRA) |
| RTX A5000 | 24 GB | ~$1,200 | 7B–14B (QLoRA) |

### Data Center / Cloud GPUs

| GPU | VRAM | Cloud Cost (hr) | Can Fine-Tune |
|-----|------|-----------------|---------------|
| NVIDIA T4 | 16 GB | $0.35 (GCP) | 7B (QLoRA) |
| NVIDIA V100 | 16/32 GB | $0.90 (GCP) | 7B (QLoRA) |
| NVIDIA A10G | 24 GB | $1.00 (AWS) | 7B–14B (QLoRA) |
| NVIDIA A100 | 40/80 GB | $2.50–$4.00 | Up to 72B (QLoRA) |
| NVIDIA H100 | 80 GB | $3.50–$5.00 | Any model |
| NVIDIA A6000 | 48 GB | $3.00 (Lambda) | Up to 30B (QLoRA) |

**Best Cloud GPU Deals:**
- [RunPod](https://runpod.io) — RTX 3090 from $0.30/hr, A100 from $1.64/hr
- [Lambda Labs](https://lambdalabs.com) — A100 from $1.10/hr
- [Vast.ai](https://vast.ai) — Community GPUs, cheapest option
- [Google Cloud](https://cloud.google.com/gpu) — T4 from $0.35/hr (spot $0.05)
- [AWS](https://aws.amazon.com/ec2/instance-types/g4/) — G4dn with T4 GPUs

---

## Performance Benchmarks

### Training Speed (tokens/second) — 7B QLoRA

| GPU | Batch 2, Seq 2048 | Batch 1, Seq 2048 | Relative Speed |
|-----|-------------------|-------------------|----------------|
| RTX 3060 | ~1,200 tok/s | ~800 tok/s | 1.0× |
| RTX 3060 Ti | ~1,800 tok/s | ~1,200 tok/s | 1.5× |
| RTX 3090 | ~3,500 tok/s | ~2,500 tok/s | 2.9× |
| RTX 4060 Ti 16GB | ~2,500 tok/s | ~1,800 tok/s | 2.1× |
| RTX 4090 | ~7,000 tok/s | ~5,000 tok/s | 5.8× |
| T4 | ~800 tok/s | ~600 tok/s | 0.7× |
| A100 40GB | ~6,000 tok/s | ~4,500 tok/s | 5.0× |
| A100 80GB | ~8,000 tok/s | ~6,000 tok/s | 6.7× |
| H100 80GB | ~14,000 tok/s | ~10,000 tok/s | 11.7× |

### Inference Speed (tokens/second) — 7B Merged (bfloat16)

| GPU | Speed (batch 1) | Speed (batch 4) |
|-----|-----------------|-----------------|
| RTX 3060 | ~45 tok/s | ~80 tok/s |
| RTX 4090 | ~120 tok/s | ~250 tok/s |
| T4 | ~30 tok/s | ~60 tok/s |
| A100 | ~100 tok/s | ~200 tok/s |

> **Note:** Benchmarks are approximate and depend on sequence length, batch size, and model architecture.

---

## Memory Optimization Techniques

### 1. Gradient Checkpointing (Saves ~40% VRAM)

Enabled by default in the pipeline. Trades compute for memory.

```yaml
training:
  gradient_checkpointing: true
```

### 2. Reduce Batch Size

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Keep effective batch size
```

### 3. Shorter Sequences

Most instruction tuning works well with shorter contexts.

```yaml
model:
  max_seq_length: 1024  # Instead of 2048 or 4096
```

### 4. Lower LoRA Rank

```yaml
lora:
  r: 8  # Instead of 16 or 32
  lora_alpha: 16
```

### 5. 8-bit Optimizer

```yaml
training:
  optim: "paged_adamw_8bit"  # Already default
```

### 6. CPU Offloading (Slow but Works)

```python
# In the script or manually:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={0: "12GiB", "cpu": "30GiB"},
    offload_folder="offload",
)
```

### 7. Use a Smaller Model

| Model | Parameters | VRAM (QLoRA) |
|-------|-----------|--------------|
| Qwen2.5-0.5B | 0.5B | ~3 GB |
| Qwen2.5-1.5B | 1.5B | ~5 GB |
| Qwen2.5-3B | 3B | ~7 GB |
| Qwen2.5-7B | 7B | ~14 GB |
| Qwen2.5-14B | 14B | ~26 GB |
| Qwen2.5-32B | 32B | ~48 GB |
| Qwen2.5-72B | 72B | ~72 GB+ |

---

## Free GPU Quick Start

### Google Colab (Easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. Upload the `nexusai-finetuning/` folder or clone from GitHub
4. Run: `!bash setup.sh`
5. Train: `!python finetune_qlora.py --config finetune_config.yaml`

> **Tip:** Colab disconnects after ~12 hours. Use `--max_steps` to control training duration.

### Kaggle (Most Free GPU Time)

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create a new notebook
3. **Settings → Accelerator → GPU T4 x2**
4. Upload files via the dataset feature
5. You get **30 hours/week** of GPU time (2 GPUs!)

### HuggingFace Spaces

1. Create a Space with a Docker SDK
2. Set GPU to T4 (free tier allows limited GPU)
3. Use for inference and small training runs

---

## Cost Estimates

| Scenario | GPU | Time | Cost |
|----------|-----|------|------|
| 7B QLoRA, 10K samples, 3 epochs | T4 (Colab) | ~2 hrs | Free |
| 7B QLoRA, 10K samples, 3 epochs | RTX 4090 | ~30 min | $0.25 (RunPod) |
| 7B QLoRA, 50K samples, 3 epochs | T4 (Colab) | ~10 hrs | Free |
| 7B QLoRA, 50K samples, 3 epochs | A100 | ~1 hr | $2.50 |
| 7B DPO, 10K pairs, 1 epoch | A100 | ~45 min | $1.90 |
| 14B QLoRA, 10K samples, 3 epochs | A100 | ~2 hrs | $5.00 |
| 72B QLoRA, 10K samples, 1 epoch | 2×A100 | ~3 hrs | $15.00 |

---

*Last updated: 2025 | A Project by Osama*