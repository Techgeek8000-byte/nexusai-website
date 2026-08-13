# NexusAI — Free AI Platform

A 100% free AI platform with multi-model chat, agent tools, and a fine-tuning pipeline — built by Osama.

**Live Website:** [nexusai-website-two.vercel.app](https://nexusai-website-two.vercel.app/)

---

## What's Inside

| Feature | Description |
|---------|-------------|
| Landing Page (`/`) | Marketing site with features, stats, API docs |
| Live AI Chat (`/chat`) | Real multi-model chat with Qwen 2.5 models |
| API Route (`/api/chat`) | Server-side HF Inference API proxy |
| Architecture PDF | System architecture documentation |
| Business Plan | Business strategy document |
| Fine-tuning Pipeline | QLoRA fine-tuning scripts for Google Colab |

## Tech Stack

- **Frontend:** Next.js 16, TypeScript, Tailwind CSS 4, shadcn/ui
- **AI Models:** Qwen 2.5 (7B, 3B, 0.5B) + CodeQwen 7B via HF Inference API
- **Deployment:** Vercel (free tier)
- **Fine-tuning:** Python + QLoRA on Google Colab (free T4 GPU)

## Project Structure

```
src/
  app/
    page.tsx              # Landing page
    chat/page.tsx          # Live AI chat
    api/chat/route.ts      # HF Inference API proxy
    api/route.ts           # Health check endpoint
    layout.tsx             # Root layout
    globals.css            # Global styles
  components/ui/          # shadcn/ui components
  hooks/                  # Custom React hooks
  lib/utils.ts            # Utility functions
public/
  docs/
    NexusAI_Architecture.pdf
    NexusAI_Business_Plan.docx
  logo.svg
  robots.txt
finetuning/               # Google Colab fine-tuning scripts
  finetune_qlora.py
  prepare_dataset.py
  evaluate_model.py
  merge_and_export.py
  run_inference.py
  advanced_finetune_dpo.py
  finetune_config.yaml
  setup.sh
  gpu_requirements.md
  README.md
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- A HuggingFace account with an access token

### 1. Clone & Install

```bash
git clone https://github.com/Techgeek8000-byte/nexusai-website.git
cd nexusai-website
npm install
```

### 2. Set Environment Variable

```bash
# Create .env.local file
echo "HF_TOKEN=hf_your_token_here" > .env.local
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Click **"Chat"** in the navbar to use the AI.

### 4. Deploy to Vercel

1. Push to GitHub
2. Connect repo at [vercel.com](https://vercel.com)
3. Add `HF_TOKEN` in Vercel Settings → Environment Variables
4. Deploy

## AI Chat Features

- **4 Models:** Qwen 2.5 7B, 3B, 0.5B + CodeQwen 7B
- **Agent Tools:** Calculator, Code Executor, Date/Time (auto-detected)
- **Quick Actions:** One-click prompts for Code, Search, Urdu, Math
- **System Prompt Editor:** Customize AI behavior
- **Temperature & Token Control:** Full parameter tuning
- **English + Urdu:** Bilingual support

## Fine-Tuning on Google Colab

The `finetuning/` folder contains a complete QLoRA pipeline:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `finetuning/` folder or clone the repo
3. Run: `chmod +x setup.sh && ./setup.sh`
4. Prepare data: `python prepare_dataset.py --dataset tatsu-lab/alpaca`
5. Fine-tune: `python finetune_qlora.py --config finetune_config.yaml`
6. Evaluate: `python evaluate_model.py --adapter_path outputs/qlora-adapters/final_adapter`
7. Merge: `python merge_and_export.py --adapter_path outputs/qlora-adapters/final_adapter`

See `finetuning/README.md` for full details.

## Cost

**$0/month** — Everything runs on free tiers:
- Vercel → website hosting
- HuggingFace Inference API → AI models
- Google Colab → fine-tuning (T4 GPU)

## License

MIT — Built with love by Osama.
