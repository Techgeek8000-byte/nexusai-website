#!/usr/bin/env bash
# ============================================================================
# NexusAI Fine-Tuning Pipeline — Setup Script
# A Project by Osama
# ============================================================================
# This script sets up the complete fine-tuning environment including:
#   - Python virtual environment
#   - All required dependencies
#   - GPU verification
#   - Directory structure
# ============================================================================

set -euo pipefail

# ---------- Color helpers ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- Project root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Configuration ----------
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_VERSION="3.12"

# ============================================================================
# 1. Check Python version
# ============================================================================
info "Checking Python installation..."

if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed. Please install Python ${PYTHON_VERSION} or higher."
    exit 1
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    error "Python ${PYTHON_VERSION}+ is required. Found Python ${PY_MAJOR}.${PY_MINOR}."
    exit 1
fi

success "Python $(python3 --version) detected"

# ============================================================================
# 2. Create virtual environment
# ============================================================================
info "Creating virtual environment at ${VENV_DIR}..."

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists. Recreating..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
success "Virtual environment created"

# Activate venv
source "${VENV_DIR}/bin/activate"
success "Virtual environment activated"

# Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
success "Pip upgraded"

# ============================================================================
# 3. Install PyTorch (with CUDA support)
# ============================================================================
info "Installing PyTorch with CUDA support..."

# Detect CUDA version if possible
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    info "Detected CUDA ${CUDA_VERSION}"
fi

# Install PyTorch — use CUDA 12.4 by default, fall back to CPU if needed
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
    info "PyTorch already installed with CUDA ${TORCH_CUDA}"
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
    success "PyTorch installed (CUDA 12.4)"
fi

# ============================================================================
# 4. Install core dependencies
# ============================================================================
info "Installing core dependencies..."

pip install \
    transformers>=4.46.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    datasets>=3.0.0 \
    bitsandbytes>=0.44.0 \
    accelerate>=1.1.0 \
    wandb>=0.18.0 \
    scipy>=1.14.0 \
    sentencepiece>=0.2.0 \
    protobuf>=5.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.67.0 \
    ninja>=1.11.0 \
    packaging>=24.0 \
    --quiet

success "Core dependencies installed"

# ============================================================================
# 5. Install optional dependencies
# ============================================================================
info "Installing optional dependencies..."

pip install \
    lm-eval>=0.4.0 \
    gguf>=0.10.0 \
    auto-gptq>=0.7.0 \
    optimum>=1.21.0 \
    --quiet 2>/dev/null || warn "Some optional dependencies failed (non-critical)"

success "Optional dependencies installed"

# ============================================================================
# 6. Verify GPU availability
# ============================================================================
info "Verifying GPU availability..."

GPU_CHECK=$(python3 << 'EOF'
import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / (1024 ** 3)
        print(f"GPU {i}: {props.name} | VRAM: {vram_gb:.1f} GB | CUDA: {torch.version.cuda}")
    print(f"AVAILABLE:{gpu_count}")
else:
    print("AVAILABLE:0")
EOF
)

if echo "$GPU_CHECK" | grep -q "AVAILABLE:0"; then
    echo ""
    error "======================================================================="
    error "  NO GPU DETECTED!"
    error "======================================================================="
    error ""
    error "Fine-tuning requires a CUDA-compatible GPU. Options:"
    error ""
    error "  1. Local GPU — Ensure you have an NVIDIA GPU with CUDA drivers installed."
    error "     Verify with: nvidia-smi"
    error ""
    error "  2. Google Colab (Free)"
    error "     Go to: https://colab.research.google.com"
    error "     Runtime → Change runtime type → T4 GPU"
    error ""
    error "  3. Kaggle Notebooks (Free)"
    error "     Go to: https://www.kaggle.com/code"
    error "     Settings → Accelerator → GPU T4 x2"
    error ""
    error "  4. RunPod / Lambda Labs (Paid, cheap)"
    error "     Rent a GPU instance for ~$0.40/hr"
    error ""
    error "  5. HuggingFace Spaces (Free tier available)"
    error "     https://huggingface.co/spaces"
    error ""
    echo ""
    warn "Dependencies are installed, but training will not work without a GPU."
    warn "You can still use CPU mode for dataset preparation and inference testing."
else
    echo "$GPU_CHECK" | while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            success "$line"
        fi
    done
fi

# ============================================================================
# 7. Create project directories
# ============================================================================
info "Creating project directories..."

for dir in data models outputs logs; do
    mkdir -p "${SCRIPT_DIR}/${dir}"
    # Add .gitkeep to preserve empty dirs
    touch "${SCRIPT_DIR}/${dir}/.gitkeep"
    success "Created ${dir}/"
done

# ============================================================================
# 8. Print summary
# ============================================================================
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  NexusAI Fine-Tuning Pipeline — Setup Complete!${NC}"
echo -e "${GREEN}  A Project by Osama${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "  ${CYAN}Virtual Environment:${NC}  ${VENV_DIR}"
echo -e "  ${CYAN}Activate with:${NC}      source ${VENV_DIR}/bin/activate"
echo ""
echo -e "  ${CYAN}Quick Start:${NC}"
echo ""
echo "    1. Prepare your dataset:"
echo "       python prepare_dataset.py --dataset tatsu-lab/alpaca --max_samples 10000"
echo ""
echo "    2. Start fine-tuning:"
echo "       python finetune_qlora.py --config finetune_config.yaml"
echo ""
echo "    3. Evaluate the model:"
echo "       python evaluate_model.py --adapter_path outputs/latest"
echo ""
echo "    4. Merge and export:"
echo "       python merge_and_export.py --adapter_path outputs/latest --export_gguf"
echo ""
echo "    5. Test inference:"
echo "       python run_inference.py --model_path outputs/merged"
echo ""
echo -e "  ${CYAN}Documentation:${NC} See README.md for detailed instructions."
echo -e "  ${CYAN}GPU Guide:${NC}     See gpu_requirements.md for hardware recommendations."
echo ""
