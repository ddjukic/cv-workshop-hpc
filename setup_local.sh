#!/usr/bin/env bash
# =============================================================================
# setup_local.sh — Local macOS (Apple Silicon) setup for CV Workshop notebooks
# =============================================================================
# Installs PyTorch with MPS support + all workshop deps into a local .venv.
# For the HPC (CUDA) environment, use setup_workshop_env.sh instead.
#
# Usage:
#   chmod +x setup_local.sh
#   ./setup_local.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ---------------------------------------------------------------------------
# Step 1: Nuke stale venv (cu121-poisoned from HPC setup)
# ---------------------------------------------------------------------------
VENV_DIR="${SCRIPT_DIR}/.venv-local"

if [ -d "$VENV_DIR" ]; then
    info "Existing local venv found at ${VENV_DIR} — reusing"
else
    info "Creating local venv with uv (Python 3.10)..."
    uv venv --python 3.10 "$VENV_DIR"
    ok "Virtual environment created at ${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

# ---------------------------------------------------------------------------
# Step 2: Install HuggingFace CLI (standalone, not from venv)
# ---------------------------------------------------------------------------
if command -v huggingface-cli &>/dev/null; then
    ok "huggingface-cli already installed: $(command -v huggingface-cli)"
else
    info "Installing HuggingFace CLI (standalone installer)..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    ok "HuggingFace CLI installed"
    info "You may need to restart your shell or run: source ~/.bashrc"
fi

# ---------------------------------------------------------------------------
# Step 3: Install PyTorch (plain PyPI — MPS on macOS ARM, no CUDA)
# ---------------------------------------------------------------------------
info "Installing PyTorch + workshop dependencies..."

# CRITICAL: Override the project's pyproject.toml [tool.uv] which sets
# extra-index-url to the cu121 PyTorch index. That index has no macOS wheels.
# --no-config tells uv to ignore pyproject.toml/uv.toml settings.
# --index-url forces PyPI only (no cu121 fallthrough).
unset UV_TORCH_BACKEND 2>/dev/null || true
unset UV_EXTRA_INDEX_URL 2>/dev/null || true

uv pip install \
    --no-config \
    --python "$VENV_PYTHON" \
    --index-url "https://pypi.org/simple" \
    "torch>=2.4" \
    "torchvision>=0.19" \
    "ultralytics>=8.4" \
    "transformers>=5.0" \
    "supervision>=0.25" \
    "matplotlib>=3.8" \
    "Pillow>=10.0" \
    "numpy>=1.26" \
    "opencv-python-headless>=4.9" \
    "ipykernel>=6.29" \
    "ipywidgets>=8.1" \
    "jupyter-bbox-widget>=0.5" \
    "huggingface-hub>=0.27" \
    "safetensors>=0.4" \
    "accelerate>=1.0" \
    "tqdm"

ok "All packages installed"

# --- CLIP for Ultralytics SAM3 text prompts ---
info "Installing CLIP for SAM3 text prompts..."
uv pip install \
    --no-config \
    --python "$VENV_PYTHON" \
    "clip @ git+https://github.com/ultralytics/CLIP.git"

ok "CLIP installed (required for SAM3 text prompts via Ultralytics)"

# ---------------------------------------------------------------------------
# Step 4: Verify
# ---------------------------------------------------------------------------
info "Verifying installation..."

"$VENV_PYTHON" -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  MPS:          {torch.backends.mps.is_available()}')
print(f'  CUDA:         {torch.cuda.is_available()}')

import transformers
print(f'  Transformers: {transformers.__version__}')

import ultralytics
print(f'  Ultralytics:  {ultralytics.__version__}')

import supervision
print(f'  Supervision:  {supervision.__version__}')

import clip
print(f'  CLIP:         installed (for SAM3 text prompts)')
"

ok "All imports verified"

# ---------------------------------------------------------------------------
# Step 5: Register Jupyter kernel
# ---------------------------------------------------------------------------
info "Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install \
    --user \
    --name cv-workshop-local \
    --display-name "CV Workshop (local)"
ok "Kernel 'cv-workshop-local' registered"

# ---------------------------------------------------------------------------
# Step 6: Pre-download model weights
# ---------------------------------------------------------------------------
info "Pre-downloading model weights (this may take a few minutes)..."

# SAM3
# NOTE: Must use Sam3Processor directly — AutoProcessor resolves to
# Sam3VideoProcessor in transformers>=5.x which lacks text= support.
"$VENV_PYTHON" -c "
from transformers import Sam3Processor, Sam3Model
import os
token = os.getenv('HF_TOKEN', None)
print('  Downloading SAM3 processor...')
Sam3Processor.from_pretrained('facebook/sam3', token=token)
print('  Downloading SAM3 model...')
Sam3Model.from_pretrained('facebook/sam3', token=token)
print('  SAM3 cached.')
" 2>&1 || {
    fail "SAM3 download failed (may need HF_TOKEN with model access)"
    info "Run: huggingface-cli login   then re-run this script"
}

# YOLOE
"$VENV_PYTHON" -c "
from ultralytics import YOLO
print('  Downloading yoloe-26n-seg.pt...')
YOLO('yoloe-26n-seg.pt')
print('  YOLOE cached.')
" 2>&1 || {
    fail "YOLOE download failed"
}

# YOLO26n
"$VENV_PYTHON" -c "
from ultralytics import YOLO
print('  Downloading yolo26n.pt...')
YOLO('yolo26n.pt')
print('  YOLO26n cached.')
" 2>&1 || {
    fail "YOLO26n download failed"
}

ok "Model weights cached"

# ---------------------------------------------------------------------------
# Step 7: Also update the HPC setup script with HF CLI installer
# ---------------------------------------------------------------------------
# (This is done by the script itself, not a runtime step)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
ok "Local setup complete!"
echo ""
info "Activate:     source ${VENV_DIR}/bin/activate"
info "HF login:     huggingface-cli login"
info "Jupyter:      jupyter lab notebooks/"
info "Kernel:       Select 'CV Workshop (local)' in notebook"
echo ""
info "NOTE: This uses .venv-local (not .venv) to avoid conflicts"
info "      with the HPC setup_workshop_env.sh which targets CUDA."
