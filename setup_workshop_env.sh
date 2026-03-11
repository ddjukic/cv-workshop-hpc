#!/usr/bin/env bash
# =============================================================================
# setup_workshop_env.sh — CV Workshop HPC Environment Setup
# =============================================================================
# Target: VSC-5 (A40 GPU, CUDA 12.3, conda Python 3.10.14, JupyterHub)
#
# This script creates a fully isolated Python environment using uv, installs
# PyTorch + workshop dependencies, registers a Jupyter kernel, and pre-downloads
# model weights (must run on a login node with internet).
#
# Usage:
#   chmod +x setup_workshop_env.sh
#   ./setup_workshop_env.sh            # Full install
#   ./setup_workshop_env.sh --dry-run  # Preview commands without executing
#   ./setup_workshop_env.sh --help     # Show help text
#
# Idempotent: safe to re-run. Existing components are detected and skipped.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
TOTAL_STEPS=12
STEP_RESULTS=()   # Tracks "ok" / "skip" / "warn" / "fail" per step
START_TIME="$(date +%s)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
step()  { echo -e "\n${GREEN}${BOLD}[Step $1/${TOTAL_STEPS}]${NC} $2"; }

run() {
    # Execute a command, or echo it in dry-run mode.
    if $DRY_RUN; then
        echo -e "  ${CYAN}[DRY-RUN]${NC} $*"
        return 0
    else
        "$@"
    fi
}

record() {
    # Record the result of a step: ok | skip | warn | fail
    STEP_RESULTS+=("$1")
}

usage() {
    cat <<'USAGE'
Usage: setup_workshop_env.sh [OPTIONS]

Options:
  --dry-run   Print commands instead of executing them
  --help      Show this help message and exit

Environment variables (optional overrides):
  UV_CACHE_DIR    Override the uv cache directory
  WORKSHOP_DIR    Override the workshop directory (default: script location)

Examples:
  ./setup_workshop_env.sh              # Full install
  ./setup_workshop_env.sh --dry-run    # Preview what would happen
USAGE
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --help|-h)  usage ;;
        *)
            fail "Unknown argument: $arg"
            usage
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} CV Workshop — HPC Environment Setup${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
info "Script directory : ${SCRIPT_DIR}"
info "Dry-run mode     : ${DRY_RUN}"
info "Date             : $(date '+%Y-%m-%d %H:%M:%S')"
info "User             : ${USER:-$(whoami)}"
info "Hostname         : $(hostname)"
echo ""

WORKSHOP_DIR="${WORKSHOP_DIR:-${SCRIPT_DIR}}"

# ============================================================================
# Step 1/12 — Install uv
# ============================================================================
step "1" "Install uv (fast Python package manager)"

if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
    record "skip"
else
    info "Installing uv via official installer..."
    if run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'; then
        # Ensure ~/.local/bin is on PATH for the rest of this script
        export PATH="${HOME}/.local/bin:${PATH}"
        if ! $DRY_RUN; then
            ok "uv installed: $(uv --version)"
        fi
        record "ok"
    else
        fail "uv installation failed. Check network connectivity (must be on login node)."
        record "fail"
        # uv is critical — abort
        exit 1
    fi
fi

# Make sure ~/.local/bin is on PATH regardless
export PATH="${HOME}/.local/bin:${PATH}"

# ============================================================================
# Step 2/12 — Install HuggingFace CLI (standalone, not pip)
# ============================================================================
step "2" "Install HuggingFace CLI (standalone installer)"

# The venv-installed huggingface-cli often doesn't work (missing from .venv/bin/).
# The standalone installer places it in ~/.local/bin, which is on PATH.
if command -v huggingface-cli &>/dev/null; then
    ok "huggingface-cli already installed: $(command -v huggingface-cli)"
    record "skip"
else
    info "Installing HuggingFace CLI via official standalone installer..."
    if run bash -c 'curl -LsSf https://hf.co/cli/install.sh | bash'; then
        export PATH="${HOME}/.local/bin:${PATH}"
        if ! $DRY_RUN && command -v huggingface-cli &>/dev/null; then
            ok "huggingface-cli installed: $(command -v huggingface-cli)"
        else
            ok "huggingface-cli installer ran (may need shell restart)"
        fi
        record "ok"
    else
        warn "HuggingFace CLI install failed — you can install it manually later:"
        warn "  curl -LsSf https://hf.co/cli/install.sh | bash"
        record "warn"
    fi
fi

info "After setup, log in with: huggingface-cli login"

# ============================================================================
# Step 3/12 — Configure UV_CACHE_DIR
# ============================================================================
step "3" "Configure UV_CACHE_DIR (avoid filling home quota)"

if [[ -n "${SCRATCH:-}" ]] && [[ -d "${SCRATCH}" ]]; then
    export UV_CACHE_DIR="${SCRATCH}/uv_cache"
    info "Using \$SCRATCH: UV_CACHE_DIR=${UV_CACHE_DIR}"
elif [[ -n "${HF_HOME:-}" ]] && [[ -d "$(dirname "${HF_HOME}")" ]]; then
    # HF_HOME (e.g. /gpfs/data/fs70824/trainee88/hf-cache) is on GPFS — same
    # filesystem as $HOME.  Placing the uv cache next to it lets uv hardlink
    # wheels into .venv instead of copying thousands of files over the network.
    export UV_CACHE_DIR="$(dirname "${HF_HOME}")/uv_cache"
    info "Using GPFS (next to HF_HOME): UV_CACHE_DIR=${UV_CACHE_DIR}"
else
    export UV_CACHE_DIR="/tmp/${USER}/uv_cache"
    warn "\$SCRATCH not set and HF_HOME not on GPFS — falling back to UV_CACHE_DIR=${UV_CACHE_DIR}"
    warn "This may be slow (cross-filesystem copy). Set \$SCRATCH to a GPFS path if possible."
fi

run mkdir -p "${UV_CACHE_DIR}"
ok "UV_CACHE_DIR=${UV_CACHE_DIR}"
record "ok"

# Tell uv to always resolve PyTorch from the CUDA 12.1 index.
# Without this, transitive torch dependencies (from ultralytics, transformers)
# get resolved from PyPI, overwriting the CUDA wheels with a broken CPU build.
export UV_LINK_MODE=copy
export UV_TORCH_BACKEND=cu121
info "UV_LINK_MODE=copy (avoids hardlink fallback overhead)"
info "UV_TORCH_BACKEND=cu121 (ensures CUDA wheels for PyTorch)"

# ============================================================================
# Step 4/12 — Create virtual environment
# ============================================================================
step "4" "Create Python 3.10 virtual environment (.venv)"

VENV_DIR="${WORKSHOP_DIR}/.venv"

if [[ -f "${VENV_DIR}/bin/python" ]]; then
    ok "Virtual environment already exists at ${VENV_DIR}"
    # Verify Python version
    if ! $DRY_RUN; then
        PYVER="$("${VENV_DIR}/bin/python" --version 2>&1)"
        info "  Python version: ${PYVER}"
    fi
    record "skip"
else
    info "Creating venv at ${VENV_DIR}..."
    if run uv venv "${VENV_DIR}" --python 3.10; then
        ok "Virtual environment created"
        record "ok"
    else
        fail "Failed to create virtual environment"
        record "fail"
        exit 1
    fi
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

# ============================================================================
# Step 5/12 — Install PyTorch + all workshop dependencies (single resolution)
# ============================================================================
step "5" "Install PyTorch 2.4.1 (CUDA 12.1) + workshop dependencies"

# Install EVERYTHING in a single uv pip install so dependency resolution
# sees torch==2.4.1 alongside ultralytics/transformers and resolves consistently.
# UV_TORCH_BACKEND=cu121 (set above) tells uv to use cu121 wheels for PyTorch.
# --extra-index-url is belt-and-suspenders: ensures cu121 index is available
# even if UV_TORCH_BACKEND is not supported by the installed uv version.

ALL_DEPS=(
    "torch==2.4.1"
    "torchvision==0.19.1"
    "ultralytics>=8.4.19"
    "transformers>=5.0"
    "accelerate"
    "qwen-vl-utils>=0.0.14"
    "supervision>=0.27.0"
    "fiftyone"
    "mlflow"
    "ipykernel"
    "ipywidgets"
    "matplotlib"
    "pandas"
    "numpy"
    "Pillow"
    "opencv-python-headless"
    "pyyaml"
    "tqdm"
    "packaging"
    "clip @ git+https://github.com/ultralytics/CLIP.git"
)

info "Installing PyTorch + ${#ALL_DEPS[@]} packages (this may take several minutes)..."
if run uv pip install \
    --python "${VENV_PYTHON}" \
    --extra-index-url "https://download.pytorch.org/whl/cu121" \
    "${ALL_DEPS[@]}"; then
    ok "All packages installed"
else
    fail "Package installation failed"
    record "fail"
    exit 1
fi

# Verify torch is intact (catches cu121 vs CPU wheel conflicts)
if ! $DRY_RUN; then
    if "${VENV_PYTHON}" -c "import torch; print(f'torch {torch.__version__}'); assert hasattr(torch, 'save'), 'torch.save missing'" 2>/dev/null; then
        ok "PyTorch verified (has core attributes)"
        record "ok"
    else
        warn "PyTorch may be corrupted — reinstalling from cu121 index..."
        if run uv pip install \
            --python "${VENV_PYTHON}" \
            --reinstall-package torch --reinstall-package torchvision \
            "torch==2.4.1" "torchvision==0.19.1" \
            --index-url "https://download.pytorch.org/whl/cu121"; then
            ok "PyTorch reinstalled from cu121"
            record "ok"
        else
            fail "PyTorch repair failed"
            record "fail"
            exit 1
        fi
    fi
else
    record "ok"
fi

# ============================================================================
# Step 6/12 — Verify torch integration with ultralytics and transformers
# ============================================================================
step "6" "Verify torch + ultralytics + transformers integration"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c '<integration test>'"
    record "ok"
else
    if "${VENV_PYTHON}" -c "
import torch
assert hasattr(torch, '__version__'), 'torch.__version__ missing'
assert hasattr(torch, 'save'), 'torch.save missing'
print(f'torch {torch.__version__} OK')

import ultralytics
print(f'ultralytics {ultralytics.__version__} OK')

import transformers
print(f'transformers {transformers.__version__} OK')

import clip
print(f'clip (for SAM3 text prompts) OK')
" 2>&1; then
        ok "Core packages verified"
        record "ok"
    else
        fail "Integration check failed — see errors above"
        record "fail"
        # Non-fatal: continue with remaining steps
    fi
fi

# ============================================================================
# Step 7/12 — Register Jupyter kernel
# ============================================================================
step "7" "Register IPython kernel for JupyterHub"

KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/cv-workshop"

if [[ -d "${KERNEL_DIR}" ]]; then
    ok "Kernel 'cv-workshop' already registered at ${KERNEL_DIR}"
    record "skip"
else
    info "Registering kernel 'cv-workshop'..."
    if run "${VENV_PYTHON}" -m ipykernel install \
        --user \
        --name cv-workshop \
        --display-name "CV Workshop (Python 3.10, CUDA)"; then
        ok "Jupyter kernel registered"
        record "ok"
    else
        fail "Kernel registration failed"
        record "fail"
        # Non-fatal — continue
    fi
fi

# ============================================================================
# Step 8/12 — Pre-download YOLO26n weights
# ============================================================================
step "8" "Pre-download YOLO26n model weights"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c 'from ultralytics import YOLO; YOLO(\"yolo26n.pt\")'"
    record "ok"
else
    if "${VENV_PYTHON}" -c "
from ultralytics import YOLO
print('Downloading yolo26n.pt...')
YOLO('yolo26n.pt')
print('Done.')
" 2>&1; then
        ok "YOLO26n weights downloaded"
        record "ok"
    else
        warn "YOLO26n download failed (will retry on first use)"
        record "warn"
    fi
fi

# ============================================================================
# Step 9/12 — Pre-download YOLOe-26n-seg weights
# ============================================================================
step "9" "Pre-download YOLOe-26n-seg model weights"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c 'from ultralytics import YOLO; YOLO(\"yoloe-26n-seg.pt\")'"
    record "ok"
else
    if "${VENV_PYTHON}" -c "
from ultralytics import YOLO
print('Downloading yoloe-26n-seg.pt...')
YOLO('yoloe-26n-seg.pt')
print('Done.')
" 2>&1; then
        ok "YOLOe-26n-seg weights downloaded"
        record "ok"
    else
        warn "YOLOe-26n-seg download failed (will retry on first use)"
        record "warn"
    fi
fi

# ============================================================================
# Step 10/12 — Pre-download SAM3 (HuggingFace) — gated model, may fail
# ============================================================================
step "10" "Pre-download SAM3 model from HuggingFace (gated — may require approval)"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c 'from transformers import Sam3Processor, Sam3Model; ...'"
    record "ok"
else
    info "SAM3 is a GATED model. Before this step can succeed you need:"
    info "  1. Create a HuggingFace account at https://huggingface.co"
    info "  2. Request access at https://huggingface.co/facebook/sam3"
    info "  3. Set your token: export HF_TOKEN=hf_... OR run huggingface-cli login"
    echo ""

    # Pre-check: is HF_TOKEN set or huggingface-cli logged in?
    HF_AUTH_OK=false
    if [[ -n "${HF_TOKEN:-}" ]]; then
        info "HF_TOKEN is set — will use it for authentication."
        HF_AUTH_OK=true
    elif command -v huggingface-cli &>/dev/null && huggingface-cli whoami &>/dev/null 2>&1; then
        info "huggingface-cli is logged in."
        HF_AUTH_OK=true
    elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
        info "Found cached HuggingFace token."
        HF_AUTH_OK=true
    else
        warn "No HF_TOKEN set and huggingface-cli not logged in."
        warn "SAM3 download will almost certainly fail."
        warn "To fix: export HF_TOKEN=hf_... or run: huggingface-cli login"
    fi

    if ! $HF_AUTH_OK; then
        info "Skipping SAM3 download (no HuggingFace credentials found)."
        info "Qwen 3.5 (Step 11) is the ungated alternative — no approval needed."
        record "warn"
    else
        info "Attempting SAM3 download..."
        # NOTE: Must use Sam3Processor directly — AutoProcessor resolves to
        # Sam3VideoProcessor in transformers>=5.x which lacks text= support.
        if "${VENV_PYTHON}" -c "
from transformers import Sam3Processor, Sam3Model
print('Downloading SAM3 processor...')
Sam3Processor.from_pretrained('facebook/sam3')
print('Downloading SAM3 model...')
Sam3Model.from_pretrained('facebook/sam3')
print('Done.')
" 2>&1; then
            ok "SAM3 model and processor downloaded"
            record "ok"
        else
            warn "SAM3 download failed — this is expected if you don't have HF access"
            warn "Use Qwen 3.5 (Step 11) as the auto-labeling alternative"
            record "warn"
        fi
    fi
fi

# ============================================================================
# Step 11/12 — Pre-download Qwen 3.5 VL (ungated SAM3 alternative)
# ============================================================================
step "11" "Pre-download Qwen 3.5 VL 8B Instruct (ungated, no approval needed)"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c 'from transformers import AutoProcessor, AutoModelForImageTextToText; ...'"
    record "ok"
else
    info "Downloading Qwen 3.5 VL 8B (~16 GB). This may take several minutes..."
    if "${VENV_PYTHON}" -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
model_id = 'Qwen/Qwen3-VL-8B-Instruct'
print(f'Downloading {model_id} processor...')
AutoProcessor.from_pretrained(model_id)
print(f'Downloading {model_id} model...')
AutoModelForImageTextToText.from_pretrained(model_id)
print('Done.')
" 2>&1; then
        ok "Qwen 3.5 VL 8B Instruct downloaded"
        record "ok"
    else
        warn "Qwen 3.5 download failed (will retry on first use)"
        record "warn"
    fi
fi

# ============================================================================
# Step 12/12 — Smoke test
# ============================================================================
step "12" "Smoke test — verify installation"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c '<smoke test snippet>'"
    record "ok"
else
    SMOKE_OUTPUT=""
    if SMOKE_OUTPUT=$("${VENV_PYTHON}" -c "
import sys
print(f'Python       : {sys.version}')

import torch
print(f'PyTorch      : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version : {torch.version.cuda}')
    print(f'GPU count    : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}      : {torch.cuda.get_device_name(i)}')
else:
    print('  (No GPU detected — expected on login nodes)')

import ultralytics
print(f'Ultralytics  : {ultralytics.__version__}')

import transformers
print(f'Transformers : {transformers.__version__}')

import fiftyone
print(f'FiftyOne     : {fiftyone.__version__}')

import mlflow
print(f'MLflow       : {mlflow.__version__}')

import cv2
print(f'OpenCV       : {cv2.__version__}')

import numpy
print(f'NumPy        : {numpy.__version__}')

print()
print('All imports successful!')
" 2>&1); then
        echo ""
        echo "${SMOKE_OUTPUT}"
        echo ""
        ok "Smoke test passed"
        record "ok"
    else
        echo ""
        echo "${SMOKE_OUTPUT}"
        echo ""
        fail "Smoke test failed — check output above"
        record "fail"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME="$(date +%s)"
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Setup Summary${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

STEP_NAMES=(
    "Install uv"
    "Install HuggingFace CLI"
    "Configure UV_CACHE_DIR"
    "Create virtual environment"
    "Install PyTorch + workshop deps"
    "Verify torch integration"
    "Register Jupyter kernel"
    "Pre-download YOLO26n"
    "Pre-download YOLOe-26n-seg"
    "Pre-download SAM3 (gated)"
    "Pre-download Qwen 3.5 (ungated)"
    "Smoke test"
)

OK_COUNT=0
SKIP_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

for i in "${!STEP_RESULTS[@]}"; do
    STATUS="${STEP_RESULTS[$i]}"
    NAME="${STEP_NAMES[$i]}"
    NUM=$(( i + 1 ))

    case "${STATUS}" in
        ok)   echo -e "  ${GREEN}[OK]${NC}   Step ${NUM}: ${NAME}"; (( OK_COUNT++ )) ;;
        skip) echo -e "  ${BLUE}[SKIP]${NC} Step ${NUM}: ${NAME} (already done)"; (( SKIP_COUNT++ )) ;;
        warn) echo -e "  ${YELLOW}[WARN]${NC} Step ${NUM}: ${NAME}"; (( WARN_COUNT++ )) ;;
        fail) echo -e "  ${RED}[FAIL]${NC} Step ${NUM}: ${NAME}"; (( FAIL_COUNT++ )) ;;
    esac
done

echo ""
echo -e "  Completed: ${GREEN}${OK_COUNT} ok${NC}, ${BLUE}${SKIP_COUNT} skipped${NC}, ${YELLOW}${WARN_COUNT} warnings${NC}, ${RED}${FAIL_COUNT} failures${NC}"
echo -e "  Elapsed  : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""

if (( FAIL_COUNT > 0 )); then
    echo -e "${RED}${BOLD}Some steps failed. Review the output above for details.${NC}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Post-install instructions
# ---------------------------------------------------------------------------
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Verify your setup:"
echo "       bash check_setup.sh"
echo ""
echo "  2. In JupyterHub, select the 'CV Workshop (Python 3.10, CUDA)' kernel"
echo "  3. If running outside JupyterHub, activate with:"
echo "       export PATH=\"${VENV_DIR}/bin:\$PATH\""
echo "  4. For AI assistance, see notebooks/cv_copilot_prompt.md"
echo ""
echo -e "${BOLD}Workshop directory:${NC} ${WORKSHOP_DIR}"
echo -e "${BOLD}Virtual environment:${NC} ${VENV_DIR}"
echo -e "${BOLD}UV cache:${NC} ${UV_CACHE_DIR}"
echo ""

# Exit with failure if any step failed
if (( FAIL_COUNT > 0 )); then
    exit 1
fi

exit 0
