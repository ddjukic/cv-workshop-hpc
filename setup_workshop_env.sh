#!/usr/bin/env bash
# =============================================================================
# setup_workshop_env.sh — CV Workshop HPC Environment Setup
# =============================================================================
# Target: VSC-5 (A40 GPU, CUDA 12.3, conda Python 3.10.14, JupyterHub)
#
# This script creates a fully isolated Python environment using uv, installs
# PyTorch + workshop dependencies, registers a Jupyter kernel, pre-downloads
# model weights (must run on a login node with internet), and optionally
# installs Claude Code via npm.
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
# Step 2/12 — Configure UV_CACHE_DIR
# ============================================================================
step "2" "Configure UV_CACHE_DIR (avoid filling home quota)"

if [[ -n "${SCRATCH:-}" ]] && [[ -d "${SCRATCH}" ]]; then
    export UV_CACHE_DIR="${SCRATCH}/uv_cache"
    info "Using \$SCRATCH: UV_CACHE_DIR=${UV_CACHE_DIR}"
else
    export UV_CACHE_DIR="/tmp/${USER}/uv_cache"
    warn "\$SCRATCH not set or missing — falling back to UV_CACHE_DIR=${UV_CACHE_DIR}"
fi

run mkdir -p "${UV_CACHE_DIR}"
ok "UV_CACHE_DIR=${UV_CACHE_DIR}"
record "ok"

# ============================================================================
# Step 3/12 — Create virtual environment
# ============================================================================
step "3" "Create Python 3.10 virtual environment (.venv)"

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
# Step 4/12 — Install PyTorch (CUDA 12.1)
# ============================================================================
step "4" "Install PyTorch 2.4.1 + torchvision 0.19.1 (CUDA 12.1)"

if ! $DRY_RUN && "${VENV_PYTHON}" -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.4.1"; then
    ok "PyTorch 2.4.1 already installed"
    record "skip"
else
    info "Installing PyTorch (this downloads ~2.5 GB, may take a few minutes)..."
    if run uv pip install \
        --python "${VENV_PYTHON}" \
        "torch==2.4.1" \
        "torchvision==0.19.1" \
        --index-url "https://download.pytorch.org/whl/cu121"; then
        ok "PyTorch installed"
        record "ok"
    else
        fail "PyTorch installation failed"
        record "fail"
        exit 1
    fi
fi

# ============================================================================
# Step 5/12 — Install workshop dependencies
# ============================================================================
step "5" "Install workshop Python dependencies"

WORKSHOP_DEPS=(
    "ultralytics>=8.4.19"
    "transformers>=4.40"
    "accelerate"
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
)

info "Installing ${#WORKSHOP_DEPS[@]} packages..."
if run uv pip install --python "${VENV_PYTHON}" "${WORKSHOP_DEPS[@]}"; then
    ok "Workshop dependencies installed"
    record "ok"
else
    fail "Some workshop dependencies failed to install"
    record "fail"
    exit 1
fi

# ============================================================================
# Step 6/12 — Register Jupyter kernel
# ============================================================================
step "6" "Register IPython kernel for JupyterHub"

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
# Step 7/12 — Pre-download YOLO26n weights
# ============================================================================
step "7" "Pre-download YOLO26n model weights"

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
# Step 8/12 — Pre-download YOLOe-26n-seg weights
# ============================================================================
step "8" "Pre-download YOLOe-26n-seg model weights"

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
# Step 9/12 — Pre-download SAM3 (HuggingFace)
# ============================================================================
step "9" "Pre-download SAM3 model from HuggingFace"

if $DRY_RUN; then
    echo -e "  ${CYAN}[DRY-RUN]${NC} ${VENV_PYTHON} -c 'from transformers import AutoProcessor, AutoModelForMaskGeneration; ...'"
    record "ok"
else
    info "This may take several minutes (large model download)..."
    if "${VENV_PYTHON}" -c "
from transformers import AutoProcessor, AutoModelForMaskGeneration
print('Downloading SAM3 processor...')
AutoProcessor.from_pretrained('facebook/sam3')
print('Downloading SAM3 model...')
AutoModelForMaskGeneration.from_pretrained('facebook/sam3')
print('Done.')
" 2>&1; then
        ok "SAM3 model and processor downloaded"
        record "ok"
    else
        warn "SAM3 download failed — check HuggingFace access (may need token)"
        record "warn"
    fi
fi

# ============================================================================
# Step 10/12 — Install Node.js + Claude Code
# ============================================================================
step "10" "Install Node.js and Claude Code (optional)"

CLAUDE_CODE_OK=false

# Check for Node.js
if command -v node &>/dev/null; then
    NODE_VER="$(node --version 2>/dev/null || echo 'unknown')"
    ok "Node.js already installed: ${NODE_VER}"
else
    info "Installing Node.js 18 via conda..."
    if run conda install -y -c conda-forge "nodejs=18"; then
        ok "Node.js installed"
    else
        warn "Node.js installation failed — skipping Claude Code"
        record "warn"
    fi
fi

if command -v node &>/dev/null || $DRY_RUN; then
    # Set npm prefix to avoid needing root
    run npm config set prefix "${HOME}/.local"

    # Check if Claude Code is already installed
    if command -v claude &>/dev/null; then
        ok "Claude Code already installed: $(claude --version 2>/dev/null || echo 'installed')"
        CLAUDE_CODE_OK=true
    else
        info "Installing Claude Code via npm..."
        if run npm install -g @anthropic-ai/claude-code; then
            ok "Claude Code installed"
            CLAUDE_CODE_OK=true
        else
            warn "Claude Code installation failed (non-critical, continuing)"
        fi
    fi
fi

if $CLAUDE_CODE_OK || $DRY_RUN; then
    record "ok"
else
    record "warn"
fi

# ============================================================================
# Step 11/12 — Copy CV engineer skill
# ============================================================================
step "11" "Copy CV engineer skill to user config"

SKILL_SRC="${WORKSHOP_DIR}/.claude/skills/cv-engineer/CLAUDE.md"
SKILL_DST="${HOME}/.claude/skills/cv-engineer/CLAUDE.md"

if [[ -f "${SKILL_DST}" ]]; then
    ok "CV engineer skill already present at ${SKILL_DST}"
    record "skip"
elif [[ ! -f "${SKILL_SRC}" ]]; then
    warn "Source skill file not found at ${SKILL_SRC} — skipping"
    record "warn"
else
    run mkdir -p "${HOME}/.claude/skills/cv-engineer/"
    if run cp "${SKILL_SRC}" "${SKILL_DST}"; then
        ok "CV engineer skill copied"
        record "ok"
    else
        warn "Failed to copy CV engineer skill"
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
    "Configure UV_CACHE_DIR"
    "Create virtual environment"
    "Install PyTorch"
    "Install workshop deps"
    "Register Jupyter kernel"
    "Pre-download YOLO26n"
    "Pre-download YOLOe-26n-seg"
    "Pre-download SAM3"
    "Install Node.js + Claude Code"
    "Copy CV engineer skill"
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
echo "     Or ask Claude Code: 'verify my workshop setup'"
echo ""
echo "  2. In JupyterHub, select the 'CV Workshop (Python 3.10, CUDA)' kernel"
echo "  3. If running outside JupyterHub, activate with:"
echo "       export PATH=\"${VENV_DIR}/bin:\$PATH\""
echo "  4. To use Claude Code:"
echo "       claude auth login"
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
