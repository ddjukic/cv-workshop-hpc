# CV Workshop — PPE Compliance Detection

**AI Factory | March 3–4, 2026 | VSC-5 HPC**

Build a PPE (Personal Protective Equipment) compliance detection system using synthetic data, foundation model auto-labeling, and YOLO26n distillation.

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> cv-workshop-hpc
cd cv-workshop-hpc

# 2. Run setup (installs uv, Python deps, PyTorch CUDA, models, Claude Code)
bash setup_workshop_env.sh

# 3. Verify everything works
bash check_setup.sh

# 4. Open JupyterHub, select "CV Workshop (Python 3.10, CUDA)" kernel
# 5. Start with notebooks/01_ppe_getting_started.ipynb
```

## Setup

### Prerequisites

- VSC-5 HPC account with GPU allocation (A40, CUDA 12.3)
- JupyterHub access
- Run setup from a **login node** (needs internet for downloads)

### Installation

```bash
bash setup_workshop_env.sh
```

This script:
1. Installs `uv` (fast Python package manager)
2. Creates an isolated `.venv` with Python 3.10
3. Installs PyTorch 2.4.1 with CUDA 12.1 support
4. Installs workshop dependencies (ultralytics, transformers, fiftyone, etc.)
5. Registers a Jupyter kernel: **"CV Workshop (Python 3.10, CUDA)"**
6. Pre-downloads model weights (YOLO26n, YOLOe-26n, SAM3) — avoids download delays during exercises
7. Installs Claude Code (your AI coding assistant)

**Dry run** (preview without installing):
```bash
bash setup_workshop_env.sh --dry-run
```

### Health Check

After setup, verify everything is working:

```bash
# Full check (includes model weight verification)
bash check_setup.sh

# Quick check (skip model weight verification)
bash check_setup.sh --quick
```

**Or ask Claude Code:**
```
> Verify my workshop setup — run check_environment.py and tell me if anything failed
```

The health check verifies:
- Python environment and virtual env activation
- Core packages (torch, ultralytics, transformers) with version checks
- GPU / CUDA availability and tensor operations
- Pre-downloaded model weights (YOLO26n, YOLOe-26n, SAM3)
- Jupyter kernel registration
- Claude Code installation and CV Engineer skill
- Workshop data files and notebooks

## Workshop Structure

### Day 1 (March 3) — Talks & Demos
- Computer vision landscape and foundation models
- Live demos: open-vocabulary detection, distillation pipeline
- Theory: attention mechanisms, CLIP, prompt engineering

### Day 2 (March 4) — Hands-On Exercise

**Goal**: Build a PPE compliance detector that identifies which construction workers are wearing hard hats.

| Phase | Notebook | Time | What You Do |
|-------|----------|------|-------------|
| 1. Baseline | `01_ppe_getting_started.ipynb` | ~10 min | Run zero-shot detection with YOLOe-26n |
| 2. Auto-Label | `01_ppe_getting_started.ipynb` | ~10 min | Generate training labels with SAM3 |
| 3. Iterate | `02_inspect_iterate_train.ipynb` | ~15 min | Inspect labels, filter noise, curate data |
| 4. Train | `02_inspect_iterate_train.ipynb` | ~10 min | Train YOLO26n on your curated labels |
| 5. Evaluate | `03_evaluate_and_deploy.ipynb` | ~20 min | Measure detection + compliance metrics |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_ppe_getting_started.ipynb` | Environment check, dataset exploration, YOLOE baseline, auto-labeling |
| `02_inspect_iterate_train.ipynb` | Label visualization, error analysis, filtering, YOLO26n training |
| `03_evaluate_and_deploy.ipynb` | Compliance post-processing, speed benchmark, model comparison |
| `reference/` | Original step-by-step tutorials (for self-study) |

## Using Claude Code

Claude Code acts as your **CV engineer partner** during the exercise. It knows the tools, the dataset, and the expected workflow — but it won't give away answers.

```bash
# Start Claude Code from the workshop directory
claude

# Example prompts:
> What tools are available for auto-labeling?
> Auto-label my dataset using SAM3 in 2-class mode
> Train a YOLO26n model on my filtered dataset
> Evaluate my model and show compliance metrics
> Why is my mAP50 stuck around 0.5?
```

## Data Scripts

All scripts are in `data/`. Full documentation is in the CV Engineer skill.

| Script | Purpose |
|--------|---------|
| `auto_label_sam3_hf.py` | Auto-label images using SAM3 (HuggingFace) |
| `auto_label_ppe_2class.py` | Auto-label using Grounding DINO (2-class) |
| `filter_tiny_labels.py` | Remove noisy tiny labels |
| `visualize_gt_annotations.py` | Visualize bounding boxes on images |
| `train_baseline_ppe.py` | Train YOLO26n baseline |
| `evaluate_2class_experiments.py` | Evaluate with detection + compliance metrics |
| `compliance_postprocessor.py` | Check PPE compliance from detections |
| `benchmark_inference_speed.py` | Compare inference speed |
| `evaluate_yoloe_26n.py` | Zero-shot YOLOe baseline evaluation |

## Troubleshooting

### "CUDA is not available"
- Make sure you're on a **GPU node**, not a login node
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- If using JupyterHub, select a GPU-enabled queue

### "Kernel not found in JupyterHub"
```bash
# Re-register the kernel
.venv/bin/python -m ipykernel install --user --name cv-workshop --display-name "CV Workshop (Python 3.10, CUDA)"
```

### "ModuleNotFoundError"
```bash
# Ensure you're using the workshop venv
.venv/bin/python -c "import ultralytics; print('OK')"

# If needed, reinstall
.venv/bin/python -m pip install ultralytics transformers
```

### "Model download is slow"
Model weights should be pre-downloaded during setup. If not:
```bash
# Re-run the model download steps
.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yoloe-26n-seg.pt'); print('OK')"
```

### Training is taking too long
- Reduce epochs to 30 for quick iteration
- Pre-baked results are available at `data/ppe_results/`
- Ask Claude Code: "Use the pre-baked results for comparison"

## Project Structure

```
cv-workshop-hpc/
├── README.md
├── setup_workshop_env.sh      # Environment setup
├── check_environment.py       # Python health checks
├── check_setup.sh             # Bash wrapper for health checks
├── pyproject.toml             # Dependency manifest
├── .claude/
│   └── skills/
│       └── cv-engineer/
│           └── CLAUDE.md      # Claude Code CV engineer skill
├── data/
│   ├── synthetic_ppe/         # 91 source images (11 categories)
│   ├── auto_label_sam3_hf.py  # Auto-labeling scripts
│   ├── train_baseline_ppe.py  # Training script
│   ├── ...                    # Other tools
│   └── EXPERIMENT_REPORT.md   # Full experiment documentation
└── notebooks/
    ├── 01_ppe_getting_started.ipynb
    ├── 02_inspect_iterate_train.ipynb
    ├── 03_evaluate_and_deploy.ipynb
    └── reference/             # Original tutorials
```
