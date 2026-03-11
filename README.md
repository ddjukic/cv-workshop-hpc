# CV Workshop — PPE Compliance Detection

**AI Factory | March 3-4, 2026 | VSC-5 HPC**

Build a PPE (Personal Protective Equipment) compliance detection system using synthetic data, foundation model auto-labeling, and YOLO26n distillation.

**Approach**: 2-class detection (hardhat + person) with post-processing compliance logic. We detect hardhats and people separately, then compute what percentage of workers are non-compliant (not wearing hardhats) via spatial overlap. This is far more reliable than trying to detect "no_hardhat" directly (which only achieves ~25% recall).

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ddjukic/cv-workshop-hpc.git
cd cv-workshop-hpc

# 2. Set up HuggingFace FIRST (required for SAM3 gated model download)
#    Option A: export token
export HF_TOKEN=hf_your_token_here
#    Option B: login interactively (after setup installs the CLI)
#    huggingface-cli login

# 3. Run setup (installs uv, Python deps, PyTorch CUDA, models)
bash setup_workshop_env.sh

# 4. Verify everything works
bash check_setup.sh

# 5. Open JupyterHub, select "CV Workshop (Python 3.10, CUDA)" kernel
# 6. Start with notebooks/01_ppe_getting_started.ipynb
```

## Setup

### Prerequisites

- VSC-5 HPC account with GPU allocation (A40, CUDA 12.3)
- JupyterHub access
- Run setup from a **login node** (needs internet for downloads)
- HuggingFace account with access token (see below)

### Step 1: HuggingFace Account & Token Setup

Do this **before** running the setup script. The script downloads SAM3, which is a gated model that requires authentication.

#### 1a. Create a HuggingFace Account

1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up with email or GitHub/Google SSO
3. Verify your email

#### 1b. Create an Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"Create new token"**
3. Name: `cv-workshop` (or anything you like)
4. Type: **Read** (sufficient for model downloads)
5. Click **"Create token"** and copy the token (`hf_...`)

#### 1c. Set Your Token

```bash
# Option A: environment variable (recommended for HPC)
export HF_TOKEN=hf_your_token_here

# Option B: login interactively (the CLI is installed by the setup script,
# so use this after Step 2 if you prefer interactive login)
huggingface-cli login
```

This stores the token in `~/.cache/huggingface/token`.

#### 1d. Request SAM3 Access (Gated Model)

SAM3 (`facebook/sam3`) is a **gated model** — Meta requires you to accept their license:

1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click **"Agree and access repository"** (or similar button)
3. Approval is usually instant, but can take up to 24 hours

> **If SAM3 access is not approved in time:** Don't worry — **Qwen 3.5** is pre-downloaded
> as an ungated alternative. It requires no approval and works out of the box.

### Step 2: Run the Setup Script

```bash
bash setup_workshop_env.sh
```

This script:
1. Installs `uv` (fast Python package manager) and `huggingface-cli`
2. Creates an isolated `.venv` with Python 3.10
3. Installs PyTorch 2.4.1 with CUDA 12.1 support
4. Installs workshop dependencies (ultralytics, transformers, fiftyone, supervision, CLIP, etc.)
5. Registers a Jupyter kernel: **"CV Workshop (Python 3.10, CUDA)"**
6. Pre-downloads model weights (YOLO26n, YOLOe-26n-seg)
7. Attempts SAM3 download (skips automatically if no HF credentials found)
8. Pre-downloads Qwen 3.5 VL 8B (ungated, always works)

**Dry run** (preview without installing):
```bash
bash setup_workshop_env.sh --dry-run
```

### Step 3: Verify HuggingFace Setup

```bash
# Check login status
huggingface-cli whoami

# Test SAM3 access (will fail if not yet approved — that's OK)
# NOTE: Must use Sam3Processor — AutoProcessor resolves to Sam3VideoProcessor
# in transformers>=5.x which lacks text= support for text-prompted segmentation.
.venv/bin/python -c "
from transformers import Sam3Processor
try:
    Sam3Processor.from_pretrained('facebook/sam3')
    print('SAM3: Access granted')
except Exception as e:
    print(f'SAM3: Not available ({e})')
    print('  Use Qwen 3.5 as alternative (already downloaded)')
"
```

### Step 4: Health Check

```bash
# Full check (includes model weight verification)
bash check_setup.sh

# Quick check (skip model weight verification)
bash check_setup.sh --quick
```

### Step 5: Test in JupyterHub

Open JupyterHub, select **"CV Workshop (Python 3.10, CUDA)"** kernel, and run:

```
notebooks/00_test_environment.ipynb
```

This notebook verifies the full stack inside JupyterHub: kernel, torch, CUDA, YOLO models, Qwen 3.5 loading, and inference test.

## Workshop Structure

### Day 1 (March 3) — Talks & Demos

- Computer vision landscape and foundation models
- Live demos: open-vocabulary detection, SAM3, distillation pipeline
- Theory: attention mechanisms, CLIP, prompt engineering

### Day 2 (March 4) — Hands-On Exercise

**Challenge**: Build a PPE compliance detector that measures what percentage of construction workers are non-compliant (not wearing hardhats).

**Approach**: 2-class detection (hardhat + person). Auto-label with SAM3 or Qwen 3.5 using single-concept prompts, train a YOLO26n detector, then use post-processing compliance logic to determine who is/isn't wearing a hardhat.

| Notebook | Time | What You Do |
|----------|------|-------------|
| `01_ppe_getting_started.ipynb` | ~20 min | Explore dataset, run YOLOe zero-shot baseline, auto-label with SAM3 or Qwen 3.5 |
| `02_inspect_iterate_train.ipynb` | ~25 min | Inspect labels in FiftyOne, filter noise, curate data, train YOLO26n |
| `03_evaluate_and_deploy.ipynb` | ~20 min | Measure detection + compliance metrics, speed benchmark, model comparison |
| `04_solution_walkthrough.ipynb` | ~15 min | Instructor-led end-of-day reveal — full solution pipeline with pre-baked results |

Notebooks 01-03 are the hands-on challenge. Notebook 04 is a guided walkthrough led by the instructor at the end of the day.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_test_environment.ipynb` | **Run first** — verifies kernel, torch, CUDA, YOLO, Qwen 3.5, SAM3 |
| `01_ppe_getting_started.ipynb` | Dataset exploration, YOLOe baseline, auto-labeling with SAM3 or Qwen 3.5 |
| `02_inspect_iterate_train.ipynb` | Label visualization, error analysis, filtering, YOLO26n training |
| `03_evaluate_and_deploy.ipynb` | Compliance post-processing, speed benchmark, model comparison |
| `04_solution_walkthrough.ipynb` | Instructor-led solution reveal with pre-baked results |
| `reference/` | Original step-by-step tutorials (for self-study) |

Day 1 demo notebooks (not part of the hands-on exercise):

| Notebook | Description |
|----------|-------------|
| `demo_01_sam3_showcase.ipynb` | SAM3 capabilities demo |
| `demo_03_distillation_pipeline.ipynb` | Foundation model to YOLO distillation |
| `demo_04_fiftyone_data_analysis.ipynb` | FiftyOne dataset analysis |
| `demo_05_model_export.ipynb` | ONNX/TensorRT model export |
| `demo_06_model_serving.ipynb` | Model serving with FastAPI/LitServe |

## AI Assistance (CV Copilot)

Use any AI coding assistant as your **CV copilot** during the exercise. We provide two ready-made guidance files in `notebooks/`:

| File | Format | Use with |
|------|--------|----------|
| `cv_copilot_skill.md` | Skill file | Claude Code, OpenCode, Cursor, etc. |
| `cv_copilot_prompt.md` | System prompt | ChatGPT, Claude web, any AI chat |

**Example prompts:**
```
> What tools are available for auto-labeling?
> Auto-label my dataset using SAM3 in 2-class mode
> Train a YOLO26n model on my filtered dataset
> Why is my mAP50 stuck around 0.5?
```

## Data Scripts

All scripts are in `data/`.

| Script | Purpose |
|--------|---------|
| `auto_label_sam3_hf.py` | Auto-label images using SAM3 (HuggingFace, gated) |
| `auto_label_qwen3_vl.py` | Auto-label images using Qwen 3.5 VL (ungated fallback) |
| `filter_tiny_labels.py` | Remove noisy tiny labels below pixel threshold |
| `visualize_gt_annotations.py` | Visualize bounding boxes on images |
| `train_baseline_ppe.py` | Train YOLO26n on curated labels |
| `evaluate_2class_experiments.py` | Evaluate with detection + compliance metrics |
| `evaluate_yoloe_26n.py` | Zero-shot YOLOe baseline evaluation |
| `compliance_postprocessor.py` | Derive PPE compliance from 2-class detections |
| `compare_sam3_vs_qwen.py` | Compare SAM3 vs Qwen 3.5 labeling quality |
| `generate_synthetic_ppe.py` | Generate synthetic PPE training images |
| `benchmark_inference_speed.py` | Compare inference speed across models |

## Troubleshooting

### "Access to model facebook/sam3 is restricted"
SAM3 is a gated model. You need to:
1. Log in: `huggingface-cli login` (or `export HF_TOKEN=hf_...`)
2. Request access at [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
3. Wait for approval (usually instant, can take up to 24h)

**If you can't wait**, use **Qwen 3.5** instead — it's ungated and pre-downloaded:
```bash
# Verify Qwen 3.5 works
python test_qwen3_vl.py --imports-only
```

### "CUDA is not available"
- Make sure you're on a **GPU node**, not a login node
- Check: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`
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
uv pip install --python .venv/bin/python ultralytics transformers
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

## Project Structure

```
cv-workshop-hpc/
├── README.md
├── setup_workshop_env.sh      # Environment setup (12-step installer)
├── check_environment.py       # Python health checks (6 categories)
├── check_setup.sh             # Bash wrapper for health checks
├── test_qwen3_vl.py           # Qwen 3.5 model verification
├── pyproject.toml             # Dependency manifest
├── .claude/
│   └── skills/
│       └── cv-engineer/
│           └── SKILL.md       # CV engineer skill (for Claude Code users)
├── data/
│   ├── synthetic_ppe/         # Source images (9 categories)
│   ├── auto_label_sam3_hf.py  # SAM3 auto-labeling (gated)
│   ├── auto_label_qwen3_vl.py # Qwen 3.5 auto-labeling (ungated)
│   ├── train_baseline_ppe.py  # YOLO26n training
│   ├── compliance_postprocessor.py
│   ├── ppe_results/           # Pre-baked results for solution walkthrough
│   └── ...                    # Other evaluation/benchmark tools
└── notebooks/
    ├── 00_test_environment.ipynb
    ├── 01_ppe_getting_started.ipynb
    ├── 02_inspect_iterate_train.ipynb
    ├── 03_evaluate_and_deploy.ipynb
    ├── 04_solution_walkthrough.ipynb
    ├── cv_copilot_skill.md
    ├── cv_copilot_prompt.md
    ├── demo_*.ipynb           # Day 1 demo notebooks
    └── reference/             # Original tutorials
```
