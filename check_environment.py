#!/usr/bin/env python3
"""CV Workshop Environment Health Check

Run this script to verify your workshop environment is correctly set up.
Usage:
    python check_environment.py           # Full check
    python check_environment.py --quick   # Skip model weight checks (faster)

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Color output ──────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
NC = "\033[0m"

PASS = f"{GREEN}✓ PASS{NC}"
FAIL = f"{RED}✗ FAIL{NC}"
WARN = f"{YELLOW}⚠ WARN{NC}"
INFO = f"{CYAN}ℹ INFO{NC}"

WORKSHOP_DIR = Path(__file__).resolve().parent


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{BOLD}{'─' * 60}{NC}")


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


def warn(name: str, detail: str = "") -> None:
    msg = f"  {WARN}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def info(name: str, detail: str = "") -> None:
    msg = f"  {INFO}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ── Check functions ───────────────────────────────────────────────────────────

def check_python() -> list[bool]:
    section("1. Python Environment")
    results = []

    v = sys.version_info
    results.append(check(
        "Python version",
        v.major == 3 and v.minor >= 10,
        f"{v.major}.{v.minor}.{v.micro}"
    ))

    in_venv = sys.prefix != sys.base_prefix
    results.append(check(
        "Running in virtual environment",
        in_venv,
        sys.prefix if in_venv else "Not in a venv — activate .venv first"
    ))

    return results


def check_core_packages() -> list[bool]:
    section("2. Core Packages")
    results = []

    required = {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "ultralytics": "8.4.19",
        "transformers": "4.40.0",
    }

    for pkg, min_ver in required.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            from packaging.version import Version
            passed = Version(version) >= Version(min_ver)
            results.append(check(pkg, passed, f"v{version}, need >={min_ver}"))
        except ImportError:
            results.append(check(pkg, False, "NOT INSTALLED"))
        except Exception as e:
            results.append(check(pkg, False, str(e)))

    # Qwen3-VL support (SAM3 fallback)
    qwen_pkgs = {
        "qwen_vl_utils": None,
        "supervision": "0.27.0",
    }
    for pkg, min_ver in qwen_pkgs.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            if min_ver:
                from packaging.version import Version
                passed = Version(version) >= Version(min_ver)
            else:
                passed = True
            results.append(check(f"{pkg} (Qwen3-VL support)", passed, f"v{version}"))
        except ImportError:
            results.append(check(f"{pkg} (Qwen3-VL support)", False, "NOT INSTALLED"))

    # Verify Qwen3-VL model class is available in transformers
    try:
        from transformers import AutoModelForImageTextToText
        info("AutoModelForImageTextToText", "available (Qwen3-VL compatible)")
    except ImportError:
        warn("AutoModelForImageTextToText", "not available — update transformers>=5.0")

    optional = ["fiftyone", "mlflow", "accelerate", "ipywidgets", "cv2"]
    for pkg in optional:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            info(pkg, f"v{version}")
        except ImportError:
            warn(pkg, "not installed — optional but recommended")

    return results


def check_gpu() -> list[bool]:
    section("3. GPU / CUDA")
    results = []

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        results.append(check("CUDA available", cuda_available))

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            info("GPU device", f"{gpu_name} ({gpu_mem:.1f} GB)")
            info("CUDA version", torch.version.cuda)

            # Quick tensor test
            try:
                t = torch.randn(100, 100, device="cuda")
                _ = t @ t.T
                results.append(check("CUDA tensor operations", True))
                del t
                torch.cuda.empty_cache()
            except Exception as e:
                results.append(check("CUDA tensor operations", False, str(e)))
        else:
            warn("No GPU detected", "Training will be slow on CPU")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info("MPS (Apple Silicon) available", "Will use MPS as fallback")

    except ImportError:
        results.append(check("PyTorch import", False, "torch not installed"))

    return results


def check_model_weights() -> list[bool]:
    section("4. Model Weights (Pre-downloaded)")
    results = []

    # YOLO models — check in current dir, ~/.config/Ultralytics, and HF cache
    from ultralytics import YOLO

    yolo_models = ["yolo26n.pt", "yoloe-26n-seg.pt"]
    for model_name in yolo_models:
        try:
            model = YOLO(model_name)
            results.append(check(model_name, True, "loaded successfully"))
            del model
        except Exception as e:
            results.append(check(model_name, False, f"download needed: {e}"))

    # SAM3 HuggingFace model (gated — may not be available)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("facebook/sam3", local_files_only=True)
        results.append(check("SAM3 (facebook/sam3)", True, "cached"))
        del processor
    except Exception:
        warn("SAM3 (facebook/sam3)", "not cached — gated model, needs HF approval")
        info("  → Use Qwen3-VL as alternative", "(ungated, no approval needed)")

    # Qwen3-VL (ungated alternative to SAM3)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", local_files_only=True
        )
        results.append(check("Qwen3-VL-8B-Instruct", True, "cached"))
        del processor
    except Exception:
        results.append(check(
            "Qwen3-VL-8B-Instruct", False,
            "not cached — run: python test_qwen3_vl.py"
        ))

    return results


def check_jupyter_kernel() -> list[bool]:
    section("5. Jupyter Kernel")
    results = []

    try:
        kernel_dir = Path.home() / ".local" / "share" / "jupyter" / "kernels" / "cv-workshop"
        kernel_json = kernel_dir / "kernel.json"

        if kernel_json.exists():
            with open(kernel_json) as f:
                spec = json.load(f)
            python_path = spec.get("argv", [None])[0]
            results.append(check(
                "CV Workshop kernel registered",
                True,
                f"python: {python_path}"
            ))
        else:
            results.append(check(
                "CV Workshop kernel registered",
                False,
                "Run: python -m ipykernel install --user --name cv-workshop"
            ))
    except Exception as e:
        results.append(check("Jupyter kernel check", False, str(e)))

    return results


def check_claude_code() -> list[bool]:
    section("6. Claude Code + CV Engineer Skill")
    results = []

    # Check Claude Code binary
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            results.append(check("Claude Code installed", True, version))
        else:
            results.append(check("Claude Code installed", False, "binary found but --version failed"))
    except FileNotFoundError:
        results.append(check("Claude Code installed", False, "not in PATH"))
    except Exception as e:
        results.append(check("Claude Code installed", False, str(e)))

    # Check CV engineer skill
    skill_locations = [
        WORKSHOP_DIR / ".claude" / "skills" / "cv-engineer" / "CLAUDE.md",
        Path.home() / ".claude" / "skills" / "cv-engineer" / "CLAUDE.md",
    ]
    found = False
    for loc in skill_locations:
        if loc.exists():
            results.append(check("CV Engineer skill", True, str(loc)))
            found = True
            break
    if not found:
        results.append(check("CV Engineer skill", False, "not found — run setup script"))

    return results


def check_workshop_files() -> list[bool]:
    section("7. Workshop Files")
    results = []

    # Check data directory
    data_dir = WORKSHOP_DIR / "data"
    if data_dir.exists():
        results.append(check("data/ directory", True))

        # Check synthetic images
        synth_dir = data_dir / "synthetic_ppe"
        if synth_dir.exists():
            images = list(synth_dir.rglob("*.jpg")) + list(synth_dir.rglob("*.png")) + list(synth_dir.rglob("*.webp"))
            results.append(check(
                "Synthetic PPE images",
                len(images) >= 80,
                f"{len(images)} images found (expect ~91)"
            ))
        else:
            results.append(check("Synthetic PPE images", False, "data/synthetic_ppe/ not found"))

        # Check key scripts
        scripts = [
            "auto_label_sam3_hf.py",
            "train_baseline_ppe.py",
            "evaluate_2class_experiments.py",
            "compliance_postprocessor.py",
            "filter_tiny_labels.py",
            "visualize_gt_annotations.py",
        ]
        for script in scripts:
            exists = (data_dir / script).exists()
            results.append(check(f"data/{script}", exists))
    else:
        results.append(check("data/ directory", False, "not found — scripts will be copied tomorrow"))
        warn("Data scripts not yet deployed", "This is expected if running setup before Day 2")

    # Check notebooks
    nb_dir = WORKSHOP_DIR / "notebooks"
    if nb_dir.exists():
        notebooks = list(nb_dir.glob("*.ipynb"))
        results.append(check(
            "Exercise notebooks",
            len(notebooks) >= 4,
            f"{len(notebooks)} notebooks found"
        ))
    else:
        results.append(check("notebooks/ directory", False, "not found"))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    quick = "--quick" in sys.argv

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  CV Workshop Environment Health Check{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")
    info("Workshop directory", str(WORKSHOP_DIR))
    info("Python executable", sys.executable)

    all_results = []
    all_results.extend(check_python())
    all_results.extend(check_core_packages())
    all_results.extend(check_gpu())

    if not quick:
        all_results.extend(check_model_weights())
    else:
        section("4. Model Weights (skipped — use full check without --quick)")

    all_results.extend(check_jupyter_kernel())
    all_results.extend(check_claude_code())
    all_results.extend(check_workshop_files())

    # Summary
    passed = sum(all_results)
    total = len(all_results)
    failed = total - passed

    section("Summary")
    print(f"  Checks passed: {GREEN}{passed}/{total}{NC}")
    if failed > 0:
        print(f"  Checks failed: {RED}{failed}{NC}")
        print(f"\n  {YELLOW}Fix the failing checks above, then re-run:{NC}")
        print(f"  {CYAN}python check_environment.py{NC}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed! You're ready for the workshop.{NC}")
        sys.exit(0)


if __name__ == "__main__":
    main()
