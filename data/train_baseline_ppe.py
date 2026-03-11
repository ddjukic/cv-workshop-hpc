"""Baseline training on the Ultralytics construction-PPE dataset.

This script trains YOLO26n on the full 1132-image Ultralytics construction-PPE
dataset in a single stage.  The goal is to establish quality targets (mAP50,
per-class AP) that synthetic-data pipelines should aim to match or exceed.

Dataset (built into Ultralytics):
    construction-ppe.yaml
    Classes: helmet(0), gloves(1), vest(2), boots(3), goggles(4), none(5),
             Person(6), no_helmet(7), no_goggle(8), no_gloves(9), no_boots(10)
    Train: 1132 images | Val: 143 images | Test: 141 images

Usage:
    uv run python data/train_baseline_ppe.py

Results are saved to:
    data/ppe_results/baseline_construction_ppe/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "yolo26n.pt"
DATA = "construction-ppe.yaml"  # built-in Ultralytics dataset config
EPOCHS = 100
PATIENCE = 20
BATCH = 8         # Smaller batch reduces MPS TAL bug probability on Apple MPS
IMGSZ = 640
WORKERS = 0       # workers=0 avoids dataloader multiprocessing issues on MPS

WORKSHOP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = WORKSHOP_DIR / "data" / "ppe_results"
RUN_NAME = "baseline_construction_ppe"
RESULTS_DIR = PROJECT_DIR / RUN_NAME

# Classes we specifically care about (subset of 11)
CLASSES_OF_INTEREST = {0: "helmet", 6: "Person", 7: "no_helmet"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  CUDA detected: {name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple MPS detected")
        return "mps"
    print("  Falling back to CPU")
    return "cpu"


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_metrics_summary(metrics, results_dir: Path) -> None:
    """Pretty-print overall and per-class AP metrics."""
    print_section("FINAL EVALUATION METRICS")

    if not hasattr(metrics, "box"):
        print("  WARNING: metrics.box not available")
        return

    box = metrics.box

    # Overall metrics
    print(f"\n  Overall (11 classes):")
    print(f"    mAP50       : {box.map50:.4f}")
    print(f"    mAP50-95    : {box.map:.4f}")
    print(f"    Precision   : {box.mp:.4f}")
    print(f"    Recall      : {box.mr:.4f}")

    # Per-class AP50
    class_names = [
        "helmet", "gloves", "vest", "boots", "goggles",
        "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots",
    ]

    if hasattr(box, "ap50") and box.ap50 is not None:
        print(f"\n  Per-class AP50 (all 11 classes):")
        for i, name in enumerate(class_names):
            if i < len(box.ap50):
                marker = " <-- KEY" if i in CLASSES_OF_INTEREST else ""
                print(f"    {i:2d} {name:12s}: {box.ap50[i]:.4f}{marker}")

    if hasattr(box, "maps") and box.maps is not None:
        print(f"\n  Per-class mAP50-95 (key classes):")
        for idx, name in CLASSES_OF_INTEREST.items():
            if idx < len(box.maps):
                print(f"    {idx:2d} {name:12s}: {box.maps[idx]:.4f}")

    # Read last row from results.csv if present
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            actual_epochs = last.get("                  epoch", "?").strip()
            print(f"\n  Training completed at epoch: {actual_epochs}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: uv pip install ultralytics")
        sys.exit(1)

    print_section("BASELINE CONSTRUCTION-PPE TRAINING")
    print(f"\n  Model    : {MODEL}")
    print(f"  Dataset  : {DATA}  (construction-ppe, 11 classes)")
    print(f"  Epochs   : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Batch    : {BATCH}  |  imgsz: {IMGSZ}  |  workers: {WORKERS}")
    print(f"  Output   : {RESULTS_DIR}")

    device = detect_device()

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"\nLoading {MODEL} ...")
    model = YOLO(MODEL)

    # ---- Training config ----
    train_kwargs: dict = {
        "data": DATA,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": device,
        "workers": WORKERS,
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        # Standard augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
    }

    print("\nTraining configuration:")
    for k, v in sorted(train_kwargs.items()):
        print(f"  {k:16s}: {v}")

    print(f"\nStarting training for up to {EPOCHS} epochs ...")
    print("-" * 60)

    model.train(**train_kwargs)

    print("-" * 60)
    print("Training complete!")

    # ---- Final evaluation on val set ----
    best_pt = RESULTS_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\nWARNING: best.pt not found at {best_pt}")
        print("Using last.pt for evaluation ...")
        best_pt = RESULTS_DIR / "weights" / "last.pt"

    if best_pt.exists():
        print(f"\nRunning final validation with {best_pt.name} ...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(
            data=DATA,
            imgsz=IMGSZ,
            batch=BATCH,
            device=device,
            verbose=True,
        )
        print_metrics_summary(metrics, RESULTS_DIR)
    else:
        print("\nERROR: No weights found for evaluation.")

    # ---- Summary ----
    print_section("TRAINING SUMMARY")
    print(f"  Model          : {MODEL}")
    print(f"  Dataset        : {DATA}")
    print(f"  Train images   : 1132")
    print(f"  Val images     : 143")
    print(f"  Device         : {device}")
    print(f"  Batch          : {BATCH}")
    print(f"  Image size     : {IMGSZ}")
    print(f"  Max epochs     : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Results dir    : {RESULTS_DIR}")
    print(f"  Best weights   : {best_pt}")
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"  Weight size    : {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
