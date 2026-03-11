"""Auto-label synthetic PPE images using SAM3 via HuggingFace Transformers.

This script uses the HuggingFace ``transformers`` library to run SAM3
(Segment Anything Model 3) with **text-prompted instance segmentation**.

Supports three labeling modes via ``--mode``:

  * **3class** (default) — 3-class detection: hardhat, no_hardhat, person
  * **exp_a** — 2-class detection: hardhat + person (compliance via post-processing)
  * **exp_b** — 2-class compliance: safe + unsafe (direct compliance labels)

Usage:
    python auto_label_sam3_hf.py --mode 3class
    python auto_label_sam3_hf.py --mode exp_a --output-dir data/ppe_dataset_exp_a
    python auto_label_sam3_hf.py --mode exp_b --output-dir data/ppe_dataset_exp_b
    python auto_label_sam3_hf.py --threshold 0.3 --device cuda

Output structure:
    <output-dir>/
    +-- images/ (train/ val/)
    +-- labels/ (train/ val/)
    +-- dataset.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared helpers (inlined from auto_label_ppe to avoid cross-module dependency)
# ---------------------------------------------------------------------------

def collect_images(source_dir: Path) -> list[Path]:
    """Recursively collect all image files from *source_dir*."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )


def build_dataset_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the standard YOLO train/val directory structure."""
    dirs = {
        "images_train": output_dir / "images" / "train",
        "images_val": output_dir / "images" / "val",
        "labels_train": output_dir / "labels" / "train",
        "labels_val": output_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# ---------------------------------------------------------------------------
# Mode-dependent configuration
# ---------------------------------------------------------------------------
MODE_CONFIG: dict[str, dict] = {
    "3class": {
        "class_names": ["hardhat", "no_hardhat", "person"],
        "prompts": {
            "hardhat": "hard hat",
            "no_hardhat": "person not wearing a hard hat",
            "person": "person",
        },
    },
    "exp_a": {
        "class_names": ["hardhat", "person"],
        "prompts": {"hardhat": "hard hat", "person": "person"},
    },
    "exp_b": {
        "class_names": ["safe", "unsafe"],
        "prompts": {
            "safe": "person wearing a hard hat",
            "unsafe": "person not wearing a hard hat",
        },
    },
}

# Default prompts used by the 3class labeling function
PROMPTS: dict[str, str] = MODE_CONFIG["3class"]["prompts"]

# ---------------------------------------------------------------------------
# Label mapping (3class mode)
# ---------------------------------------------------------------------------
CLASS_ID_HARDHAT = 0
CLASS_ID_NO_HARDHAT = 1
CLASS_ID_PERSON = 2

# Fraction of a detected bbox (measured from the top) used as the "head
# region" when emitting no_hardhat labels.  This matches the convention
# in ``auto_label_ppe.py``.
HEAD_REGION_FRACTION = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_device(requested: str | None) -> str:
    """Return the requested device, falling back to CPU when MPS is broken."""
    device = requested or _detect_device()
    if device == "mps":
        try:
            import torch
            _ = torch.zeros(1, device="mps")
        except Exception:
            print("WARNING: MPS device requested but unavailable; falling back to CPU.")
            device = "cpu"
    return device


def _box_to_yolo(box: list[float], img_w: int, img_h: int) -> str:
    """Convert an [x1, y1, x2, y2] pixel box to ``x_center y_center w h`` (normalised).

    Returns the four normalised values as a space-separated string (no class id).
    """
    x_center = ((box[0] + box[2]) / 2.0) / img_w
    y_center = ((box[1] + box[3]) / 2.0) / img_h
    width = (box[2] - box[0]) / img_w
    height = (box[3] - box[1]) / img_h
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _head_region(box: list[float]) -> list[float]:
    """Return the top 30 % of an [x1, y1, x2, y2] bounding box (head region)."""
    return [
        box[0],
        box[1],
        box[2],
        box[1] + HEAD_REGION_FRACTION * (box[3] - box[1]),
    ]


def _is_valid_box(box: list[float], img_w: int, img_h: int) -> bool:
    """Reject degenerate boxes (zero-area or out of bounds)."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return False
    return True


def _clamp_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    """Clamp box coordinates to image boundaries."""
    return [
        max(0.0, min(box[0], float(img_w))),
        max(0.0, min(box[1], float(img_h))),
        max(0.0, min(box[2], float(img_w))),
        max(0.0, min(box[3], float(img_h))),
    ]


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# Minimum IoU between a no_hardhat person's head region and a hardhat box
# to consider them conflicting (i.e. the person IS wearing a hardhat).
CONFLICT_IOU_THRESHOLD = 0.1

# IoU threshold for deduplicating person boxes from different prompts.
PERSON_DEDUP_IOU_THRESHOLD = 0.5


def _run_prompt(
    img,
    text: str,
    processor,
    model,
    device: str,
    threshold: float,
    mask_threshold: float,
    img_w: int,
    img_h: int,
) -> list[list[float]]:
    """Run a single SAM3 prompt and return valid, clamped [x1,y1,x2,y2] boxes."""
    import torch

    inputs = processor(images=img, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if target_sizes is not None:
        target_sizes = target_sizes.tolist()
    else:
        target_sizes = [(img_h, img_w)]

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results.get("boxes", [])
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if not boxes:
        return []

    valid_boxes: list[list[float]] = []
    for box in boxes:
        box = _clamp_box(box, img_w, img_h)
        if _is_valid_box(box, img_w, img_h):
            valid_boxes.append(box)
    return valid_boxes


# ---------------------------------------------------------------------------
# Per-image labeling
# ---------------------------------------------------------------------------

def _label_image(
    img_path: Path,
    processor,  # Sam3Processor
    model,      # Sam3Model
    device: str,
    threshold: float,
    mask_threshold: float,
) -> tuple[list[str], int, int]:
    """Run SAM3 on a single image and return YOLO label lines.

    Uses a three-phase approach:
      1. Collect raw detections from all three prompts.
      2. Resolve conflicts: drop no_hardhat labels when the person's head
         region overlaps a detected hardhat (IoU >= CONFLICT_IOU_THRESHOLD).
      3. Deduplicate person boxes across prompts (IoU > PERSON_DEDUP_IOU_THRESHOLD).

    Returns:
        A tuple of ``(label_lines, img_w, img_h)``.
    """
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    # ---- Phase 1: Collect raw detections ----
    hardhat_boxes = _run_prompt(
        img, PROMPTS["hardhat"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )
    no_hardhat_person_boxes = _run_prompt(
        img, PROMPTS["no_hardhat"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )
    person_boxes = _run_prompt(
        img, PROMPTS["person"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )

    # ---- Phase 2: Conflict resolution ----
    # For each no_hardhat person, check if their head region overlaps any
    # hardhat box.  If so, this person IS wearing a hardhat — drop the
    # no_hardhat label.
    resolved_no_hardhat: list[list[float]] = []
    conflicts_resolved = 0

    for person_box in no_hardhat_person_boxes:
        head_box = _head_region(person_box)
        head_box = _clamp_box(head_box, img_w, img_h)
        has_hardhat = any(
            _iou(head_box, hh_box) >= CONFLICT_IOU_THRESHOLD
            for hh_box in hardhat_boxes
        )
        if has_hardhat:
            conflicts_resolved += 1
        else:
            resolved_no_hardhat.append(person_box)

    if conflicts_resolved > 0:
        print(f"    Resolved {conflicts_resolved} hardhat/no_hardhat conflict(s) in {img_path.name}")

    # ---- Phase 3: Emit YOLO lines with person deduplication ----
    lines: list[str] = []

    # 3a. Hardhat boxes → class 0
    for box in hardhat_boxes:
        yolo = _box_to_yolo(box, img_w, img_h)
        lines.append(f"{CLASS_ID_HARDHAT} {yolo}")

    # 3b. Surviving no_hardhat persons → class 1 (head region) + class 2 (full body)
    emitted_person_boxes: list[list[float]] = []
    for person_box in resolved_no_hardhat:
        head_box = _head_region(person_box)
        head_box = _clamp_box(head_box, img_w, img_h)
        if _is_valid_box(head_box, img_w, img_h):
            yolo_head = _box_to_yolo(head_box, img_w, img_h)
            lines.append(f"{CLASS_ID_NO_HARDHAT} {yolo_head}")
        yolo_person = _box_to_yolo(person_box, img_w, img_h)
        lines.append(f"{CLASS_ID_PERSON} {yolo_person}")
        emitted_person_boxes.append(person_box)

    # 3c. Person boxes from "person" prompt → class 2, deduplicated
    for p_box in person_boxes:
        is_duplicate = any(
            _iou(p_box, ep_box) > PERSON_DEDUP_IOU_THRESHOLD
            for ep_box in emitted_person_boxes
        )
        if not is_duplicate:
            yolo = _box_to_yolo(p_box, img_w, img_h)
            lines.append(f"{CLASS_ID_PERSON} {yolo}")
            emitted_person_boxes.append(p_box)

    return lines, img_w, img_h


# ---------------------------------------------------------------------------
# Per-image labeling — Experiment A (hardhat + person, 2-class detection)
# ---------------------------------------------------------------------------

def _label_image_exp_a(
    img_path: Path,
    processor,
    model,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> tuple[list[str], int, int]:
    """Run SAM3 with 2 prompts: 'hard hat' (class 0) + 'person' (class 1).

    No conflict resolution — emit all detected hardhats and persons as-is.
    """
    from PIL import Image

    prompts = MODE_CONFIG["exp_a"]["prompts"]
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    hardhat_boxes = _run_prompt(
        img, prompts["hardhat"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )
    person_boxes = _run_prompt(
        img, prompts["person"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )

    lines: list[str] = []
    for box in hardhat_boxes:
        lines.append(f"0 {_box_to_yolo(box, img_w, img_h)}")
    for box in person_boxes:
        lines.append(f"1 {_box_to_yolo(box, img_w, img_h)}")

    return lines, img_w, img_h


# ---------------------------------------------------------------------------
# Per-image labeling — Experiment B (safe + unsafe, 2-class compliance)
# ---------------------------------------------------------------------------

EXP_B_CONFLICT_IOU_THRESHOLD = 0.5

def _label_image_exp_b(
    img_path: Path,
    processor,
    model,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> tuple[list[str], int, int]:
    """Run SAM3 with 2 prompts: 'person wearing a hard hat' (class 0 = safe)
    and 'person not wearing a hard hat' (class 1 = unsafe).

    Conflict resolution: if the same person (IoU > 0.5) is detected by both
    prompts, prefer 'unsafe' (conservative for workplace safety).
    Emit whole-person bboxes for both classes.
    """
    from PIL import Image

    prompts = MODE_CONFIG["exp_b"]["prompts"]
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    safe_boxes = _run_prompt(
        img, prompts["safe"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )
    unsafe_boxes = _run_prompt(
        img, prompts["unsafe"], processor, model, device,
        threshold, mask_threshold, img_w, img_h,
    )

    # Conflict resolution: remove safe boxes that overlap with any unsafe box
    resolved_safe: list[list[float]] = []
    conflicts = 0
    for s_box in safe_boxes:
        has_conflict = any(
            _iou(s_box, u_box) > EXP_B_CONFLICT_IOU_THRESHOLD
            for u_box in unsafe_boxes
        )
        if has_conflict:
            conflicts += 1
        else:
            resolved_safe.append(s_box)

    if conflicts > 0:
        print(f"    Resolved {conflicts} safe/unsafe conflict(s) in {img_path.name} (kept unsafe)")

    lines: list[str] = []
    for box in resolved_safe:
        lines.append(f"0 {_box_to_yolo(box, img_w, img_h)}")
    for box in unsafe_boxes:
        lines.append(f"1 {_box_to_yolo(box, img_w, img_h)}")

    return lines, img_w, img_h


# ---------------------------------------------------------------------------
# Local dataset.yaml writer (mode-aware class names)
# ---------------------------------------------------------------------------

def _write_dataset_yaml(output_dir: Path, class_names: list[str], labeler: str = "sam3-hf") -> Path:
    """Write a ``dataset.yaml`` with the given class names."""
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml_content = (
        f"# Auto-generated PPE dataset configuration\n"
        f"# Created by auto_label_sam3_hf.py ({labeler}, mode classes: {class_names})\n"
        f"\n"
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
        f"\n"
        f"nc: {len(class_names)}\n"
    )
    yaml_path.write_text(yaml_content)
    return yaml_path


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_with_sam3_hf(args: argparse.Namespace) -> None:
    """Label PPE images using SAM3 via HuggingFace Transformers.

    Args:
        args: Namespace with ``source_dir``, ``output_dir``, ``threshold``,
              ``split_ratio``, ``seed``, ``device``, and ``mode``.
    """
    # ---- Deferred heavy imports ----
    try:
        import torch  # noqa: F401
        from transformers import Sam3Model, Sam3Processor
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency: {exc}\n"
            f"Install with:  uv pip install transformers torch torchvision pillow\n"
        )
        sys.exit(1)

    # ---- Mode config ----
    mode: str = getattr(args, "mode", "3class")
    if mode not in MODE_CONFIG:
        print(f"ERROR: Unknown mode '{mode}'. Choose from: {list(MODE_CONFIG.keys())}")
        sys.exit(1)
    mode_cfg = MODE_CONFIG[mode]
    class_names = mode_cfg["class_names"]
    mode_prompts = mode_cfg["prompts"]

    # Select the labeling function for this mode
    label_fn_map = {
        "3class": _label_image,
        "exp_a": _label_image_exp_a,
        "exp_b": _label_image_exp_b,
    }
    label_fn = label_fn_map[mode]

    print(f"Mode: {mode} — classes: {class_names}")

    # ---- Collect images ----
    source_dir: Path = Path(args.source_dir)
    output_dir: Path = Path(args.output_dir)

    images = collect_images(source_dir)
    if not images:
        print(f"ERROR: No images found in {source_dir}")
        sys.exit(1)
    print(f"Found {len(images)} images in {source_dir}")

    # ---- Device ----
    device = _safe_device(getattr(args, "device", None))
    print(f"Using device: {device}")

    # ---- Load SAM3 model via HuggingFace ----
    model_id = "facebook/sam3"
    print(f"Loading SAM3 model: {model_id} ...")
    try:
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id).to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load SAM3 model: {exc}")
        print("Ensure you have internet access and sufficient disk space.")
        print(f"Install/update with:  uv pip install -U transformers torch")
        sys.exit(1)
    print("SAM3 model loaded")

    # ---- Prepare output ----
    dirs = build_dataset_dirs(output_dir)

    # ---- Train / val split ----
    random.seed(args.seed)
    shuffled = list(images)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * args.split_ratio)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]
    print(f"Train/val split: {len(train_images)} / {len(val_images)}")

    # ---- Process images ----
    stats: Counter = Counter()
    images_with_labels = 0
    images_without_labels = 0
    threshold: float = getattr(args, "threshold", 0.5)
    mask_threshold: float = getattr(args, "mask_threshold", 0.5)

    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        img_dir = dirs[f"images_{split_name}"]
        lbl_dir = dirs[f"labels_{split_name}"]

        for i, img_path in enumerate(split_images):
            try:
                label_lines, img_w, img_h = label_fn(
                    img_path,
                    processor,
                    model,
                    device,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                )
            except RuntimeError as exc:
                # MPS fallback: if inference fails on MPS, retry on CPU.
                if device == "mps":
                    print(f"WARNING: MPS inference failed ({exc}); retrying on CPU.")
                    device = "cpu"
                    model = model.to(device)
                    label_lines, img_w, img_h = label_fn(
                        img_path,
                        processor,
                        model,
                        device,
                        threshold=threshold,
                        mask_threshold=mask_threshold,
                    )
                else:
                    raise

            # ---- Write image + label ----
            relative = img_path.relative_to(source_dir)
            safe_name = relative.as_posix().replace("/", "_")
            stem = Path(safe_name).stem
            suffix = img_path.suffix

            dst_img = img_dir / f"{stem}{suffix}"
            shutil.copy2(img_path, dst_img)

            dst_lbl = lbl_dir / f"{stem}.txt"
            dst_lbl.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else "")
            )

            if label_lines:
                images_with_labels += 1
                for line in label_lines:
                    cls_id = int(line.split()[0])
                    stats[class_names[cls_id]] += 1
            else:
                images_without_labels += 1

            total_so_far = i + 1
            if total_so_far % 5 == 0 or total_so_far == len(split_images):
                print(
                    f"  [{split_name}] {total_so_far}/{len(split_images)} images processed"
                )

    # ---- Write dataset.yaml (mode-aware) ----
    yaml_path = _write_dataset_yaml(output_dir, class_names, labeler="sam3-hf")

    # ---- Summary ----
    total_images = len(images)
    total_labels = sum(stats.values())

    print("\n" + "=" * 60)
    print(f"AUTO-LABELING COMPLETE (SAM3 HuggingFace, mode={mode})")
    print("=" * 60)
    print(f"  Source directory : {source_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Dataset config   : {yaml_path}")
    print(f"  Model            : facebook/sam3 (HuggingFace Transformers)")
    print(f"  Device           : {device}")
    print(f"  Threshold        : {threshold}")
    print(f"  Mask threshold   : {mask_threshold}")
    print(f"  Mode             : {mode}")
    print()
    print(f"  Total images     : {total_images}")
    print(f"  Train images     : {len(train_images)}")
    print(f"  Val images       : {len(val_images)}")
    print(f"  Images w/ labels : {images_with_labels}")
    print(f"  Images w/o labels: {images_without_labels}")
    print()
    print("  Prompts used:")
    for key, text in mode_prompts.items():
        print(f"    {key:12s}: \"{text}\"")
    print()
    print("  Detections per class:")
    for cls_name in class_names:
        count = stats.get(cls_name, 0)
        print(f"    {cls_name:12s}: {count:5d}")
    print(f"    {'TOTAL':12s}: {total_labels:5d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label PPE images with SAM3 via HuggingFace Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(MODE_CONFIG.keys()),
        default="3class",
        help="Labeling mode: 3class (default), exp_a (hardhat+person), exp_b (safe+unsafe).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_ppe",
        help="Root directory containing source images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ppe_dataset_sam3hf",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for SAM3 detections.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold for SAM3.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of images for training (rest goes to val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (auto-detected if omitted).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    label_with_sam3_hf(_parse_args())
