"""Auto-label synthetic PPE images using Qwen3-VL for open-vocabulary detection.

This script uses Qwen3-VL (an ungated VLM on HuggingFace) to perform
open-vocabulary object detection via grounding prompts, producing YOLO-format
bounding box labels. It serves as the primary teacher model for the workshop
because SAM3 (facebook/sam3) is gated and requires access approval.

Supports two labeling modes via ``--mode``:

  * **exp_a** (default) -- 2-class detection: hardhat (0) + person (1)
  * **3class** -- 3-class detection: hardhat (0), no_hardhat (1), person (2)
    Runs 2-class detection then infers no_hardhat from spatial analysis.

Usage:
    python auto_label_qwen3_vl.py --mode exp_a
    python auto_label_qwen3_vl.py --mode 3class
    python auto_label_qwen3_vl.py --model-id Qwen/Qwen3-VL-4B-Instruct
    python auto_label_qwen3_vl.py --source-dir data/synthetic_ppe --output-dir data/ppe_dataset
    python auto_label_qwen3_vl.py --target "helmet, person"

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
import time
import warnings
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Mode-dependent configuration
# ---------------------------------------------------------------------------
MODE_CONFIG: dict[str, dict] = {
    "exp_a": {
        "class_names": ["hardhat", "person"],
        "target": "hard hat, person",
        "parse_classes": ["hard hat", "person"],
    },
    "3class": {
        "class_names": ["hardhat", "no_hardhat", "person"],
        "target": "hard hat, person",
        "parse_classes": ["hard hat", "person"],
    },
}

# Fraction of a person bbox (from the top) treated as the "head region"
# for no_hardhat inference.
HEAD_REGION_FRACTION = 0.30

# Minimum IoU between a person's head region and a hardhat box to consider
# that person as wearing a hardhat.
CONFLICT_IOU_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Helpers (self-contained, no imports from other scripts)
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_images(source_dir: Path) -> list[Path]:
    """Recursively find jpg/png/webp images in source_dir."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )


def build_dataset_dirs(output_dir: Path) -> dict[str, Path]:
    """Create images/{train,val} labels/{train,val} directories."""
    dirs = {
        "images_train": output_dir / "images" / "train",
        "images_val": output_dir / "images" / "val",
        "labels_train": output_dir / "labels" / "train",
        "labels_val": output_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def box_to_yolo(xyxy: list[float], img_w: int, img_h: int) -> str:
    """Convert [x1, y1, x2, y2] pixel box to normalised ``cx cy w h`` string."""
    x_center = ((xyxy[0] + xyxy[2]) / 2.0) / img_w
    y_center = ((xyxy[1] + xyxy[3]) / 2.0) / img_h
    width = (xyxy[2] - xyxy[0]) / img_w
    height = (xyxy[3] - xyxy[1]) / img_h
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def head_region(box: list[float]) -> list[float]:
    """Return the top 30% of an [x1, y1, x2, y2] bounding box."""
    return [
        box[0],
        box[1],
        box[2],
        box[1] + HEAD_REGION_FRACTION * (box[3] - box[1]),
    ]


def clamp_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    """Clamp box coordinates to image boundaries."""
    return [
        max(0.0, min(box[0], float(img_w))),
        max(0.0, min(box[1], float(img_h))),
        max(0.0, min(box[2], float(img_w))),
        max(0.0, min(box[3], float(img_h))),
    ]


def is_valid_box(box: list[float], img_w: int, img_h: int) -> bool:
    """Reject degenerate boxes (zero-area or fully out of bounds)."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return False
    return True


def iou(box_a: list[float], box_b: list[float]) -> float:
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


# ---------------------------------------------------------------------------
# Qwen3-VL detection
# ---------------------------------------------------------------------------

def detect_objects(
    image,  # PIL.Image
    target: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 1024,
    parse_classes: list[str] | None = None,
):
    """Run Qwen3-VL grounding on an image and return supervision Detections.

    Args:
        image: PIL Image (RGB).
        target: Comma-separated object names for the grounding prompt.
        model: Loaded Qwen3-VL model.
        processor: Loaded AutoProcessor.
        device: Torch device string.
        max_new_tokens: Maximum tokens to generate.
        parse_classes: Class names for supervision parsing (enables class_id).

    Returns:
        sv.Detections with .xyxy in pixel coords and .class_id if classes given.
    """
    import torch
    import supervision as sv

    prompt = f"Outline the position of {target} and output all the coordinates in JSON format."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim the input tokens from the generated output
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # Parse bounding boxes via supervision
    kwargs = {"vlm": sv.VLM.QWEN_3_VL, "result": text, "resolution_wh": image.size}
    if parse_classes:
        kwargs["classes"] = parse_classes

    detections = sv.Detections.from_vlm(**kwargs)
    return detections


# ---------------------------------------------------------------------------
# Per-image labeling — exp_a (2-class: hardhat + person)
# ---------------------------------------------------------------------------

def label_image_exp_a(
    img_path: Path,
    model,
    processor,
    device: str,
    max_new_tokens: int,
) -> tuple[list[str], int, int]:
    """Detect hard hats (class 0) and persons (class 1). No conflict resolution."""
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    cfg = MODE_CONFIG["exp_a"]
    try:
        detections = detect_objects(
            img, cfg["target"], model, processor, device,
            max_new_tokens=max_new_tokens,
            parse_classes=cfg["parse_classes"],
        )
    except Exception as exc:
        warnings.warn(f"Detection failed for {img_path.name}: {exc}")
        return [], img_w, img_h

    lines: list[str] = []

    if detections.xyxy is None or len(detections) == 0:
        return lines, img_w, img_h

    for i in range(len(detections)):
        box = detections.xyxy[i].tolist()
        box = clamp_box(box, img_w, img_h)
        if not is_valid_box(box, img_w, img_h):
            continue

        # Map: "hard hat" (parse class 0) -> YOLO class 0 (hardhat)
        #       "person"   (parse class 1) -> YOLO class 1 (person)
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        yolo = box_to_yolo(box, img_w, img_h)
        lines.append(f"{cls_id} {yolo}")

    return lines, img_w, img_h


# ---------------------------------------------------------------------------
# Per-image labeling — 3class (hardhat + no_hardhat + person)
# ---------------------------------------------------------------------------

def label_image_3class(
    img_path: Path,
    model,
    processor,
    device: str,
    max_new_tokens: int,
) -> tuple[list[str], int, int]:
    """Detect hard hats and persons, then infer no_hardhat from spatial analysis.

    Strategy:
      1. Detect "hard hat" and "person" (same as exp_a).
      2. For each person, check if any hardhat overlaps the head region (top 30%).
      3. If no overlap -> emit no_hardhat head-region box (class 1).
      4. Emit: hardhat=0, no_hardhat=1, person=2.
    """
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    cfg = MODE_CONFIG["3class"]
    try:
        detections = detect_objects(
            img, cfg["target"], model, processor, device,
            max_new_tokens=max_new_tokens,
            parse_classes=cfg["parse_classes"],
        )
    except Exception as exc:
        warnings.warn(f"Detection failed for {img_path.name}: {exc}")
        return [], img_w, img_h

    if detections.xyxy is None or len(detections) == 0:
        return [], img_w, img_h

    # Separate hardhat and person detections
    hardhat_boxes: list[list[float]] = []
    person_boxes: list[list[float]] = []

    for i in range(len(detections)):
        box = detections.xyxy[i].tolist()
        box = clamp_box(box, img_w, img_h)
        if not is_valid_box(box, img_w, img_h):
            continue
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        if cls_id == 0:  # hard hat
            hardhat_boxes.append(box)
        elif cls_id == 1:  # person
            person_boxes.append(box)

    lines: list[str] = []

    # Emit all hardhats as class 0
    for box in hardhat_boxes:
        lines.append(f"0 {box_to_yolo(box, img_w, img_h)}")

    # For each person, check head region overlap with hardhats
    conflicts_resolved = 0
    for p_box in person_boxes:
        h_box = head_region(p_box)
        h_box = clamp_box(h_box, img_w, img_h)

        has_hardhat = any(
            iou(h_box, hh_box) >= CONFLICT_IOU_THRESHOLD
            for hh_box in hardhat_boxes
        )

        if not has_hardhat:
            # Emit no_hardhat head-region box as class 1
            if is_valid_box(h_box, img_w, img_h):
                lines.append(f"1 {box_to_yolo(h_box, img_w, img_h)}")
        else:
            conflicts_resolved += 1

        # Emit full person box as class 2
        lines.append(f"2 {box_to_yolo(p_box, img_w, img_h)}")

    if conflicts_resolved > 0:
        print(f"    {conflicts_resolved} person(s) with hardhat in {img_path.name}")

    return lines, img_w, img_h


# ---------------------------------------------------------------------------
# Dataset YAML writer
# ---------------------------------------------------------------------------

def write_dataset_yaml(
    output_dir: Path,
    class_names: list[str],
    labeler: str = "qwen3-vl",
) -> Path:
    """Write a dataset.yaml with the given class names."""
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml_content = (
        f"# Auto-generated PPE dataset configuration\n"
        f"# Created by auto_label_qwen3_vl.py ({labeler}, classes: {class_names})\n"
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

def label_dataset(args: argparse.Namespace) -> None:
    """Label PPE images using Qwen3-VL open-vocabulary detection.

    Args:
        args: Namespace with source_dir, output_dir, model_id, mode,
              split_ratio, seed, device, max_new_tokens, target.
    """
    # ---- Deferred heavy imports ----
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from PIL import Image  # noqa: F401
        import supervision as sv  # noqa: F401
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency: {exc}\n"
            f"Install with:  uv pip install transformers torch torchvision "
            f"pillow supervision qwen-vl-utils\n"
        )
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # ---- Mode config ----
    mode: str = args.mode
    if mode not in MODE_CONFIG:
        print(f"ERROR: Unknown mode '{mode}'. Choose from: {list(MODE_CONFIG.keys())}")
        sys.exit(1)
    mode_cfg = MODE_CONFIG[mode]

    # --target override: replace the mode's default detection prompt
    if args.target is not None:
        target_str = args.target.strip()
        mode_cfg["target"] = target_str
        mode_cfg["parse_classes"] = [t.strip() for t in target_str.split(",")]
        print(f"Target override: \"{target_str}\" -> parse_classes={mode_cfg['parse_classes']}")

    class_names = mode_cfg["class_names"]

    label_fn = label_image_exp_a if mode == "exp_a" else label_image_3class

    print(f"Mode: {mode} -- classes: {class_names}")

    # ---- Collect images ----
    source_dir: Path = Path(args.source_dir)
    output_dir: Path = Path(args.output_dir)

    images = collect_images(source_dir)
    if not images:
        print(f"ERROR: No images found in {source_dir}")
        sys.exit(1)
    print(f"Found {len(images)} images in {source_dir}")

    # ---- Device ----
    device = args.device or detect_device()
    print(f"Using device: {device}")

    # ---- Load Qwen3-VL model ----
    model_id = args.model_id
    print(f"Loading model: {model_id} ...")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load model: {exc}")
        print("Ensure you have internet access and sufficient disk space.")
        print(f"  8B model: ~16GB, 4B model: ~8GB")
        print(f"Install/update with:  uv pip install -U transformers torch")
        sys.exit(1)
    print(f"Model loaded ({dtype})")

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
    total_time = 0.0
    total_processed = 0
    max_new_tokens = args.max_new_tokens

    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        img_dir = dirs[f"images_{split_name}"]
        lbl_dir = dirs[f"labels_{split_name}"]

        iterator = enumerate(split_images)
        if tqdm is not None:
            iterator = tqdm(
                iterator, total=len(split_images),
                desc=f"  {split_name}", unit="img",
            )

        for i, img_path in iterator:
            t0 = time.time()

            try:
                label_lines, img_w, img_h = label_fn(
                    img_path, model, processor, device, max_new_tokens,
                )
            except RuntimeError as exc:
                if device == "mps":
                    print(f"WARNING: MPS inference failed ({exc}); retrying on CPU.")
                    device = "cpu"
                    model = model.to(device)
                    label_lines, img_w, img_h = label_fn(
                        img_path, model, processor, device, max_new_tokens,
                    )
                else:
                    raise

            elapsed = time.time() - t0
            total_time += elapsed
            total_processed += 1

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

            # Progress when tqdm is not available
            if tqdm is None:
                total_so_far = i + 1
                if total_so_far % 5 == 0 or total_so_far == len(split_images):
                    avg = total_time / total_processed
                    print(
                        f"  [{split_name}] {total_so_far}/{len(split_images)} "
                        f"({avg:.1f}s/img)"
                    )

    # ---- Write dataset.yaml ----
    yaml_path = write_dataset_yaml(output_dir, class_names)

    # ---- Summary ----
    total_images = len(images)
    total_labels = sum(stats.values())
    avg_time = total_time / total_processed if total_processed > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"AUTO-LABELING COMPLETE (Qwen3-VL, mode={mode})")
    print("=" * 60)
    print(f"  Source directory : {source_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Dataset config   : {yaml_path}")
    print(f"  Model            : {model_id}")
    print(f"  Device           : {device}")
    print(f"  Dtype            : {dtype}")
    print(f"  Max new tokens   : {max_new_tokens}")
    print(f"  Mode             : {mode}")
    print()
    print(f"  Total images     : {total_images}")
    print(f"  Train images     : {len(train_images)}")
    print(f"  Val images       : {len(val_images)}")
    print(f"  Images w/ labels : {images_with_labels}")
    print(f"  Images w/o labels: {images_without_labels}")
    print(f"  Avg time/image   : {avg_time:.1f}s")
    print()
    print(f"  Prompt target    : \"{mode_cfg['target']}\"")
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
        description="Auto-label PPE images with Qwen3-VL open-vocabulary detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(MODE_CONFIG.keys()),
        default="exp_a",
        help="Labeling mode: exp_a (2-class, recommended), 3class (3-class with no_hardhat inference).",
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
        default=Path(__file__).resolve().parent / "ppe_dataset_qwen3vl",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID. Use Qwen/Qwen3-VL-4B-Instruct for less VRAM.",
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
        help="Device for inference: cuda, mps, cpu (auto-detected if omitted).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for VLM generation per image.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Override the detection prompt target string. "
            "Comma-separated object names, e.g. 'helmet, person' or "
            "'hard hat, safety vest, person'. If omitted, uses the mode default."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    label_dataset(_parse_args())
