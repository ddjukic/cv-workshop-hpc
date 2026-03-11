"""Visualize ground-truth YOLO annotations on validation images.

Draws bounding boxes on images with class-specific colors:
  - GREEN  = hardhat (class 0)
  - RED    = no_hardhat (class 1)
  - BLUE   = person (class 2)
  - YELLOW dashed outline = head region (top 40% of person bbox)

Usage:
    uv run python data/visualize_gt_annotations.py \
        --dataset-dir data/ppe_dataset_sam3_v2_filtered \
        --output-dir data/error_analysis/gt_annotations_sam3_v2_val \
        --split val

    uv run python data/visualize_gt_annotations.py \
        --dataset-dir data/ppe_dataset_sam3_v2_filtered \
        --output-dir data/error_analysis/gt_annotations_sam3_v2_val \
        --split val --max-images 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "hardhat", 1: "no_hardhat", 2: "person"}
CLASS_COLORS = {
    0: (0, 200, 0),       # GREEN  - hardhat
    1: (220, 30, 30),     # RED    - no_hardhat
    2: (30, 100, 220),    # BLUE   - person
}
HEAD_REGION_COLOR = (255, 220, 0)  # YELLOW - head region overlay
HEAD_FRACTION = 0.40  # Top 40% of person bbox

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into (class_id, cx, cy, w, h) tuples."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            labels.append((cls_id, cx, cy, w, h))
        except ValueError:
            continue
    return labels


def yolo_to_pixel(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalised YOLO coords to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_annotations(
    img: Image.Image,
    labels: list[tuple[int, float, float, float, float]],
) -> Image.Image:
    """Draw bounding boxes and head regions on a copy of the image."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = img.size

    # Try to get a decent font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw person boxes first (bottom layer), then hardhats, then no_hardhat
    draw_order = sorted(labels, key=lambda l: {2: 0, 0: 1, 1: 2}.get(l[0], 3))

    for cls_id, cx, cy, w, h in draw_order:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
        color = CLASS_COLORS.get(cls_id, (128, 128, 128))
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        # Main bounding box
        line_width = 3 if cls_id != 2 else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label text
        text = name
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        # Background for text
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), text, fill=(255, 255, 255), font=font)

        # For person boxes, also draw head region outline
        if cls_id == 2:
            head_y2 = y1 + int(HEAD_FRACTION * (y2 - y1))
            # Dashed-like effect: draw thinner yellow rectangle
            draw.rectangle(
                [x1 + 1, y1 + 1, x2 - 1, head_y2],
                outline=HEAD_REGION_COLOR,
                width=1,
            )

    return annotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GT annotations on YOLO dataset images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Root of YOLO dataset (with images/ and labels/ subdirs).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Which split to visualize (train or val).",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to process (0 = all).",
    )
    args = parser.parse_args()

    images_dir = args.dataset_dir / "images" / args.split
    labels_dir = args.dataset_dir / "labels" / args.split

    if not images_dir.is_dir():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    if args.max_images > 0:
        image_files = image_files[:args.max_images]

    print(f"Visualizing {len(image_files)} {args.split} images from {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Colors: GREEN=hardhat, RED=no_hardhat, BLUE=person, YELLOW=head_region")
    print()

    # Stats
    total_labels = {0: 0, 1: 0, 2: 0}

    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        labels = parse_yolo_label(label_path)

        img = Image.open(img_path).convert("RGB")
        annotated = draw_annotations(img, labels)

        out_path = args.output_dir / f"{stem}_annotated.jpg"
        annotated.save(out_path, quality=90)

        # Stats
        for cls_id, *_ in labels:
            total_labels[cls_id] = total_labels.get(cls_id, 0) + 1

        if (i + 1) % 5 == 0 or (i + 1) == len(image_files):
            print(f"  [{i + 1}/{len(image_files)}] {stem}: {len(labels)} labels")

    print()
    print("=" * 50)
    print("ANNOTATION VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"  Images processed : {len(image_files)}")
    print(f"  Output directory : {args.output_dir}")
    print(f"  Label counts:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"    {name:12s}: {total_labels.get(cls_id, 0)}")
    print(f"    {'TOTAL':12s}: {sum(total_labels.values())}")
    print("=" * 50)


if __name__ == "__main__":
    main()
