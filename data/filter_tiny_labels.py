"""Filter tiny/noise labels from a YOLO dataset.

Removes labels where either normalised dimension (width or height) is below a
configurable threshold (default 0.03, roughly 20 px at 640 resolution).  These
micro-boxes are typically false positives from auto-labelling pipelines or
truncated objects at image edges that hurt training stability.

The script copies images and writes filtered labels to a new output directory,
preserving the standard YOLO ``images/{train,val}/`` and ``labels/{train,val}/``
split structure.

Usage:
    uv run python data/filter_tiny_labels.py \\
        --input-dir data/ppe_dataset_v3 \\
        --output-dir data/ppe_dataset_v3_filtered

    uv run python data/filter_tiny_labels.py \\
        --input-dir data/ppe_dataset_v3 \\
        --output-dir data/ppe_dataset_v3_filtered \\
        --min-dim 0.05

Classes:
    0: hardhat
    1: no_hardhat
    2: person
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Class ontology (shared across the PPE pipeline)
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = ["hardhat", "no_hardhat", "person"]

# Image file extensions recognised when copying source images
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------

def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse a single YOLO label line.

    Returns (class_id, x_center, y_center, width, height) or ``None`` if the
    line is malformed or empty.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    return class_id, cx, cy, w, h


def filter_labels(
    label_path: Path,
    min_dim: float,
) -> tuple[list[str], dict[int, int], dict[int, int]]:
    """Read a YOLO label file and filter out tiny boxes.

    Returns:
        kept_lines: Label lines that passed the size threshold.
        removed_per_class: {class_id: count} of removed labels.
        kept_per_class: {class_id: count} of kept labels.
    """
    kept_lines: list[str] = []
    removed_per_class: dict[int, int] = defaultdict(int)
    kept_per_class: dict[int, int] = defaultdict(int)

    text = label_path.read_text()
    for line in text.splitlines():
        parsed = parse_yolo_line(line)
        if parsed is None:
            # Skip blank / malformed lines silently
            continue

        class_id, cx, cy, w, h = parsed

        if w < min_dim or h < min_dim:
            removed_per_class[class_id] += 1
        else:
            kept_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            kept_per_class[class_id] += 1

    return kept_lines, dict(removed_per_class), dict(kept_per_class)


# ---------------------------------------------------------------------------
# Dataset structure helpers
# ---------------------------------------------------------------------------

def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    """Find the image file matching a label stem in the given directory."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def write_dataset_yaml(
    output_dir: Path,
    class_names: list[str] | None = None,
) -> Path:
    """Write a YOLO dataset.yaml configuration file.

    Args:
        output_dir: Root directory of the filtered dataset.
        class_names: Optional list of class names to write.  If None,
            falls back to the module-level CLASS_NAMES (3-class default).
            Pass a 2-element list for 2-class datasets.
    """
    names = class_names if class_names is not None else CLASS_NAMES
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
        f"\n"
        f"nc: {len(names)}\n"
    )
    yaml_path.write_text(yaml_content)
    return yaml_path


def _read_class_names_from_yaml(yaml_path: Path) -> list[str] | None:
    """Parse class names from a YOLO dataset.yaml without a YAML library.

    Returns a list of class names in order, or None if parsing fails.
    """
    try:
        text = yaml_path.read_text()
    except OSError:
        return None

    names: dict[int, str] = {}
    in_names_block = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names_block = True
            continue
        if in_names_block:
            # Stop at the next top-level key (no leading whitespace)
            if line and not line[0].isspace():
                in_names_block = False
                continue
            # Parse "  <int>: <name>" lines
            if ":" in stripped:
                try:
                    idx_str, name = stripped.split(":", 1)
                    idx = int(idx_str.strip())
                    names[idx] = name.strip()
                except ValueError:
                    pass

    if not names:
        return None
    return [names[i] for i in sorted(names)]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class FilterStats:
    """Accumulates per-class and per-split filtering statistics."""

    def __init__(self) -> None:
        # {class_id: count} aggregated across all splits
        self.before: dict[int, int] = defaultdict(int)
        self.after: dict[int, int] = defaultdict(int)
        self.removed: dict[int, int] = defaultdict(int)

        # Image-level counters
        self.images_processed: int = 0
        self.images_with_removals: int = 0
        self.images_fully_emptied: int = 0
        self.images_missing_source: int = 0

    def record(
        self,
        kept_per_class: dict[int, int],
        removed_per_class: dict[int, int],
    ) -> None:
        had_removals = False
        for cls_id, count in kept_per_class.items():
            self.before[cls_id] += count
            self.after[cls_id] += count
        for cls_id, count in removed_per_class.items():
            self.before[cls_id] += count
            self.removed[cls_id] += count
            had_removals = True

        self.images_processed += 1
        if had_removals:
            self.images_with_removals += 1
        if had_removals and not kept_per_class:
            self.images_fully_emptied += 1


def format_stats_report(
    stats: FilterStats,
    min_dim: float,
    class_names: list[str] | None = None,
) -> str:
    """Produce a human-readable filtering report.

    Args:
        stats: Accumulated filtering statistics.
        min_dim: The minimum dimension threshold used.
        class_names: Optional class name list for pretty-printing class IDs.
            Defaults to the module-level CLASS_NAMES if None.
    """
    _names = class_names if class_names is not None else CLASS_NAMES
    lines: list[str] = []
    W = 70

    def line(s: str = "") -> None:
        lines.append(s)

    def pct(num: int, den: int) -> str:
        if den == 0:
            return "  --%"
        return f"{num / den * 100:5.1f}%"

    def class_name(cid: int) -> str:
        if 0 <= cid < len(_names):
            return _names[cid]
        return f"class_{cid}"

    all_classes = sorted(
        set(stats.before.keys()) | set(stats.after.keys()) | set(stats.removed.keys())
    )

    # Header
    line("=" * W)
    line("TINY LABEL FILTERING REPORT")
    line("=" * W)
    line(f"  Min dimension threshold : {min_dim}")
    line(f"  Images processed        : {stats.images_processed}")
    line(f"  Images with removals    : {stats.images_with_removals}")
    line(f"  Images fully emptied    : {stats.images_fully_emptied}")
    line(f"  Images missing source   : {stats.images_missing_source}")

    # Per-class table
    line()
    line("-" * W)
    line("  Per-class breakdown:")
    line("-" * W)
    line(f"  {'Class':<15} {'Before':>10} {'After':>10} {'Removed':>10} {'% Removed':>12}")
    line(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    total_before = total_after = total_removed = 0
    for cls_id in all_classes:
        b = stats.before.get(cls_id, 0)
        a = stats.after.get(cls_id, 0)
        r = stats.removed.get(cls_id, 0)
        total_before += b
        total_after += a
        total_removed += r
        line(f"  {class_name(cls_id):<15} {b:>10} {a:>10} {r:>10} {pct(r, b):>12}")

    line(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    line(f"  {'TOTAL':<15} {total_before:>10} {total_after:>10} {total_removed:>10} {pct(total_removed, total_before):>12}")

    # Summary
    line()
    line("=" * W)
    line(f"  Labels before : {total_before}")
    line(f"  Labels after  : {total_after}")
    line(f"  Labels removed: {total_removed} ({pct(total_removed, total_before).strip()})")
    line("=" * W)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def filter_dataset(
    input_dir: Path,
    output_dir: Path,
    min_dim: float,
) -> FilterStats:
    """Filter tiny labels from a YOLO dataset, preserving train/val structure.

    For each split (train, val):
      - Read each label file from ``input_dir/labels/{split}/``
      - Remove lines where normalised width < min_dim OR height < min_dim
      - Write filtered labels to ``output_dir/labels/{split}/``
      - Copy corresponding image from ``input_dir/images/{split}/``
    """
    stats = FilterStats()

    for split in ("train", "val"):
        input_labels_dir = input_dir / "labels" / split
        input_images_dir = input_dir / "images" / split
        output_labels_dir = output_dir / "labels" / split
        output_images_dir = output_dir / "images" / split

        if not input_labels_dir.is_dir():
            print(f"  [{split}] No labels directory found at {input_labels_dir}, skipping.")
            continue

        # Create output directories
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(input_labels_dir.glob("*.txt"))
        if not label_files:
            print(f"  [{split}] No label files found, skipping.")
            continue

        print(f"\n  [{split}] Processing {len(label_files)} label files ...")

        for i, label_path in enumerate(label_files):
            stem = label_path.stem

            # Filter labels
            kept_lines, removed_per_class, kept_per_class = filter_labels(
                label_path, min_dim
            )
            stats.record(kept_per_class, removed_per_class)

            # Write filtered labels (always write, even if empty)
            dst_label = output_labels_dir / label_path.name
            dst_label.write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else "")
            )

            # Copy corresponding image
            src_image = find_image_for_stem(input_images_dir, stem)
            if src_image is not None:
                dst_image = output_images_dir / src_image.name
                shutil.copy2(src_image, dst_image)
            else:
                stats.images_missing_source += 1
                print(f"    WARNING: No image found for '{stem}' in {input_images_dir}")

            # Progress
            count = i + 1
            if count % 50 == 0 or count == len(label_files):
                print(f"  [{split}] {count}/{len(label_files)} files processed")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter tiny/noise labels from a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input dataset directory (must have images/{train,val}/ and labels/{train,val}/ structure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the filtered dataset.",
    )
    parser.add_argument(
        "--min-dim",
        type=float,
        default=0.03,
        help="Minimum normalised dimension threshold.  Labels with width OR height below this are removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input
    if not args.input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    has_labels = False
    for split in ("train", "val"):
        if (args.input_dir / "labels" / split).is_dir():
            has_labels = True
            break
    if not has_labels:
        print(f"ERROR: No labels/train or labels/val directory found in {args.input_dir}")
        sys.exit(1)

    if args.min_dim <= 0 or args.min_dim >= 1:
        print(f"ERROR: --min-dim must be between 0 and 1 (exclusive), got {args.min_dim}")
        sys.exit(1)

    # Warn if output directory already exists
    if args.output_dir.is_dir():
        print(f"WARNING: Output directory already exists: {args.output_dir}")
        print("  Existing files may be overwritten.")

    print(f"Filtering tiny labels from YOLO dataset:")
    print(f"  Input directory  : {args.input_dir}")
    print(f"  Output directory : {args.output_dir}")
    print(f"  Min dimension    : {args.min_dim} (approx {args.min_dim * 640:.0f}px at 640 resolution)")

    # Run filtering pipeline
    stats = filter_dataset(args.input_dir, args.output_dir, args.min_dim)

    if stats.images_processed == 0:
        print("\nERROR: No label files were processed.")
        sys.exit(1)

    # Write dataset.yaml — read class names from source dataset.yaml if available
    src_yaml = args.input_dir / "dataset.yaml"
    class_names_from_src = _read_class_names_from_yaml(src_yaml) if src_yaml.exists() else None
    if class_names_from_src is not None:
        print(f"\n  Reading class names from source dataset.yaml: {class_names_from_src}")
    yaml_path = write_dataset_yaml(args.output_dir, class_names=class_names_from_src)
    print(f"\n  Dataset config written to: {yaml_path}")

    # Print report
    report = format_stats_report(stats, args.min_dim, class_names=class_names_from_src)
    print()
    print(report)

    print(f"Filtered dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
