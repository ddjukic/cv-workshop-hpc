"""PPE Compliance Post-processor for 2-class YOLO models.

This module converts 2-class (hardhat + person) YOLO detection results into
PPE compliance assessments via spatial overlap analysis.

The 3-class approach (hardhat/no_hardhat/person) suffered from ~29% recall on
the 'no_hardhat' class — an inherently ambiguous detection task.  The 2-class
fallback trains a cleaner model that detects only *hardhat* objects and *person*
objects, then derives compliance here:

    For each detected person:
        1. Extract the head region = top ``head_fraction`` (default 40%) of
           the person bbox.
        2. Compute area-weighted overlap between each detected hardhat and the
           head region.
        3. If any hardhat overlaps the head region with IoU >= ``min_overlap``
           (default 0.1): the person is COMPLIANT.
        4. Otherwise: NON-COMPLIANT.

Importable API:
    from compliance_postprocessor import check_compliance, run_compliance_check

CLI usage:
    uv run python data/compliance_postprocessor.py \\
        --model data/ppe_results/v4_2class/weights/best.pt \\
        --image data/synthetic_ppe/easy/easy_01.jpg \\
        --conf 0.25

    uv run python data/compliance_postprocessor.py \\
        --model data/ppe_results/v4_2class/weights/best.pt \\
        --source-dir data/synthetic_ppe/easy \\
        --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

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


def _head_region(
    person_bbox: list[float],
    head_fraction: float = 0.4,
) -> list[float]:
    """Return the head sub-region (top fraction) of a person bounding box.

    Args:
        person_bbox: [x1, y1, x2, y2] absolute pixel coordinates.
        head_fraction: Fraction of person height to treat as head (default 0.4).

    Returns:
        [x1, y1, x2, y2] of the head sub-region.
    """
    x1, y1, x2, y2 = person_bbox
    head_y2 = y1 + head_fraction * (y2 - y1)
    return [x1, y1, x2, head_y2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_compliance(
    person_bbox: list[float],
    hardhat_bboxes: list[list[float]],
    head_fraction: float = 0.4,
    min_overlap: float = 0.1,
) -> bool:
    """Check if a person has a hardhat overlapping their head region.

    Args:
        person_bbox: [x1, y1, x2, y2] absolute pixel coordinates of person.
        hardhat_bboxes: List of [x1, y1, x2, y2] absolute pixel coordinates
            of detected hardhat objects.
        head_fraction: Top fraction of person bbox to treat as head region.
            Default 0.4 (top 40%).
        min_overlap: Minimum IoU between head region and hardhat to count
            as compliant.  Default 0.1 (10% overlap).

    Returns:
        True if the person is COMPLIANT (wearing a hardhat), False otherwise.
    """
    if not hardhat_bboxes:
        return False

    head = _head_region(person_bbox, head_fraction)
    for hh_bbox in hardhat_bboxes:
        if _iou(head, hh_bbox) >= min_overlap:
            return True
    return False


class PersonResult(NamedTuple):
    """Result for a single detected person."""
    person_bbox: list[float]    # [x1, y1, x2, y2] absolute pixels
    person_conf: float          # Detection confidence
    compliant: bool             # True = wearing hardhat
    best_hardhat_iou: float     # Highest IoU with any hardhat (0 if none)


class ComplianceReport(NamedTuple):
    """Aggregate compliance report for one image."""
    image_path: str
    total_persons: int
    compliant_count: int
    non_compliant_count: int
    compliance_rate: float      # compliant / total_persons (NaN if 0 persons)
    persons: list[PersonResult]
    total_hardhats: int


def run_compliance_check(
    person_bboxes: list[list[float]],
    person_confs: list[float],
    hardhat_bboxes: list[list[float]],
    head_fraction: float = 0.4,
    min_overlap: float = 0.1,
    image_path: str = "",
) -> ComplianceReport:
    """Run compliance check on all detected persons in an image.

    Args:
        person_bboxes: List of [x1, y1, x2, y2] person detections.
        person_confs: Confidence score for each person detection.
        hardhat_bboxes: List of [x1, y1, x2, y2] hardhat detections.
        head_fraction: Top fraction of person bbox considered as head.
        min_overlap: Minimum IoU for hardhat-head overlap to count as compliant.
        image_path: Optional image path for reporting.

    Returns:
        ComplianceReport with per-person results and aggregate statistics.
    """
    persons: list[PersonResult] = []

    for pb, pc in zip(person_bboxes, person_confs):
        head = _head_region(pb, head_fraction)
        best_iou = 0.0
        for hh in hardhat_bboxes:
            ov = _iou(head, hh)
            if ov > best_iou:
                best_iou = ov
        compliant = best_iou >= min_overlap
        persons.append(PersonResult(
            person_bbox=pb,
            person_conf=pc,
            compliant=compliant,
            best_hardhat_iou=best_iou,
        ))

    total = len(persons)
    compliant_count = sum(1 for p in persons if p.compliant)
    non_compliant_count = total - compliant_count
    compliance_rate = compliant_count / total if total > 0 else float("nan")

    return ComplianceReport(
        image_path=image_path,
        total_persons=total,
        compliant_count=compliant_count,
        non_compliant_count=non_compliant_count,
        compliance_rate=compliance_rate,
        persons=persons,
        total_hardhats=len(hardhat_bboxes),
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _load_model_and_predict(model_path: Path, image_path: Path, conf: float):
    """Run YOLO inference and return (person_bboxes, person_confs, hardhat_bboxes).

    Returns:
        tuple: (
            person_bboxes: list of [x1, y1, x2, y2],
            person_confs: list of float,
            hardhat_bboxes: list of [x1, y1, x2, y2],
        )
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: uv pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf, verbose=False)

    person_bboxes: list[list[float]] = []
    person_confs: list[float] = []
    hardhat_bboxes: list[list[float]] = []

    if results:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # Class mapping: 0=hardhat, 1=person
                if cls_id == 0:
                    hardhat_bboxes.append(xyxy)
                elif cls_id == 1:
                    person_bboxes.append(xyxy)
                    person_confs.append(conf_val)

    return person_bboxes, person_confs if person_confs else [1.0] * len(person_bboxes), hardhat_bboxes


def _format_report(report: ComplianceReport) -> str:
    """Format a compliance report for human-readable terminal output."""
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"COMPLIANCE REPORT: {Path(report.image_path).name}")
    lines.append(sep)
    lines.append(f"  Total persons   : {report.total_persons}")
    lines.append(f"  Total hardhats  : {report.total_hardhats}")
    lines.append(f"  Compliant       : {report.compliant_count}")
    lines.append(f"  Non-compliant   : {report.non_compliant_count}")

    if report.total_persons > 0:
        rate_pct = report.compliance_rate * 100
        lines.append(f"  Compliance rate : {rate_pct:.1f}%")
    else:
        lines.append("  Compliance rate : N/A (no persons detected)")

    if report.persons:
        lines.append("")
        lines.append("  Per-person detail:")
        lines.append(f"  {'#':>3}  {'Status':<14} {'Conf':>6}  {'Best HH IoU':>11}  Person bbox")
        lines.append(f"  {'-'*3}  {'-'*14} {'-'*6}  {'-'*11}  {'-'*30}")
        for i, p in enumerate(report.persons, 1):
            status = "COMPLIANT    " if p.compliant else "NON-COMPLIANT"
            bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*p.person_bbox)
            lines.append(
                f"  {i:>3}  {status}  {p.person_conf:>6.3f}  {p.best_hardhat_iou:>11.3f}  {bbox_str}"
            )

    lines.append(sep)
    return "\n".join(lines)


def _collect_images(source_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in extensions and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPE compliance post-processor for 2-class YOLO models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained 2-class YOLO model weights (.pt).",
    )

    # Input: either a single image or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Single image to run compliance check on.",
    )
    input_group.add_argument(
        "--source-dir",
        type=Path,
        help="Directory of images — runs on all images and reports aggregate stats.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--head-fraction",
        type=float,
        default=0.4,
        help="Top fraction of person bbox to treat as head region.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.1,
        help="Minimum IoU between head region and hardhat bbox to classify as compliant.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-person detail for every image (directory mode only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"ERROR: Model weights not found: {args.model}")
        sys.exit(1)

    # ---- Single image mode ----
    if args.image is not None:
        if not args.image.exists():
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)

        print(f"Running inference: {args.image}")
        person_bboxes, person_confs, hardhat_bboxes = _load_model_and_predict(
            args.model, args.image, args.conf
        )

        report = run_compliance_check(
            person_bboxes=person_bboxes,
            person_confs=person_confs,
            hardhat_bboxes=hardhat_bboxes,
            head_fraction=args.head_fraction,
            min_overlap=args.min_overlap,
            image_path=str(args.image),
        )
        print(_format_report(report))
        return

    # ---- Directory mode ----
    images = _collect_images(args.source_dir)
    if not images:
        print(f"ERROR: No images found in {args.source_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} images from {args.source_dir}")
    print(f"Model: {args.model}")
    print(f"Settings: conf={args.conf}, head_fraction={args.head_fraction}, min_overlap={args.min_overlap}")
    print()

    all_reports: list[ComplianceReport] = []

    for img_path in images:
        person_bboxes, person_confs, hardhat_bboxes = _load_model_and_predict(
            args.model, img_path, args.conf
        )
        report = run_compliance_check(
            person_bboxes=person_bboxes,
            person_confs=person_confs,
            hardhat_bboxes=hardhat_bboxes,
            head_fraction=args.head_fraction,
            min_overlap=args.min_overlap,
            image_path=str(img_path),
        )
        all_reports.append(report)

        if args.verbose:
            print(_format_report(report))
            print()
        else:
            rate_str = f"{report.compliance_rate * 100:.1f}%" if report.total_persons > 0 else "N/A"
            print(
                f"  {img_path.name:<40} persons={report.total_persons:>3}  "
                f"compliant={report.compliant_count:>3}  rate={rate_str}"
            )

    # ---- Aggregate stats ----
    images_with_persons = [r for r in all_reports if r.total_persons > 0]
    total_persons = sum(r.total_persons for r in all_reports)
    total_compliant = sum(r.compliant_count for r in all_reports)
    total_non_compliant = sum(r.non_compliant_count for r in all_reports)
    total_hardhats = sum(r.total_hardhats for r in all_reports)
    overall_rate = total_compliant / total_persons if total_persons > 0 else float("nan")

    print()
    print("=" * 60)
    print("AGGREGATE COMPLIANCE SUMMARY")
    print("=" * 60)
    print(f"  Images processed     : {len(all_reports)}")
    print(f"  Images with persons  : {len(images_with_persons)}")
    print(f"  Total persons        : {total_persons}")
    print(f"  Total hardhats       : {total_hardhats}")
    print(f"  Compliant persons    : {total_compliant}")
    print(f"  Non-compliant persons: {total_non_compliant}")
    if total_persons > 0:
        print(f"  Overall compliance   : {overall_rate * 100:.1f}%")
    else:
        print("  Overall compliance   : N/A (no persons detected)")
    print("=" * 60)


if __name__ == "__main__":
    main()
