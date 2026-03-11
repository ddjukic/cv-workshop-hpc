"""Evaluate 2-class PPE detection experiments against 3-class ground truth.

Supports two evaluation modes:

  * **exp_a** — Model detects hardhat (cls 0) + person (cls 1). Compliance
    derived via spatial post-processing: person is compliant if a hardhat
    overlaps their head region.

  * **exp_b** — Model detects safe (cls 0) + unsafe (cls 1). Compliance
    read directly from class label.

Both modes compare against the same 3-class ground truth (v3_filtered),
computing: mAP50, per-class AP50, compliance accuracy, catch rate, false
alarm rate.

Usage:
    uv run python data/evaluate_2class_experiments.py \
        --model data/ppe_results/exp_a_2class_detect/weights/best.pt \
        --eval-dataset data/ppe_dataset_exp_a_filtered \
        --gt-dataset data/ppe_dataset_sam3_v3_filtered \
        --mode exp_a --conf 0.25

    uv run python data/evaluate_2class_experiments.py \
        --model data/ppe_results/exp_b_2class_compliance/weights/best.pt \
        --eval-dataset data/ppe_dataset_exp_b_filtered \
        --gt-dataset data/ppe_dataset_sam3_v3_filtered \
        --mode exp_b --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label file into list of (cls_id, cx, cy, w, h)."""
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


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> list[float]:
    """Convert YOLO normalised (cx, cy, w, h) to pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def head_region(person_bbox: list[float], head_fraction: float = 0.4) -> list[float]:
    """Return top fraction of person bbox as head region."""
    x1, y1, x2, y2 = person_bbox
    head_y2 = y1 + head_fraction * (y2 - y1)
    return [x1, y1, x2, head_y2]


def check_compliance(person_bbox: list[float],
                      hardhat_bboxes: list[list[float]],
                      min_overlap: float = 0.1) -> bool:
    """Check if person has a hardhat overlapping their head region."""
    head = head_region(person_bbox)
    for hh in hardhat_bboxes:
        if iou(head, hh) >= min_overlap:
            return True
    return False


def compute_per_class_metrics(
    gt_labels: list[tuple[int, float, float, float, float]],
    pred_boxes: list[dict],
    img_w: int, img_h: int,
    iou_threshold: float = 0.5,
) -> dict[int, dict[str, int]]:
    """Compute TP/FP/FN per class using IoU matching."""
    metrics: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    gt_by_class: dict[int, list[list[float]]] = defaultdict(list)
    pred_by_class: dict[int, list[dict]] = defaultdict(list)

    for cls_id, cx, cy, w, h in gt_labels:
        box = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        gt_by_class[cls_id].append(box)

    for pred in pred_boxes:
        pred_by_class[pred["cls"]].append(pred)

    all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

    for cls_id in all_classes:
        gt_boxes = gt_by_class.get(cls_id, [])
        preds = sorted(pred_by_class.get(cls_id, []), key=lambda x: -x["conf"])

        matched_gt = set()
        tp = fp = 0

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                ov = iou(pred["bbox"], gt_box)
                if ov > best_iou:
                    best_iou = ov
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        metrics[cls_id]["tp"] += tp
        metrics[cls_id]["fp"] += fp
        metrics[cls_id]["fn"] += fn

    return dict(metrics)


# ---------------------------------------------------------------------------
# GT compliance derivation from 3-class labels
# ---------------------------------------------------------------------------

def derive_gt_compliance(
    gt_labels_3class: list[tuple[int, float, float, float, float]],
    img_w: int, img_h: int,
) -> list[tuple[list[float], bool]]:
    """From 3-class GT labels, derive per-person compliance status.

    Returns list of (person_bbox_xyxy, is_compliant).
    """
    gt_hardhat_boxes = [
        yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        for cls_id, cx, cy, w, h in gt_labels_3class if cls_id == 0
    ]
    gt_person_boxes = [
        yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        for cls_id, cx, cy, w, h in gt_labels_3class if cls_id == 2
    ]

    result = []
    for person_box in gt_person_boxes:
        compliant = check_compliance(person_box, gt_hardhat_boxes)
        result.append((person_box, compliant))
    return result


# ---------------------------------------------------------------------------
# Exp A compliance: hardhat(0) + person(1) → spatial overlap
# ---------------------------------------------------------------------------

def evaluate_compliance_exp_a(
    pred_boxes: list[dict],
    gt_compliance: list[tuple[list[float], bool]],
) -> dict[str, int]:
    """Evaluate Exp A predictions against GT compliance.

    pred_boxes have cls 0 (hardhat) and cls 1 (person).
    """
    pred_hardhat_boxes = [p["bbox"] for p in pred_boxes if p["cls"] == 0]
    pred_person_boxes = [p["bbox"] for p in pred_boxes if p["cls"] == 1]

    counts = {"gt_compliant": 0, "gt_non_compliant": 0,
              "correct_compliant": 0, "correct_non_compliant": 0,
              "false_alarm": 0, "missed_catch": 0}

    for gt_person_box, gt_compliant in gt_compliance:
        # Find best matching predicted person
        best_iou = 0.0
        best_pred = None
        for pred_person in pred_person_boxes:
            ov = iou(gt_person_box, pred_person)
            if ov > best_iou:
                best_iou = ov
                best_pred = pred_person

        if best_pred is not None and best_iou >= 0.3:
            pred_compliant = check_compliance(best_pred, pred_hardhat_boxes)
        else:
            pred_compliant = False

        if gt_compliant:
            counts["gt_compliant"] += 1
            if pred_compliant:
                counts["correct_compliant"] += 1
            else:
                counts["missed_catch"] += 1
        else:
            counts["gt_non_compliant"] += 1
            if not pred_compliant:
                counts["correct_non_compliant"] += 1
            else:
                counts["false_alarm"] += 1

    return counts


# ---------------------------------------------------------------------------
# Exp B compliance: safe(0) + unsafe(1) → class label is compliance
# ---------------------------------------------------------------------------

def evaluate_compliance_exp_b(
    pred_boxes: list[dict],
    gt_compliance: list[tuple[list[float], bool]],
) -> dict[str, int]:
    """Evaluate Exp B predictions against GT compliance.

    pred_boxes have cls 0 (safe) and cls 1 (unsafe).
    Match predicted persons to GT persons by IoU, read compliance from class.
    """
    # All predictions are person bboxes with compliance encoded in class
    all_preds = [(p["bbox"], p["cls"] == 0, p["conf"]) for p in pred_boxes]

    counts = {"gt_compliant": 0, "gt_non_compliant": 0,
              "correct_compliant": 0, "correct_non_compliant": 0,
              "false_alarm": 0, "missed_catch": 0}

    for gt_person_box, gt_compliant in gt_compliance:
        best_iou = 0.0
        best_pred_compliant = None
        for pred_box, pred_is_safe, _ in all_preds:
            ov = iou(gt_person_box, pred_box)
            if ov > best_iou:
                best_iou = ov
                best_pred_compliant = pred_is_safe

        if best_iou >= 0.3 and best_pred_compliant is not None:
            pred_compliant = best_pred_compliant
        else:
            pred_compliant = False

        if gt_compliant:
            counts["gt_compliant"] += 1
            if pred_compliant:
                counts["correct_compliant"] += 1
            else:
                counts["missed_catch"] += 1
        else:
            counts["gt_non_compliant"] += 1
            if not pred_compliant:
                counts["correct_non_compliant"] += 1
            else:
                counts["false_alarm"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 2-class PPE experiments against 3-class GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to trained 2-class model weights (.pt).")
    parser.add_argument("--eval-dataset", type=Path, required=True,
                        help="2-class YOLO dataset dir (for model.val() mAP).")
    parser.add_argument("--gt-dataset", type=Path, required=True,
                        help="3-class GT dataset dir (for compliance derivation).")
    parser.add_argument("--mode", type=str, required=True, choices=["exp_a", "exp_b"],
                        help="Experiment mode.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for inference.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device for model inference.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for results.")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    gt_val_images_dir = args.gt_dataset / "images" / "val"
    gt_val_labels_dir = args.gt_dataset / "labels" / "val"

    if not gt_val_images_dir.is_dir():
        print(f"ERROR: GT val images dir not found: {gt_val_images_dir}")
        sys.exit(1)

    # --- Load model ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        sys.exit(1)

    from PIL import Image

    model = YOLO(str(args.model))
    mode_label = "Exp A (hardhat+person → postprocess)" if args.mode == "exp_a" \
        else "Exp B (safe+unsafe → direct)"

    print(f"Model loaded: {args.model}")
    print(f"Mode: {mode_label}")
    print(f"Eval dataset (mAP): {args.eval_dataset}")
    print(f"GT dataset (compliance): {args.gt_dataset}")

    # --- model.val() for official mAP on the 2-class eval dataset ---
    print("\n--- Running model.val() for official mAP metrics ---")
    official_map50 = None
    official_map50_95 = None
    per_class_ap50: dict[str, float] = {}

    try:
        val_results = model.val(
            data=str(args.eval_dataset / "dataset.yaml"),
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        official_map50 = float(val_results.box.map50)
        official_map50_95 = float(val_results.box.map)

        if hasattr(val_results.box, 'ap50') and val_results.box.ap50 is not None:
            ap50_array = val_results.box.ap50
            class_names = val_results.names if hasattr(val_results, 'names') else {}
            for i, ap in enumerate(ap50_array):
                cls_name = class_names.get(i, f"class_{i}") if isinstance(class_names, dict) else str(i)
                per_class_ap50[cls_name] = float(ap)

        print(f"  Official mAP50   : {official_map50:.4f}")
        print(f"  Official mAP50-95: {official_map50_95:.4f}")
        for name, ap in per_class_ap50.items():
            print(f"  AP50 {name:12s}: {ap:.4f}")
    except Exception as exc:
        print(f"  WARNING: model.val() failed: {exc}")
        print("  Falling back to manual evaluation only.")

    # --- Per-image compliance evaluation on GT images ---
    print("\n--- Running compliance evaluation on GT val images ---")

    gt_image_files = sorted(
        p for p in gt_val_images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not gt_image_files:
        print(f"No images found in {gt_val_images_dir}")
        sys.exit(1)

    print(f"Evaluating compliance on {len(gt_image_files)} GT val images...")

    # Aggregate detection metrics (on 2-class eval dataset labels)
    total_metrics: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Compliance aggregation
    total_compliance = {"gt_compliant": 0, "gt_non_compliant": 0,
                        "correct_compliant": 0, "correct_non_compliant": 0,
                        "false_alarm": 0, "missed_catch": 0}

    eval_fn = evaluate_compliance_exp_a if args.mode == "exp_a" else evaluate_compliance_exp_b

    for img_path in gt_image_files:
        stem = img_path.stem
        gt_label_path = gt_val_labels_dir / f"{stem}.txt"
        gt_labels_3class = parse_yolo_label(gt_label_path)

        img = Image.open(img_path)
        img_w, img_h = img.size

        # Derive GT compliance from 3-class labels
        gt_compliance = derive_gt_compliance(gt_labels_3class, img_w, img_h)

        # Run 2-class model inference on GT image
        results = model.predict(str(img_path), conf=args.conf, verbose=False, device=args.device)

        pred_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append({"cls": cls_id, "bbox": xyxy, "conf": conf_val})

        # Compliance evaluation
        img_compliance = eval_fn(pred_boxes, gt_compliance)
        for k in total_compliance:
            total_compliance[k] += img_compliance[k]

        # Also compute detection metrics against 2-class eval labels (if they exist)
        eval_label_path = args.eval_dataset / "labels" / "val" / f"{stem}.txt"
        if eval_label_path.exists():
            eval_labels = parse_yolo_label(eval_label_path)
            img_metrics = compute_per_class_metrics(eval_labels, pred_boxes, img_w, img_h)
            for cls_id, counts in img_metrics.items():
                total_metrics[cls_id]["tp"] += counts["tp"]
                total_metrics[cls_id]["fp"] += counts["fp"]
                total_metrics[cls_id]["fn"] += counts["fn"]

    # --- Format results ---
    W = 70
    lines: list[str] = []

    def line(s: str = "") -> None:
        lines.append(s)

    line("=" * W)
    line(f"2-CLASS EXPERIMENT EVALUATION: {mode_label}")
    line("=" * W)
    line(f"  Model          : {args.model}")
    line(f"  Eval dataset   : {args.eval_dataset}")
    line(f"  GT dataset     : {args.gt_dataset}")
    line(f"  Conf threshold : {args.conf}")
    line(f"  GT val images  : {len(gt_image_files)}")

    # Official mAP
    if official_map50 is not None:
        line()
        line("-" * W)
        line("  Official mAP (model.val() on 2-class eval dataset):")
        line("-" * W)
        line(f"  mAP50       : {official_map50:.4f}")
        line(f"  mAP50-95    : {official_map50_95:.4f}")
        for name, ap in per_class_ap50.items():
            line(f"  AP50 {name:12s}: {ap:.4f}")

    # Per-class detection metrics
    if total_metrics:
        line()
        line("-" * W)
        line("  Per-class detection metrics (IoU>=0.5, on eval dataset labels):")
        line("-" * W)
        line(f"  {'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        line(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

        for cls_id in sorted(total_metrics.keys()):
            tp = total_metrics[cls_id]["tp"]
            fp = total_metrics[cls_id]["fp"]
            fn = total_metrics[cls_id]["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            line(f"  class_{cls_id:<9} {tp:>6} {fp:>6} {fn:>6} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    # Compliance metrics
    line()
    line("-" * W)
    line("  Compliance Assessment (vs 3-class GT):")
    line("-" * W)
    total_persons = total_compliance["gt_compliant"] + total_compliance["gt_non_compliant"]
    total_correct = total_compliance["correct_compliant"] + total_compliance["correct_non_compliant"]
    accuracy = total_correct / total_persons if total_persons > 0 else 0.0
    catch_rate = (total_compliance["correct_non_compliant"] / total_compliance["gt_non_compliant"]
                  if total_compliance["gt_non_compliant"] > 0 else 0.0)
    false_alarm_rate = (total_compliance["false_alarm"] / total_compliance["gt_non_compliant"]
                        if total_compliance["gt_non_compliant"] > 0 else 0.0)
    miss_rate = (total_compliance["missed_catch"] / total_compliance["gt_compliant"]
                 if total_compliance["gt_compliant"] > 0 else 0.0)

    line(f"  GT compliant persons     : {total_compliance['gt_compliant']}")
    line(f"  GT non-compliant persons : {total_compliance['gt_non_compliant']}")
    line(f"  Total persons            : {total_persons}")
    line()
    line(f"  Correctly predicted compliant     : {total_compliance['correct_compliant']}")
    line(f"  Correctly predicted non-compliant : {total_compliance['correct_non_compliant']}")
    line(f"  False alarms (predicted safe, actually unsafe): {total_compliance['false_alarm']}")
    line(f"  Missed catches (predicted unsafe, actually safe): {total_compliance['missed_catch']}")
    line()
    line(f"  Overall accuracy  : {accuracy:.4f} ({accuracy*100:.1f}%)")
    line(f"  Catch rate        : {catch_rate:.4f} ({catch_rate*100:.1f}%)")
    line(f"  False alarm rate  : {false_alarm_rate:.4f} ({false_alarm_rate*100:.1f}%)")
    line(f"  Miss rate         : {miss_rate:.4f} ({miss_rate*100:.1f}%)")

    line()
    line("=" * W)

    # Print and save
    report = "\n".join(lines)
    print()
    print(report)

    output_path = args.output or Path(f"data/ppe_results/{args.mode}_evaluation.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
