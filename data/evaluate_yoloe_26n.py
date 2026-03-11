"""Evaluate YOLOe-26n-seg (open-vocabulary) on PPE detection val set.

YOLOe is an open-vocabulary model that cannot use model.val() on a custom
dataset. Instead, this script:

  1. Runs YOLOe inference on all val images from the 2-class eval dataset
  2. Matches predictions to YOLO-format ground truth labels at IoU>=0.5
  3. Computes per-class AP50 (area under precision-recall curve)
  4. Computes mAP50 and mAP50-95 (averaging over IoU thresholds 0.50..0.95)
  5. Runs compliance evaluation against the 3-class GT dataset

Usage:
    uv run python data/evaluate_yoloe_26n.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class mapping: YOLOe set_classes(["hard hat", "person"]) -> 0=hard hat, 1=person
# Eval dataset: 0=hardhat, 1=person
# These are already aligned.

EVAL_DATASET = Path(__file__).resolve().parent / "ppe_dataset_exp_a_filtered"
GT_DATASET = Path(__file__).resolve().parent / "ppe_dataset_sam3_v3_filtered"
MODEL_PATH = Path(__file__).resolve().parent.parent / "yoloe-26n-seg.pt"
OUTPUT_PATH = Path(__file__).resolve().parent / "ppe_results" / "yoloe_26n_evaluation.txt"


# ---------------------------------------------------------------------------
# Geometry utilities (reused from evaluate_2class_experiments.py)
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label file into list of (cls_id, cx, cy, w, h)."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
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


# ---------------------------------------------------------------------------
# AP computation (COCO-style, all-point interpolation)
# ---------------------------------------------------------------------------

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using all-point interpolation (COCO style).

    Args:
        recalls: array of recall values (sorted ascending)
        precisions: array of precision values corresponding to each recall

    Returns:
        AP value
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (delta recall) * precision
    ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])
    return float(ap)


def compute_class_ap(
    all_preds: list[dict],
    all_gt: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP for a single class at a given IoU threshold.

    Args:
        all_preds: list of {"bbox": [x1,y1,x2,y2], "conf": float, "img_id": str}
        all_gt: list of {"bbox": [x1,y1,x2,y2], "img_id": str}
        iou_threshold: IoU threshold for TP

    Returns:
        AP value
    """
    if len(all_gt) == 0:
        return 0.0 if len(all_preds) > 0 else 1.0

    # Sort predictions by confidence (descending)
    preds_sorted = sorted(all_preds, key=lambda x: -x["conf"])

    # Build GT lookup: img_id -> list of GT boxes (with matched flag)
    gt_by_img: dict[str, list[dict]] = defaultdict(list)
    for gt in all_gt:
        gt_by_img[gt["img_id"]].append({"bbox": gt["bbox"], "matched": False})

    n_gt = len(all_gt)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        img_gts = gt_by_img.get(pred["img_id"], [])

        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(img_gts):
            ov = iou(pred["bbox"], gt["bbox"])
            if ov > best_iou:
                best_iou = ov
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0 and not img_gts[best_gt_idx]["matched"]:
            tp[i] = 1.0
            img_gts[best_gt_idx]["matched"] = True
        else:
            fp[i] = 1.0

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    return compute_ap(recalls, precisions)


# ---------------------------------------------------------------------------
# GT compliance derivation from 3-class labels
# ---------------------------------------------------------------------------

def derive_gt_compliance(
    gt_labels_3class: list[tuple[int, float, float, float, float]],
    img_w: int, img_h: int,
) -> list[tuple[list[float], bool]]:
    """From 3-class GT labels, derive per-person compliance status."""
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


def evaluate_compliance(
    pred_boxes: list[dict],
    gt_compliance: list[tuple[list[float], bool]],
) -> dict[str, int]:
    """Evaluate predictions (hardhat+person) against GT compliance via spatial overlap."""
    pred_hardhat_boxes = [p["bbox"] for p in pred_boxes if p["cls"] == 0]
    pred_person_boxes = [p["bbox"] for p in pred_boxes if p["cls"] == 1]

    counts = {"gt_compliant": 0, "gt_non_compliant": 0,
              "correct_compliant": 0, "correct_non_compliant": 0,
              "false_alarm": 0, "missed_catch": 0}

    for gt_person_box, gt_compliant in gt_compliance:
        # Find best matching predicted person
        best_iou_val = 0.0
        best_pred = None
        for pred_person in pred_person_boxes:
            ov = iou(gt_person_box, pred_person)
            if ov > best_iou_val:
                best_iou_val = ov
                best_pred = pred_person

        if best_pred is not None and best_iou_val >= 0.3:
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate paths
    eval_val_images_dir = EVAL_DATASET / "images" / "val"
    eval_val_labels_dir = EVAL_DATASET / "labels" / "val"
    gt_val_images_dir = GT_DATASET / "images" / "val"
    gt_val_labels_dir = GT_DATASET / "labels" / "val"

    for d, label in [
        (eval_val_images_dir, "Eval val images"),
        (eval_val_labels_dir, "Eval val labels"),
    ]:
        if not d.is_dir():
            print(f"ERROR: {label} dir not found: {d}")
            sys.exit(1)

    # 3-class GT dataset is optional (used for compliance evaluation only)
    has_gt_dataset = gt_val_images_dir.is_dir() and gt_val_labels_dir.is_dir()
    if not has_gt_dataset:
        print(f"NOTE: 3-class GT dataset not found at {GT_DATASET}")
        print("  Compliance evaluation will be skipped.")

    # Load model (Ultralytics will auto-download if not found locally)
    from ultralytics import YOLO
    from PIL import Image

    model_str = str(MODEL_PATH) if MODEL_PATH.exists() else "yoloe-26n-seg.pt"
    print(f"Loading YOLOe-26n-seg from: {model_str}")
    model = YOLO(model_str)
    model.set_classes(["hard hat", "person"])

    CONF = 0.25
    CLASS_NAMES = {0: "hardhat", 1: "person"}

    # Collect eval val images
    eval_image_files = sorted(
        p for p in eval_val_images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Eval val images: {len(eval_image_files)}")

    # Collect GT val images (optional — only if 3-class GT dataset exists)
    gt_image_files = []
    if has_gt_dataset:
        gt_image_files = sorted(
            p for p in gt_val_images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"GT val images: {len(gt_image_files)}")

    # -----------------------------------------------------------------------
    # Part 1: Run inference on all eval val images & compute mAP
    # -----------------------------------------------------------------------
    print("\n--- Running YOLOe inference on eval val images ---")

    # Collect all predictions and ground truths per class for AP computation
    all_preds_by_class: dict[int, list[dict]] = defaultdict(list)
    all_gt_by_class: dict[int, list[dict]] = defaultdict(list)

    # Also track aggregate TP/FP/FN for simple precision/recall
    total_metrics: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    total_inference_ms = 0.0
    total_detections = 0

    for img_path in eval_image_files:
        stem = img_path.stem
        label_path = eval_val_labels_dir / f"{stem}.txt"
        gt_labels = parse_yolo_label(label_path)

        img = Image.open(img_path)
        img_w, img_h = img.size

        # Run inference
        t0 = time.perf_counter()
        results = model.predict(str(img_path), conf=CONF, verbose=False, device="cpu")
        t1 = time.perf_counter()
        total_inference_ms += (t1 - t0) * 1000

        pred_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                pred_boxes.append({"cls": cls_id, "bbox": xyxy, "conf": conf_val})
                total_detections += 1

        # Collect for AP computation
        for cls_id, cx, cy, w, h in gt_labels:
            gt_box = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            all_gt_by_class[cls_id].append({"bbox": gt_box, "img_id": stem})

        for pred in pred_boxes:
            all_preds_by_class[pred["cls"]].append({
                "bbox": pred["bbox"],
                "conf": pred["conf"],
                "img_id": stem,
            })

        # Simple TP/FP/FN per image (for aggregate precision/recall/F1)
        gt_by_class: dict[int, list[list[float]]] = defaultdict(list)
        pred_by_class: dict[int, list[dict]] = defaultdict(list)

        for cls_id, cx, cy, w, h in gt_labels:
            box = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            gt_by_class[cls_id].append(box)
        for pred in pred_boxes:
            pred_by_class[pred["cls"]].append(pred)

        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
        for cls_id in all_classes:
            gt_boxes_list = gt_by_class.get(cls_id, [])
            preds = sorted(pred_by_class.get(cls_id, []), key=lambda x: -x["conf"])
            matched_gt = set()
            tp = fp = 0
            for pred in preds:
                best_iou_val = 0.0
                best_gt_idx = -1
                for i, gt_box in enumerate(gt_boxes_list):
                    if i in matched_gt:
                        continue
                    ov = iou(pred["bbox"], gt_box)
                    if ov > best_iou_val:
                        best_iou_val = ov
                        best_gt_idx = i
                if best_iou_val >= 0.5 and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn = len(gt_boxes_list) - len(matched_gt)
            total_metrics[cls_id]["tp"] += tp
            total_metrics[cls_id]["fp"] += fp
            total_metrics[cls_id]["fn"] += fn

    # Compute AP50 per class
    ap50_per_class: dict[int, float] = {}
    for cls_id in sorted(set(all_gt_by_class.keys()) | set(all_preds_by_class.keys())):
        ap = compute_class_ap(
            all_preds_by_class.get(cls_id, []),
            all_gt_by_class.get(cls_id, []),
            iou_threshold=0.5,
        )
        ap50_per_class[cls_id] = ap

    map50 = np.mean(list(ap50_per_class.values())) if ap50_per_class else 0.0

    # Compute mAP50-95 (IoU thresholds from 0.50 to 0.95, step 0.05)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    ap_per_class_per_iou: dict[int, list[float]] = defaultdict(list)

    for iou_thresh in iou_thresholds:
        for cls_id in sorted(set(all_gt_by_class.keys()) | set(all_preds_by_class.keys())):
            # Need fresh matched state for each IoU threshold
            ap = compute_class_ap(
                all_preds_by_class.get(cls_id, []),
                all_gt_by_class.get(cls_id, []),
                iou_threshold=float(iou_thresh),
            )
            ap_per_class_per_iou[cls_id].append(ap)

    map50_95_per_class: dict[int, float] = {}
    for cls_id in ap_per_class_per_iou:
        map50_95_per_class[cls_id] = float(np.mean(ap_per_class_per_iou[cls_id]))

    map50_95 = np.mean(list(map50_95_per_class.values())) if map50_95_per_class else 0.0

    avg_inference_ms = total_inference_ms / len(eval_image_files) if eval_image_files else 0.0

    print(f"  mAP50     : {map50:.4f}")
    print(f"  mAP50-95  : {map50_95:.4f}")
    for cls_id in sorted(ap50_per_class.keys()):
        print(f"  AP50 {CLASS_NAMES.get(cls_id, f'cls_{cls_id}'):12s}: {ap50_per_class[cls_id]:.4f}")
    print(f"  Avg inference: {avg_inference_ms:.1f} ms/img (CPU)")

    # -----------------------------------------------------------------------
    # Part 2: Compliance evaluation on GT val images (optional)
    # -----------------------------------------------------------------------
    total_compliance = None
    if has_gt_dataset and gt_image_files:
        print("\n--- Running compliance evaluation on GT val images ---")

        total_compliance = {"gt_compliant": 0, "gt_non_compliant": 0,
                            "correct_compliant": 0, "correct_non_compliant": 0,
                            "false_alarm": 0, "missed_catch": 0}

        for img_path in gt_image_files:
            stem = img_path.stem
            gt_label_path = gt_val_labels_dir / f"{stem}.txt"
            gt_labels_3class = parse_yolo_label(gt_label_path)

            img = Image.open(img_path)
            img_w, img_h = img.size

            # Derive GT compliance from 3-class labels
            gt_compliance = derive_gt_compliance(gt_labels_3class, img_w, img_h)

            # Run inference
            results = model.predict(str(img_path), conf=CONF, verbose=False, device="cpu")

            pred_boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    conf_val = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()
                    pred_boxes.append({"cls": cls_id, "bbox": xyxy, "conf": conf_val})

            # Compliance evaluation
            img_compliance = evaluate_compliance(pred_boxes, gt_compliance)
            for k in total_compliance:
                total_compliance[k] += img_compliance[k]

    # -----------------------------------------------------------------------
    # Format results
    # -----------------------------------------------------------------------
    W = 70
    lines: list[str] = []

    def line(s: str = "") -> None:
        lines.append(s)

    line("=" * W)
    line("YOLOe-26n-seg EVALUATION (Open-Vocabulary, Zero-Shot)")
    line("=" * W)
    line(f"  Model          : {MODEL_PATH}")
    line(f"  Prompts        : ['hard hat', 'person']")
    line(f"  Eval dataset   : {EVAL_DATASET}")
    line(f"  GT dataset     : {GT_DATASET}{'' if has_gt_dataset else ' (not found, compliance skipped)'}")
    line(f"  Conf threshold : {CONF}")
    line(f"  Eval val images: {len(eval_image_files)}")
    if has_gt_dataset:
        line(f"  GT val images  : {len(gt_image_files)}")

    # mAP metrics
    line()
    line("-" * W)
    line("  Detection mAP (manual AP computation, on 2-class eval dataset):")
    line("-" * W)
    line(f"  mAP50       : {map50:.4f}")
    line(f"  mAP50-95    : {map50_95:.4f}")
    for cls_id in sorted(ap50_per_class.keys()):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        line(f"  AP50 {name:12s}: {ap50_per_class[cls_id]:.4f}")
    line()
    line(f"  Avg inference time: {avg_inference_ms:.1f} ms/img (CPU)")
    line(f"  Total detections  : {total_detections} ({total_detections / len(eval_image_files):.1f} per image)")

    # Per-class detection metrics (simple precision/recall/F1)
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
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            line(f"  {name:<15} {tp:>6} {fp:>6} {fn:>6} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    # Compliance metrics (only if GT dataset was available)
    accuracy = 0.0
    if total_compliance is not None:
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
    else:
        line()
        line("-" * W)
        line("  Compliance Assessment: SKIPPED (3-class GT dataset not found)")
        line("-" * W)

    # Comparison summary
    line()
    line("-" * W)
    line("  Comparison with fine-tuned models:")
    line("-" * W)
    compliance_str = f"{accuracy*100:.1f}%" if total_compliance is not None else "N/A"
    line(f"  {'Model':<35} {'mAP50':>8} {'AP50 hat':>10} {'AP50 person':>12} {'Compliance':>12}")
    line(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
    line(f"  {'YOLOe-26n-seg (zero-shot)':<35} {map50:>8.3f} {ap50_per_class.get(0, 0.0):>10.3f} {ap50_per_class.get(1, 0.0):>12.3f} {compliance_str:>12}")
    line(f"  {'YOLO26n@640 (Exp A, fine-tuned)':<35} {'0.689':>8} {'0.522':>10} {'0.855':>12} {'85.4%':>12}")
    line(f"  {'YOLO26n@1280 (Exp C, fine-tuned)':<35} {'0.772':>8} {'0.624':>10} {'0.920':>12} {'85.4%':>12}")

    line()
    line("=" * W)

    # Print and save
    report = "\n".join(lines)
    print()
    print(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report + "\n")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
