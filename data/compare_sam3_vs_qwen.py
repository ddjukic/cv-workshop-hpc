#!/usr/bin/env python3
"""Compare SAM3 vs Qwen 3.5 VL teacher models side-by-side.

Runs both models on the same images and produces side-by-side annotated
outputs showing bounding boxes, labels, detection counts, and timing.
Designed for the CV workshop to visually demonstrate teacher model options.

Usage:
    # Bottles on MPS (Apple Silicon):
    uv run python compare_sam3_vs_qwen.py \
        --images data/milk_images/ \
        --targets "bottle" \
        --qwen-model Qwen/Qwen3.5-9B \
        --device mps \
        --output data/comparison_outputs/

    # PPE detection:
    uv run python compare_sam3_vs_qwen.py \
        --images data/scaling_experiment/synthetic_ppe/easy/ \
        --targets "person,hard hat" \
        --qwen-model Qwen/Qwen3.5-4B \
        --device cuda

    # Single image:
    uv run python compare_sam3_vs_qwen.py \
        --images data/demo_images/traffic_jam.jpg \
        --targets "car,person" \
        --device mps

    # Multiple target sets:
    uv run python compare_sam3_vs_qwen.py \
        --images data/demo_images/ \
        --targets "person" \
        --max-images 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_IMAGE_DIM = 1280  # Resize large images to prevent OOM
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    """Synchronize GPU for accurate timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def collect_images(path: Path, max_images: int | None = None) -> list[Path]:
    """Collect image files from a path (file or directory)."""
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]

    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )
        if max_images:
            images = images[:max_images]
        return images

    print(f"{RED}ERROR: {path} is not a valid file or directory{NC}")
    return []


def resize_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Resize image if longest side exceeds max_dim."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# SAM3 detection
# ---------------------------------------------------------------------------

def load_sam3(device: str):
    """Load SAM3 model and processor. Returns (model, processor, load_time_s)."""
    from transformers import Sam3Model, Sam3Processor

    model_id = "facebook/sam3"
    print(f"  Loading SAM3 from {model_id}...")
    t0 = time.time()
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()
    load_time = time.time() - t0
    print(f"  {GREEN}SAM3 loaded in {load_time:.1f}s{NC}")
    return model, processor, load_time


def detect_sam3(
    image: Image.Image,
    targets: list[str],
    model,
    processor,
    device: str,
    threshold: float = 0.3,
) -> dict:
    """Run SAM3 text-prompted detection.

    Returns dict with keys: boxes (list of [x1,y1,x2,y2]), scores, labels, elapsed_s.
    """
    all_boxes = []
    all_scores = []
    all_labels = []
    total_elapsed = 0.0

    for target in targets:
        inputs = processor(images=image, text=target, return_tensors="pt").to(device)

        sync_device(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        sync_device(device)
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        # Post-process to get bounding boxes
        h, w = image.size[1], image.size[0]
        results = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(h, w)],
        )

        if results and len(results) > 0:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()

            for i in range(len(boxes)):
                all_boxes.append(boxes[i].tolist())
                all_scores.append(float(scores[i]))
                all_labels.append(target)

    return {
        "boxes": all_boxes,
        "scores": all_scores,
        "labels": all_labels,
        "elapsed_s": total_elapsed,
    }


# ---------------------------------------------------------------------------
# Qwen 3.5 VL detection
# ---------------------------------------------------------------------------

def load_qwen(model_id: str, device: str):
    """Load Qwen 3.5 VL model and processor. Returns (model, processor, load_time_s)."""
    import warnings

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"  Loading Qwen from {model_id}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_id)

    # Choose dtype based on device
    if device == "cuda":
        model_dtype = torch.bfloat16
    elif device == "mps":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*dtype.*")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)

    load_time = time.time() - t0
    print(f"  {GREEN}Qwen loaded in {load_time:.1f}s (dtype={model_dtype}){NC}")
    return model, processor, load_time


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> chain-of-thought prefix from model output.

    Qwen 3.5 models with enable_thinking=True emit a reasoning trace before the
    actual answer. The supervision parser cannot handle this prefix, so we strip
    everything up to and including the closing </think> tag.
    """
    import re

    return re.sub(r".*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def detect_qwen(
    image: Image.Image,
    targets: list[str],
    model,
    processor,
    device: str,
) -> dict:
    """Run Qwen 3.5 VL grounding detection.

    Returns dict with keys: boxes (list of [x1,y1,x2,y2]), scores, labels, elapsed_s.
    """
    import warnings

    import supervision as sv

    target_str = ", ".join(targets)
    prompt = f"Outline the position of {target_str} and output all the coordinates in JSON format."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*enable_thinking.*")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

    sync_device(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=1024)
    sync_device(device)
    elapsed = time.perf_counter() - t0

    # Trim input tokens
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # Strip any residual thinking prefix (safety net)
    text = _strip_thinking(text)

    # Parse with supervision
    w, h = image.size
    try:
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.QWEN_3_VL,
            result=text,
            resolution_wh=(w, h),
            classes=targets,
        )
    except Exception:
        # Retry without classes if class-based parsing fails
        try:
            detections = sv.Detections.from_vlm(
                vlm=sv.VLM.QWEN_3_VL,
                result=text,
                resolution_wh=(w, h),
            )
        except Exception:
            return {
                "boxes": [],
                "scores": [],
                "labels": [],
                "elapsed_s": elapsed,
                "raw_text": text,
            }

    boxes = detections.xyxy.tolist() if len(detections) > 0 else []
    scores = detections.confidence.tolist() if detections.confidence is not None and len(detections) > 0 else [1.0] * len(boxes)
    if detections.data and "class_name" in detections.data and len(detections) > 0:
        labels = list(detections.data["class_name"])
    elif detections.class_id is not None and len(detections) > 0:
        labels = [targets[cid] if cid < len(targets) else f"class_{cid}" for cid in detections.class_id]
    else:
        labels = [target_str] * len(boxes)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "elapsed_s": elapsed,
        "raw_text": text,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def annotate_image(
    image: Image.Image,
    result: dict,
    model_name: str,
) -> np.ndarray:
    """Annotate image with bounding boxes using supervision.

    Returns numpy array (H, W, 3).
    """
    import supervision as sv

    img_np = np.array(image)
    boxes = result["boxes"]
    labels = result["labels"]
    scores = result["scores"]

    if not boxes:
        return img_np

    xyxy = np.array(boxes, dtype=np.float32)
    confidence = np.array(scores, dtype=np.float32)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
    )

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.INDEX,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_padding=4,
        color_lookup=sv.ColorLookup.INDEX,
    )

    # Build label strings with confidence
    label_texts = []
    for i in range(len(detections)):
        lbl = labels[i] if i < len(labels) else "object"
        conf = scores[i] if i < len(scores) else 0.0
        label_texts.append(f"{lbl} ({conf:.2f})")

    annotated = box_annotator.annotate(img_np.copy(), detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=label_texts)

    return annotated


def create_side_by_side(
    image: Image.Image,
    sam3_result: dict,
    qwen_result: dict,
    qwen_model_name: str,
    image_name: str,
    targets: list[str],
) -> Image.Image:
    """Create side-by-side comparison image with title bars."""
    # Annotate both
    sam3_annotated = annotate_image(image, sam3_result, "SAM3")
    qwen_annotated = annotate_image(image, qwen_result, qwen_model_name)

    h, w = sam3_annotated.shape[:2]

    # Title bar height
    title_h = 50
    gap = 10  # gap between images

    # Create canvas
    canvas_w = w * 2 + gap
    canvas_h = h + title_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # dark gray background

    # Place annotated images below title bar
    canvas[title_h:title_h + h, :w] = sam3_annotated
    canvas[title_h:title_h + h, w + gap:] = qwen_annotated

    # Convert to PIL for text rendering
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

    targets_str = ", ".join(targets)

    # SAM3 title (left half)
    sam3_n = len(sam3_result["boxes"])
    sam3_time = sam3_result["elapsed_s"]
    sam3_avg_conf = np.mean(sam3_result["scores"]) if sam3_result["scores"] else 0.0
    sam3_title = f"SAM3  |  {sam3_n} detections  |  {sam3_time:.2f}s"
    draw.text((10, 8), sam3_title, fill=(100, 255, 100), font=font)
    draw.text((10, 30), f"targets: {targets_str}  |  avg conf: {sam3_avg_conf:.2f}", fill=(180, 180, 180), font=font_small)

    # Qwen title (right half)
    qwen_n = len(qwen_result["boxes"])
    qwen_time = qwen_result["elapsed_s"]
    qwen_avg_conf = np.mean(qwen_result["scores"]) if qwen_result["scores"] else 0.0
    qwen_short = qwen_model_name.split("/")[-1]
    qwen_title = f"{qwen_short}  |  {qwen_n} detections  |  {qwen_time:.2f}s"
    draw.text((w + gap + 10, 8), qwen_title, fill=(100, 200, 255), font=font)
    draw.text((w + gap + 10, 30), f"targets: {targets_str}  |  avg conf: {qwen_avg_conf:.2f}", fill=(180, 180, 180), font=font_small)

    return canvas_pil


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(
    all_results: list[dict],
    qwen_model_name: str,
) -> None:
    """Print summary comparison table."""
    sam3_results = [r for r in all_results if r["model"] == "SAM3"]
    qwen_results = [r for r in all_results if r["model"] == "Qwen"]

    print(f"\n{'=' * 80}")
    print(f"{BOLD}  COMPARISON SUMMARY{NC}")
    print(f"{'=' * 80}")

    # Per-image details
    header = f"  {'Image':<30} {'Model':<20} {'Detections':>10} {'Avg Conf':>10} {'Time (s)':>10}"
    print(header)
    print(f"  {'-' * 78}")

    for r in all_results:
        avg_conf = np.mean(r["scores"]) if r["scores"] else 0.0
        model_label = "SAM3" if r["model"] == "SAM3" else qwen_model_name.split("/")[-1]
        print(
            f"  {r['image']:<30} {model_label:<20} "
            f"{len(r['boxes']):>10} {avg_conf:>10.3f} {r['elapsed_s']:>10.2f}"
        )

    print(f"  {'-' * 78}")

    # Aggregated stats
    for model_label, results in [("SAM3", sam3_results), (qwen_model_name.split("/")[-1], qwen_results)]:
        if not results:
            continue
        total_dets = sum(len(r["boxes"]) for r in results)
        all_scores = [s for r in results for s in r["scores"]]
        avg_conf = np.mean(all_scores) if all_scores else 0.0
        avg_latency = np.mean([r["elapsed_s"] for r in results])
        total_time = sum(r["elapsed_s"] for r in results)
        n_images = len(results)

        print(f"\n  {BOLD}{model_label}{NC}")
        print(f"    Images processed:    {n_images}")
        print(f"    Total detections:    {total_dets}")
        print(f"    Avg detections/img:  {total_dets / n_images:.1f}")
        print(f"    Avg confidence:      {avg_conf:.3f}")
        print(f"    Avg latency:         {avg_latency:.2f}s")
        print(f"    Total time:          {total_time:.2f}s")

    print(f"\n{'=' * 80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SAM3 vs Qwen 3.5 VL teacher models side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to image file or directory of images.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Comma-separated detection targets (e.g., 'person,hard hat').",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Qwen model ID. Supports 4B, 9B, 27B variants.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for comparison images. Defaults to data/comparison_outputs/.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--sam3-threshold",
        type=float,
        default=0.3,
        help="SAM3 confidence threshold for detections.",
    )
    args = parser.parse_args()

    # ---- Setup ----
    device = args.device or detect_device()
    targets = [t.strip() for t in args.targets.split(",")]
    images_path = Path(args.images)
    output_dir = Path(args.output) if args.output else Path(__file__).resolve().parent / "comparison_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  SAM3 vs Qwen 3.5 VL Comparison{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"  Device:       {device}")
    print(f"  Targets:      {targets}")
    print(f"  Qwen model:   {args.qwen_model}")
    print(f"  SAM3 thresh:  {args.sam3_threshold}")
    print(f"  Output dir:   {output_dir}")
    print(f"  PyTorch:      {torch.__version__}")

    # ---- Collect images ----
    image_paths = collect_images(images_path, max_images=args.max_images)
    if not image_paths:
        print(f"\n{RED}ERROR: No images found at {images_path}{NC}")
        sys.exit(1)
    print(f"  Images found: {len(image_paths)}")
    for p in image_paths:
        print(f"    - {p.name}")

    # ---- Load models ----
    print(f"\n{BOLD}Loading models...{NC}")

    sam3_model, sam3_processor, sam3_load_time = load_sam3(device)
    print(f"  SAM3 load time: {sam3_load_time:.1f}s")

    qwen_model, qwen_processor, qwen_load_time = load_qwen(args.qwen_model, device)
    print(f"  Qwen load time: {qwen_load_time:.1f}s")

    # ---- Process images ----
    all_results: list[dict] = []

    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n{BOLD}[{idx}/{len(image_paths)}] {img_path.name}{NC}")

        image = Image.open(img_path).convert("RGB")
        orig_size = image.size
        image = resize_if_needed(image)
        if image.size != orig_size:
            print(f"  Resized: {orig_size[0]}x{orig_size[1]} -> {image.size[0]}x{image.size[1]}")
        else:
            print(f"  Size: {image.size[0]}x{image.size[1]}")

        # --- SAM3 ---
        print(f"  Running SAM3...", end=" ", flush=True)
        sam3_result = detect_sam3(
            image, targets, sam3_model, sam3_processor, device,
            threshold=args.sam3_threshold,
        )
        sam3_n = len(sam3_result["boxes"])
        print(f"{GREEN}{sam3_n} detections in {sam3_result['elapsed_s']:.2f}s{NC}")
        sam3_result["model"] = "SAM3"
        sam3_result["image"] = img_path.name
        all_results.append(sam3_result)

        # --- Qwen ---
        print(f"  Running Qwen...", end=" ", flush=True)
        qwen_result = detect_qwen(
            image, targets, qwen_model, qwen_processor, device,
        )
        qwen_n = len(qwen_result["boxes"])
        print(f"{GREEN}{qwen_n} detections in {qwen_result['elapsed_s']:.2f}s{NC}")
        qwen_result["model"] = "Qwen"
        qwen_result["image"] = img_path.name
        all_results.append(qwen_result)

        # --- Side-by-side ---
        comparison = create_side_by_side(
            image, sam3_result, qwen_result,
            args.qwen_model, img_path.name, targets,
        )
        out_name = f"compare_{img_path.stem}.jpg"
        out_path = output_dir / out_name
        comparison.save(str(out_path), quality=95)
        print(f"  {GREEN}Saved: {out_path}{NC}")

    # ---- Summary ----
    print_summary(all_results, args.qwen_model)
    print(f"\n  Comparison images saved to: {output_dir}")
    print(f"  {GREEN}Done!{NC}\n")


if __name__ == "__main__":
    main()
