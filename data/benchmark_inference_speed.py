#!/usr/bin/env python3
"""
Benchmark inference speed: YOLO26n vs YOLOe-26n-seg vs SAM3

Compares the three key workshop models across 3 image resolutions
(320x320, 640x640, 1280x1280) and reports per-image timing statistics.

Usage:
    uv run python benchmark_inference_speed.py [--device mps|cuda|cpu] [--warmup 5] [--runs 20]
    uv run python benchmark_inference_speed.py --device mps --warmup 3 --runs 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZES = [(320, 320), (640, 640), (1280, 1280)]

# Search paths for a test image (first found wins)
CANDIDATE_IMAGE_DIRS = [
    Path(__file__).resolve().parent.parent / "workshop" / "datasets" / "construction-ppe" / "images" / "test",
    Path(__file__).resolve().parent.parent / "workshop" / "datasets" / "construction-ppe" / "images" / "val",
    Path(__file__).resolve().parent / "demo_images",
    Path(__file__).resolve().parent.parent / "workshop" / "data" / "synthetic_ppe",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Model weight paths (search relative to common workshop locations)
MODEL_SEARCH_DIRS = [
    Path(__file__).resolve().parent.parent,       # cv-workshop-hpc/
    Path(__file__).resolve().parent.parent / "workshop",  # cv-workshop/workshop/
    Path.cwd(),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect best available device: MPS (macOS) > CUDA > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    """Synchronize GPU to get accurate timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def find_test_image() -> Path | None:
    """Find a suitable test image from the workshop dataset."""
    for d in CANDIDATE_IMAGE_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                return p
    return None


def find_model_weights(filename: str) -> Path | None:
    """Search common directories for a model weight file."""
    for d in MODEL_SEARCH_DIRS:
        candidate = d / filename
        if candidate.is_file():
            return candidate
    return None


def generate_test_image() -> Image.Image:
    """Generate a solid-color test image as a fallback."""
    import numpy as np
    arr = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def prepare_image(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize image to the target size."""
    return img.resize(target_size, Image.BILINEAR)


def format_table(results: list[dict]) -> str:
    """Format benchmark results as a clean ASCII table for slides."""
    header = (
        f"{'Model':<22} {'Resolution':>10} {'Mean ms':>9} {'Std ms':>8} "
        f"{'Min ms':>8} {'Max ms':>8} {'img/sec':>9}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    current_model = None
    for r in results:
        model_label = r["model"] if r["model"] != current_model else ""
        current_model = r["model"]
        res = f"{r['width']}x{r['height']}"
        throughput = 1000.0 / r["mean_ms"] if r["mean_ms"] > 0 else 0
        lines.append(
            f"{model_label:<22} {res:>10} {r['mean_ms']:>9.1f} {r['std_ms']:>8.1f} "
            f"{r['min_ms']:>8.1f} {r['max_ms']:>8.1f} {throughput:>9.1f}"
        )
        if r == results[-1] or results[results.index(r) + 1]["model"] != current_model:
            lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_yolo(
    model_path: str,
    model_name: str,
    img: Image.Image,
    sizes: list[tuple[int, int]],
    device: str,
    warmup: int,
    runs: int,
    is_open_vocab: bool = False,
) -> list[dict]:
    """Benchmark a YOLO model across multiple image sizes."""
    from ultralytics import YOLO

    print(f"\n  Loading {model_name} from {model_path}...")
    model = YOLO(model_path)

    if is_open_vocab:
        model.set_classes(["hard hat", "person"])
        print(f"  Set open-vocab classes: ['hard hat', 'person']")

    all_results = []

    for w, h in sizes:
        resized = prepare_image(img, (w, h))
        label = f"{model_name} @ {w}x{h}"
        print(f"\n  Benchmarking {label}...")

        # Warmup
        for _ in range(warmup):
            sync_device(device)
            model.predict(resized, device=device, verbose=False, imgsz=(h, w))
            sync_device(device)

        # Timed runs
        times_ms = []
        for _ in range(runs):
            sync_device(device)
            t0 = time.perf_counter()
            model.predict(resized, device=device, verbose=False, imgsz=(h, w))
            sync_device(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        throughput = 1000.0 / mean_ms if mean_ms > 0 else 0

        print(f"    Mean: {mean_ms:.1f} ms  Std: {std_ms:.1f} ms  "
              f"Min: {min_ms:.1f} ms  Max: {max_ms:.1f} ms  "
              f"Throughput: {throughput:.1f} img/s")

        all_results.append({
            "model": model_name,
            "width": w,
            "height": h,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "throughput_ips": round(throughput, 2),
            "runs": runs,
            "warmup": warmup,
        })

    return all_results


def benchmark_sam3(
    img: Image.Image,
    sizes: list[tuple[int, int]],
    device: str,
    warmup: int,
    runs: int,
) -> list[dict]:
    """Benchmark SAM3 via HuggingFace Transformers with text prompt mode.

    Uses Sam3Processor + Sam3Model directly (not Auto classes) because
    AutoProcessor resolves to Sam3VideoProcessor in transformers>=5.x
    which does not support the ``text=`` parameter.
    """
    from transformers import Sam3Model, Sam3Processor

    model_id = "facebook/sam3"
    print(f"\n  Loading SAM3 from {model_id}...")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()
    print(f"  SAM3 loaded on {device}")

    text_prompt = "person"

    all_results = []

    for w, h in sizes:
        resized = prepare_image(img, (w, h))
        label = f"SAM3 @ {w}x{h}"
        print(f"\n  Benchmarking {label} (text prompt: '{text_prompt}')...")

        def run_once():
            inputs = processor(images=resized, text=text_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Post-process to get actual detections (mirrors the workshop pipeline)
            target_sizes = inputs.get("original_sizes")
            if target_sizes is not None:
                target_sizes = target_sizes.tolist()
            else:
                target_sizes = [(h, w)]
            _ = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )

        # Warmup
        for _ in range(warmup):
            sync_device(device)
            run_once()
            sync_device(device)

        # Timed runs
        times_ms = []
        for _ in range(runs):
            sync_device(device)
            t0 = time.perf_counter()
            run_once()
            sync_device(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        throughput = 1000.0 / mean_ms if mean_ms > 0 else 0

        print(f"    Mean: {mean_ms:.1f} ms  Std: {std_ms:.1f} ms  "
              f"Min: {min_ms:.1f} ms  Max: {max_ms:.1f} ms  "
              f"Throughput: {throughput:.1f} img/s")

        all_results.append({
            "model": "SAM3",
            "width": w,
            "height": h,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "throughput_ips": round(throughput, 2),
            "runs": runs,
            "warmup": warmup,
        })

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed: YOLO26n vs YOLOe-26n-seg vs SAM3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (auto-detected if omitted: MPS > CUDA > CPU).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations per model+resolution (not timed).",
    )
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of timed iterations per model+resolution.",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a test image. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--yolo26n", type=str, default=None,
        help="Path to yolo26n.pt weights. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--yoloe", type=str, default=None,
        help="Path to yoloe-26n-seg.pt weights. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--skip-sam3", action="store_true",
        help="Skip SAM3 benchmark (useful for quick YOLO-only runs).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for JSON results file. Defaults to benchmark_results.json next to script.",
    )
    args = parser.parse_args()

    # ---- Device ----
    device = args.device or detect_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # ---- Test image ----
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"ERROR: Image not found: {img_path}")
            sys.exit(1)
        img = Image.open(img_path).convert("RGB")
        print(f"Test image: {img_path} ({img.size[0]}x{img.size[1]})")
    else:
        img_path = find_test_image()
        if img_path:
            img = Image.open(img_path).convert("RGB")
            print(f"Test image (auto): {img_path} ({img.size[0]}x{img.size[1]})")
        else:
            print("No test image found — generating a random test image (640x640)")
            img = generate_test_image()
            img_path = None

    # ---- Locate model weights ----
    yolo26n_path = args.yolo26n or find_model_weights("yolo26n.pt")
    yoloe_path = args.yoloe or find_model_weights("yoloe-26n-seg.pt")

    if yolo26n_path:
        print(f"YOLO26n weights: {yolo26n_path}")
    else:
        print("WARNING: yolo26n.pt not found — will attempt download via Ultralytics")
        yolo26n_path = "yolo26n.pt"

    if yoloe_path:
        print(f"YOLOe-26n-seg weights: {yoloe_path}")
    else:
        print("WARNING: yoloe-26n-seg.pt not found — will attempt download via Ultralytics")
        yoloe_path = "yoloe-26n-seg.pt"

    print(f"\nWarmup: {args.warmup} iterations")
    print(f"Timed:  {args.runs} iterations")
    print(f"Image sizes: {', '.join(f'{w}x{h}' for w, h in IMAGE_SIZES)}")

    # ---- Run benchmarks ----
    all_results: list[dict] = []

    print("\n" + "=" * 70)
    print("  BENCHMARK 1: YOLO26n (distilled student model)")
    print("=" * 70)
    yolo26n_results = benchmark_yolo(
        model_path=str(yolo26n_path),
        model_name="YOLO26n",
        img=img,
        sizes=IMAGE_SIZES,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )
    all_results.extend(yolo26n_results)

    print("\n" + "=" * 70)
    print("  BENCHMARK 2: YOLOe-26n-seg (open-vocabulary)")
    print("=" * 70)
    yoloe_results = benchmark_yolo(
        model_path=str(yoloe_path),
        model_name="YOLOe-26n-seg",
        img=img,
        sizes=IMAGE_SIZES,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        is_open_vocab=True,
    )
    all_results.extend(yoloe_results)

    if not args.skip_sam3:
        print("\n" + "=" * 70)
        print("  BENCHMARK 3: SAM3 (facebook/sam3, text-prompted)")
        print("=" * 70)
        sam3_results = benchmark_sam3(
            img=img,
            sizes=IMAGE_SIZES,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )
        all_results.extend(sam3_results)

    # ---- Print comparison table ----
    print("\n\n" + "=" * 70)
    print("  INFERENCE SPEED COMPARISON")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Warmup:     {args.warmup} iterations")
    print(f"  Timed:      {args.runs} iterations per config")
    if img_path:
        print(f"  Test image: {Path(img_path).name}")
    print()
    print(format_table(all_results))

    # ---- Speedup summary ----
    print("\n  SPEEDUP SUMMARY (at 640x640, mean latency):")
    baseline = None
    entries_640 = [r for r in all_results if r["width"] == 640]
    if entries_640:
        baseline = entries_640[0]  # YOLO26n
        for r in entries_640:
            ratio = r["mean_ms"] / baseline["mean_ms"] if baseline["mean_ms"] > 0 else 0
            if r["model"] == baseline["model"]:
                print(f"    {r['model']:<22} {r['mean_ms']:>8.1f} ms  (baseline)")
            else:
                print(f"    {r['model']:<22} {r['mean_ms']:>8.1f} ms  ({ratio:.1f}x slower)")

    # ---- Save JSON ----
    output_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "benchmark_results.json"
    output_data = {
        "device": device,
        "pytorch_version": torch.__version__,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.runs,
        "test_image": str(img_path) if img_path else "generated",
        "image_sizes": [{"width": w, "height": h} for w, h in IMAGE_SIZES],
        "results": all_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2) + "\n")
    print(f"\n  Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
