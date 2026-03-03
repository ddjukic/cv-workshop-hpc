#!/usr/bin/env python3
"""Test Qwen3-VL model loading and inference in the workshop environment.

This script verifies that:
1. transformers can import Qwen3-VL classes
2. The model + processor can be loaded (downloads ~16GB on first run)
3. The model can run inference on a test image with grounding
4. qwen-vl-utils and supervision are available for post-processing

Usage:
    python test_qwen3_vl.py                    # Full test (loads model, runs inference)
    python test_qwen3_vl.py --imports-only      # Just check imports, skip model download
    python test_qwen3_vl.py --model-id Qwen/Qwen3-VL-4B-Instruct  # Test smaller variant

Why Qwen3-VL?
    SAM3 (facebook/sam3) is a gated model on HuggingFace — participants need to
    request access, and approval may not arrive in time. Qwen3-VL is publicly
    available, requires no approval, and can do open-vocabulary object detection
    via VLM grounding (returns bounding box coordinates in JSON).
"""

import argparse
import sys
import time
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
NC = "\033[0m"


def check_imports():
    """Verify all required packages can be imported."""
    print(f"\n{BOLD}1. Checking imports...{NC}")
    results = []

    checks = [
        ("torch", "import torch"),
        ("transformers", "import transformers"),
        ("Qwen3VLForConditionalGeneration", "from transformers import Qwen3VLForConditionalGeneration"),
        ("AutoModelForImageTextToText", "from transformers import AutoModelForImageTextToText"),
        ("AutoProcessor", "from transformers import AutoProcessor"),
        ("qwen_vl_utils", "import qwen_vl_utils"),
        ("supervision", "import supervision"),
        ("PIL", "from PIL import Image"),
    ]

    for name, stmt in checks:
        try:
            exec(stmt)
            print(f"  {GREEN}✓{NC} {name}")
            results.append(True)
        except ImportError as e:
            print(f"  {RED}✗{NC} {name}: {e}")
            results.append(False)
        except Exception as e:
            print(f"  {YELLOW}⚠{NC} {name}: {e}")
            results.append(False)

    # Version info
    try:
        import torch
        import transformers
        print(f"\n  torch:        {torch.__version__}")
        print(f"  transformers: {transformers.__version__}")
        print(f"  CUDA:         {torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f" ({torch.cuda.get_device_name(0)})")
        else:
            print()
    except Exception:
        pass

    return all(results)


def check_model_loading(model_id: str):
    """Load model and processor."""
    print(f"\n{BOLD}2. Loading model: {model_id}...{NC}")
    print(f"  {CYAN}ℹ{NC} First run downloads ~16GB for 8B, ~8GB for 4B")

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_id)
    print(f"  {GREEN}✓{NC} Processor loaded ({time.time() - t0:.1f}s)")

    t1 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    elapsed = time.time() - t1
    print(f"  {GREEN}✓{NC} Model loaded on {device} ({elapsed:.1f}s)")

    # Memory info
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  {CYAN}ℹ{NC} GPU memory used: {mem_gb:.1f} GB")

    return model, processor


def check_inference(model, processor, model_id: str):
    """Run a test inference with grounding."""
    print(f"\n{BOLD}3. Running test inference (grounding)...{NC}")

    import torch
    from PIL import Image
    import urllib.request
    import tempfile

    # Download a test image
    test_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
    print(f"  Downloading test image...")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        urllib.request.urlretrieve(test_url, f.name)
        test_image_path = f.name

    image = Image.open(test_image_path).convert("RGB")
    print(f"  Image size: {image.size}")

    # Test 1: Basic VQA
    print(f"\n  {BOLD}Test A: Basic VQA{NC}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see in this image? Answer in one sentence."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64)
    elapsed = time.time() - t0

    trimmed = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(trimmed, skip_special_tokens=True)
    print(f"  {GREEN}✓{NC} Response: {response}")
    print(f"  {CYAN}ℹ{NC} Inference time: {elapsed:.1f}s")

    # Test 2: Grounding (bounding box detection)
    print(f"\n  {BOLD}Test B: Grounding / Object Detection{NC}")
    grounding_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Outline the position of candy and output all the coordinates in JSON format."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        grounding_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512)
    elapsed = time.time() - t0

    trimmed = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(trimmed, skip_special_tokens=True)
    print(f"  {GREEN}✓{NC} Grounding response: {response[:200]}...")
    print(f"  {CYAN}ℹ{NC} Inference time: {elapsed:.1f}s")

    # Test 3: Parse with supervision
    print(f"\n  {BOLD}Test C: Supervision parsing{NC}")
    try:
        import supervision as sv
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.QWEN_3_VL,
            result=response,
            resolution_wh=image.size,
        )
        print(f"  {GREEN}✓{NC} Parsed {len(detections)} detections via supervision")
        if len(detections) > 0:
            print(f"  {CYAN}ℹ{NC} Bounding boxes: {detections.xyxy[:3].tolist()}...")
    except Exception as e:
        print(f"  {YELLOW}⚠{NC} Supervision parsing failed: {e}")
        print(f"      Raw response can still be parsed manually via JSON")

    # Cleanup
    Path(test_image_path).unlink(missing_ok=True)

    return True


def check_ppe_grounding(model, processor):
    """Test PPE-specific grounding with a construction site scenario."""
    print(f"\n{BOLD}4. PPE-specific grounding test...{NC}")

    import torch
    from PIL import Image

    # Check if we have local synthetic images
    workshop_dir = Path(__file__).resolve().parent
    synth_dir = workshop_dir / "data" / "synthetic_ppe"

    if synth_dir.exists():
        # Use a real synthetic PPE image
        images = list(synth_dir.rglob("*.jpg")) + list(synth_dir.rglob("*.png")) + list(synth_dir.rglob("*.webp"))
        if images:
            image = Image.open(images[0]).convert("RGB")
            print(f"  Using local image: {images[0].name}")
        else:
            print(f"  {YELLOW}⚠{NC} No images in synthetic_ppe/. Skipping PPE test.")
            return True
    else:
        print(f"  {YELLOW}⚠{NC} data/synthetic_ppe/ not found. Skipping PPE test.")
        print(f"      (This is expected if data hasn't been deployed yet)")
        return True

    # Test: detect hard hats and persons
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Outline the position of hard hat, person and output all the coordinates in JSON format."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024)
    elapsed = time.time() - t0

    trimmed = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(trimmed, skip_special_tokens=True)
    print(f"  {GREEN}✓{NC} PPE grounding response ({elapsed:.1f}s):")
    print(f"      {response[:300]}")

    try:
        import supervision as sv
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.QWEN_3_VL,
            result=response,
            resolution_wh=image.size,
            classes=["hard hat", "person"],
        )
        print(f"  {GREEN}✓{NC} Detected: {len(detections)} objects")
        if detections.class_id is not None:
            from collections import Counter
            class_counts = Counter(detections.class_id.tolist())
            for cls_id, count in sorted(class_counts.items()):
                cls_name = ["hard hat", "person"][cls_id] if cls_id < 2 else f"class_{cls_id}"
                print(f"      {cls_name}: {count}")
    except Exception as e:
        print(f"  {YELLOW}⚠{NC} Supervision parsing: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen3-VL model loading and inference for the CV workshop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID. Use Qwen/Qwen3-VL-4B-Instruct for less VRAM.",
    )
    parser.add_argument(
        "--imports-only", action="store_true",
        help="Only check imports, skip model download and inference.",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  Qwen3-VL Workshop Environment Test{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")
    print(f"  Model: {args.model_id}")
    print(f"  Mode:  {'imports only' if args.imports_only else 'full test'}")

    # Step 1: Imports
    if not check_imports():
        print(f"\n{RED}{BOLD}Import check failed. Install missing packages:{NC}")
        print(f"  uv pip install transformers>=5.0 qwen-vl-utils>=0.0.14 supervision>=0.27.0")
        sys.exit(1)

    if args.imports_only:
        print(f"\n{GREEN}{BOLD}All imports OK.{NC}")
        sys.exit(0)

    # Step 2: Model loading
    try:
        model, processor = check_model_loading(args.model_id)
    except Exception as e:
        print(f"\n{RED}{BOLD}Model loading failed:{NC} {e}")
        print(f"  Check internet connectivity and disk space (~16GB for 8B model)")
        sys.exit(1)

    # Step 3: Inference
    try:
        check_inference(model, processor, args.model_id)
    except Exception as e:
        print(f"\n{RED}{BOLD}Inference failed:{NC} {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: PPE grounding
    try:
        check_ppe_grounding(model, processor)
    except Exception as e:
        print(f"\n{YELLOW}⚠ PPE test failed (non-critical):{NC} {e}")

    # Summary
    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{GREEN}{BOLD}  All Qwen3-VL tests passed!{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")
    print(f"\n  This model can be used as an alternative to SAM3 for:")
    print(f"  - Open-vocabulary object detection (hard hat, person)")
    print(f"  - Visual question answering about scenes")
    print(f"  - Grounding with bounding box coordinates")
    print(f"\n  Note: Qwen3-VL is ~5-10x slower than YOLOE for detection,")
    print(f"  but provides richer understanding and no gating restrictions.")


if __name__ == "__main__":
    main()
