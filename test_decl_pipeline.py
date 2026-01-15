#!/usr/bin/env python3
"""
DeCl Pipeline QA Test Script

Firebase URL 없이 로컬 이미지로 파이프라인 각 단계를 테스트합니다.

테스트 항목:
1. YOLO-World 객체 탐지
2. CLIP 세부 유형 분류
3. Knowledge Base 조회
4. Volume Calculator (PLY 파일 있을 경우)

Usage:
    python test_decl_pipeline.py
"""

import sys
import os
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Add DeCl directory to path (required for DeCl's internal imports)
DECL_PATH = PROJECT_ROOT / "DeCl"
if str(DECL_PATH) not in sys.path:
    sys.path.insert(0, str(DECL_PATH))

# Test configuration
TEST_IMAGES = [
    "DeCl/imgs/test.jpg",      # Living room with sofa, mirror, tables
    "DeCl/imgs/test2.jpg",     # Dining room with table, chairs, cabinet
]


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title: str):
    print(f"\n--- {title} ---")


def test_imports():
    """Test that all required modules can be imported"""
    print_header("1. Testing Imports")

    errors = []

    # Test core dependencies
    try:
        import torch
        print(f"[OK] torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        from PIL import Image
        print("[OK] PIL (Pillow)")
    except ImportError as e:
        errors.append(f"PIL: {e}")

    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")

    try:
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")

    # Test DeCl modules (using relative imports from DeCl path)
    try:
        from config import Config
        print(f"[OK] DeCl.config (Device: {Config.DEVICE})")
    except ImportError as e:
        errors.append(f"DeCl.config: {e}")

    try:
        from data.knowledge_base import FURNITURE_DB, get_dimensions_for_subtype, estimate_size_variant
        print(f"[OK] DeCl.knowledge_base ({len(FURNITURE_DB)} furniture types)")
    except ImportError as e:
        errors.append(f"DeCl.knowledge_base: {e}")

    try:
        from models.detector import YoloDetector
        print("[OK] DeCl.models.detector")
    except ImportError as e:
        errors.append(f"DeCl.models.detector: {e}")

    try:
        from models.classifier import ClipClassifier
        print("[OK] DeCl.models.classifier")
    except ImportError as e:
        errors.append(f"DeCl.models.classifier: {e}")

    try:
        from utils.volume_calculator import VolumeCalculator
        print("[OK] DeCl.utils.volume_calculator")
    except ImportError as e:
        errors.append(f"DeCl.utils.volume_calculator: {e}")

    if errors:
        print(f"\n[FAIL] {len(errors)} import errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("\n[PASS] All imports successful")
    return True


def test_knowledge_base():
    """Test knowledge base functionality"""
    print_header("2. Testing Knowledge Base")

    from data.knowledge_base import (
        FURNITURE_DB,
        get_dimensions_for_subtype,
        estimate_size_variant
    )

    # List all furniture types
    print_section("Registered Furniture Types")
    for key, info in FURNITURE_DB.items():
        movable_flag = info.get('is_movable', True)
        movable = "이동가능" if movable_flag else "고정"
        subtypes = len(info.get('subtypes', []))
        print(f"  - {key}: {info['base_name']} ({movable}, {subtypes} subtypes)")

    # Test get_dimensions_for_subtype
    print_section("Test get_dimensions_for_subtype()")
    test_cases = [
        ("bed", "퀸 사이즈 침대"),
        ("sofa", "3인용 소파"),
        ("refrigerator", "일반 냉장고"),
    ]
    for db_key, subtype_name in test_cases:
        dims = get_dimensions_for_subtype(db_key, subtype_name)
        if dims:
            print(f"  '{db_key}' / '{subtype_name}' -> {dims}")
        else:
            print(f"  '{db_key}' / '{subtype_name}' -> Not found")

    # Test variant estimation
    print_section("Test estimate_size_variant()")
    test_cases = [
        ("bed", "퀸 사이즈 침대", {"w": 0.75, "h": 0.225, "d": 1.0}),
        ("box", None, {"w": 1.0, "h": 1.0, "d": 0.66}),
    ]
    for db_key, subtype_name, ratio in test_cases:
        variant_name, dims = estimate_size_variant(db_key, subtype_name, ratio)
        if variant_name:
            print(f"  {db_key}/{subtype_name} ratio {ratio} -> {variant_name}: {dims}")
        else:
            print(f"  {db_key}/{subtype_name} ratio {ratio} -> default: {dims}")

    print("\n[PASS] Knowledge base tests completed")
    return True


def test_yolo_detection(image_path: str):
    """Test YOLO-World detection"""
    print_header("3. Testing YOLO-World Detection")

    from PIL import Image
    from models.detector import YoloDetector
    from config import Config
    from data.knowledge_base import FURNITURE_DB

    # Load image
    print_section(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    print(f"  Image size: {img.size}")

    # Initialize detector
    print_section("Initializing YOLO-World detector")
    start_time = time.time()
    detector = YoloDetector()
    init_time = time.time() - start_time
    print(f"  Initialization time: {init_time:.2f}s")

    # Set furniture classes and get class list
    all_classes = []
    for info in FURNITURE_DB.values():
        all_classes.extend(info.get('synonyms', []))
    class_list = list(set(all_classes))
    detector.set_classes(class_list)
    print(f"  Monitoring {len(class_list)} furniture classes")

    # Run detection
    print_section("Running detection")
    start_time = time.time()
    raw_results = detector.detect_smart(img)
    detect_time = time.time() - start_time

    # Convert raw results to list of detection dicts
    detections = []
    if raw_results is not None:
        boxes = raw_results.get('boxes', [])
        scores = raw_results.get('scores', [])
        classes = raw_results.get('classes', [])

        for i in range(len(boxes)):
            bbox = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
            conf = float(scores[i])
            cls_idx = int(classes[i])
            # Get label from class index (YOLO-World uses set_classes order)
            label = class_list[cls_idx] if cls_idx < len(class_list) else f"class_{cls_idx}"

            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'label': label,
                'class_idx': cls_idx
            })

    print(f"  Detection time: {detect_time:.2f}s")
    print(f"  Objects detected: {len(detections)}")

    # Display results
    print_section("Detection Results")
    for i, det in enumerate(detections):
        bbox = det['bbox']
        label = det['label']
        conf = det['confidence']
        print(f"  {i+1}. {label}: confidence={conf:.3f}, bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")

    if len(detections) > 0:
        print(f"\n[PASS] Detected {len(detections)} objects")
        return detections, img, detector
    else:
        print("\n[WARN] No objects detected - this may be expected for some images")
        return [], img, detector


def test_clip_classification(image: "Image", detections: list):
    """Test CLIP classification on detected objects"""
    print_header("4. Testing CLIP Classification")

    if not detections:
        print("[SKIP] No detections to classify")
        return []

    from data.knowledge_base import FURNITURE_DB

    # Find DB key for a label
    def find_db_key(label: str) -> str:
        label_lower = label.lower()
        for db_key, info in FURNITURE_DB.items():
            synonyms = [s.lower() for s in info.get('synonyms', [])]
            if label_lower in synonyms or label_lower == db_key:
                return db_key
        return None

    # Initialize classifier (with torch version workaround)
    print_section("Initializing CLIP classifier")
    try:
        from models.classifier import ClipClassifier
        start_time = time.time()
        classifier = ClipClassifier()
        init_time = time.time() - start_time
        print(f"  Initialization time: {init_time:.2f}s")
    except ValueError as e:
        if "torch" in str(e) and "2.6" in str(e):
            print(f"  [SKIP] CLIP requires torch >= 2.6 (current: {__import__('torch').__version__})")
            print("  Workaround: Run 'pip install torch>=2.6' or use safetensors")
            return []
        raise

    # Classify each detection
    print_section("Classification Results")
    results = []

    for i, det in enumerate(detections[:5]):  # Limit to first 5 for speed
        bbox = det.get('bbox', [])
        label = det.get('label', 'unknown')

        if len(bbox) < 4:
            continue

        # Find DB key
        db_key = find_db_key(label)
        if not db_key:
            print(f"  {i+1}. {label} -> Not in knowledge base (skipped)")
            continue

        info = FURNITURE_DB.get(db_key, {})
        subtypes = info.get('subtypes', [])

        if not subtypes:
            base_name = info.get('base_name', label)
            print(f"  {i+1}. {label} -> {base_name} (no subtypes)")
            continue

        # Crop image
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        crop = image.crop((x1, y1, x2, y2))

        # Classify
        start_time = time.time()
        result = classifier.classify(crop, subtypes)
        classify_time = time.time() - start_time

        if result:
            best_match = result.get('name', 'unknown')
            score = result.get('score', 0)
            is_movable = result.get('is_movable', True)
            status = "이동가능" if is_movable else "고정"

            print(f"  {i+1}. {label} -> {best_match} ({status}, score: {score:.3f}, time: {classify_time:.2f}s)")
            results.append({
                'original_label': label,
                'refined_label': best_match,
                'score': score,
                'is_movable': is_movable
            })
        else:
            print(f"  {i+1}. {label} -> Classification failed")

    print(f"\n[PASS] Classified {len(results)} objects")
    return results


def test_volume_calculator():
    """Test volume calculator with sample PLY if available"""
    print_header("5. Testing Volume Calculator")

    from utils.volume_calculator import VolumeCalculator

    # Initialize calculator
    calculator = VolumeCalculator()
    print("[OK] VolumeCalculator initialized")

    # Check for existing PLY files in assets
    assets_dir = PROJECT_ROOT / "assets"
    ply_files = list(assets_dir.glob("*.ply")) if assets_dir.exists() else []

    if ply_files:
        print_section(f"Testing with existing PLY: {ply_files[0].name}")
        try:
            result = calculator.calculate_from_ply(str(ply_files[0]))
            print(f"  Volume: {result.get('volume', 'N/A')}")
            print(f"  Bounding box: {result.get('bounding_box', 'N/A')}")
            print(f"  Ratio: {result.get('ratio', 'N/A')}")
        except Exception as e:
            print(f"  [WARN] Could not process PLY: {e}")
    else:
        print("[INFO] No PLY files found in assets/ - testing with synthetic data")

    # Test scale_to_absolute
    print_section("Testing scale_to_absolute()")
    relative_dims = {
        'bounding_box': {'width': 1.0, 'depth': 0.66, 'height': 0.5},
        'ratio': {'w': 1.0, 'd': 0.66, 'h': 0.5}
    }
    reference_dims = {'width': 2000, 'depth': 1000, 'height': 500}  # mm

    absolute = calculator.scale_to_absolute(relative_dims, reference_dims)
    print(f"  Input relative: {relative_dims['bounding_box']}")
    print(f"  Reference (mm): {reference_dims}")
    print(f"  Output absolute: w={absolute.get('width', 0):.0f}mm, d={absolute.get('depth', 0):.0f}mm, h={absolute.get('height', 0):.0f}mm")
    print(f"  Volume: {absolute.get('volume_liters', 0):.2f} liters")

    print("\n[PASS] Volume calculator tests completed")
    return True


def test_full_pipeline_detection_only(image_path: str, reuse_detector=None):
    """Test the detection-only pipeline (no SAM2/SAM3D)"""
    print_header("6. Full Pipeline Test (Detection Only)")

    from PIL import Image
    from models.detector import YoloDetector
    from data.knowledge_base import FURNITURE_DB

    # Find DB key for a label
    def find_db_key(label: str) -> str:
        label_lower = label.lower()
        for db_key, info in FURNITURE_DB.items():
            synonyms = [s.lower() for s in info.get('synonyms', [])]
            if label_lower in synonyms or label_lower == db_key:
                return db_key
        return None

    # Load image
    img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({img.size[0]}x{img.size[1]})")

    # Detection
    print_section("Step 1: Object Detection (YOLO-World)")
    if reuse_detector:
        detector = reuse_detector
        print("  (Reusing existing detector)")
    else:
        detector = YoloDetector()

    # Set furniture classes
    all_classes = []
    for info in FURNITURE_DB.values():
        all_classes.extend(info.get('synonyms', []))
    class_list = list(set(all_classes))
    detector.set_classes(class_list)

    raw_results = detector.detect_smart(img)

    # Convert raw results to list of detection dicts
    detections = []
    if raw_results is not None:
        boxes = raw_results.get('boxes', [])
        scores = raw_results.get('scores', [])
        classes = raw_results.get('classes', [])

        for i in range(len(boxes)):
            bbox = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
            conf = float(scores[i])
            cls_idx = int(classes[i])
            label = class_list[cls_idx] if cls_idx < len(class_list) else f"class_{cls_idx}"

            detections.append({
                'bbox': bbox,
                'confidence': conf,
                'label': label,
                'class_idx': cls_idx
            })

    print(f"  Detected: {len(detections)} objects")

    # Classification and movability check
    print_section("Step 2: Classification & Movability Check")

    # Try to load CLIP classifier
    classifier = None
    try:
        from models.classifier import ClipClassifier
        classifier = ClipClassifier()
    except ValueError as e:
        if "torch" in str(e) and "2.6" in str(e):
            print(f"  [WARN] CLIP skipped (requires torch >= 2.6)")
            print("  Using base classification only (no subtype refinement)")
        else:
            raise

    final_results = []
    movable_count = 0

    for det in detections:
        bbox = det.get('bbox', [])
        label = det.get('label', 'unknown')
        conf = det.get('confidence', 0)

        if len(bbox) < 4:
            continue

        # Find DB key
        db_key = find_db_key(label)
        is_movable = True
        refined_label = label

        if db_key:
            info = FURNITURE_DB.get(db_key, {})
            is_movable = info.get('is_movable', True)
            base_name = info.get('base_name', label)
            subtypes = info.get('subtypes', [])

            # Try to classify subtype if classifier available
            if classifier and subtypes:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                crop = img.crop((x1, y1, x2, y2))
                result = classifier.classify(crop, subtypes)
                if result:
                    refined_label = result.get('name', base_name)
                    is_movable = result.get('is_movable', is_movable)
                else:
                    refined_label = base_name
            else:
                refined_label = base_name

        if is_movable:
            movable_count += 1

        final_results.append({
            'label': refined_label,
            'original_label': label,
            'confidence': conf,
            'is_movable': is_movable,
            'bbox': bbox
        })

        status = "이동가능" if is_movable else "고정"
        print(f"  - {refined_label} ({status}, conf: {conf:.2f})")

    # Summary
    print_section("Pipeline Summary")
    print(f"  Total objects: {len(final_results)}")
    print(f"  Movable objects: {movable_count}")
    print(f"  Fixed objects: {len(final_results) - movable_count}")

    # Generate JSON-like output
    print_section("Output JSON Preview")
    output = {
        "objects": [
            {
                "label": r['label'],
                "is_movable": r['is_movable'],
                "confidence": round(r['confidence'], 3),
                "bbox": [round(v, 1) for v in r['bbox'][:4]]
            }
            for r in final_results
        ],
        "summary": {
            "total_objects": len(final_results),
            "movable_objects": movable_count,
            "fixed_objects": len(final_results) - movable_count
        }
    }

    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n[PASS] Full pipeline test completed")
    return final_results


def main():
    print("\n" + "=" * 60)
    print(" DeCl Pipeline QA Test")
    print(" Testing AI Logic without Firebase/SAM2/SAM3D")
    print("=" * 60)

    # Check test images exist
    for img_path in TEST_IMAGES:
        full_path = PROJECT_ROOT / img_path
        if not full_path.exists():
            print(f"[ERROR] Test image not found: {img_path}")
            return 1

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        print("\n[FATAL] Import tests failed. Cannot continue.")
        return 1

    # Test 2: Knowledge Base
    try:
        test_knowledge_base()
    except Exception as e:
        print(f"\n[FAIL] Knowledge base test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: YOLO Detection
    detector = None
    try:
        test_image = str(PROJECT_ROOT / TEST_IMAGES[0])
        detections, img, detector = test_yolo_detection(test_image)
    except Exception as e:
        print(f"\n[FAIL] YOLO detection test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        detections, img = [], None

    # Test 4: CLIP Classification
    if img and detections:
        try:
            test_clip_classification(img, detections)
        except Exception as e:
            print(f"\n[FAIL] CLIP classification test error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Test 5: Volume Calculator
    try:
        test_volume_calculator()
    except Exception as e:
        print(f"\n[FAIL] Volume calculator test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 6: Full Pipeline (Detection Only)
    try:
        test_image = str(PROJECT_ROOT / TEST_IMAGES[1])
        test_full_pipeline_detection_only(test_image, reuse_detector=detector)
    except Exception as e:
        print(f"\n[FAIL] Full pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Final summary
    print_header("QA Test Summary")
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print("\nNote: This test covers detection + classification.")
        print("SAM2 mask generation and SAM-3D 3D conversion require")
        print("the API server to be running (uvicorn api:app).")
        return 0
    else:
        print("[PARTIAL] Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
