#!/usr/bin/env python3
"""
YOLOE vs SAM2 Mask Comparison Test

Compares segmentation mask quality between:
1. YOLOE-seg: Direct instance segmentation from object detection
2. SAM2: Point-prompted segmentation using bbox center points

Output Structure:
    test_outputs/{image_name}/
    ├── 00_original.jpg               # Original image
    ├── 01_yoloe_detection.jpg        # YOLOE bbox + labels
    ├── 02_yoloe_masks.png            # YOLOE masks (colored overlay)
    ├── 03_sam2_masks.png             # SAM2 masks (colored overlay)
    ├── mask_XX_{label}.png           # Individual YOLOE masks (B/W)
    ├── crop_XX_{label}.jpg           # Cropped image from bbox (debug)
    ├── sam2_crop_mask_XX_{label}.png # SAM2 mask on cropped image (debug)
    └── sam2_mask_XX_{label}.png      # SAM2 mask (full size, restored)

Usage:
    python test_yoloe_vs_sam2_masks.py
"""

import os
import sys

# Environment setup BEFORE imports
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# SAM-3D Converter
from ai.processors import SAM3DConverter

# Set PyTorch defaults
torch.set_default_dtype(torch.float32)

# Project root
PROJECT_ROOT = Path(__file__).parent

# Test images to process
TEST_IMAGES = [
    "ai/imgs/bedroom-5772286_1920.jpg",
    "ai/imgs/bed-1834327_1920.jpg",
    "ai/imgs/apartment-1835482_1920.jpg",
]

# Color palet2te for mask visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """Get color for index (cycles through palette)"""
    return COLORS[idx % len(COLORS)]


def load_font(size: int = 20):
    """Load font for text drawing"""
    font_paths = [
        PROJECT_ROOT / "ai/fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except:
                pass
    return ImageFont.load_default()


class YOLOEDetector:
    """YOLOE-seg detector wrapper"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOE-seg model"""
        try:
            from ultralytics import YOLOE
        except ImportError:
            from ultralytics import YOLO as YOLOE
            print("[YOLOEDetector] YOLOE not available, using YOLO")

        from ai.config import Config
        import importlib
        _yolo_module = importlib.import_module('ai.processors.2_YOLO_detect')
        FURNITURE_CLASSES = _yolo_module.FURNITURE_CLASSES

        model_path = Config.YOLO_MODEL_PATH
        print(f"[YOLOEDetector] Loading model: {model_path}")

        self.model = YOLOE(model_path)

        # Set furniture classes
        class_names = list(FURNITURE_CLASSES.keys())
        self.model.set_classes(class_names)

        if "cuda" in self.device:
            self.model.to(self.device)

        print(f"[YOLOEDetector] Model loaded with {len(self.model.names)} classes")

    def detect(self, image: Image.Image, conf: float = 0.25) -> Dict:
        """
        Detect objects with segmentation masks

        Returns:
            {
                "boxes": np.ndarray [[x1,y1,x2,y2], ...],
                "scores": np.ndarray,
                "labels": [str, ...],
                "masks": [np.ndarray, ...] (binary masks, image size)
            }
        """
        if self.model is None:
            return {"boxes": [], "scores": [], "labels": [], "masks": []}

        results = self.model.predict(
            image,
            conf=conf,
            verbose=False,
            device=self.device
        )[0]

        if len(results.boxes) == 0:
            return {"boxes": [], "scores": [], "labels": [], "masks": []}

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        labels = [self.model.names[int(c)] for c in classes]

        # Extract masks
        masks = []
        if results.masks is not None:
            for mask in results.masks.data:
                mask_np = mask.cpu().numpy()
                # Resize to image dimensions
                if mask_np.shape != (image.height, image.width):
                    mask_np = cv2.resize(
                        mask_np,
                        (image.width, image.height),
                        interpolation=cv2.INTER_NEAREST
                    )
                masks.append((mask_np > 0.5).astype(np.uint8) * 255)

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks
        }


class SAM2Segmenter:
    """SAM2 segmentation wrapper"""

    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SAM2 model from HuggingFace"""
        from transformers import Sam2Processor, Sam2Model

        model_id = "facebook/sam2.1-hiera-large"
        print(f"[SAM2Segmenter] Loading model: {model_id}")

        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device)

        print(f"[SAM2Segmenter] Model loaded on {self.device}")

    def segment_point(
        self,
        image: Image.Image,
        x: float,
        y: float
    ) -> Optional[np.ndarray]:
        """
        Generate mask from single point

        Args:
            image: PIL Image
            x, y: Point coordinates

        Returns:
            Binary mask (H, W) as uint8, values 0 or 255
        """
        if self.model is None or self.processor is None:
            return None

        # Format: [[[[x, y]]]]
        input_points = [[[[x, y]]]]
        input_labels = [[[1]]]  # positive click

        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]

        # Get scores
        scores = (
            outputs.iou_preds[0].cpu().numpy()
            if hasattr(outputs, "iou_preds")
            else np.array([0.95] * masks.shape[0])
        )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].numpy()
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            mask = mask[0] if mask.ndim > 2 else mask

        # Threshold and convert
        mask = (mask > 0).astype(np.uint8) * 255

        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask


def draw_detection_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray
) -> Image.Image:
    """Draw bounding boxes with labels on image"""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    font = load_font(18)

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box.astype(int)
        color = get_color(i)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        text = f"{label} ({score:.2f})"
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=color
        )
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return draw_img


def create_mask_overlay(
    image: Image.Image,
    masks: List[np.ndarray],
    labels: List[str],
    alpha: float = 0.5
) -> Image.Image:
    """Create colored mask overlay on image"""
    img_np = np.array(image)
    overlay = img_np.copy()

    for i, (mask, label) in enumerate(zip(masks, labels)):
        if mask is None:
            continue

        color = get_color(i)

        # Ensure mask is binary
        mask_bool = mask > 127

        # Apply color overlay
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    # Add labels
    result = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = load_font(14)

    # Draw legend
    y_offset = 10
    for i, label in enumerate(labels):
        color = get_color(i)
        text = f"{i:02d}: {label}"
        draw.rectangle([10, y_offset, 25, y_offset + 15], fill=color)
        draw.text((30, y_offset), text, fill=(255, 255, 255), font=font)
        y_offset += 20

    return result


def save_individual_mask(
    mask: np.ndarray,
    output_path: Path,
    label: str,
    idx: int,
    prefix: str = "mask"
) -> str:
    """Save individual binary mask"""
    # Sanitize label for filename
    safe_label = label.replace("/", "_").replace(" ", "_")
    filename = f"{prefix}_{idx:02d}_{safe_label}.png"
    filepath = output_path / filename

    # Save as grayscale PNG
    mask_img = Image.fromarray(mask, mode="L")
    mask_img.save(filepath)

    return filename


def get_bbox_center(box: np.ndarray) -> Tuple[float, float]:
    """Get center point of bounding box"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def process_image(
    image_path: Path,
    output_dir: Path,
    yolo_detector: YOLOEDetector,
    sam2_segmenter: SAM2Segmenter
) -> Dict:
    """
    Process single image: compare YOLOE vs SAM2 masks

    Returns:
        Summary dictionary with detection counts and paths
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")

    # Create output directory
    img_name = image_path.stem
    img_output_dir = output_dir / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Save original
    image.save(img_output_dir / "00_original.jpg", quality=95)

    # YOLOE detection
    print("\n--- YOLOE Detection ---")
    yolo_result = yolo_detector.detect(image, conf=0.25)

    boxes = yolo_result["boxes"]
    scores = yolo_result["scores"]
    labels = yolo_result["labels"]
    yolo_masks = yolo_result["masks"]

    num_detections = len(boxes) if hasattr(boxes, '__len__') else 0
    print(f"Detected {num_detections} objects")

    if num_detections == 0:
        print("No objects detected!")
        return {
            "image": img_name,
            "detections": 0,
            "yolo_masks": 0,
            "sam2_masks": 0
        }

    # Print detections
    for i, (label, score) in enumerate(zip(labels, scores)):
        print(f"  {i:02d}: {label} (conf: {score:.3f})")

    # Save detection visualization
    detection_img = draw_detection_boxes(image, boxes, labels, scores)
    detection_img.save(img_output_dir / "01_yoloe_detection.jpg", quality=95)

    # Save YOLOE masks
    print("\n--- YOLOE Masks ---")
    yolo_mask_files = []
    for i, (mask, label) in enumerate(zip(yolo_masks, labels)):
        if mask is not None:
            filename = save_individual_mask(mask, img_output_dir, label, i, "mask")
            yolo_mask_files.append(filename)
            print(f"  Saved: {filename}")

    # Create YOLOE mask overlay
    if yolo_masks:
        yolo_overlay = create_mask_overlay(image, yolo_masks, labels)
        yolo_overlay.save(img_output_dir / "02_yoloe_masks.png")
        print(f"  Overlay saved: 02_yoloe_masks.png")

    # SAM2 segmentation from cropped images
    print("\n--- SAM2 Masks (from cropped images) ---")
    sam2_masks = []
    sam2_mask_files = []

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)

        # 1. Crop image from bbox
        cropped = image.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size

        # 2. Center point of cropped image
        cx, cy = crop_w / 2, crop_h / 2
        print(f"  {i:02d}: {label} - crop size ({crop_w}x{crop_h}), center ({cx:.1f}, {cy:.1f})")

        # 3. SAM2 segment on cropped image
        crop_mask = sam2_segmenter.segment_point(cropped, cx, cy)

        if crop_mask is not None:
            # 4. Create full-size mask (restore to original size)
            full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = cv2.resize(
                crop_mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST
            )
            sam2_masks.append(full_mask)

            # Save full-size mask
            filename = save_individual_mask(full_mask, img_output_dir, label, i, "sam2_mask")
            sam2_mask_files.append(filename)

            # Also save cropped image and mask for debugging
            safe_label = label.replace("/", "_").replace(" ", "_")
            cropped.save(img_output_dir / f"crop_{i:02d}_{safe_label}.jpg", quality=95)
            save_individual_mask(crop_mask, img_output_dir, label, i, "sam2_crop_mask")

            print(f"      Saved: {filename}")
        else:
            sam2_masks.append(None)
            print(f"      Failed to generate mask")

    # Create SAM2 mask overlay
    valid_sam2_masks = [m for m in sam2_masks if m is not None]
    valid_labels = [labels[i] for i, m in enumerate(sam2_masks) if m is not None]

    if valid_sam2_masks:
        sam2_overlay = create_mask_overlay(image, valid_sam2_masks, valid_labels)
        sam2_overlay.save(img_output_dir / "03_sam2_masks.png")
        print(f"\n  Overlay saved: 03_sam2_masks.png")

    # Step 3: YOLO-seg mask → SAM-3D 직접 변환 테스트 (대표 객체 1개)
    print("\n--- YOLO-seg Mask → SAM-3D Direct Test ---")
    sam3d_result = None
    sam3d_success = False
    sam3d_label = None

    # 가장 큰 마스크 찾기 (픽셀 수 기준)
    valid_masks = [(i, m, l) for i, (m, l) in enumerate(zip(yolo_masks, labels)) if m is not None]
    if valid_masks:
        # 마스크 픽셀 수로 정렬, 가장 큰 것 선택
        best_idx, best_mask, best_label = max(valid_masks, key=lambda x: np.sum(x[1] > 0))
        sam3d_label = best_label

        pixel_count = np.sum(best_mask > 0)
        print(f"  Selected: {best_idx:02d} - {best_label} (largest mask)")
        print(f"  Mask pixels: {pixel_count}")

        # 1. YOLO-seg 마스크를 파일로 저장
        safe_label = best_label.replace("/", "_").replace(" ", "_")
        mask_save_path = img_output_dir / f"yolo_mask_{best_idx:02d}_{safe_label}.png"
        Image.fromarray(best_mask).save(mask_save_path)
        print(f"  Saved mask: {mask_save_path.name}")

        # 2. SAM-3D 변환기 초기화
        sam3d_output_dir = img_output_dir / "sam3d_outputs"
        sam3d_output_dir.mkdir(parents=True, exist_ok=True)
        converter = SAM3DConverter(assets_dir=str(sam3d_output_dir))

        # 3. SAM-3D 변환 호출
        print(f"  Running SAM-3D conversion...")
        try:
            sam3d_result = converter.convert(
                image_path=str(image_path),
                mask_path=str(mask_save_path),
                seed=42
            )

            if sam3d_result.success:
                sam3d_success = True
                print(f"  ✓ SAM-3D 성공!")
                if sam3d_result.ply_size_bytes:
                    print(f"    PLY: {sam3d_result.ply_size_bytes} bytes")
                if sam3d_result.gif_b64:
                    # Save GIF for visual inspection
                    import base64
                    gif_bytes = base64.b64decode(sam3d_result.gif_b64)
                    gif_path = sam3d_output_dir / f"preview_{best_idx:02d}_{safe_label}.gif"
                    with open(gif_path, "wb") as f:
                        f.write(gif_bytes)
                    print(f"    GIF: {len(sam3d_result.gif_b64)} chars (base64) → {gif_path.name}")
                if sam3d_result.mesh_url:
                    print(f"    Mesh: {sam3d_result.mesh_url}")
                if sam3d_result.ply_url:
                    print(f"    PLY URL: {sam3d_result.ply_url}")
            else:
                print(f"  ✗ SAM-3D 실패: {sam3d_result.error}")
        except Exception as e:
            print(f"  ✗ SAM-3D 예외 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  No valid YOLO-seg masks found!")

    # Summary
    summary = {
        "image": img_name,
        "detections": num_detections,
        "yolo_masks": len(yolo_mask_files),
        "sam2_masks": len(sam2_mask_files),
        "sam3d_success": sam3d_success,
        "sam3d_label": sam3d_label,
        "sam3d_ply_bytes": sam3d_result.ply_size_bytes if sam3d_result and sam3d_result.ply_size_bytes else None,
        "output_dir": str(img_output_dir)
    }

    print(f"\n--- Summary ---")
    print(f"  Detections: {num_detections}")
    print(f"  YOLOE masks: {len(yolo_mask_files)}")
    print(f"  SAM2 masks: {len(sam2_mask_files)}")
    print(f"  SAM-3D: {'✓ Success' if sam3d_success else '✗ Failed'}" + (f" ({sam3d_label})" if sam3d_label else ""))

    return summary


def main():
    print("\n" + "=" * 60)
    print(" YOLOE vs SAM2 Mask Comparison Test")
    print("=" * 60)

    # Verify test images exist
    images_to_process = []
    for img_path in TEST_IMAGES:
        full_path = PROJECT_ROOT / img_path
        if full_path.exists():
            images_to_process.append(full_path)
            print(f"[OK] {img_path}")
        else:
            print(f"[SKIP] Not found: {img_path}")

    if not images_to_process:
        print("\n[ERROR] No test images found!")
        return 1

    # Output directory
    output_dir = PROJECT_ROOT / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize models
    print("\n--- Initializing Models ---")

    print("\nLoading YOLOE detector...")
    yolo_detector = YOLOEDetector(device_id=0)

    print("\nLoading SAM2 segmenter...")
    sam2_segmenter = SAM2Segmenter()

    # Process each image
    results = []
    for img_path in images_to_process:
        try:
            result = process_image(
                img_path,
                output_dir,
                yolo_detector,
                sam2_segmenter
            )
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "image": img_path.stem,
                "error": str(e)
            })

    # Final summary
    print("\n" + "=" * 60)
    print(" Test Complete - Results Summary")
    print("=" * 60)

    for result in results:
        img_name = result.get("image", "unknown")
        if "error" in result:
            print(f"\n{img_name}: ERROR - {result['error']}")
        else:
            print(f"\n{img_name}:")
            print(f"  - Detections: {result.get('detections', 0)}")
            print(f"  - YOLOE masks: {result.get('yolo_masks', 0)}")
            print(f"  - SAM2 masks: {result.get('sam2_masks', 0)}")
            sam3d_status = "✓ Success" if result.get("sam3d_success") else "✗ Failed"
            sam3d_label = result.get("sam3d_label", "N/A")
            sam3d_ply = result.get("sam3d_ply_bytes")
            print(f"  - SAM-3D: {sam3d_status} ({sam3d_label})" + (f" - PLY: {sam3d_ply} bytes" if sam3d_ply else ""))

    print(f"\nOutputs saved to: {output_dir}")
    print("\nTo compare masks:")
    print("  - YOLOE: test_outputs/{image}/02_yoloe_masks.png")
    print("  - SAM2:  test_outputs/{image}/03_sam2_masks.png")
    print("\nSAM-3D outputs:")
    print("  - 3D models: test_outputs/{image}/sam3d_outputs/")
    print("  - Preview GIF: test_outputs/{image}/sam3d_outputs/preview_*.gif")

    return 0


if __name__ == "__main__":
    sys.exit(main())
