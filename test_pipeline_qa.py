#!/usr/bin/env python3
"""
Pipeline V2 QA Test Script

전체 파이프라인 품질 검증:
1. YOLOE-seg 탐지 정확도
2. YOLOE-seg → SAM-3D 직접 연결
3. 3D 모델 생성 성공률
4. 부피 계산 정확도
5. 전체 데이터 플로우 검증

Usage:
    python test_pipeline_qa.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Environment setup BEFORE imports
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from PIL import Image

# Set PyTorch defaults
torch.set_default_dtype(torch.float32)

# Project root
PROJECT_ROOT = Path(__file__).parent

# Test images
TEST_IMAGES = [
    "ai/imgs/bedroom-5772286_1920.jpg",
    "ai/imgs/bed-1834327_1920.jpg",
    "ai/imgs/apartment-1835482_1920.jpg",
]


@dataclass
class QAStageResult:
    """Individual stage test result"""
    stage_name: str
    passed: bool = False
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class QAImageResult:
    """Single image QA result"""
    image_name: str
    image_path: str
    stages: List[QAStageResult] = field(default_factory=list)
    overall_passed: bool = False
    total_duration_seconds: float = 0.0


@dataclass
class QAReport:
    """Full QA report"""
    timestamp: str
    pipeline_version: str = "V2"
    total_images: int = 0
    passed_images: int = 0
    failed_images: int = 0
    image_results: List[QAImageResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


class PipelineQATester:
    """Pipeline V2 QA Tester"""

    def __init__(self):
        self.report = QAReport(
            timestamp=datetime.now().isoformat()
        )
        self.yolo_detector = None
        self.sam3d_converter = None
        self.volume_calculator = None
        self.movability_checker = None

    def _init_components(self):
        """Initialize pipeline components"""
        print("\n--- Initializing Pipeline Components ---")

        # YOLO Detector
        print("Loading YOLOE-seg detector...")
        from ai.processors import YoloDetector
        self.yolo_detector = YoloDetector(device_id=0)
        print(f"  Model loaded: {self.yolo_detector.model_path}")

        # SAM-3D Converter
        print("Loading SAM-3D converter...")
        from ai.processors import SAM3DConverter
        self.sam3d_converter = SAM3DConverter(
            assets_dir=str(PROJECT_ROOT / "test_outputs" / "qa_assets")
        )
        print(f"  Assets dir: {self.sam3d_converter.assets_dir}")

        # Volume Calculator
        print("Loading Volume calculator...")
        from ai.processors import VolumeCalculator
        self.volume_calculator = VolumeCalculator()

        # Movability Checker
        print("Loading Movability checker...")
        from ai.processors import MovabilityChecker
        self.movability_checker = MovabilityChecker()
        print(f"  Class mappings: {len(self.movability_checker.class_map)}")

        print("All components loaded successfully\n")

    def test_stage1_yolo_detection(self, image: Image.Image) -> QAStageResult:
        """
        Stage 1: YOLOE-seg Detection Test

        Validates:
        - Detection returns results
        - Bounding boxes are valid
        - Labels are mapped correctly
        - Segmentation masks are present
        """
        start = time.time()
        result = QAStageResult(stage_name="1_YOLO_Detection")

        try:
            # Run detection
            detection = self.yolo_detector.detect_smart(image, return_masks=True)

            if detection is None or len(detection.get("boxes", [])) == 0:
                result.passed = False
                result.error = "No objects detected"
                return result

            boxes = detection["boxes"]
            labels = detection["labels"]
            scores = detection["scores"]
            masks = detection.get("masks", [])

            # Validate results
            num_detections = len(boxes)
            num_with_masks = sum(1 for m in masks if m is not None)
            valid_boxes = sum(1 for b in boxes if b[2] > b[0] and b[3] > b[1])

            result.details = {
                "total_detections": num_detections,
                "detections_with_masks": num_with_masks,
                "valid_boxes": valid_boxes,
                "labels": labels[:5],  # First 5 labels
                "avg_confidence": float(np.mean(scores)) if len(scores) > 0 else 0,
                "mask_coverage": num_with_masks / num_detections if num_detections > 0 else 0
            }

            # Pass criteria: at least 1 detection with mask
            result.passed = num_with_masks > 0 and valid_boxes == num_detections

        except Exception as e:
            result.passed = False
            result.error = str(e)

        result.duration_seconds = time.time() - start
        return result

    def test_stage2_db_matching(self, labels: List[str]) -> QAStageResult:
        """
        Stage 2: DB Matching Test

        Validates:
        - Labels are mapped to DB keys
        - is_movable determination works
        """
        start = time.time()
        result = QAStageResult(stage_name="2_DB_Matching")

        try:
            matched_count = 0
            movable_count = 0
            unmapped_labels = []

            for label in labels:
                # Check DB mapping
                movability = self.movability_checker.check_from_label(label, 0.9)

                if movability.db_key:
                    matched_count += 1
                    if movability.is_movable:
                        movable_count += 1
                else:
                    unmapped_labels.append(label)

            result.details = {
                "total_labels": len(labels),
                "matched_to_db": matched_count,
                "movable_objects": movable_count,
                "unmapped_labels": unmapped_labels[:5],
                "match_rate": matched_count / len(labels) if labels else 0
            }

            # Pass criteria: at least 50% of labels mapped
            result.passed = matched_count > 0 and (matched_count / len(labels) >= 0.5 if labels else False)

        except Exception as e:
            result.passed = False
            result.error = str(e)

        result.duration_seconds = time.time() - start
        return result

    def test_stage3_mask_quality(self, masks: List[np.ndarray], image_size: tuple) -> QAStageResult:
        """
        Stage 3: Mask Quality Test

        Validates:
        - Masks have correct dimensions
        - Masks have sufficient pixel coverage
        - Masks are valid binary masks
        """
        start = time.time()
        result = QAStageResult(stage_name="3_Mask_Quality")

        try:
            valid_masks = []
            img_w, img_h = image_size

            for i, mask in enumerate(masks):
                if mask is None:
                    continue

                # Check dimensions
                if mask.shape != (img_h, img_w):
                    continue

                # Check pixel coverage
                pixel_count = np.sum(mask > 0)
                coverage = pixel_count / (img_w * img_h)

                if pixel_count > 100 and coverage < 0.9:  # Not too small, not entire image
                    valid_masks.append({
                        "index": i,
                        "pixel_count": int(pixel_count),
                        "coverage": float(coverage)
                    })

            result.details = {
                "total_masks": len(masks),
                "valid_masks": len(valid_masks),
                "avg_coverage": np.mean([m["coverage"] for m in valid_masks]) if valid_masks else 0,
                "mask_samples": valid_masks[:3]
            }

            # Pass criteria: at least 1 valid mask
            result.passed = len(valid_masks) > 0

        except Exception as e:
            result.passed = False
            result.error = str(e)

        result.duration_seconds = time.time() - start
        return result

    def test_stage4_sam3d_conversion(
        self,
        image_path: str,
        mask: np.ndarray,
        output_dir: Path
    ) -> QAStageResult:
        """
        Stage 4: SAM-3D Conversion Test

        Validates:
        - 3D conversion succeeds
        - PLY file is generated
        - GIF preview is generated
        - GLB mesh is generated (optional)
        """
        start = time.time()
        result = QAStageResult(stage_name="4_SAM3D_Conversion")

        try:
            # Save mask to temp file
            mask_path = output_dir / "qa_test_mask.png"
            Image.fromarray(mask).save(mask_path)

            # Run SAM-3D conversion
            sam3d_result = self.sam3d_converter.convert(
                image_path=image_path,
                mask_path=str(mask_path),
                seed=42
            )

            result.details = {
                "success": sam3d_result.success,
                "ply_size_bytes": sam3d_result.ply_size_bytes,
                "has_gif": sam3d_result.gif_b64 is not None,
                "has_mesh": sam3d_result.mesh_url is not None,
                "error": sam3d_result.error
            }

            # Save GIF if available
            if sam3d_result.gif_b64:
                import base64
                gif_path = output_dir / "qa_preview.gif"
                with open(gif_path, "wb") as f:
                    f.write(base64.b64decode(sam3d_result.gif_b64))
                result.details["gif_path"] = str(gif_path)

            # Pass criteria: successful conversion with PLY
            result.passed = sam3d_result.success and sam3d_result.ply_size_bytes is not None

        except Exception as e:
            result.passed = False
            result.error = str(e)
            import traceback
            traceback.print_exc()

        result.duration_seconds = time.time() - start
        return result

    def test_stage5_volume_calculation(self, ply_b64: Optional[str]) -> QAStageResult:
        """
        Stage 5: Volume Calculation Test

        Validates:
        - Volume calculator processes PLY
        - Relative dimensions are extracted
        - Ratios are reasonable
        """
        start = time.time()
        result = QAStageResult(stage_name="5_Volume_Calculation")

        if not ply_b64:
            result.passed = False
            result.error = "No PLY data available"
            return result

        try:
            import base64
            import tempfile

            # Save PLY to temp file
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                tmp.write(base64.b64decode(ply_b64))
                ply_path = tmp.name

            # Calculate dimensions
            dims = self.volume_calculator.calculate_from_ply(ply_path)

            os.unlink(ply_path)

            if dims is None:
                result.passed = False
                result.error = "Volume calculation returned None"
                return result

            result.details = {
                "relative_width": dims.get("relative_width"),
                "relative_height": dims.get("relative_height"),
                "relative_depth": dims.get("relative_depth"),
                "ratio": dims.get("ratio"),
                "bounding_box": dims.get("bounding_box")
            }

            # Pass criteria: all dimensions are positive
            ratio = dims.get("ratio", {})
            result.passed = all(v > 0 for v in ratio.values()) if ratio else False

        except Exception as e:
            result.passed = False
            result.error = str(e)

        result.duration_seconds = time.time() - start
        return result

    def test_image(self, image_path: Path) -> QAImageResult:
        """Run all QA tests for a single image"""
        print(f"\n{'='*60}")
        print(f"Testing: {image_path.name}")
        print(f"{'='*60}")

        img_result = QAImageResult(
            image_name=image_path.stem,
            image_path=str(image_path)
        )
        start_time = time.time()

        # Create output directory
        output_dir = PROJECT_ROOT / "test_outputs" / "qa" / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")

        # Stage 1: YOLO Detection
        print("\n[Stage 1] YOLOE-seg Detection...")
        stage1 = self.test_stage1_yolo_detection(image)
        img_result.stages.append(stage1)
        self._print_stage_result(stage1)

        if not stage1.passed:
            img_result.overall_passed = False
            img_result.total_duration_seconds = time.time() - start_time
            return img_result

        # Get detection results for subsequent tests
        detection = self.yolo_detector.detect_smart(image, return_masks=True)
        labels = detection["labels"]
        masks = detection.get("masks", [])

        # Stage 2: DB Matching
        print("\n[Stage 2] DB Matching...")
        stage2 = self.test_stage2_db_matching(labels)
        img_result.stages.append(stage2)
        self._print_stage_result(stage2)

        # Stage 3: Mask Quality
        print("\n[Stage 3] Mask Quality...")
        stage3 = self.test_stage3_mask_quality(masks, image.size)
        img_result.stages.append(stage3)
        self._print_stage_result(stage3)

        if not stage3.passed:
            img_result.overall_passed = False
            img_result.total_duration_seconds = time.time() - start_time
            return img_result

        # Find best mask for SAM-3D test
        best_mask = None
        best_pixel_count = 0
        for mask in masks:
            if mask is not None:
                pixel_count = np.sum(mask > 0)
                if pixel_count > best_pixel_count:
                    best_pixel_count = pixel_count
                    best_mask = mask

        # Stage 4: SAM-3D Conversion
        print("\n[Stage 4] SAM-3D Conversion...")
        stage4 = self.test_stage4_sam3d_conversion(
            str(image_path),
            best_mask,
            output_dir
        )
        img_result.stages.append(stage4)
        self._print_stage_result(stage4)

        # Stage 5: Volume Calculation (if SAM-3D succeeded)
        if stage4.passed:
            print("\n[Stage 5] Volume Calculation...")
            # Get PLY from SAM-3D result
            sam3d_result = self.sam3d_converter.convert(
                image_path=str(image_path),
                mask_path=str(output_dir / "qa_test_mask.png"),
                seed=42
            )
            stage5 = self.test_stage5_volume_calculation(sam3d_result.ply_b64)
            img_result.stages.append(stage5)
            self._print_stage_result(stage5)

        # Determine overall result
        img_result.overall_passed = all(s.passed for s in img_result.stages)
        img_result.total_duration_seconds = time.time() - start_time

        return img_result

    def _print_stage_result(self, stage: QAStageResult):
        """Print stage result to console"""
        status = "✓ PASS" if stage.passed else "✗ FAIL"
        print(f"  {status} ({stage.duration_seconds:.2f}s)")

        if stage.details:
            for key, value in list(stage.details.items())[:3]:
                print(f"    - {key}: {value}")

        if stage.error:
            print(f"    - Error: {stage.error}")

    def run_full_qa(self) -> QAReport:
        """Run full QA test suite"""
        print("\n" + "=" * 60)
        print(" Pipeline V2 Full QA Test")
        print("=" * 60)

        # Initialize components
        self._init_components()

        # Find test images
        images_to_test = []
        for img_path in TEST_IMAGES:
            full_path = PROJECT_ROOT / img_path
            if full_path.exists():
                images_to_test.append(full_path)
                print(f"[OK] {img_path}")
            else:
                print(f"[SKIP] Not found: {img_path}")

        if not images_to_test:
            print("\n[ERROR] No test images found!")
            return self.report

        self.report.total_images = len(images_to_test)

        # Test each image
        for img_path in images_to_test:
            try:
                img_result = self.test_image(img_path)
                self.report.image_results.append(img_result)

                if img_result.overall_passed:
                    self.report.passed_images += 1
                else:
                    self.report.failed_images += 1
            except Exception as e:
                print(f"\n[ERROR] Testing {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
                self.report.failed_images += 1

        # Generate summary
        self._generate_summary()

        return self.report

    def _generate_summary(self):
        """Generate QA summary"""
        self.report.summary = {
            "total_images": self.report.total_images,
            "passed": self.report.passed_images,
            "failed": self.report.failed_images,
            "pass_rate": self.report.passed_images / self.report.total_images if self.report.total_images > 0 else 0,
            "pipeline_version": "V2 (YOLOE-seg → SAM-3D Direct)",
            "stages_tested": [
                "1_YOLO_Detection",
                "2_DB_Matching",
                "3_Mask_Quality",
                "4_SAM3D_Conversion",
                "5_Volume_Calculation"
            ]
        }

    def save_report(self, output_path: Optional[Path] = None):
        """Save QA report to JSON file"""
        if output_path is None:
            output_path = PROJECT_ROOT / "test_outputs" / "qa" / "qa_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict
        report_dict = {
            "timestamp": self.report.timestamp,
            "pipeline_version": self.report.pipeline_version,
            "total_images": self.report.total_images,
            "passed_images": self.report.passed_images,
            "failed_images": self.report.failed_images,
            "summary": self.report.summary,
            "image_results": [
                {
                    "image_name": r.image_name,
                    "image_path": r.image_path,
                    "overall_passed": r.overall_passed,
                    "total_duration_seconds": r.total_duration_seconds,
                    "stages": [
                        {
                            "stage_name": s.stage_name,
                            "passed": s.passed,
                            "details": s.details,
                            "error": s.error,
                            "duration_seconds": s.duration_seconds
                        }
                        for s in r.stages
                    ]
                }
                for r in self.report.image_results
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"\nQA Report saved to: {output_path}")
        return output_path


def main():
    tester = PipelineQATester()
    report = tester.run_full_qa()

    # Print final summary
    print("\n" + "=" * 60)
    print(" QA Test Complete - Final Summary")
    print("=" * 60)

    print(f"\nPipeline Version: {report.pipeline_version}")
    print(f"Total Images: {report.total_images}")
    print(f"Passed: {report.passed_images}")
    print(f"Failed: {report.failed_images}")
    print(f"Pass Rate: {report.summary.get('pass_rate', 0) * 100:.1f}%")

    print("\nImage Results:")
    for img_result in report.image_results:
        status = "✓" if img_result.overall_passed else "✗"
        print(f"  {status} {img_result.image_name}: {img_result.total_duration_seconds:.2f}s")
        for stage in img_result.stages:
            s_status = "✓" if stage.passed else "✗"
            print(f"      {s_status} {stage.stage_name}")

    # Save report
    report_path = tester.save_report()

    print(f"\nOutputs saved to: {PROJECT_ROOT / 'test_outputs' / 'qa'}")

    return 0 if report.passed_images == report.total_images else 1


if __name__ == "__main__":
    sys.exit(main())
