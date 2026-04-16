"""
Boxer A/B/C Benchmark Script

3가지 절대 치수 결정 방식의 정확도를 비교합니다:
  Mode A: Boxer 우선 + 규칙기반 fallback
  Mode B: Boxer only (실패 시 skip)
  Mode C: 규칙기반 only (baseline, V2.5 동일)

Usage:
    python scripts/benchmark_boxer_abc.py \
        --ground-truth tests/fixtures/ground_truth_dimensions.json \
        --image-dir tests/fixtures/benchmark_images/ \
        --device 0

Ground Truth JSON format:
    [
        {
            "image": "living_room_01.jpg",
            "objects": [
                {
                    "label": "SOFA",
                    "type": "THREE_SEATER_SOFA",
                    "bbox": [100, 200, 500, 600],
                    "width_mm": 3000,
                    "depth_mm": 1000,
                    "height_mm": 900
                }
            ]
        }
    ]
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# 프로젝트 루트를 PATH에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

_stage8 = importlib.import_module('ai.processors.8_absolute_volume_calculate')
_stage9 = importlib.import_module('ai.processors.9_boxer_dimension')

AbsoluteVolumeCalculator = _stage8.AbsoluteVolumeCalculator
BoxerDimensionEstimator = _stage9.BoxerDimensionEstimator
BoxerResult = _stage9.BoxerResult
is_boxer_available = _stage9.is_boxer_available

from ai.processors.dimension_strategy import DimensionMode, DimensionStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectGT:
    """Ground Truth 객체"""
    label: str
    type_name: Optional[str]
    bbox: List[int]
    width_mm: float
    depth_mm: float
    height_mm: float


@dataclass
class BenchmarkResult:
    """단일 객체 벤치마크 결과"""
    label: str
    gt_width: float
    gt_depth: float
    gt_height: float
    pred_width: float
    pred_depth: float
    pred_height: float
    width_error_pct: float
    depth_error_pct: float
    height_error_pct: float
    volume_error_pct: float
    skipped: bool = False


@dataclass
class ModeResults:
    """모드별 집계 결과"""
    mode: str
    results: List[BenchmarkResult] = field(default_factory=list)
    skipped_count: int = 0
    elapsed_seconds: float = 0.0

    @property
    def valid_results(self) -> List[BenchmarkResult]:
        return [r for r in self.results if not r.skipped]

    def mean_dim_error(self) -> float:
        valid = self.valid_results
        if not valid:
            return float("inf")
        errors = []
        for r in valid:
            errors.extend([r.width_error_pct, r.depth_error_pct, r.height_error_pct])
        return float(np.mean(errors))

    def std_dim_error(self) -> float:
        valid = self.valid_results
        if not valid:
            return 0.0
        errors = []
        for r in valid:
            errors.extend([r.width_error_pct, r.depth_error_pct, r.height_error_pct])
        return float(np.std(errors))

    def mean_volume_error(self) -> float:
        valid = self.valid_results
        if not valid:
            return float("inf")
        return float(np.mean([r.volume_error_pct for r in valid]))

    def std_volume_error(self) -> float:
        valid = self.valid_results
        if not valid:
            return 0.0
        return float(np.std([r.volume_error_pct for r in valid]))


def compute_error_pct(predicted: float, ground_truth: float) -> float:
    """오차율 (%) 계산"""
    if ground_truth == 0:
        return 0.0 if predicted == 0 else 100.0
    return abs(predicted - ground_truth) / ground_truth * 100


def evaluate_object(
    gt: ObjectGT,
    pred_width: float,
    pred_depth: float,
    pred_height: float,
) -> BenchmarkResult:
    """단일 객체에 대한 오차 계산"""
    w_err = compute_error_pct(pred_width, gt.width_mm)
    d_err = compute_error_pct(pred_depth, gt.depth_mm)
    h_err = compute_error_pct(pred_height, gt.height_mm)

    gt_vol = gt.width_mm * gt.depth_mm * gt.height_mm
    pred_vol = pred_width * pred_depth * pred_height
    vol_err = compute_error_pct(pred_vol, gt_vol)

    return BenchmarkResult(
        label=gt.label,
        gt_width=gt.width_mm,
        gt_depth=gt.depth_mm,
        gt_height=gt.height_mm,
        pred_width=pred_width,
        pred_depth=pred_depth,
        pred_height=pred_height,
        width_error_pct=w_err,
        depth_error_pct=d_err,
        height_error_pct=h_err,
        volume_error_pct=vol_err,
    )


def run_benchmark(
    image_dir: str,
    ground_truths: List[dict],
    device_id: int = 0,
) -> Dict[str, ModeResults]:
    """A/B/C 3가지 모드 벤치마크 실행"""

    # 초기화
    rule_based = AbsoluteVolumeCalculator()
    boxer_estimator = None
    if is_boxer_available():
        try:
            boxer_estimator = BoxerDimensionEstimator(device_id=device_id)
            logger.info("Boxer model loaded")
        except Exception as e:
            logger.error(f"Boxer init failed: {e}")

    modes = {
        "A": DimensionStrategy(mode=DimensionMode.BOXER_WITH_FALLBACK, rule_based=rule_based),
        "B": DimensionStrategy(mode=DimensionMode.BOXER_ONLY, rule_based=rule_based),
        "C": DimensionStrategy(mode=DimensionMode.RULE_BASED_ONLY, rule_based=rule_based),
    }

    all_results: Dict[str, ModeResults] = {
        "A": ModeResults(mode="A"),
        "B": ModeResults(mode="B"),
        "C": ModeResults(mode="C"),
    }

    for gt_entry in ground_truths:
        image_path = os.path.join(image_dir, gt_entry["image"])
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}, skipping")
            continue

        image = Image.open(image_path).convert("RGB")
        objects_gt = [
            ObjectGT(
                label=o["label"],
                type_name=o.get("type"),
                bbox=o["bbox"],
                width_mm=o["width_mm"],
                depth_mm=o["depth_mm"],
                height_mm=o["height_mm"],
            )
            for o in gt_entry["objects"]
        ]

        # Boxer 추론 (1회만, A/B 공유)
        bboxes = [o.bbox for o in objects_gt]
        boxer_results: List[Optional[BoxerResult]] = [None] * len(objects_gt)
        if boxer_estimator:
            try:
                raw = boxer_estimator.predict(image, bboxes)
                for i in range(min(len(raw), len(objects_gt))):
                    boxer_results[i] = raw[i]
            except Exception as e:
                logger.error(f"Boxer predict failed: {e}")

        # 각 모드 실행
        for mode_name, strategy in modes.items():
            t0 = time.perf_counter()
            for i, gt_obj in enumerate(objects_gt):
                # Mode C (규칙기반)용: GT 치수를 상대 치수로 사용
                # 실제 파이프라인에서는 SAM3D PLY에서 추출하지만,
                # 벤치마크에서는 GT의 비율 관계를 유지한 상대값 사용
                rel_w = gt_obj.width_mm / 1000.0
                rel_d = gt_obj.depth_mm / 1000.0
                rel_h = gt_obj.height_mm / 1000.0

                result = strategy.resolve(
                    boxer_result=boxer_results[i] if mode_name != "C" else None,
                    label=gt_obj.label,
                    type_name=gt_obj.type_name,
                    rel_width=rel_w, rel_depth=rel_d, rel_height=rel_h,
                )

                if result is None:
                    br = BenchmarkResult(
                        label=gt_obj.label,
                        gt_width=gt_obj.width_mm, gt_depth=gt_obj.depth_mm,
                        gt_height=gt_obj.height_mm,
                        pred_width=0, pred_depth=0, pred_height=0,
                        width_error_pct=100, depth_error_pct=100,
                        height_error_pct=100, volume_error_pct=100,
                        skipped=True,
                    )
                    all_results[mode_name].skipped_count += 1
                else:
                    br = evaluate_object(
                        gt_obj,
                        result.width_mm,
                        result.depth_mm,
                        result.height_mm,
                    )
                all_results[mode_name].results.append(br)
            all_results[mode_name].elapsed_seconds += time.perf_counter() - t0

    return all_results


def print_report(all_results: Dict[str, ModeResults]):
    """벤치마크 결과 출력"""
    total = max(len(list(all_results.values())[0].results), 1)

    print("\n" + "=" * 70)
    print(f"  Boxer A/B/C Benchmark (N={total} objects)")
    print("=" * 70)
    print(f"  {'Mode':<8} | {'치수 오차(%)':<16} | {'부피 오차(%)':<16} | {'실패율':<10} | {'시간(s)'}")
    print("-" * 70)

    for mode_name in ["A", "B", "C"]:
        r = all_results[mode_name]
        dim_mean = r.mean_dim_error()
        dim_std = r.std_dim_error()
        vol_mean = r.mean_volume_error()
        vol_std = r.std_volume_error()
        skip_pct = r.skipped_count * 100 // total

        mode_label = {
            "A": "A(Box+FB)",
            "B": "B(Box)",
            "C": "C(Rules)",
        }[mode_name]

        print(
            f"  {mode_label:<8} | "
            f"{dim_mean:6.1f} ± {dim_std:5.1f}   | "
            f"{vol_mean:6.1f} ± {vol_std:5.1f}   | "
            f"{r.skipped_count}/{total} ({skip_pct}%) | "
            f"{r.elapsed_seconds:.2f}"
        )

    print("=" * 70)

    # DimensionStrategy 통계
    print("\n  DimensionStrategy Stats:")
    for mode_name in ["A", "B", "C"]:
        # Reconstruct stats from results
        r = all_results[mode_name]
        boxer_used = len(r.valid_results) - r.skipped_count
        print(f"    Mode {mode_name}: {len(r.results)} total, {r.skipped_count} skipped")

    # 개별 객체 결과 (상위 오차)
    print("\n  Worst cases (Mode A):")
    a_valid = all_results["A"].valid_results
    if a_valid:
        sorted_by_vol = sorted(a_valid, key=lambda x: x.volume_error_pct, reverse=True)
        for br in sorted_by_vol[:5]:
            print(
                f"    {br.label}: "
                f"GT=({br.gt_width:.0f},{br.gt_depth:.0f},{br.gt_height:.0f}) "
                f"Pred=({br.pred_width:.0f},{br.pred_depth:.0f},{br.pred_height:.0f}) "
                f"VolErr={br.volume_error_pct:.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="Boxer A/B/C Benchmark")
    parser.add_argument(
        "--ground-truth", required=True,
        help="Ground truth JSON file path"
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Directory containing benchmark images"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file for results (optional)"
    )
    args = parser.parse_args()

    # Ground truth 로드
    with open(args.ground_truth) as f:
        ground_truths = json.load(f)

    logger.info(f"Loaded {len(ground_truths)} images from {args.ground_truth}")

    # 벤치마크 실행
    results = run_benchmark(args.image_dir, ground_truths, device_id=args.device)

    # 결과 출력
    print_report(results)

    # JSON 저장 (선택)
    if args.output:
        output_data = {}
        for mode_name, mode_result in results.items():
            output_data[mode_name] = {
                "mean_dim_error_pct": mode_result.mean_dim_error(),
                "std_dim_error_pct": mode_result.std_dim_error(),
                "mean_volume_error_pct": mode_result.mean_volume_error(),
                "std_volume_error_pct": mode_result.std_volume_error(),
                "skipped_count": mode_result.skipped_count,
                "total_count": len(mode_result.results),
                "elapsed_seconds": mode_result.elapsed_seconds,
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
