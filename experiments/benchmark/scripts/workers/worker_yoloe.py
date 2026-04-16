"""
YOLOE Inference Benchmark Worker — detect_smart() 추론 시간 측정

SAM3D 벤치마크와 동일한 실험 조건:
  - 동일 500개 Pix3D 샘플 (benchmark_samples.json)
  - torch.cuda.synchronize() + time.perf_counter() 정밀 타이밍
  - peak VRAM 추적 (torch.cuda.max_memory_allocated)
  - warmup 1회 후 측정 (latency 편향 방지)
  - 중단 복구 (이미 완료된 샘플 스킵)

Usage:
    python experiments/benchmark/scripts/workers/worker_yoloe.py \
        --gpu 0 \
        --samples data/benchmark_samples.json \
        --output results/yoloe.csv
"""

import sys
import os
import argparse

# Parse args early to set CUDA_VISIBLE_DEVICES before any torch import
parser = argparse.ArgumentParser(description="YOLOE Inference Benchmark")
parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
parser.add_argument("--samples", type=str, required=True, help="Path to benchmark_samples.json")
parser.add_argument("--output", type=str, required=True, help="Output CSV path")
parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (default: 0.10, matches production)")
args = parser.parse_args()

# CRITICAL: Set environment variables BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import csv
import gc
import json
import logging
import time

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_default_dtype(torch.float32)

from pathlib import Path
from PIL import Image

# Path setup — worker_ablation.py와 동일 패턴
# __file__ = experiments/benchmark/scripts/workers/worker_yoloe.py
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent  # experiments/benchmark/
PROJECT_ROOT = BENCHMARK_ROOT.parent.parent                      # Isajjim-AI/
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import PIX3D_DIR, FLUSH_INTERVAL, SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Worker:yoloe] %(message)s",
)
logger = logging.getLogger(__name__)


class YoloeWorker:
    """YOLOE detect_smart() 추론 시간 측정 워커."""

    def __init__(self, confidence_threshold: float = 0.10):
        self.confidence_threshold = confidence_threshold
        self.detector = None

    def load_model(self):
        """YOLOE 모델 로드."""
        logger.info("Loading YOLOE-seg model...")
        t_start = time.time()

        from ai.processors import YoloDetector

        self.detector = YoloDetector(
            confidence_threshold=self.confidence_threshold,
            device_id=0,  # CUDA_VISIBLE_DEVICES로 remap되므로 항상 0
        )

        torch.set_grad_enabled(False)

        load_time = time.time() - t_start
        vram_after_load = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Model loaded in {load_time:.1f}s, VRAM: {vram_after_load:.3f}GB")

    def run_single(self, sample: dict) -> dict:
        """단일 샘플 detect_smart() 추론 + 타이밍."""
        img_path = PIX3D_DIR / sample["img_path"]

        result = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "img_path": sample["img_path"],
            "img_width": 0,
            "img_height": 0,
            "num_objects": 0,
            "latency_seconds": 0.0,
            "vram_peak_mb": 0.0,
            "success": False,
            "error": "",
        }

        if not img_path.exists():
            result["error"] = "File not found"
            return result

        # Load image
        image = Image.open(img_path).convert("RGB")
        result["img_width"] = image.width
        result["img_height"] = image.height

        # Seed 설정 (재현성 — worker_ablation.py와 동일)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Reset VRAM stats
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure inference
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        try:
            with torch.no_grad():
                detection = self.detector.detect_smart(image, return_masks=True)

            torch.cuda.synchronize()
            latency = time.perf_counter() - t_start
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB

            num_objects = 0
            if detection is not None and len(detection.get("boxes", [])) > 0:
                num_objects = len(detection["boxes"])

            result.update({
                "num_objects": num_objects,
                "latency_seconds": round(latency, 4),
                "vram_peak_mb": round(vram_peak, 1),
                "success": True,
            })

        except torch.cuda.OutOfMemoryError as e:
            result["error"] = f"OOM: {e}"
            result["latency_seconds"] = round(time.perf_counter() - t_start, 4)
            result["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            result["error"] = str(e)[:200]
            result["latency_seconds"] = round(time.perf_counter() - t_start, 4)
            result["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

        return result


def load_completed_ids(output_path: str) -> set:
    """이미 처리된 sample_id 목록 (중단 복구용)."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["sample_id"]))
    return completed


def main():
    logger.info("=" * 60)
    logger.info("YOLOE Inference Benchmark")
    logger.info(f"  GPU: {args.gpu}")
    logger.info(f"  Conf threshold: {args.conf}")
    logger.info(f"  Samples: {args.samples}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    # GPU check
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # Load samples
    with open(args.samples) as f:
        samples = json.load(f)
    logger.info(f"Total samples: {len(samples)}")

    # 중단 복구
    completed = load_completed_ids(args.output)
    if completed:
        logger.info(f"Resuming: {len(completed)} already completed, skipping")
    remaining = [s for s in samples if s["sample_id"] not in completed]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All samples already completed!")
        return

    # Load model
    worker = YoloeWorker(confidence_threshold=args.conf)
    worker.load_model()

    # Warmup (별도 dummy — latency 편향 방지)
    logger.info("Warmup run (not recorded)...")
    warmup_img_path = PIX3D_DIR / remaining[0]["img_path"]
    if warmup_img_path.exists():
        warmup_image = Image.open(warmup_img_path).convert("RGB")
        _ = worker.detector.detect_smart(warmup_image, return_masks=True)
        torch.cuda.synchronize()
    logger.info("Warmup done")

    # CSV output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id", "category", "img_path",
        "img_width", "img_height", "num_objects",
        "latency_seconds", "vram_peak_mb", "success", "error",
    ]

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    success_count = 0
    fail_count = 0
    latencies = []
    t_total_start = time.time()

    with open(output_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, sample in enumerate(remaining):
            sid = sample["sample_id"]
            cat = sample["category"]

            result = worker.run_single(sample)
            writer.writerow(result)

            if result["success"]:
                success_count += 1
                latencies.append(result["latency_seconds"])
                status = f"{result['latency_seconds']:.4f}s ({result['num_objects']} objs)"
            else:
                fail_count += 1
                status = f"FAIL: {result['error'][:40]}"

            # Progress
            done = i + 1
            total = len(remaining)
            avg_lat = np.mean(latencies) if latencies else 0

            if done % 50 == 0 or done == total:
                logger.info(
                    f"[{done}/{total}] {cat}/{sid} → {status} | "
                    f"avg={avg_lat:.4f}s | "
                    f"ok={success_count} fail={fail_count}"
                )

            # Flush (장시간 실행 안정성)
            if done % FLUSH_INTERVAL == 0:
                csv_file.flush()
                os.fsync(csv_file.fileno())

    # Summary
    elapsed_total = time.time() - t_total_start
    logger.info(f"\n{'=' * 60}")
    logger.info("YOLOE Benchmark Complete")
    logger.info(f"  성공: {success_count}/{len(remaining)}")
    logger.info(f"  실패: {fail_count}")
    if latencies:
        lat_arr = np.array(latencies)
        logger.info(f"  평균 Latency: {lat_arr.mean():.4f}s")
        logger.info(f"  중간값 Latency: {np.median(lat_arr):.4f}s")
        logger.info(f"  표준편차: {lat_arr.std():.4f}s")
        logger.info(f"  P95 Latency: {np.percentile(lat_arr, 95):.4f}s")
        logger.info(f"  최소/최대: {lat_arr.min():.4f}s / {lat_arr.max():.4f}s")
    logger.info(f"  총 소요: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}m)")
    logger.info(f"  결과: {args.output}")


if __name__ == "__main__":
    main()
