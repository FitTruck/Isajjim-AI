#!/usr/bin/env python3
"""
Multi-GPU Pipeline Full Test (8 GPUs)

8개 GPU에서 실제 FurniturePipeline 전 과정을 테스트합니다.

테스트 항목:
1. 8개 GPU에 YOLOE 모델 사전 로드 (모델 로딩 시간 측정)
2. ai/imgs 이미지들을 8개 GPU에 분배하여 병렬 처리
3. 순수 추론 시간 측정 (모델 로딩 제외)
4. 각 단계별 시간 분석

파이프라인 단계:
- Stage 1: 이미지 로드 (로컬 파일)
- Stage 2: YOLOE-seg 탐지 (bbox + class + 세그멘테이션 마스크)
- Stage 3: DB 매칭 (한국어 라벨)
- (SAM-3D는 외부 API이므로 제외)
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 설정 (torch import 전에)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["OMP_NUM_THREADS"] = "4"

import torch
from PIL import Image

from ai.gpu import GPUPoolManager, initialize_gpu_pool, get_gpu_pool
from ai.pipeline import FurniturePipeline
from ai.config import Config


@dataclass
class TimingResult:
    """타이밍 측정 결과"""
    image_name: str
    gpu_id: int
    image_load_ms: float = 0.0
    yolo_detect_ms: float = 0.0
    mask_process_ms: float = 0.0
    total_inference_ms: float = 0.0
    objects_detected: int = 0
    object_labels: List[str] = field(default_factory=list)


def print_header(title: str):
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_timing_table(results: List[TimingResult]):
    """타이밍 결과 테이블 출력"""
    print("\n" + "-" * 100)
    print(f"{'Image':<35} {'GPU':>4} {'Load':>10} {'YOLO':>10} {'Mask':>10} {'Total':>10} {'Objects':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.gpu_id):
        print(f"{r.image_name:<35} {r.gpu_id:>4} "
              f"{r.image_load_ms:>8.1f}ms {r.yolo_detect_ms:>8.1f}ms "
              f"{r.mask_process_ms:>8.1f}ms {r.total_inference_ms:>8.1f}ms "
              f"{r.objects_detected:>8}")

    print("-" * 100)

    # 평균 계산
    if results:
        avg_load = sum(r.image_load_ms for r in results) / len(results)
        avg_yolo = sum(r.yolo_detect_ms for r in results) / len(results)
        avg_mask = sum(r.mask_process_ms for r in results) / len(results)
        avg_total = sum(r.total_inference_ms for r in results) / len(results)
        total_objects = sum(r.objects_detected for r in results)

        print(f"{'AVERAGE':<35} {'':>4} "
              f"{avg_load:>8.1f}ms {avg_yolo:>8.1f}ms "
              f"{avg_mask:>8.1f}ms {avg_total:>8.1f}ms "
              f"{total_objects:>8} total")


class PipelineProfiler:
    """파이프라인 프로파일러 - 각 단계별 시간 측정"""

    def __init__(self, pipeline: FurniturePipeline, gpu_id: int):
        self.pipeline = pipeline
        self.gpu_id = gpu_id

    def process_image_with_timing(self, image_path: Path) -> TimingResult:
        """이미지 처리 및 각 단계별 시간 측정"""
        result = TimingResult(
            image_name=image_path.name,
            gpu_id=self.gpu_id
        )

        total_start = time.perf_counter()

        # Stage 1: 이미지 로드
        load_start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        result.image_load_ms = (time.perf_counter() - load_start) * 1000

        # Stage 2: YOLO 탐지 (세그멘테이션 마스크 포함)
        yolo_start = time.perf_counter()
        detected_objects = self.pipeline.detect_objects(image)
        result.yolo_detect_ms = (time.perf_counter() - yolo_start) * 1000

        # Stage 3: 마스크 처리 (base64 변환)
        mask_start = time.perf_counter()
        for obj in detected_objects:
            if obj.yolo_mask is not None:
                _ = self.pipeline._yolo_mask_to_base64(obj.yolo_mask)
        result.mask_process_ms = (time.perf_counter() - mask_start) * 1000

        # 결과 집계
        result.total_inference_ms = (time.perf_counter() - total_start) * 1000
        result.objects_detected = len(detected_objects)
        result.object_labels = [obj.label for obj in detected_objects]

        return result


async def test_model_loading_time():
    """테스트 1: 8개 GPU에 모델 로딩 시간 측정"""
    print_header("Test 1: Model Loading Time (8 GPUs)")

    gpu_count = torch.cuda.device_count()
    gpu_ids = list(range(min(8, gpu_count)))

    print(f"\nInitializing GPU pool with {len(gpu_ids)} GPUs...")

    # GPU 풀 초기화
    pool = initialize_gpu_pool(gpu_ids=gpu_ids)

    # 각 GPU별 모델 로딩 시간 측정
    loading_times = {}

    print("\nLoading YOLOE models on each GPU (this will take time)...")
    print("-" * 60)

    total_load_start = time.perf_counter()

    for gpu_id in gpu_ids:
        print(f"\n[GPU {gpu_id}] Loading FurniturePipeline...", end=" ", flush=True)
        start = time.perf_counter()

        pipeline = FurniturePipeline(
            enable_3d_generation=False,  # 3D 생성 비활성화 (테스트용)
            device_id=gpu_id,
            gpu_pool=pool
        )

        elapsed = time.perf_counter() - start
        loading_times[gpu_id] = elapsed
        print(f"Done in {elapsed:.2f}s")

        # 풀에 파이프라인 등록
        pool.register_pipeline(gpu_id, pipeline)

    total_load_time = time.perf_counter() - total_load_start

    print("-" * 60)
    print(f"\nModel Loading Summary:")
    for gpu_id, t in loading_times.items():
        print(f"  GPU {gpu_id}: {t:.2f}s")
    print(f"\nTotal sequential loading time: {total_load_time:.2f}s")
    print(f"Average per GPU: {total_load_time / len(gpu_ids):.2f}s")

    # GPU 메모리 확인
    print("\nGPU Memory after model loading:")
    for gpu_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"  GPU {gpu_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return pool, loading_times


async def test_pure_inference_time(pool: GPUPoolManager):
    """테스트 2: 순수 추론 시간 측정 (모델 로딩 제외)"""
    print_header("Test 2: Pure Inference Time (Model Already Loaded)")

    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_files = sorted(imgs_dir.glob("*.jpg"))

    print(f"\nFound {len(image_files)} images")
    print(f"GPUs available: {len(pool.gpu_ids)}")

    # 워밍업 - 첫 번째 추론은 느릴 수 있음
    print("\nWarming up (first inference on each GPU)...")
    warmup_image = image_files[0]

    for gpu_id in pool.gpu_ids:
        pipeline = pool.get_pipeline(gpu_id)
        if pipeline:
            profiler = PipelineProfiler(pipeline, gpu_id)
            _ = profiler.process_image_with_timing(warmup_image)
            print(f"  GPU {gpu_id} warmed up")

    # 동기화
    for gpu_id in pool.gpu_ids:
        torch.cuda.synchronize(gpu_id)

    print("\n" + "=" * 70)
    print(" Sequential Processing (Single GPU baseline)")
    print("=" * 70)

    # 단일 GPU 순차 처리 (baseline)
    single_gpu_results = []
    single_gpu_pipeline = pool.get_pipeline(0)
    single_gpu_profiler = PipelineProfiler(single_gpu_pipeline, 0)

    single_start = time.perf_counter()
    for img_path in image_files:
        result = single_gpu_profiler.process_image_with_timing(img_path)
        single_gpu_results.append(result)
    single_total_time = time.perf_counter() - single_start

    print_timing_table(single_gpu_results)
    print(f"\nSingle GPU Total Time: {single_total_time*1000:.1f}ms ({single_total_time:.2f}s)")

    print("\n" + "=" * 70)
    print(" Parallel Processing (8 GPUs)")
    print("=" * 70)

    # 8개 GPU 병렬 처리
    parallel_results: List[TimingResult] = []

    async def process_on_gpu(img_path: Path, gpu_id: int) -> TimingResult:
        """특정 GPU에서 이미지 처리"""
        pipeline = pool.get_pipeline(gpu_id)
        profiler = PipelineProfiler(pipeline, gpu_id)
        return profiler.process_image_with_timing(img_path)

    # 이미지를 GPU에 라운드로빈 분배
    gpu_assignments = []
    for i, img_path in enumerate(image_files):
        gpu_id = pool.gpu_ids[i % len(pool.gpu_ids)]
        gpu_assignments.append((img_path, gpu_id))

    print(f"\nImage → GPU assignments:")
    for img_path, gpu_id in gpu_assignments:
        print(f"  {img_path.name} → GPU {gpu_id}")

    # 병렬 실행
    parallel_start = time.perf_counter()

    # asyncio로 병렬 실행 (실제로는 GIL 때문에 완전 병렬은 아님)
    # 하지만 GPU 연산은 GIL을 해제하므로 병렬 효과 있음
    tasks = []
    for img_path, gpu_id in gpu_assignments:
        tasks.append(process_on_gpu(img_path, gpu_id))

    parallel_results = await asyncio.gather(*tasks)
    parallel_total_time = time.perf_counter() - parallel_start

    print_timing_table(parallel_results)
    print(f"\n8-GPU Parallel Total Time: {parallel_total_time*1000:.1f}ms ({parallel_total_time:.2f}s)")

    # 성능 비교
    speedup = single_total_time / parallel_total_time if parallel_total_time > 0 else 0

    print("\n" + "=" * 70)
    print(" Performance Comparison")
    print("=" * 70)
    print(f"\n  Single GPU (sequential): {single_total_time*1000:.1f}ms")
    print(f"  8-GPU (parallel):        {parallel_total_time*1000:.1f}ms")
    print(f"  Speedup:                 {speedup:.2f}x")

    return single_gpu_results, parallel_results, single_total_time, parallel_total_time


async def test_throughput(pool: GPUPoolManager):
    """테스트 3: Throughput 측정 (이미지/초)"""
    print_header("Test 3: Throughput Measurement")

    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_files = sorted(imgs_dir.glob("*.jpg"))

    # 여러 번 반복하여 throughput 측정
    num_iterations = 3
    total_images = len(image_files) * num_iterations

    print(f"\nProcessing {len(image_files)} images × {num_iterations} iterations = {total_images} total")

    # 단일 GPU throughput
    print("\n[Single GPU Throughput]")
    single_pipeline = pool.get_pipeline(0)
    single_profiler = PipelineProfiler(single_pipeline, 0)

    single_start = time.perf_counter()
    for _ in range(num_iterations):
        for img_path in image_files:
            _ = single_profiler.process_image_with_timing(img_path)
    single_elapsed = time.perf_counter() - single_start
    single_throughput = total_images / single_elapsed

    print(f"  Total time: {single_elapsed:.2f}s")
    print(f"  Throughput: {single_throughput:.2f} images/sec")

    # 8 GPU throughput
    print("\n[8-GPU Parallel Throughput]")

    async def batch_process(iteration: int):
        tasks = []
        for i, img_path in enumerate(image_files):
            gpu_id = pool.gpu_ids[i % len(pool.gpu_ids)]
            pipeline = pool.get_pipeline(gpu_id)
            profiler = PipelineProfiler(pipeline, gpu_id)
            tasks.append(asyncio.to_thread(profiler.process_image_with_timing, img_path))
        return await asyncio.gather(*tasks)

    parallel_start = time.perf_counter()
    for i in range(num_iterations):
        await batch_process(i)
    parallel_elapsed = time.perf_counter() - parallel_start
    parallel_throughput = total_images / parallel_elapsed

    print(f"  Total time: {parallel_elapsed:.2f}s")
    print(f"  Throughput: {parallel_throughput:.2f} images/sec")

    print(f"\n  Throughput improvement: {parallel_throughput / single_throughput:.2f}x")

    return single_throughput, parallel_throughput


async def test_detection_results(pool: GPUPoolManager):
    """테스트 4: 탐지 결과 상세 분석"""
    print_header("Test 4: Detection Results Analysis")

    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_files = sorted(imgs_dir.glob("*.jpg"))

    print("\nDetected Objects per Image:")
    print("-" * 80)

    all_labels = []

    for i, img_path in enumerate(image_files):
        gpu_id = pool.gpu_ids[i % len(pool.gpu_ids)]
        pipeline = pool.get_pipeline(gpu_id)

        image = Image.open(img_path).convert("RGB")
        detected = pipeline.detect_objects(image)

        labels = [obj.label for obj in detected]
        all_labels.extend(labels)

        print(f"\n{img_path.name} (GPU {gpu_id}):")
        print(f"  Detected {len(detected)} objects: {labels}")

        for obj in detected:
            conf = obj.confidence
            bbox = obj.bbox
            has_mask = obj.yolo_mask is not None
            mask_size = obj.yolo_mask.shape if has_mask else "N/A"
            print(f"    - {obj.label}: conf={conf:.2f}, bbox={bbox}, mask={mask_size}")

    print("\n" + "-" * 80)
    print("\nLabel Distribution:")

    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in label_counts.most_common():
        bar = "█" * count
        print(f"  {label:<20} {bar} ({count})")

    return label_counts


async def main():
    """메인 테스트 실행"""
    print_header("Multi-GPU Pipeline Full Test (8 GPUs)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    results_summary = {}

    try:
        # 테스트 1: 모델 로딩 시간
        pool, loading_times = await test_model_loading_time()
        results_summary["model_loading"] = {
            "times": loading_times,
            "total": sum(loading_times.values())
        }

        # 테스트 2: 순수 추론 시간
        single_results, parallel_results, single_time, parallel_time = await test_pure_inference_time(pool)
        results_summary["inference"] = {
            "single_gpu_ms": single_time * 1000,
            "parallel_8gpu_ms": parallel_time * 1000,
            "speedup": single_time / parallel_time if parallel_time > 0 else 0
        }

        # 테스트 3: Throughput
        single_tp, parallel_tp = await test_throughput(pool)
        results_summary["throughput"] = {
            "single_gpu": single_tp,
            "parallel_8gpu": parallel_tp,
            "improvement": parallel_tp / single_tp if single_tp > 0 else 0
        }

        # 테스트 4: 탐지 결과
        label_counts = await test_detection_results(pool)
        results_summary["detection"] = dict(label_counts)

        # 최종 요약
        print_header("FINAL SUMMARY")

        print("\n📊 Model Loading:")
        print(f"   Total loading time: {results_summary['model_loading']['total']:.2f}s")
        print(f"   Average per GPU: {results_summary['model_loading']['total']/8:.2f}s")

        print("\n⚡ Pure Inference Time (excluding model loading):")
        print(f"   Single GPU: {results_summary['inference']['single_gpu_ms']:.1f}ms")
        print(f"   8-GPU Parallel: {results_summary['inference']['parallel_8gpu_ms']:.1f}ms")
        print(f"   Speedup: {results_summary['inference']['speedup']:.2f}x")

        print("\n🚀 Throughput:")
        print(f"   Single GPU: {results_summary['throughput']['single_gpu']:.2f} images/sec")
        print(f"   8-GPU Parallel: {results_summary['throughput']['parallel_8gpu']:.2f} images/sec")
        print(f"   Improvement: {results_summary['throughput']['improvement']:.2f}x")

        print("\n🔍 Total Objects Detected: ", sum(label_counts.values()))

        print("\n" + "=" * 70)
        print(" ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
