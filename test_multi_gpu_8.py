#!/usr/bin/env python3
"""
Multi-GPU Architecture Test Script (8 GPUs)

8개 GPU에서 병렬 이미지 처리가 올바르게 동작하는지 테스트합니다.

테스트 항목:
1. 8개 GPU 초기화 확인
2. Round-robin GPU 할당 확인
3. 병렬 이미지 처리 성능 측정
4. GPU 메모리 사용량 모니터링
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 설정 (torch import 전에)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["OMP_NUM_THREADS"] = "4"

import torch

from ai.gpu import GPUPoolManager, initialize_gpu_pool, get_gpu_pool


def print_header(title: str):
    """섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_gpu_status(pool: GPUPoolManager):
    """GPU 풀 상태 출력"""
    status = pool.get_status()
    print(f"\nTotal GPUs: {status['total_gpus']}")
    print(f"Available GPUs: {status['available_gpus']}")
    print("\nGPU Details:")
    for gpu_id, info in status['gpus'].items():
        avail = "✓" if info['available'] else "✗"
        task = info['task_id'] or "-"
        mem = f"{info['memory_used_mb']:.0f}/{info['memory_total_mb']:.0f} MB"
        print(f"  GPU {gpu_id}: [{avail}] task={task:<20} mem={mem}")


async def test_gpu_initialization():
    """테스트 1: 8개 GPU 초기화"""
    print_header("Test 1: GPU Pool Initialization (8 GPUs)")

    # 현재 사용 가능한 GPU 수 확인
    gpu_count = torch.cuda.device_count()
    print(f"\nDetected {gpu_count} GPUs")

    # 8개 GPU로 풀 초기화
    gpu_ids = list(range(min(8, gpu_count)))
    pool = initialize_gpu_pool(gpu_ids=gpu_ids)

    print_gpu_status(pool)

    assert len(pool.gpu_ids) == len(gpu_ids), f"Expected {len(gpu_ids)} GPUs, got {len(pool.gpu_ids)}"
    print(f"\n✓ Successfully initialized pool with {len(pool.gpu_ids)} GPUs")

    return pool


async def test_round_robin_allocation(pool: GPUPoolManager):
    """테스트 2: Round-robin GPU 할당"""
    print_header("Test 2: Round-Robin GPU Allocation")

    allocation_order = []
    num_allocations = 16  # 8개 GPU × 2 사이클

    print(f"\nAllocating {num_allocations} GPUs sequentially...")

    for i in range(num_allocations):
        gpu_id = await pool.acquire(task_id=f"test_{i}")
        allocation_order.append(gpu_id)
        print(f"  Allocation {i+1}: GPU {gpu_id}")
        await pool.release(gpu_id)

    # Round-robin 패턴 검증
    expected_pattern = list(range(len(pool.gpu_ids))) * 2

    print(f"\nAllocation order: {allocation_order}")
    print(f"Expected pattern: {expected_pattern}")

    # 첫 사이클 확인 (순서대로 0,1,2,3,4,5,6,7 할당)
    first_cycle = allocation_order[:len(pool.gpu_ids)]
    unique_gpus = set(first_cycle)

    assert len(unique_gpus) == len(pool.gpu_ids), "First cycle should use all GPUs"
    print(f"\n✓ Round-robin allocation working correctly")

    return allocation_order


async def test_parallel_acquisition(pool: GPUPoolManager):
    """테스트 3: 병렬 GPU 획득"""
    print_header("Test 3: Parallel GPU Acquisition")

    num_tasks = len(pool.gpu_ids)  # 모든 GPU 동시 사용
    acquired_gpus = []
    results = []

    async def acquire_and_process(task_id: str, delay: float):
        """GPU 획득 후 작업 시뮬레이션"""
        start = time.perf_counter()

        async with pool.gpu_context(task_id=task_id) as gpu_id:
            acquire_time = time.perf_counter() - start
            acquired_gpus.append(gpu_id)

            # GPU에서 작업 시뮬레이션 (텐서 연산)
            with torch.cuda.device(gpu_id):
                x = torch.randn(1000, 1000, device=f"cuda:{gpu_id}")
                y = torch.matmul(x, x)
                torch.cuda.synchronize(gpu_id)
                del x, y
                torch.cuda.empty_cache()

            await asyncio.sleep(delay)  # 작업 시간 시뮬레이션

        total_time = time.perf_counter() - start
        return {
            "task_id": task_id,
            "gpu_id": gpu_id,
            "acquire_time": acquire_time,
            "total_time": total_time
        }

    print(f"\nLaunching {num_tasks} parallel tasks...")

    # 모든 태스크 병렬 실행
    start_time = time.perf_counter()
    tasks = [
        acquire_and_process(f"parallel_{i}", delay=0.5)
        for i in range(num_tasks)
    ]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - start_time

    # 결과 출력
    print("\nTask Results:")
    for r in sorted(results, key=lambda x: x['gpu_id']):
        print(f"  {r['task_id']}: GPU {r['gpu_id']}, "
              f"acquire={r['acquire_time']:.3f}s, total={r['total_time']:.3f}s")

    # 검증: 모든 GPU가 사용되었는지
    unique_gpus = set(r['gpu_id'] for r in results)
    print(f"\nUnique GPUs used: {sorted(unique_gpus)}")
    print(f"Total elapsed time: {total_elapsed:.2f}s")

    # 병렬 실행 검증: 총 시간이 순차 실행보다 훨씬 짧아야 함
    # 순차 실행 시 8개 × 0.5s = 4s, 병렬 실행 시 ~0.5s
    assert total_elapsed < 2.0, f"Parallel execution took too long: {total_elapsed:.2f}s"
    assert len(unique_gpus) == num_tasks, f"Expected {num_tasks} unique GPUs"

    print(f"\n✓ Parallel GPU acquisition working correctly")
    print(f"  Speedup: {(num_tasks * 0.5) / total_elapsed:.1f}x vs sequential")

    return results


async def test_image_paths_distribution(pool: GPUPoolManager):
    """테스트 4: 이미지 경로를 GPU에 분배"""
    print_header("Test 4: Image Distribution to 8 GPUs")

    # ai/imgs 디렉토리의 이미지 목록
    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_files = sorted(imgs_dir.glob("*.jpg"))

    print(f"\nFound {len(image_files)} images in {imgs_dir}")
    for img in image_files:
        print(f"  - {img.name}")

    # 이미지를 GPU에 할당 (실제 처리 없이 할당만 테스트)
    gpu_assignments: Dict[int, List[str]] = {i: [] for i in pool.gpu_ids}

    print(f"\nDistributing {len(image_files)} images to {len(pool.gpu_ids)} GPUs:")

    for i, img_path in enumerate(image_files):
        gpu_id = await pool.acquire(task_id=f"img_{i}")
        gpu_assignments[gpu_id].append(img_path.name)
        print(f"  {img_path.name} → GPU {gpu_id}")
        await pool.release(gpu_id)

    # 분배 결과 요약
    print("\nDistribution Summary:")
    for gpu_id, images in sorted(gpu_assignments.items()):
        print(f"  GPU {gpu_id}: {len(images)} images - {images}")

    # 검증: 균등 분배 확인
    counts = [len(imgs) for imgs in gpu_assignments.values()]
    max_diff = max(counts) - min(counts)

    print(f"\nImage count per GPU: min={min(counts)}, max={max(counts)}, diff={max_diff}")
    assert max_diff <= 1, "Images should be distributed evenly (±1)"

    print(f"\n✓ Image distribution working correctly")

    return gpu_assignments


async def test_concurrent_image_simulation(pool: GPUPoolManager):
    """테스트 5: 동시 이미지 처리 시뮬레이션"""
    print_header("Test 5: Concurrent Image Processing Simulation")

    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_files = sorted(imgs_dir.glob("*.jpg"))[:8]  # 최대 8개 이미지

    if len(image_files) < len(pool.gpu_ids):
        print(f"\nWarning: Only {len(image_files)} images, using all available")

    results = []

    async def simulate_process_image(img_path: Path, idx: int):
        """이미지 처리 시뮬레이션 (실제 모델 없이)"""
        start = time.perf_counter()

        async with pool.gpu_context(task_id=f"img_{idx}") as gpu_id:
            acquire_time = time.perf_counter() - start

            # GPU 텐서 연산으로 처리 시뮬레이션
            with torch.cuda.device(gpu_id):
                # 이미지 크기 정도의 텐서 생성
                img_tensor = torch.randn(3, 1920, 1920, device=f"cuda:{gpu_id}")

                # 간단한 CNN 연산 시뮬레이션
                conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(f"cuda:{gpu_id}")
                output = conv(img_tensor.unsqueeze(0))

                torch.cuda.synchronize(gpu_id)

                # 메모리 사용량 측정
                mem_used = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024 / 1024

                del img_tensor, output, conv
                torch.cuda.empty_cache()

            process_time = time.perf_counter() - start

        return {
            "image": img_path.name,
            "gpu_id": gpu_id,
            "acquire_time": acquire_time,
            "process_time": process_time,
            "memory_used_mb": mem_used,
            "memory_reserved_mb": mem_reserved
        }

    print(f"\nProcessing {len(image_files)} images in parallel...")

    start_time = time.perf_counter()
    tasks = [
        simulate_process_image(img, i)
        for i, img in enumerate(image_files)
    ]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    # 결과 출력
    print("\nProcessing Results:")
    print("-" * 80)
    print(f"{'Image':<30} {'GPU':>4} {'Acquire':>10} {'Process':>10} {'Memory':>12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['gpu_id']):
        print(f"{r['image']:<30} {r['gpu_id']:>4} "
              f"{r['acquire_time']*1000:>8.1f}ms {r['process_time']*1000:>8.1f}ms "
              f"{r['memory_used_mb']:>8.1f}MB")

    print("-" * 80)
    print(f"Total time: {total_time:.2f}s")

    # GPU별 사용 통계
    gpu_usage = {}
    for r in results:
        gpu_id = r['gpu_id']
        if gpu_id not in gpu_usage:
            gpu_usage[gpu_id] = 0
        gpu_usage[gpu_id] += 1

    print("\nGPU Usage Statistics:")
    for gpu_id in sorted(gpu_usage.keys()):
        count = gpu_usage[gpu_id]
        bar = "█" * count
        print(f"  GPU {gpu_id}: {bar} ({count})")

    print(f"\n✓ Concurrent image processing simulation completed")

    return results


async def test_gpu_memory_monitoring(pool: GPUPoolManager):
    """테스트 6: GPU 메모리 모니터링"""
    print_header("Test 6: GPU Memory Monitoring")

    print("\nCurrent GPU Memory Status:")
    print("-" * 60)

    for gpu_id in pool.gpu_ids:
        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        free = total - reserved

        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Free:      {free:.2f} GB")
        print()

    print("✓ Memory monitoring completed")


async def main():
    """메인 테스트 실행"""
    print_header("Multi-GPU Architecture Test Suite (8 GPUs)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    try:
        # 테스트 실행
        pool = await test_gpu_initialization()
        await test_round_robin_allocation(pool)
        await test_parallel_acquisition(pool)
        await test_image_paths_distribution(pool)
        await test_concurrent_image_simulation(pool)
        await test_gpu_memory_monitoring(pool)

        # 최종 상태 출력
        print_header("Final GPU Pool Status")
        print_gpu_status(pool)

        print_header("ALL TESTS PASSED ✓")
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
