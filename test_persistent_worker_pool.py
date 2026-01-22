#!/usr/bin/env python3
"""
Persistent Worker Pool 및 병렬 3D 생성 통합 테스트

테스트 항목:
1. Worker Pool 초기화
2. 병렬 3D 생성
3. 순차 vs 병렬 성능 비교
4. 이미지별 객체 분리 검증
"""

import asyncio
import time
import os
import sys
import base64
from PIL import Image
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_worker_pool_initialization():
    """Worker Pool 초기화 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Worker Pool Initialization")
    print("=" * 60)

    from ai.gpu import SAM3DWorkerPool

    # 사용 가능한 GPU 확인
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
        print(f"Available GPUs: {gpu_count} ({gpu_ids})")
    else:
        print("No CUDA available, using CPU fallback")
        gpu_ids = [0]

    # Worker Pool 생성
    pool = SAM3DWorkerPool(gpu_ids=gpu_ids, init_timeout=180.0)

    print(f"\nStarting {len(gpu_ids)} workers...")
    start_time = time.time()

    try:
        await pool.start_workers()
        init_time = time.time() - start_time
        print(f"\nWorker Pool initialized in {init_time:.2f}s")

        status = pool.get_status()
        print(f"Status: {status}")

        return pool
    except Exception as e:
        print(f"\nFailed to initialize Worker Pool: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_single_3d_generation(pool):
    """단일 3D 생성 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Single 3D Generation")
    print("=" * 60)

    if pool is None:
        print("Skipping: Worker Pool not available")
        return

    # 테스트 이미지 로드
    test_images_dir = "ai/imgs"
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not test_images:
        print("No test images found")
        return

    image_path = os.path.join(test_images_dir, test_images[0])
    print(f"Using test image: {image_path}")

    # 이미지를 base64로 변환
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    # 간단한 마스크 생성 (이미지 중앙 영역)
    img = Image.open(image_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    # 중앙 50% 영역을 마스크로
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)

    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    print(f"Image size: {w}x{h}")
    print(f"Mask size: {w}x{h} (center 50%)")

    # 3D 생성
    print("\nSubmitting 3D generation task...")
    start_time = time.time()

    result = await pool.submit_task(
        task_id="test_single",
        image_b64=image_b64,
        mask_b64=mask_b64,
        seed=42,
        skip_gif=True
    )

    elapsed = time.time() - start_time

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Processing time: {elapsed:.2f}s")
    if result.success:
        print(f"  PLY size: {result.ply_size_bytes} bytes")
    else:
        print(f"  Error: {result.error}")

    return result.success


async def test_parallel_3d_generation(pool, num_tasks=3):
    """병렬 3D 생성 테스트"""
    print("\n" + "=" * 60)
    print(f"TEST 3: Parallel 3D Generation ({num_tasks} tasks)")
    print("=" * 60)

    if pool is None:
        print("Skipping: Worker Pool not available")
        return

    # 테스트 이미지 로드
    test_images_dir = "ai/imgs"
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:num_tasks]

    if len(test_images) < num_tasks:
        print(f"Warning: Only {len(test_images)} images available")
        num_tasks = len(test_images)

    tasks = []
    for i, img_name in enumerate(test_images):
        image_path = os.path.join(test_images_dir, img_name)

        # 이미지 로드
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        # 마스크 생성
        img = Image.open(image_path)
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)

        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        tasks.append({
            "task_id": f"parallel_{i}",
            "image_b64": image_b64,
            "mask_b64": mask_b64,
            "seed": 42,
            "skip_gif": True
        })

        print(f"Task {i}: {img_name} ({w}x{h})")

    # 병렬 실행
    print(f"\nSubmitting {num_tasks} tasks in parallel...")
    start_time = time.time()

    results = await pool.submit_tasks_parallel(tasks)

    elapsed = time.time() - start_time

    # 결과 분석
    success_count = sum(1 for r in results if r.success)
    total_ply_size = sum(r.ply_size_bytes or 0 for r in results)

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average per task: {elapsed/num_tasks:.2f}s")
    print(f"  Success: {success_count}/{num_tasks}")
    print(f"  Total PLY size: {total_ply_size} bytes")

    for i, r in enumerate(results):
        status = "OK" if r.success else f"FAIL: {r.error}"
        ply_info = f"{r.ply_size_bytes} bytes" if r.ply_size_bytes else "N/A"
        print(f"    Task {i}: {status} ({ply_info})")

    return success_count == num_tasks


async def test_sequential_vs_parallel_comparison(pool):
    """순차 vs 병렬 성능 비교"""
    print("\n" + "=" * 60)
    print("TEST 4: Sequential vs Parallel Comparison")
    print("=" * 60)

    if pool is None:
        print("Skipping: Worker Pool not available")
        return

    # 테스트 이미지 준비
    test_images_dir = "ai/imgs"
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:3]

    if len(test_images) < 3:
        print("Not enough test images for comparison")
        return

    tasks = []
    for i, img_name in enumerate(test_images):
        image_path = os.path.join(test_images_dir, img_name)

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        img = Image.open(image_path)
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)

        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        tasks.append({
            "task_id": f"compare_{i}",
            "image_b64": image_b64,
            "mask_b64": mask_b64,
            "seed": 42,
            "skip_gif": True
        })

    # 순차 실행
    print(f"\n1. Sequential execution ({len(tasks)} tasks)...")
    start_time = time.time()
    sequential_results = []
    for task in tasks:
        result = await pool.submit_task(**task)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"   Sequential time: {sequential_time:.2f}s")

    # 병렬 실행
    print(f"\n2. Parallel execution ({len(tasks)} tasks)...")
    start_time = time.time()
    parallel_results = await pool.submit_tasks_parallel(tasks)
    parallel_time = time.time() - start_time
    print(f"   Parallel time: {parallel_time:.2f}s")

    # 비교
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"\nComparison:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Parallel:   {parallel_time:.2f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    return speedup > 1.0


async def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("Persistent Worker Pool & Parallel 3D Generation Test")
    print("=" * 60)

    # Test 1: Worker Pool 초기화
    pool = await test_worker_pool_initialization()

    if pool is None:
        print("\nFATAL: Worker Pool initialization failed. Cannot continue tests.")
        return

    try:
        # Test 2: 단일 3D 생성
        await test_single_3d_generation(pool)

        # Test 3: 병렬 3D 생성
        await test_parallel_3d_generation(pool, num_tasks=3)

        # Test 4: 순차 vs 병렬 비교
        await test_sequential_vs_parallel_comparison(pool)

    finally:
        # 정리
        print("\n" + "=" * 60)
        print("Cleaning up...")
        print("=" * 60)
        await pool.shutdown()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
