#!/usr/bin/env python3
"""
Optimization Comparison Test - 최적화 적용 전후 상대 길이 비교

동일한 이미지에 대해:
1. 최적화 OFF (원본 설정)
2. 최적화 ON (새 설정)
결과를 비교하여 상대 길이의 일관성 검증
"""

import asyncio
import time
import os
import sys
import json
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_single_image_test(sam3d_pool, pipeline, image_path: str, image_id: int):
    """단일 이미지 테스트"""
    image_url = f"file://{os.path.abspath(image_path)}"

    result = await pipeline.process_single_image(
        image_url=image_url,
        enable_mask=True,
        enable_3d=True,
        use_parallel_3d=True
    )
    result.user_image_id = image_id

    # 결과에서 dimensions 추출
    dimensions = []
    for obj in result.objects:
        if obj.relative_dimensions:
            dimensions.append({
                "label": obj.label,
                "width": obj.relative_dimensions.get("width", 0),
                "depth": obj.relative_dimensions.get("depth", 0),
                "height": obj.relative_dimensions.get("height", 0),
                "volume": obj.relative_dimensions.get("volume", 0),
            })

    return dimensions


async def run_comparison_test():
    """최적화 전후 비교 테스트"""
    print("=" * 70)
    print("Optimization Comparison Test - 상대 길이 일관성 검증")
    print("=" * 70)

    # 테스트 이미지 선택 (2개만 사용하여 빠르게 테스트)
    test_images_dir = "ai/imgs"
    all_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    test_images = all_images[:2]  # 2개만 선택

    print(f"\nTest images: {test_images}")

    # Import 및 초기화
    from ai.gpu import SAM3DWorkerPool, initialize_gpu_pool
    import torch

    gpu_ids = list(range(min(4, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"Using GPUs: {gpu_ids}")

    # =========================================================================
    # Test 1: 최적화 OFF (원본 설정)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Test 1] 최적화 OFF (MAX_IMAGE_SIZE=2048, STAGE2_INFERENCE_STEPS=12)")
    print("=" * 70)

    # 설정 변경 (최적화 OFF)
    import ai.subprocess.persistent_3d_worker as worker_module

    # 원본 설정 백업
    original_max_size = getattr(worker_module, 'MAX_IMAGE_SIZE', 512)
    original_steps = getattr(worker_module, 'STAGE2_INFERENCE_STEPS', 8)
    original_binary = getattr(worker_module, 'USE_BINARY_PLY', True)

    # 실제로는 워커 프로세스가 별도로 실행되므로, 환경변수로 전달해야 함
    # 여기서는 워커를 재시작하여 설정 변경

    # GPU Pool 초기화
    gpu_pool = initialize_gpu_pool(gpu_ids)

    # 테스트를 위해 설정 파일 임시 수정
    worker_file = "ai/subprocess/persistent_3d_worker.py"

    # 원본 파일 백업
    with open(worker_file, "r") as f:
        original_content = f.read()

    # 최적화 OFF 설정으로 변경
    no_opt_content = original_content.replace(
        "MAX_IMAGE_SIZE = 768",
        "MAX_IMAGE_SIZE = 2048  # TEMP: optimization OFF"
    ).replace(
        "STAGE2_INFERENCE_STEPS = 10",
        "STAGE2_INFERENCE_STEPS = 12  # TEMP: optimization OFF"
    )

    with open(worker_file, "w") as f:
        f.write(no_opt_content)

    print("Worker config: MAX_IMAGE_SIZE=2048, STAGE2_INFERENCE_STEPS=12")

    # SAM3D Worker Pool 시작
    sam3d_pool_off = SAM3DWorkerPool(gpu_ids=gpu_ids, init_timeout=180.0)
    await sam3d_pool_off.start_workers()

    from ai.pipeline import FurniturePipeline
    import ai.gpu.sam3d_worker_pool as sam3d_module
    sam3d_module._global_sam3d_pool = sam3d_pool_off

    pipeline_off = FurniturePipeline(
        sam2_api_url="http://localhost:8000",
        enable_3d_generation=True,
        device_id=0,
        gpu_pool=gpu_pool
    )

    # 테스트 실행
    results_off = {}
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\nProcessing (OFF): {img_name}")
        start = time.time()
        dims = await run_single_image_test(sam3d_pool_off, pipeline_off, img_path, i+1)
        elapsed = time.time() - start
        results_off[img_name] = {"dims": dims, "time": elapsed}
        print(f"  Objects: {len(dims)}, Time: {elapsed:.2f}s")

    # 워커 종료
    await sam3d_pool_off.shutdown()

    # =========================================================================
    # Test 2: 최적화 ON (새 설정)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Test 2] 최적화 ON (MAX_IMAGE_SIZE=768, STAGE2_INFERENCE_STEPS=10)")
    print("=" * 70)

    # 최적화 ON 설정으로 복원
    with open(worker_file, "w") as f:
        f.write(original_content)

    print("Worker config: MAX_IMAGE_SIZE=768, STAGE2_INFERENCE_STEPS=10")

    # SAM3D Worker Pool 재시작
    sam3d_pool_on = SAM3DWorkerPool(gpu_ids=gpu_ids, init_timeout=180.0)
    await sam3d_pool_on.start_workers()

    sam3d_module._global_sam3d_pool = sam3d_pool_on

    pipeline_on = FurniturePipeline(
        sam2_api_url="http://localhost:8000",
        enable_3d_generation=True,
        device_id=0,
        gpu_pool=gpu_pool
    )

    # 테스트 실행
    results_on = {}
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\nProcessing (ON): {img_name}")
        start = time.time()
        dims = await run_single_image_test(sam3d_pool_on, pipeline_on, img_path, i+1)
        elapsed = time.time() - start
        results_on[img_name] = {"dims": dims, "time": elapsed}
        print(f"  Objects: {len(dims)}, Time: {elapsed:.2f}s")

    # 워커 종료
    await sam3d_pool_on.shutdown()

    # =========================================================================
    # 결과 비교
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Results Comparison] 상대 길이 비교")
    print("=" * 70)

    comparison_results = []

    for img_name in test_images:
        print(f"\n--- {img_name} ---")

        dims_off = results_off[img_name]["dims"]
        dims_on = results_on[img_name]["dims"]
        time_off = results_off[img_name]["time"]
        time_on = results_on[img_name]["time"]

        print(f"Time: OFF={time_off:.2f}s, ON={time_on:.2f}s, Speedup={time_off/time_on:.2f}x")
        print(f"Objects: OFF={len(dims_off)}, ON={len(dims_on)}")

        # 객체별 비교 (라벨 기준 매칭)
        for i, (d_off, d_on) in enumerate(zip(dims_off, dims_on)):
            if d_off["label"] == d_on["label"]:
                # 상대 오차 계산
                width_diff = abs(d_off["width"] - d_on["width"]) / max(d_off["width"], 0.001) * 100
                depth_diff = abs(d_off["depth"] - d_on["depth"]) / max(d_off["depth"], 0.001) * 100
                height_diff = abs(d_off["height"] - d_on["height"]) / max(d_off["height"], 0.001) * 100
                volume_diff = abs(d_off["volume"] - d_on["volume"]) / max(d_off["volume"], 0.001) * 100

                print(f"\n  [{i}] {d_off['label']}:")
                print(f"      Width:  OFF={d_off['width']:.3f}, ON={d_on['width']:.3f}, Diff={width_diff:.1f}%")
                print(f"      Depth:  OFF={d_off['depth']:.3f}, ON={d_on['depth']:.3f}, Diff={depth_diff:.1f}%")
                print(f"      Height: OFF={d_off['height']:.3f}, ON={d_on['height']:.3f}, Diff={height_diff:.1f}%")
                print(f"      Volume: OFF={d_off['volume']:.4f}, ON={d_on['volume']:.4f}, Diff={volume_diff:.1f}%")

                comparison_results.append({
                    "image": img_name,
                    "label": d_off["label"],
                    "width_diff_pct": width_diff,
                    "depth_diff_pct": depth_diff,
                    "height_diff_pct": height_diff,
                    "volume_diff_pct": volume_diff,
                })

    # =========================================================================
    # 요약 통계
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Summary Statistics]")
    print("=" * 70)

    if comparison_results:
        avg_width_diff = np.mean([r["width_diff_pct"] for r in comparison_results])
        avg_depth_diff = np.mean([r["depth_diff_pct"] for r in comparison_results])
        avg_height_diff = np.mean([r["height_diff_pct"] for r in comparison_results])
        avg_volume_diff = np.mean([r["volume_diff_pct"] for r in comparison_results])

        max_width_diff = np.max([r["width_diff_pct"] for r in comparison_results])
        max_depth_diff = np.max([r["depth_diff_pct"] for r in comparison_results])
        max_height_diff = np.max([r["height_diff_pct"] for r in comparison_results])
        max_volume_diff = np.max([r["volume_diff_pct"] for r in comparison_results])

        print(f"\nAverage Differences:")
        print(f"  Width:  {avg_width_diff:.1f}% (max: {max_width_diff:.1f}%)")
        print(f"  Depth:  {avg_depth_diff:.1f}% (max: {max_depth_diff:.1f}%)")
        print(f"  Height: {avg_height_diff:.1f}% (max: {max_height_diff:.1f}%)")
        print(f"  Volume: {avg_volume_diff:.1f}% (max: {max_volume_diff:.1f}%)")

        # 허용 오차 체크 (5% 이내)
        tolerance = 5.0
        within_tolerance = all(
            r["volume_diff_pct"] <= tolerance for r in comparison_results
        )

        print(f"\n[Validation] Volume difference within {tolerance}%: {'PASS' if within_tolerance else 'FAIL'}")

        # 결과 저장
        with open("test_optimization_comparison_results.json", "w") as f:
            json.dump({
                "comparison": comparison_results,
                "summary": {
                    "avg_width_diff": avg_width_diff,
                    "avg_depth_diff": avg_depth_diff,
                    "avg_height_diff": avg_height_diff,
                    "avg_volume_diff": avg_volume_diff,
                },
                "passed": within_tolerance
            }, f, indent=2)

        print("\nResults saved to: test_optimization_comparison_results.json")

        return within_tolerance
    else:
        print("No comparable objects found")
        return False


if __name__ == "__main__":
    passed = asyncio.run(run_comparison_test())
    sys.exit(0 if passed else 1)
