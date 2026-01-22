#!/usr/bin/env python3
"""
Factor Isolation Test - 다운샘플링 vs Inference Steps 영향 분리 테스트

테스트 시나리오:
1. Baseline: 다운샘플링 OFF + steps=12 (원본)
2. Test A: 다운샘플링 ON (768px) + steps=12 (다운샘플링만 적용)
3. Test B: 다운샘플링 OFF + steps=8 (steps만 감소)

각 요소가 부피 정확도에 미치는 영향을 분리하여 측정
"""

import asyncio
import time
import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_test_with_config(config_name: str, max_size: str, steps: str, test_image: str, gpu_ids: list):
    """특정 설정으로 테스트 실행"""
    print(f"\n{'='*60}")
    print(f"[{config_name}] MAX_IMAGE_SIZE={max_size}, STEPS={steps}")
    print(f"{'='*60}")

    # 워커 파일 수정
    worker_file = "ai/subprocess/persistent_3d_worker.py"

    with open(worker_file, "r") as f:
        original_content = f.read()

    # 설정 변경
    modified_content = original_content

    # MAX_IMAGE_SIZE 변경
    if "MAX_IMAGE_SIZE = 768" in modified_content:
        modified_content = modified_content.replace(
            "MAX_IMAGE_SIZE = 768",
            f"MAX_IMAGE_SIZE = {max_size}"
        )
    elif "MAX_IMAGE_SIZE = None" in modified_content:
        modified_content = modified_content.replace(
            "MAX_IMAGE_SIZE = None",
            f"MAX_IMAGE_SIZE = {max_size}"
        )

    # STAGE2_INFERENCE_STEPS 변경
    if "STAGE2_INFERENCE_STEPS = 10" in modified_content:
        modified_content = modified_content.replace(
            "STAGE2_INFERENCE_STEPS = 10",
            f"STAGE2_INFERENCE_STEPS = {steps}"
        )
    elif "STAGE2_INFERENCE_STEPS = 12" in modified_content:
        modified_content = modified_content.replace(
            "STAGE2_INFERENCE_STEPS = 12",
            f"STAGE2_INFERENCE_STEPS = {steps}"
        )
    elif "STAGE2_INFERENCE_STEPS = 8" in modified_content:
        modified_content = modified_content.replace(
            "STAGE2_INFERENCE_STEPS = 8",
            f"STAGE2_INFERENCE_STEPS = {steps}"
        )

    with open(worker_file, "w") as f:
        f.write(modified_content)

    print(f"Config applied: MAX_IMAGE_SIZE={max_size}, STEPS={steps}")

    try:
        from ai.gpu import SAM3DWorkerPool, initialize_gpu_pool, get_gpu_pool
        from ai.pipeline import FurniturePipeline
        import ai.gpu.sam3d_worker_pool as sam3d_module

        # 기존 풀이 있으면 초기화
        try:
            existing_pool = get_gpu_pool()
        except:
            pass

        gpu_pool = initialize_gpu_pool(gpu_ids)

        # Worker Pool 시작
        sam3d_pool = SAM3DWorkerPool(gpu_ids=gpu_ids, init_timeout=180.0)
        await sam3d_pool.start_workers()

        sam3d_module._global_sam3d_pool = sam3d_pool

        pipeline = FurniturePipeline(
            sam2_api_url="http://localhost:8000",
            enable_3d_generation=True,
            device_id=0,
            gpu_pool=gpu_pool
        )

        # 테스트 실행
        image_url = f"file://{os.path.abspath(test_image)}"
        start = time.time()

        result = await pipeline.process_single_image(
            image_url=image_url,
            enable_mask=True,
            enable_3d=True,
            use_parallel_3d=True
        )

        elapsed = time.time() - start

        # 결과 수집
        volumes = []
        for obj in result.objects:
            if obj.relative_dimensions:
                volumes.append({
                    "label": obj.label,
                    "volume": obj.relative_dimensions.get("volume", 0)
                })

        print(f"Time: {elapsed:.2f}s, Objects: {len(volumes)}")

        # 워커 종료
        await sam3d_pool.shutdown()

        return {
            "config": config_name,
            "max_size": max_size,
            "steps": steps,
            "time": elapsed,
            "volumes": volumes
        }

    finally:
        # 원본 복원
        with open(worker_file, "w") as f:
            f.write(original_content)


async def main():
    print("=" * 70)
    print("Factor Isolation Test - 다운샘플링 vs Inference Steps 영향 분리")
    print("=" * 70)

    import torch
    gpu_ids = list(range(min(4, torch.cuda.device_count()))) if torch.cuda.is_available() else [0]
    print(f"Using GPUs: {gpu_ids}")

    # 테스트 이미지 선택 (1개만 사용)
    test_image = "ai/imgs/bed-1834327_1920.jpg"
    print(f"Test image: {test_image}")

    results = {}

    # Test 1: Baseline (다운샘플링 OFF + steps=12)
    results["baseline"] = await run_test_with_config(
        "Baseline", "None", "12", test_image, gpu_ids
    )

    # Test 2: 다운샘플링만 적용 (768px + steps=12)
    results["downsample_only"] = await run_test_with_config(
        "Downsample Only", "768", "12", test_image, gpu_ids
    )

    # Test 3: Steps만 감소 (다운샘플링 OFF + steps=8)
    results["steps_only"] = await run_test_with_config(
        "Steps Only", "None", "8", test_image, gpu_ids
    )

    # 결과 분석
    print("\n" + "=" * 70)
    print("[Results Analysis] 요소별 영향 분석")
    print("=" * 70)

    baseline_vols = {v["label"]: v["volume"] for v in results["baseline"]["volumes"]}
    downsample_vols = {v["label"]: v["volume"] for v in results["downsample_only"]["volumes"]}
    steps_vols = {v["label"]: v["volume"] for v in results["steps_only"]["volumes"]}

    print(f"\n{'객체':<20} {'Baseline':<12} {'다운샘플링':<12} {'Steps감소':<12} {'다운샘플링차이':<15} {'Steps차이':<15}")
    print("-" * 90)

    downsample_diffs = []
    steps_diffs = []

    for label in baseline_vols:
        baseline_v = baseline_vols.get(label, 0)
        downsample_v = downsample_vols.get(label, 0)
        steps_v = steps_vols.get(label, 0)

        if baseline_v > 0:
            downsample_diff = abs(baseline_v - downsample_v) / baseline_v * 100
            steps_diff = abs(baseline_v - steps_v) / baseline_v * 100

            downsample_diffs.append(downsample_diff)
            steps_diffs.append(steps_diff)

            print(f"{label:<20} {baseline_v:<12.4f} {downsample_v:<12.4f} {steps_v:<12.4f} {downsample_diff:<15.1f}% {steps_diff:<15.1f}%")

    print("-" * 90)

    avg_downsample = np.mean(downsample_diffs) if downsample_diffs else 0
    avg_steps = np.mean(steps_diffs) if steps_diffs else 0
    max_downsample = np.max(downsample_diffs) if downsample_diffs else 0
    max_steps = np.max(steps_diffs) if steps_diffs else 0

    print(f"\n{'평균 차이:':<20} {'':<12} {'':<12} {'':<12} {avg_downsample:<15.1f}% {avg_steps:<15.1f}%")
    print(f"{'최대 차이:':<20} {'':<12} {'':<12} {'':<12} {max_downsample:<15.1f}% {max_steps:<15.1f}%")

    print("\n" + "=" * 70)
    print("[Conclusion] 결론")
    print("=" * 70)

    if avg_downsample > avg_steps:
        print(f"\n🔴 다운샘플링이 부피 정확도에 더 큰 영향 ({avg_downsample:.1f}% vs {avg_steps:.1f}%)")
        print("   → 다운샘플링을 비활성화하고 inference steps만 조정하는 것을 권장")
    elif avg_steps > avg_downsample:
        print(f"\n🔴 Inference Steps 감소가 부피 정확도에 더 큰 영향 ({avg_steps:.1f}% vs {avg_downsample:.1f}%)")
        print("   → Steps를 12로 유지하고 다운샘플링만 적용하는 것을 권장")
    else:
        print(f"\n🟡 두 요소의 영향이 비슷함 ({avg_downsample:.1f}% vs {avg_steps:.1f}%)")

    # 처리 시간 비교
    print(f"\n처리 시간 비교:")
    print(f"  Baseline: {results['baseline']['time']:.2f}s")
    print(f"  다운샘플링만: {results['downsample_only']['time']:.2f}s (차이: {results['baseline']['time'] - results['downsample_only']['time']:.2f}s)")
    print(f"  Steps감소만: {results['steps_only']['time']:.2f}s (차이: {results['baseline']['time'] - results['steps_only']['time']:.2f}s)")

    # 결과 저장
    with open("test_factor_isolation_results.json", "w") as f:
        json.dump({
            "results": results,
            "analysis": {
                "avg_downsample_diff": avg_downsample,
                "avg_steps_diff": avg_steps,
                "max_downsample_diff": max_downsample,
                "max_steps_diff": max_steps,
                "conclusion": "downsample" if avg_downsample > avg_steps else "steps"
            }
        }, f, indent=2)

    print("\nResults saved to: test_factor_isolation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
