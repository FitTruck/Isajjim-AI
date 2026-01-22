#!/usr/bin/env python3
"""
E2E Full Pipeline Test - Persistent Worker Pool + Parallel 3D Generation

실제 파이프라인 전체 테스트:
1. 8개 이미지 로드
2. YOLOE-seg 탐지 (모든 객체)
3. Worker Pool을 사용한 병렬 3D 생성 (모든 객체)
4. 부피 계산
5. 이미지별 객체 분리 검증
6. TDD Section 4.1 JSON 형식 출력
"""

import asyncio
import time
import os
import sys
import json
from PIL import Image
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def run_e2e_test():
    """E2E 전체 파이프라인 테스트"""
    print("=" * 70)
    print("E2E Full Pipeline Test - Persistent Worker Pool + Parallel 3D")
    print("=" * 70)

    # =========================================================================
    # Step 1: 테스트 이미지 준비 (8개)
    # =========================================================================
    print("\n[Step 1] Loading test images...")

    test_images_dir = "ai/imgs"
    all_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    test_images = all_images[:8]  # 8개 선택

    print(f"Selected {len(test_images)} images:")
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_name)
        img = Image.open(img_path)
        print(f"  [{i+1}] {img_name} ({img.size[0]}x{img.size[1]})")

    # 이미지 URL 형식으로 변환 (로컬 파일 경로를 file:// URL로)
    image_items = []
    for i, img_name in enumerate(test_images):
        img_path = os.path.abspath(os.path.join(test_images_dir, img_name))
        image_items.append((i + 1, f"file://{img_path}"))  # (user_image_id, url)

    # =========================================================================
    # Step 2: Worker Pool 초기화
    # =========================================================================
    print("\n[Step 2] Initializing SAM3D Worker Pool...")

    from ai.gpu import SAM3DWorkerPool, initialize_gpu_pool
    from ai.config import Config
    import torch

    gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    print(f"Available GPUs: {len(gpu_ids)} ({gpu_ids})")

    # GPU Pool 초기화 (YOLOE 탐지용)
    gpu_pool = initialize_gpu_pool(gpu_ids)

    # SAM3D Worker Pool 초기화
    sam3d_pool = SAM3DWorkerPool(gpu_ids=gpu_ids, init_timeout=180.0)

    print("Starting SAM3D workers (model loading)...")
    worker_init_start = time.time()
    await sam3d_pool.start_workers()
    worker_init_time = time.time() - worker_init_start
    print(f"SAM3D Worker Pool ready in {worker_init_time:.2f}s")
    print(f"Worker status: {sam3d_pool.get_status()}")

    # =========================================================================
    # Step 3: FurniturePipeline 생성
    # =========================================================================
    print("\n[Step 3] Creating FurniturePipeline...")

    from ai.pipeline import FurniturePipeline
    from ai.gpu import get_gpu_pool

    # Worker Pool을 사용하도록 글로벌 설정
    import ai.gpu.sam3d_worker_pool as sam3d_module
    sam3d_module._global_sam3d_pool = sam3d_pool

    pipeline = FurniturePipeline(
        sam2_api_url="http://localhost:8000",
        enable_3d_generation=True,
        device_id=0,
        gpu_pool=gpu_pool
    )

    print("FurniturePipeline created")

    # =========================================================================
    # Step 4: 전체 파이프라인 실행 (탐지 + 병렬 3D + 부피 계산)
    # =========================================================================
    print("\n[Step 4] Running full pipeline...")
    print("=" * 70)

    total_start = time.time()

    # 각 이미지 처리 결과 수집
    all_results = []
    total_objects_detected = 0
    total_objects_with_3d = 0

    for user_image_id, image_url in image_items:
        print(f"\n--- Processing Image {user_image_id}: {os.path.basename(image_url)} ---")

        img_start = time.time()

        # 파이프라인 실행 (use_parallel_3d=True)
        result = await pipeline.process_single_image(
            image_url=image_url,
            enable_mask=True,
            enable_3d=True,
            use_parallel_3d=True  # Worker Pool 사용
        )

        result.user_image_id = user_image_id
        img_time = time.time() - img_start

        # 결과 분석
        objects = result.objects
        objects_with_dims = [o for o in objects if o.relative_dimensions]

        total_objects_detected += len(objects)
        total_objects_with_3d += len(objects_with_dims)

        print(f"  Detected: {len(objects)} objects")
        print(f"  3D Generated: {len(objects_with_dims)} objects")
        print(f"  Processing time: {img_time:.2f}s")

        for obj in objects:
            mask_status = "mask" if obj.mask_base64 else "no-mask"
            dims_status = "3D OK" if obj.relative_dimensions else "no-3D"
            volume = obj.relative_dimensions.get("volume", 0) if obj.relative_dimensions else 0
            print(f"    - [{obj.id}] {obj.label} (conf={obj.confidence:.2f}, {mask_status}, {dims_status}, vol={volume:.4f})")

        all_results.append(result)

    total_time = time.time() - total_start

    # =========================================================================
    # Step 5: 결과 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 5] Results Summary")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Total images processed: {len(all_results)}")
    print(f"  Total objects detected: {total_objects_detected}")
    print(f"  Total objects with 3D: {total_objects_with_3d}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average per image: {total_time/len(all_results):.2f}s")
    if total_objects_with_3d > 0:
        print(f"  Average per 3D object: {total_time/total_objects_with_3d:.2f}s")

    # 이미지별 요약
    print(f"\nPer-Image Summary:")
    for result in all_results:
        objects_with_3d = [o for o in result.objects if o.relative_dimensions]
        print(f"  Image {result.user_image_id}: {len(result.objects)} detected, {len(objects_with_3d)} with 3D, {result.processing_time_seconds:.2f}s")

    # =========================================================================
    # Step 6: TDD Section 4.1 JSON 형식 출력
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 6] TDD Section 4.1 JSON Output")
    print("=" * 70)

    json_response = pipeline.to_json_response_v2(all_results)

    print("\nJSON Response:")
    print(json.dumps(json_response, indent=2, ensure_ascii=False))

    # JSON 파일로 저장
    output_path = "test_e2e_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_response, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to: {output_path}")

    # =========================================================================
    # Step 7: 검증
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Step 7] Validation")
    print("=" * 70)

    # 검증 항목
    validations = []

    # 1. 모든 이미지가 처리되었는지
    all_images_processed = len(json_response["results"]) == len(image_items)
    validations.append(("All images processed", all_images_processed, f"{len(json_response['results'])}/{len(image_items)}"))

    # 2. image_id가 올바르게 매핑되었는지
    expected_ids = set(item[0] for item in image_items)
    actual_ids = set(r["image_id"] for r in json_response["results"])
    ids_correct = expected_ids == actual_ids
    validations.append(("Image IDs correct", ids_correct, f"expected={expected_ids}, actual={actual_ids}"))

    # 3. 3D가 생성된 객체가 있는지
    has_3d_objects = total_objects_with_3d > 0
    validations.append(("Has 3D objects", has_3d_objects, f"{total_objects_with_3d} objects"))

    # 4. 부피가 계산되었는지
    total_volume = sum(
        obj.get("volume", 0)
        for result in json_response["results"]
        for obj in result["objects"]
    )
    has_volume = total_volume > 0
    validations.append(("Volume calculated", has_volume, f"total={total_volume:.4f}"))

    # 결과 출력
    print("\nValidation Results:")
    all_passed = True
    for name, passed, detail in validations:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Cleanup] Shutting down Worker Pool...")
    print("=" * 70)

    await sam3d_pool.shutdown()
    print("Done.")

    # 최종 결과
    print("\n" + "=" * 70)
    if all_passed:
        print("E2E TEST PASSED")
    else:
        print("E2E TEST FAILED")
    print("=" * 70)

    return all_passed, json_response


if __name__ == "__main__":
    passed, results = asyncio.run(run_e2e_test())
    sys.exit(0 if passed else 1)
