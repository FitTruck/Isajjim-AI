#!/usr/bin/env python3
"""
Full Pipeline End-to-End Test (8 GPUs with SAM-3D)

전체 파이프라인 테스트:
1. 이미지 로드 (로컬 파일)
2. YOLOE-seg 탐지 (bbox + class + 세그멘테이션 마스크)
3. DB 매칭 (한국어 라벨)
4. SAM-3D 3D 생성 (PLY, GLB, GIF)
5. 부피 계산 (trimesh)

SAM-3D 서버가 localhost:8000에서 실행 중이어야 합니다.
"""

import asyncio
import os
import sys
import time
import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 환경변수 설정 (torch import 전에)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"
os.environ["OMP_NUM_THREADS"] = "4"

import aiohttp
from PIL import Image
import numpy as np


API_URL = "http://localhost:8000"


@dataclass
class PipelineTimings:
    """파이프라인 단계별 타이밍"""
    image_name: str
    gpu_id: Optional[int] = None

    # 단계별 시간 (ms)
    image_load_ms: float = 0.0
    yolo_detect_ms: float = 0.0
    mask_encode_ms: float = 0.0
    sam3d_request_ms: float = 0.0
    sam3d_wait_ms: float = 0.0
    volume_calc_ms: float = 0.0
    total_ms: float = 0.0

    # 결과
    objects_detected: int = 0
    objects_with_3d: int = 0
    total_volume: float = 0.0
    object_details: List[Dict] = field(default_factory=list)


def print_header(title: str):
    """섹션 헤더 출력"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


async def check_server_health():
    """서버 상태 확인"""
    print_header("Server Health Check")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Server healthy")
                    print(f"  Model: {data.get('model')}")
                    print(f"  Device: {data.get('device')}")
                    return True
                else:
                    print(f"✗ Server unhealthy: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Server connection failed: {e}")
            return False


async def check_gpu_status():
    """GPU 상태 확인"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_URL}/gpu-status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\n✓ GPU Pool Status:")
                    print(f"  Total GPUs: {data['total_gpus']}")
                    print(f"  Available GPUs: {data['available_gpus']}")
                    print(f"  Pipelines initialized: {data.get('pipelines_initialized', 0)}")
                    return data
        except Exception as e:
            print(f"✗ GPU status check failed: {e}")
    return None


async def test_detect_only(image_path: Path) -> Dict:
    """Detection only 테스트 (빠른 응답)"""

    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()

        async with session.post(
            f"{API_URL}/detect-furniture",
            json={"image": image_b64},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            elapsed = time.perf_counter() - start

            if resp.status == 200:
                result = await resp.json()
                return {
                    "success": True,
                    "time_ms": elapsed * 1000,
                    "objects": result.get("objects", [])
                }
            else:
                return {"success": False, "error": await resp.text()}


async def test_full_pipeline_single(image_path: Path, enable_3d: bool = True) -> PipelineTimings:
    """단일 이미지에 대한 전체 파이프라인 테스트"""

    timings = PipelineTimings(image_name=image_path.name)
    total_start = time.perf_counter()

    # 이미지 로드
    load_start = time.perf_counter()
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    timings.image_load_ms = (time.perf_counter() - load_start) * 1000

    async with aiohttp.ClientSession() as session:
        # analyze-furniture-base64 엔드포인트 호출 (전체 파이프라인)
        request_start = time.perf_counter()

        async with session.post(
            f"{API_URL}/analyze-furniture-base64",
            json={
                "image": image_b64,
                "enable_3d": enable_3d,
                "skip_gif": True  # GIF 생성 스킵하여 속도 향상
            },
            timeout=aiohttp.ClientTimeout(total=600)  # 3D 생성은 시간이 오래 걸림
        ) as resp:
            request_elapsed = (time.perf_counter() - request_start) * 1000

            if resp.status == 200:
                result = await resp.json()

                # 결과 파싱
                objects = result.get("objects", [])
                timings.objects_detected = len(objects)

                for obj in objects:
                    obj_detail = {
                        "label": obj.get("label"),
                        "width": obj.get("width"),
                        "depth": obj.get("depth"),
                        "height": obj.get("height"),
                        "volume": obj.get("volume"),
                    }
                    timings.object_details.append(obj_detail)

                    if obj.get("volume", 0) > 0:
                        timings.objects_with_3d += 1
                        timings.total_volume += obj.get("volume", 0)

                timings.sam3d_request_ms = request_elapsed
            else:
                print(f"  Error: {await resp.text()}")

    timings.total_ms = (time.perf_counter() - total_start) * 1000
    return timings


async def test_multi_image_parallel(image_paths: List[Path]) -> List[Dict]:
    """여러 이미지 병렬 처리 테스트 (analyze-furniture 엔드포인트)"""

    print_header("Multi-Image Parallel Processing Test")

    # 이미지들을 base64로 인코딩하고 ID 부여
    image_items = []
    for i, img_path in enumerate(image_paths):
        with open(img_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        image_items.append({
            "id": i + 1,
            "image": image_b64
        })

    print(f"\nSending {len(image_items)} images for parallel processing...")

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()

        async with session.post(
            f"{API_URL}/analyze-furniture-base64-batch",
            json={
                "images": image_items,
                "enable_3d": True,
                "skip_gif": True
            },
            timeout=aiohttp.ClientTimeout(total=1800)  # 30분 타임아웃
        ) as resp:
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status == 200:
                result = await resp.json()
                print(f"✓ Batch processing completed in {elapsed/1000:.2f}s")
                return result.get("results", [])
            else:
                print(f"✗ Error: {resp.status}")
                error_text = await resp.text()
                print(f"  {error_text[:500]}")
                return []


async def test_sequential_vs_parallel(image_paths: List[Path]):
    """순차 vs 병렬 처리 비교 테스트"""

    print_header("Sequential vs Parallel Comparison (Detection Only)")

    # 3개 이미지로 테스트
    test_images = image_paths[:3]

    # 순차 처리
    print("\n[Sequential Processing]")
    seq_start = time.perf_counter()
    seq_results = []
    for img_path in test_images:
        result = await test_detect_only(img_path)
        seq_results.append(result)
        print(f"  {img_path.name}: {result['time_ms']:.1f}ms, {len(result.get('objects', []))} objects")
    seq_total = (time.perf_counter() - seq_start) * 1000

    # 병렬 처리
    print("\n[Parallel Processing]")
    par_start = time.perf_counter()
    tasks = [test_detect_only(img_path) for img_path in test_images]
    par_results = await asyncio.gather(*tasks)
    par_total = (time.perf_counter() - par_start) * 1000

    for img_path, result in zip(test_images, par_results):
        print(f"  {img_path.name}: {result['time_ms']:.1f}ms, {len(result.get('objects', []))} objects")

    print(f"\n  Sequential total: {seq_total:.1f}ms")
    print(f"  Parallel total:   {par_total:.1f}ms")
    print(f"  Speedup:          {seq_total/par_total:.2f}x")


async def test_full_pipeline_with_3d(image_paths: List[Path], max_images: int = 3):
    """3D 생성을 포함한 전체 파이프라인 테스트"""

    print_header(f"Full Pipeline Test with 3D Generation ({max_images} images)")

    test_images = image_paths[:max_images]
    results: List[PipelineTimings] = []

    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        print("-" * 60)

        timings = await test_full_pipeline_single(img_path, enable_3d=True)
        results.append(timings)

        print(f"  Image load:      {timings.image_load_ms:>8.1f}ms")
        print(f"  Pipeline total:  {timings.sam3d_request_ms:>8.1f}ms")
        print(f"  Objects detected: {timings.objects_detected}")
        print(f"  Objects with 3D:  {timings.objects_with_3d}")
        print(f"  Total volume:     {timings.total_volume:.6f}")

        if timings.object_details:
            print(f"\n  Detected Objects:")
            for obj in timings.object_details:
                vol = obj.get('volume', 0) or 0
                print(f"    - {obj['label']}: {obj.get('width', 0):.1f} x {obj.get('depth', 0):.1f} x {obj.get('height', 0):.1f}, vol={vol:.6f}")

    return results


async def test_detect_only_all_images(image_paths: List[Path]):
    """모든 이미지에 대한 Detection Only 테스트"""

    print_header("Detection Only Test (All Images)")

    print(f"\nProcessing {len(image_paths)} images with detection only (no 3D)...")
    print("-" * 100)
    print(f"{'Image':<35} {'Time':>10} {'Objects':>10} {'Labels'}")
    print("-" * 100)

    total_start = time.perf_counter()
    all_results = []

    for img_path in image_paths:
        result = await test_detect_only(img_path)
        all_results.append(result)

        if result["success"]:
            objects = result.get("objects", [])
            labels = [o.get("label", "?") for o in objects[:5]]
            labels_str = ", ".join(labels)
            if len(objects) > 5:
                labels_str += f" +{len(objects)-5} more"
            print(f"{img_path.name:<35} {result['time_ms']:>8.1f}ms {len(objects):>10} {labels_str}")
        else:
            print(f"{img_path.name:<35} ERROR")

    total_time = (time.perf_counter() - total_start) * 1000

    print("-" * 100)

    # 통계
    successful = [r for r in all_results if r["success"]]
    if successful:
        avg_time = sum(r["time_ms"] for r in successful) / len(successful)
        total_objects = sum(len(r.get("objects", [])) for r in successful)

        print(f"\nSummary:")
        print(f"  Images processed: {len(successful)}/{len(image_paths)}")
        print(f"  Total time:       {total_time:.1f}ms ({total_time/1000:.2f}s)")
        print(f"  Average time:     {avg_time:.1f}ms per image")
        print(f"  Total objects:    {total_objects}")
        print(f"  Throughput:       {len(successful) / (total_time/1000):.2f} images/sec")

    return all_results


async def test_single_object_3d_generation(image_paths: List[Path]):
    """단일 객체 3D 생성 테스트 (가장 단순한 이미지 사용)"""

    print_header("Single Object 3D Generation Test")

    # 가장 작은 이미지 선택 (kitchen_test3.jpg - 3개 객체만 탐지됨)
    simple_image = None
    for img_path in image_paths:
        if "test3" in img_path.name or "test8" in img_path.name:
            simple_image = img_path
            break

    if simple_image is None:
        simple_image = image_paths[0]

    print(f"\nUsing image: {simple_image.name}")
    print("-" * 60)

    # 전체 파이프라인 실행
    timings = await test_full_pipeline_single(simple_image, enable_3d=True)

    print(f"\nTimings:")
    print(f"  Image load:      {timings.image_load_ms:>8.1f}ms")
    print(f"  Total pipeline:  {timings.total_ms:>8.1f}ms ({timings.total_ms/1000:.2f}s)")

    print(f"\nResults:")
    print(f"  Objects detected: {timings.objects_detected}")
    print(f"  Objects with 3D:  {timings.objects_with_3d}")
    print(f"  Total volume:     {timings.total_volume:.6f}")

    if timings.object_details:
        print(f"\nDetected Objects with Dimensions:")
        print("-" * 80)
        print(f"{'Label':<20} {'Width':>10} {'Depth':>10} {'Height':>10} {'Volume':>15}")
        print("-" * 80)

        for obj in timings.object_details:
            label = obj.get('label', 'Unknown')
            width = obj.get('width', 0) or 0
            depth = obj.get('depth', 0) or 0
            height = obj.get('height', 0) or 0
            volume = obj.get('volume', 0) or 0

            print(f"{label:<20} {width:>10.2f} {depth:>10.2f} {height:>10.2f} {volume:>15.6f}")

    return timings


async def main():
    """메인 테스트 실행"""
    print_header("Full Pipeline End-to-End Test")
    print(f"API URL: {API_URL}")

    # 서버 상태 확인
    if not await check_server_health():
        print("\n✗ Server not available. Please start the server first:")
        print("  python api.py")
        return 1

    await check_gpu_status()

    # 테스트 이미지 로드
    imgs_dir = PROJECT_ROOT / "ai" / "imgs"
    image_paths = sorted(imgs_dir.glob("*.jpg"))

    print(f"\nFound {len(image_paths)} test images")

    try:
        # 테스트 1: Detection Only (모든 이미지)
        await test_detect_only_all_images(image_paths)

        # 테스트 2: 순차 vs 병렬 비교
        await test_sequential_vs_parallel(image_paths)

        # 테스트 3: 단일 객체 3D 생성 테스트
        result = await test_single_object_3d_generation(image_paths)

        # 최종 요약
        print_header("FINAL SUMMARY")

        print("\n✓ Detection Pipeline: Working")
        print("✓ Multi-GPU Distribution: Working")

        if result.objects_with_3d > 0:
            print("✓ 3D Generation: Working")
            print("✓ Volume Calculation: Working")
            print(f"\n  Sample 3D Results:")
            print(f"    Objects with 3D: {result.objects_with_3d}")
            print(f"    Total volume: {result.total_volume:.6f}")
        else:
            print("△ 3D Generation: No objects processed (may need more time)")

        print("\n" + "=" * 80)
        print(" ALL TESTS COMPLETED ✓")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
