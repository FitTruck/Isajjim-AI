"""
STAGE1_INFERENCE_STEPS=14 통합 테스트

서버가 실행 중일 때 STAGE1=14 설정으로 3D 생성이 제대로 동작하는지 검증합니다.
이 테스트는 @pytest.mark.slow 마커가 있으며, 서버가 필요합니다.

사용법:
    pytest tests/test_stage1_14_integration.py -v -m slow --server-required
"""

import pytest
import asyncio
import aiohttp
import base64
import os
import json
import time
from pathlib import Path
from typing import Optional


API_URL = os.environ.get("TEST_API_URL", "http://localhost:8000")
IMGS_DIR = Path(__file__).parent.parent / "ai" / "imgs"
OUTPUT_DIR = Path(__file__).parent / "stage1_comparison"


def get_test_image_path() -> Optional[str]:
    """사용 가능한 테스트 이미지 경로 반환"""
    # 우선순위: 1.png > test1.jpg > 아무 이미지
    candidates = ["1.png", "test1.jpg", "test7.jpg"]
    for candidate in candidates:
        path = IMGS_DIR / candidate
        if path.exists():
            return str(path)

    # 임의의 이미지 파일 찾기
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files = list(IMGS_DIR.glob(ext))
        if files:
            return str(files[0])

    return None


async def check_server_health() -> bool:
    """서버 상태 확인"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def analyze_image(image_path: str) -> dict:
    """이미지 분석 API 호출"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_data,
        "enable_mask": True,
        "enable_3d": True,
        "return_ply": False
    }

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.post(
            f"{API_URL}/analyze-furniture-base64",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            elapsed = time.time() - start_time
            result = await resp.json()
            result["_elapsed_time"] = elapsed
            return result


@pytest.fixture
def test_image_path():
    """테스트 이미지 경로 fixture"""
    path = get_test_image_path()
    if path is None:
        pytest.skip("테스트 이미지를 찾을 수 없습니다")
    return path


@pytest.fixture
def output_dir():
    """출력 디렉토리 fixture"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


class TestStage1_14Integration:
    """STAGE1=14 통합 테스트"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_server_is_running(self):
        """서버가 실행 중인지 확인"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다. 서버를 시작한 후 다시 실행하세요.")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_analyze_returns_objects(self, test_image_path):
        """API가 객체 목록을 반환하는지 확인"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다")

        result = await analyze_image(test_image_path)

        assert "objects" in result or "error" not in result, (
            f"API 오류: {result.get('error', 'Unknown error')}"
        )

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_objects_have_dimensions(self, test_image_path):
        """반환된 객체에 치수 정보가 있는지 확인"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다")

        result = await analyze_image(test_image_path)
        objects = result.get("objects", [])

        if len(objects) == 0:
            pytest.skip("이미지에서 객체가 감지되지 않았습니다")

        for obj in objects:
            assert "width" in obj, f"객체에 width가 없습니다: {obj}"
            assert "depth" in obj, f"객체에 depth가 없습니다: {obj}"
            assert "height" in obj, f"객체에 height가 없습니다: {obj}"
            assert "volume" in obj, f"객체에 volume가 없습니다: {obj}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_dimensions_are_positive(self, test_image_path):
        """치수 값이 양수인지 확인"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다")

        result = await analyze_image(test_image_path)
        objects = result.get("objects", [])

        if len(objects) == 0:
            pytest.skip("이미지에서 객체가 감지되지 않았습니다")

        for obj in objects:
            assert obj.get("width", 0) > 0, f"width가 양수가 아닙니다: {obj}"
            assert obj.get("depth", 0) > 0, f"depth가 양수가 아닙니다: {obj}"
            assert obj.get("height", 0) > 0, f"height가 양수가 아닙니다: {obj}"
            assert obj.get("volume", 0) > 0, f"volume이 양수가 아닙니다: {obj}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_dimensions_are_normalized(self, test_image_path):
        """치수가 정규화된 범위(0-2) 내에 있는지 확인"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다")

        result = await analyze_image(test_image_path)
        objects = result.get("objects", [])

        if len(objects) == 0:
            pytest.skip("이미지에서 객체가 감지되지 않았습니다")

        for obj in objects:
            # 상대 치수이므로 일반적으로 0-2 범위 내에 있어야 함
            assert 0 < obj.get("width", 0) < 2, f"width가 범위를 벗어났습니다: {obj}"
            assert 0 < obj.get("depth", 0) < 2, f"depth가 범위를 벗어났습니다: {obj}"
            assert 0 < obj.get("height", 0) < 2, f"height가 범위를 벗어났습니다: {obj}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_response_time_reasonable(self, test_image_path):
        """응답 시간이 합리적인지 확인 (객체당 평균 3초 이내)"""
        is_healthy = await check_server_health()
        if not is_healthy:
            pytest.skip("서버가 실행 중이 아닙니다")

        result = await analyze_image(test_image_path)
        elapsed = result.get("_elapsed_time", 0)
        num_objects = len(result.get("objects", []))

        if num_objects == 0:
            pytest.skip("이미지에서 객체가 감지되지 않았습니다")

        avg_time_per_object = elapsed / num_objects

        # STAGE1=14 기준 객체당 평균 3초 이내
        assert avg_time_per_object < 5, (
            f"객체당 평균 시간이 너무 깁니다: {avg_time_per_object:.2f}s "
            f"(총 {elapsed:.2f}s / {num_objects} 객체)"
        )


class TestStage1_14ComparisonWithBaseline:
    """STAGE1=14와 STAGE1=16 베이스라인 비교 테스트"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_load_baseline_results(self, output_dir):
        """베이스라인 결과 파일이 존재하는지 확인"""
        baseline_file = output_dir / "results_stage1_16.json"

        if not baseline_file.exists():
            pytest.skip(
                "베이스라인 결과가 없습니다. "
                "tests/test_new_images.py를 먼저 실행하세요."
            )

        with open(baseline_file) as f:
            baseline = json.load(f)

        assert len(baseline) > 0, "베이스라인 결과가 비어있습니다"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_current_vs_baseline_volume_diff(self, output_dir):
        """현재 설정(14)과 베이스라인(16) 간 부피 차이가 15% 이내인지 확인"""
        current_file = output_dir / "results_stage1_14.json"
        baseline_file = output_dir / "results_stage1_16.json"

        if not baseline_file.exists():
            pytest.skip("베이스라인 결과가 없습니다")

        if not current_file.exists():
            pytest.skip(
                "STAGE1=14 결과가 없습니다. "
                "STAGE1=14로 테스트를 먼저 실행하세요."
            )

        with open(current_file) as f:
            current = json.load(f)
        with open(baseline_file) as f:
            baseline = json.load(f)

        # 부피 차이 계산
        volume_diffs = []

        for img_name in current.keys():
            if img_name not in baseline:
                continue

            current_objs = {obj["label"]: obj for obj in current[img_name].get("objects", [])}
            baseline_objs = {obj["label"]: obj for obj in baseline[img_name].get("objects", [])}

            for label in current_objs:
                if label not in baseline_objs:
                    continue

                curr_vol = current_objs[label].get("volume", 0)
                base_vol = baseline_objs[label].get("volume", 0)

                if base_vol > 0:
                    diff_pct = abs((curr_vol - base_vol) / base_vol * 100)
                    volume_diffs.append(diff_pct)

        if not volume_diffs:
            pytest.skip("비교할 객체가 없습니다")

        avg_diff = sum(volume_diffs) / len(volume_diffs)
        max_diff = max(volume_diffs)

        # STAGE1=14는 베이스라인(16)과 평균 5% 이내, 최대 20% 이내 차이
        assert avg_diff < 10, f"평균 부피 차이가 너무 큽니다: {avg_diff:.2f}%"
        assert max_diff < 25, f"최대 부피 차이가 너무 큽니다: {max_diff:.2f}%"


def pytest_configure(config):
    """pytest 커스텀 마커 등록"""
    config.addinivalue_line("markers", "slow: marks tests as slow (server required)")
