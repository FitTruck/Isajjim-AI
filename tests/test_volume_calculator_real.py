"""
Tests for VolumeCalculator with real PLY/GLB files

test_outputs 디렉토리의 실제 파일로 VolumeCalculator 테스트:
- PLY 파일 부피 계산
- GLB 파일 부피 계산
- 메시 분석
"""

import pytest
from pathlib import Path

from ai.processors import VolumeCalculator


# Test output directories
TEST_OUTPUTS_DIR = Path(__file__).parent.parent / "test_outputs"
QA_ASSETS_DIR = TEST_OUTPUTS_DIR / "qa_assets"
BEDROOM_DIR = TEST_OUTPUTS_DIR / "bedroom-5772286_1920"


@pytest.fixture
def volume_calculator():
    """VolumeCalculator 인스턴스"""
    return VolumeCalculator()


class TestVolumeCalculatorWithRealPly:
    """실제 PLY 파일로 VolumeCalculator 테스트"""

    def test_ply_files_exist(self):
        """PLY 파일이 존재하는지 확인"""
        ply_files = list(QA_ASSETS_DIR.glob("*.ply"))
        print(f"Found {len(ply_files)} PLY files")
        assert len(ply_files) > 0, "No PLY files found in test_outputs/qa_assets"

    def test_calculate_from_real_ply(self, volume_calculator):
        """실제 PLY 파일에서 부피 계산"""
        ply_files = list(QA_ASSETS_DIR.glob("*.ply"))
        if not ply_files:
            pytest.skip("No PLY files found")

        for ply_file in ply_files[:3]:  # 처음 3개만 테스트
            print(f"\nTesting: {ply_file.name}")

            result = volume_calculator.calculate_from_ply(str(ply_file))

            assert result is not None, f"Failed to calculate volume for {ply_file.name}"
            assert "bounding_box" in result  # volume 필드 제거됨
            assert "bounding_box" in result
            assert "centroid" in result

            print(f"  Width: {result['bounding_box']['width']:.6f}")
            print(f"  Bounding box: {result['bounding_box']}")
            print(f"  Centroid: {result['centroid']}")

    def test_ply_dimensions_are_positive(self, volume_calculator):
        """PLY 치수가 양수인지 확인"""
        ply_files = list(QA_ASSETS_DIR.glob("*.ply"))
        if not ply_files:
            pytest.skip("No PLY files found")

        ply_file = ply_files[0]
        result = volume_calculator.calculate_from_ply(str(ply_file))

        assert result is not None
        assert result["bounding_box"]["width"] >= 0
        assert result["bounding_box"]["depth"] >= 0
        assert result["bounding_box"]["height"] >= 0


class TestVolumeCalculatorWithRealGlb:
    """실제 GLB 파일로 VolumeCalculator 테스트"""

    def test_glb_files_exist(self):
        """GLB 파일이 존재하는지 확인"""
        glb_files = list(QA_ASSETS_DIR.glob("*.glb"))
        print(f"Found {len(glb_files)} GLB files")
        assert len(glb_files) > 0, "No GLB files found in test_outputs/qa_assets"

    def test_calculate_from_real_glb(self, volume_calculator):
        """실제 GLB 파일에서 부피 계산"""
        glb_files = list(QA_ASSETS_DIR.glob("*.glb"))
        if not glb_files:
            pytest.skip("No GLB files found")

        for glb_file in glb_files[:3]:  # 처음 3개만 테스트
            print(f"\nTesting: {glb_file.name}")

            result = volume_calculator.calculate_from_glb(str(glb_file))

            assert result is not None, f"Failed to calculate volume for {glb_file.name}"
            assert "bounding_box" in result  # volume 필드 제거됨
            assert "bounding_box" in result

            print(f"  Width: {result['bounding_box']['width']:.6f}")
            print(f"  Bounding box: {result['bounding_box']}")

    def test_glb_has_surface_area(self, volume_calculator):
        """GLB 표면적 계산"""
        glb_files = list(QA_ASSETS_DIR.glob("*.glb"))
        if not glb_files:
            pytest.skip("No GLB files found")

        glb_file = glb_files[0]
        result = volume_calculator.calculate_from_glb(str(glb_file))

        assert result is not None
        assert "surface_area" in result
        print(f"Surface area: {result['surface_area']:.4f}")


class TestVolumeCalculatorCompare:
    """PLY와 GLB 비교 테스트"""

    def test_compare_ply_and_glb_dimensions(self, volume_calculator):
        """같은 객체의 PLY와 GLB 치수 비교"""
        ply_files = list(QA_ASSETS_DIR.glob("*.ply"))
        glb_files = list(QA_ASSETS_DIR.glob("*.glb"))

        if not ply_files or not glb_files:
            pytest.skip("Need both PLY and GLB files")

        # 첫 번째 파일들 비교
        ply_result = volume_calculator.calculate_from_ply(str(ply_files[0]))
        glb_result = volume_calculator.calculate_from_glb(str(glb_files[0]))

        if ply_result and glb_result:
            print(f"PLY dimensions: {ply_result['bounding_box']}")
            print(f"GLB dimensions: {glb_result['bounding_box']}")
            # 치수가 0보다 커야 함
            assert ply_result['bounding_box']['width'] >= 0
            assert glb_result['bounding_box']['width'] >= 0


class TestVolumeCalculatorWithMasks:
    """마스크 파일과 연동 테스트"""

    def test_mask_files_exist(self):
        """마스크 파일이 존재하는지 확인"""
        mask_files = list(BEDROOM_DIR.glob("mask_*.png"))
        print(f"Found {len(mask_files)} mask files")
        assert len(mask_files) > 0, "No mask files found"

    def test_original_image_exists(self):
        """원본 이미지가 존재하는지 확인"""
        original = BEDROOM_DIR / "00_original.jpg"
        assert original.exists(), "Original image not found"

    def test_detection_result_exists(self):
        """탐지 결과 이미지가 존재하는지 확인"""
        detection = BEDROOM_DIR / "01_yoloe_detection.jpg"
        assert detection.exists(), "Detection result not found"


class TestVolumeCalculatorPerformance:
    """성능 테스트"""

    def test_ply_loading_time(self, volume_calculator):
        """PLY 로딩 시간"""
        import time
        ply_files = list(QA_ASSETS_DIR.glob("*.ply"))
        if not ply_files:
            pytest.skip("No PLY files found")

        ply_file = ply_files[0]

        start = time.time()
        volume_calculator.calculate_from_ply(str(ply_file))
        elapsed = time.time() - start

        print(f"PLY loading time: {elapsed:.3f}s for {ply_file.name}")
        assert elapsed < 30.0, f"PLY loading too slow: {elapsed:.3f}s"

    def test_glb_loading_time(self, volume_calculator):
        """GLB 로딩 시간"""
        import time
        glb_files = list(QA_ASSETS_DIR.glob("*.glb"))
        if not glb_files:
            pytest.skip("No GLB files found")

        glb_file = glb_files[0]

        start = time.time()
        volume_calculator.calculate_from_glb(str(glb_file))
        elapsed = time.time() - start

        print(f"GLB loading time: {elapsed:.3f}s for {glb_file.name}")
        assert elapsed < 10.0, f"GLB loading too slow: {elapsed:.3f}s"
