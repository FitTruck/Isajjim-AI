"""
Tests for ai/processors/7_volume_calculate.py

VolumeCalculator 클래스의 단위 테스트:
- PLY 파일 부피 계산
- GLB 파일 부피 계산
- Gaussian Splat 부피 계산
- 점군 기반 치수 계산
"""

import pytest
import numpy as np
import tempfile
import os
import importlib
from unittest.mock import MagicMock

from ai.processors import VolumeCalculator

# 숫자로 시작하는 모듈은 직접 import
_stage7 = importlib.import_module('.7_volume_calculate', package='ai.processors')


class TestVolumeCalculatorInit:
    """VolumeCalculator 초기화 테스트"""

    def test_init(self):
        """초기화 테스트"""
        calc = VolumeCalculator()
        assert calc is not None


class TestCalculateFromPoints:
    """_calculate_from_points 내부 함수 테스트"""

    def test_empty_points(self):
        """빈 점군"""
        calc = VolumeCalculator()
        points = np.array([]).reshape(0, 3)
        result = calc._calculate_from_points(points)

        assert result["bounding_box"]["width"] == 0
        assert result["bounding_box"]["depth"] == 0
        assert result["bounding_box"]["height"] == 0

    def test_single_point(self):
        """단일 점"""
        calc = VolumeCalculator()
        points = np.array([[1.0, 2.0, 3.0]])
        result = calc._calculate_from_points(points)

        assert result["bounding_box"]["width"] == 0
        assert result["centroid"] == [1.0, 2.0, 3.0]

    def test_cube_points(self):
        """정육면체 점군"""
        calc = VolumeCalculator()
        # 1x1x1 정육면체의 꼭지점들
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=float)
        result = calc._calculate_from_points(points)

        assert result["bounding_box"]["width"] == pytest.approx(1.0)
        assert result["bounding_box"]["depth"] == pytest.approx(1.0)
        assert result["bounding_box"]["height"] == pytest.approx(1.0)

    def test_rectangular_box_points(self):
        """직육면체 점군"""
        calc = VolumeCalculator()
        # 2x3x4 직육면체 (X=width, Y=height, Z=depth)
        points = np.array([
            [0, 0, 0], [2, 0, 0], [0, 4, 0], [2, 4, 0],
            [0, 0, 3], [2, 0, 3], [0, 4, 3], [2, 4, 3]
        ], dtype=float)
        result = calc._calculate_from_points(points)

        assert result["bounding_box"]["width"] == pytest.approx(2.0)
        assert result["bounding_box"]["depth"] == pytest.approx(3.0)
        assert result["bounding_box"]["height"] == pytest.approx(4.0)

    def test_centroid_calculation(self):
        """중심점 계산"""
        calc = VolumeCalculator()
        points = np.array([
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
            [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2]
        ], dtype=float)
        result = calc._calculate_from_points(points)

        assert result["centroid"] == pytest.approx([1.0, 1.0, 1.0])


class TestCalculateFromPly:
    """calculate_from_ply 함수 테스트"""

    def test_nonexistent_file(self):
        """존재하지 않는 파일"""
        calc = VolumeCalculator()
        result = calc.calculate_from_ply("/nonexistent/path.ply")
        assert result is None

    def test_no_trimesh(self):
        """trimesh가 없는 경우"""
        calc = VolumeCalculator()
        original_value = _stage7.HAS_TRIMESH
        try:
            _stage7.HAS_TRIMESH = False
            result = calc.calculate_from_ply("test.ply")
            assert result is None
        finally:
            _stage7.HAS_TRIMESH = original_value

    def test_with_mock_trimesh(self):
        """trimesh 모킹 테스트"""
        calc = VolumeCalculator()

        # 임시 PLY 파일 생성 (ASCII format)
        ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
3 0 1 2
3 0 2 3
3 4 6 5
3 4 7 6
3 0 4 5
3 0 5 1
3 2 6 7
3 2 7 3
3 0 3 7
3 0 7 4
3 1 5 6
3 1 6 2
"""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False, mode='w') as f:
            f.write(ply_content)
            ply_path = f.name

        try:
            result = calc.calculate_from_ply(ply_path)
            if result is not None:  # trimesh가 설치된 경우
                assert "bounding_box" in result
                assert "centroid" in result
                assert "surface_area" in result
        finally:
            os.unlink(ply_path)


class TestCalculateFromGlb:
    """calculate_from_glb 함수 테스트"""

    def test_nonexistent_file(self):
        """존재하지 않는 파일"""
        calc = VolumeCalculator()
        result = calc.calculate_from_glb("/nonexistent/path.glb")
        assert result is None

    def test_no_trimesh(self):
        """trimesh가 없는 경우"""
        calc = VolumeCalculator()
        original_value = _stage7.HAS_TRIMESH
        try:
            _stage7.HAS_TRIMESH = False
            result = calc.calculate_from_glb("test.glb")
            assert result is None
        finally:
            _stage7.HAS_TRIMESH = original_value


class TestCalculateFromGaussianSplat:
    """calculate_from_gaussian_splat 함수 테스트"""

    def test_no_torch(self):
        """torch가 없는 경우"""
        calc = VolumeCalculator()
        original_value = _stage7.HAS_TORCH
        try:
            _stage7.HAS_TORCH = False
            result = calc.calculate_from_gaussian_splat(None)
            assert result is None
        finally:
            _stage7.HAS_TORCH = original_value

    def test_with_mock_gaussian_splat(self):
        """Gaussian Splat 모킹 테스트"""
        import torch
        calc = VolumeCalculator()

        # Mock Gaussian Splat 객체
        mock_splat = MagicMock()
        mock_splat.get_xyz = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        result = calc.calculate_from_gaussian_splat(mock_splat)

        if result is not None:  # trimesh가 설치된 경우
            assert "bounding_box" in result
            assert "centroid" in result

    def test_with_numpy_array_splat(self):
        """NumPy 배열로 된 Gaussian Splat"""
        calc = VolumeCalculator()

        mock_splat = MagicMock()
        mock_splat.get_xyz = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])

        result = calc.calculate_from_gaussian_splat(mock_splat)

        if result is not None:
            # OBB 기반이므로 정확한 축 매핑은 점 분포에 따라 다름
            # 최소한 결과가 올바른 구조를 가지는지 확인
            assert "bounding_box" in result
            assert result["bounding_box"]["width"] >= 0
            assert result["bounding_box"]["depth"] >= 0
            assert result["bounding_box"]["height"] >= 0

    def test_with_few_points(self):
        """점이 너무 적은 경우"""
        calc = VolumeCalculator()

        mock_splat = MagicMock()
        mock_splat.get_xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        result = calc.calculate_from_gaussian_splat(mock_splat)

        if result is not None:
            # 4개 미만이면 _calculate_from_points 사용
            assert "bounding_box" in result


class TestAnalyzeMesh:
    """_analyze_mesh 내부 함수 테스트 (trimesh 필요)"""

    def _create_mock_obb(self, width, depth, height):
        """OBB mock 생성 (axis-aligned)"""
        mock_obb = MagicMock()
        mock_obb.primitive.extents = np.array([width, height, depth])  # OBB 순서: width, height, depth
        mock_obb.primitive.transform = np.eye(4)  # 단위 행렬 = 축 정렬
        return mock_obb

    def test_with_mock_mesh(self):
        """Mock 메시로 테스트"""
        calc = VolumeCalculator()

        # Mock trimesh 객체 (OBB 기반)
        mock_mesh = MagicMock()
        mock_mesh.bounding_box_oriented = self._create_mock_obb(2.0, 3.0, 4.0)
        mock_mesh.centroid = np.array([1.0, 1.5, 2.0])
        mock_mesh.area = 52.0

        result = calc._analyze_mesh(mock_mesh)

        assert result["bounding_box"]["width"] == pytest.approx(2.0)
        assert result["bounding_box"]["depth"] == pytest.approx(3.0)
        assert result["bounding_box"]["height"] == pytest.approx(4.0)
        assert result["centroid"] == pytest.approx([1.0, 1.5, 2.0])
        assert result["surface_area"] == pytest.approx(52.0)

    def test_obb_fallback_to_aabb(self):
        """OBB 실패 시 AABB fallback"""
        calc = VolumeCalculator()

        mock_mesh = MagicMock()
        # OBB 접근 시 예외 발생
        type(mock_mesh).bounding_box_oriented = property(
            lambda self: (_ for _ in ()).throw(Exception("OBB error"))
        )
        mock_mesh.bounds = np.array([[0, 0, 0], [1, 2, 3]])
        mock_mesh.centroid = np.array([0.5, 1.0, 1.5])
        mock_mesh.area = 6.0

        result = calc._analyze_mesh(mock_mesh)

        # AABB fallback: X=width, Y=height, Z=depth
        assert result["bounding_box"]["width"] == pytest.approx(1.0)
        assert result["bounding_box"]["height"] == pytest.approx(2.0)
        assert result["bounding_box"]["depth"] == pytest.approx(3.0)

    def test_zero_dimensions(self):
        """0 크기 메시"""
        calc = VolumeCalculator()

        mock_mesh = MagicMock()
        mock_mesh.bounding_box_oriented = self._create_mock_obb(0.0, 0.0, 0.0)
        mock_mesh.centroid = np.array([0, 0, 0])
        mock_mesh.area = 0.0

        result = calc._analyze_mesh(mock_mesh)

        # 0 크기 메시도 처리 가능
        assert result["bounding_box"]["width"] == pytest.approx(0.0)
        assert result["bounding_box"]["depth"] == pytest.approx(0.0)
        assert result["bounding_box"]["height"] == pytest.approx(0.0)


class TestResultStructure:
    """결과 구조 테스트"""

    def test_result_has_all_fields(self):
        """결과에 필요한 모든 필드가 있는지 확인"""
        calc = VolumeCalculator()
        points = np.array([
            [0, 0, 0], [1, 1, 1]
        ], dtype=float)
        result = calc._calculate_from_points(points)

        assert "bounding_box" in result
        assert "width" in result["bounding_box"]
        assert "depth" in result["bounding_box"]
        assert "height" in result["bounding_box"]
        assert "centroid" in result
        assert "surface_area" in result

    def test_result_types(self):
        """결과 필드 타입 확인"""
        calc = VolumeCalculator()
        points = np.array([
            [0, 0, 0], [1, 1, 1]
        ], dtype=float)
        result = calc._calculate_from_points(points)

        assert isinstance(result["bounding_box"]["width"], (int, float))
        assert isinstance(result["centroid"], list)
        assert len(result["centroid"]) == 3
