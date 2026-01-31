"""PLY 정렬 서비스 테스트"""

import pytest
import numpy as np
from pathlib import Path

# Open3D 선택적 import
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


# ==================== 유틸리티 함수 테스트 ====================

class TestCoordinateConversion:
    """좌표계 변환 함수 테스트"""

    def test_zup_to_yup_single_point(self):
        """단일 포인트 Z-up → Y-up 변환"""
        from simulation.ply_alignment import convert_zup_to_yup

        # Z-up: (1, 2, 3) = right=1, forward=2, up=3
        # Y-up: (1, 3, -2) = right=1, up=3, forward=-2
        points = np.array([[1.0, 2.0, 3.0]])
        result = convert_zup_to_yup(points)

        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [1.0, 3.0, -2.0])

    def test_zup_to_yup_multiple_points(self):
        """다중 포인트 변환"""
        from simulation.ply_alignment import convert_zup_to_yup

        points = np.array([
            [1.0, 0.0, 0.0],  # X축 (right)
            [0.0, 1.0, 0.0],  # Y축 (forward in Z-up)
            [0.0, 0.0, 1.0],  # Z축 (up in Z-up)
        ])
        result = convert_zup_to_yup(points)

        expected = np.array([
            [1.0, 0.0, 0.0],   # X축 유지
            [0.0, 0.0, -1.0],  # Y축 → -Z축
            [0.0, 1.0, 0.0],   # Z축 → Y축
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_yup_to_zup_single_point(self):
        """Y-up → Z-up 역변환"""
        from simulation.ply_alignment import convert_yup_to_zup

        # Y-up: (1, 3, -2)
        # Z-up: (1, 2, 3)
        points = np.array([[1.0, 3.0, -2.0]])
        result = convert_yup_to_zup(points)

        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    def test_roundtrip_conversion(self):
        """왕복 변환 테스트 (Z-up → Y-up → Z-up)"""
        from simulation.ply_alignment import convert_zup_to_yup, convert_yup_to_zup

        original = np.array([
            [1.5, 2.3, 4.7],
            [-0.5, 1.2, 3.3],
            [0.0, 0.0, 0.0],
        ])

        yup = convert_zup_to_yup(original)
        back_to_zup = convert_yup_to_zup(yup)

        np.testing.assert_array_almost_equal(back_to_zup, original)


# ==================== AlignmentResult 테스트 ====================

class TestAlignmentResult:
    """AlignmentResult 데이터 클래스 테스트"""

    def test_success_result(self):
        """성공 결과 생성"""
        from simulation.ply_alignment import AlignmentResult

        result = AlignmentResult(
            success=True,
            width=1.5,
            depth=2.0,
            height=0.8,
            center=(0.0, 0.4, 0.0),
            min_bound=(-0.75, 0.0, -1.0),
            max_bound=(0.75, 0.8, 1.0),
            point_count=10000,
            message="OK"
        )

        assert result.success is True
        assert result.width == 1.5
        assert result.depth == 2.0
        assert result.height == 0.8
        assert result.point_count == 10000

    def test_failure_result(self):
        """실패 결과 생성"""
        from simulation.ply_alignment import AlignmentResult

        result = AlignmentResult(
            success=False,
            width=0, depth=0, height=0,
            center=(0, 0, 0),
            min_bound=(0, 0, 0),
            max_bound=(0, 0, 0),
            point_count=0,
            message="File not found"
        )

        assert result.success is False
        assert "not found" in result.message


# ==================== PLYAlignmentService 테스트 (Open3D 필요) ====================

@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not installed")
class TestPLYAlignmentService:
    """PLY 정렬 서비스 테스트"""

    @pytest.fixture
    def service(self):
        """PLYAlignmentService 인스턴스"""
        from simulation.ply_alignment import PLYAlignmentService
        return PLYAlignmentService(convert_to_yup=True)

    @pytest.fixture
    def service_zup(self):
        """Z-up 유지 서비스"""
        from simulation.ply_alignment import PLYAlignmentService
        return PLYAlignmentService(convert_to_yup=False)

    @pytest.fixture
    def sample_ply_path(self, tmp_path):
        """테스트용 임시 PLY 파일 생성"""
        # 간단한 큐브 포인트 클라우드 생성
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)

        # 임의 회전 적용 (테스트용)
        angle = np.pi / 6  # 30도
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_points = points @ rotation.T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotated_points)

        ply_path = tmp_path / "test_cube.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)

        return str(ply_path)

    def test_align_single_success(self, service, sample_ply_path, tmp_path):
        """단일 파일 정렬 성공"""
        output_path = str(tmp_path / "aligned.ply")
        result = service.align_single(sample_ply_path, output_path)

        assert result.success is True
        assert result.point_count == 8
        assert result.width > 0
        assert result.depth > 0
        assert result.height > 0
        assert Path(output_path).exists()

    def test_align_single_no_output(self, service, sample_ply_path):
        """출력 파일 없이 정렬 (치수만 계산)"""
        result = service.align_single(sample_ply_path, output_path=None)

        assert result.success is True
        assert result.point_count == 8

    def test_align_nonexistent_file(self, service):
        """존재하지 않는 파일"""
        result = service.align_single("/nonexistent/path.ply")

        assert result.success is False
        assert result.point_count == 0

    def test_get_dimensions(self, service, sample_ply_path):
        """치수 조회"""
        result = service.get_dimensions(sample_ply_path)

        assert result.success is True
        # 큐브는 대략 1x1x1 크기여야 함
        assert 0.5 < result.width < 2.0
        assert 0.5 < result.depth < 2.0
        assert 0.5 < result.height < 2.0

    def test_floor_placement_yup(self, service, sample_ply_path):
        """Y-up 좌표계에서 바닥 배치 (Y-min ≈ 0)"""
        result = service.align_single(sample_ply_path)

        assert result.success is True
        # Y-up에서는 min_bound[1] (Y)가 0에 가까워야 함
        assert abs(result.min_bound[1]) < 0.01

    def test_floor_placement_zup(self, service_zup, sample_ply_path):
        """Z-up 좌표계에서 바닥 배치 (Z-min ≈ 0)"""
        result = service_zup.align_single(sample_ply_path)

        assert result.success is True
        # Z-up에서는 min_bound[2] (Z)가 0에 가까워야 함
        assert abs(result.min_bound[2]) < 0.01


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not installed")
class TestPLYAlignmentServiceBase64:
    """Base64 인터페이스 테스트"""

    @pytest.fixture
    def service(self):
        from simulation.ply_alignment import PLYAlignmentService
        return PLYAlignmentService(convert_to_yup=True)

    @pytest.fixture
    def sample_ply_base64(self, tmp_path):
        """Base64 인코딩된 테스트 PLY"""
        import base64

        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        ply_path = tmp_path / "test.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)

        with open(ply_path, "rb") as f:
            ply_data = f.read()

        return base64.b64encode(ply_data).decode('utf-8')

    def test_align_from_base64(self, service, sample_ply_base64):
        """Base64 입력 정렬"""
        aligned_base64, result = service.align_from_base64(sample_ply_base64)

        assert result.success is True
        assert len(aligned_base64) > 0
        assert result.point_count == 4


# ==================== 디렉토리 처리 테스트 ====================

@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not installed")
class TestProcessDirectory:
    """디렉토리 일괄 처리 테스트"""

    def test_process_directory(self, tmp_path):
        """디렉토리 내 PLY 파일 일괄 처리"""
        from simulation.ply_alignment import process_directory

        # 테스트 PLY 파일 생성
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(3):
            points = np.random.rand(10, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(input_dir / f"test_{i}.ply"), pcd)

        results = process_directory(str(input_dir), str(output_dir))

        assert results["total"] == 3
        assert results["success"] == 3
        assert results["failed"] == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.ply"))) == 3

    def test_process_empty_directory(self, tmp_path):
        """빈 디렉토리 처리"""
        from simulation.ply_alignment import process_directory

        input_dir = tmp_path / "empty_input"
        output_dir = tmp_path / "empty_output"
        input_dir.mkdir()

        results = process_directory(str(input_dir), str(output_dir))

        assert results["total"] == 0
        assert results["success"] == 0
