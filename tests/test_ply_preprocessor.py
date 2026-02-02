"""
Tests for PLY Preprocessor
"""

import base64
import math
import pytest
import numpy as np

from ai.processors.ply_preprocessor import (
    PLYPreprocessor,
    PreprocessResult,
    preprocess_ply
)


def create_test_ply_ascii(num_points: int = 100, with_colors: bool = True) -> bytes:
    """테스트용 ASCII PLY 파일 생성."""
    np.random.seed(42)
    points = np.random.randn(num_points, 3)

    # Normalize to unit cube
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if with_colors:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    body_lines = []
    for i in range(num_points):
        x, y, z = points[i]
        if with_colors:
            r, g, b = np.random.randint(0, 256, 3)
            body_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}")
        else:
            body_lines.append(f"{x:.6f} {y:.6f} {z:.6f}")

    body = "\n".join(body_lines)
    return (header + body).encode('ascii')


def create_test_ply_binary(num_points: int = 100, with_colors: bool = True) -> bytes:
    """테스트용 Binary PLY 파일 생성."""
    np.random.seed(42)
    points = np.random.randn(num_points, 3).astype(np.float32)

    # Normalize to unit cube
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if with_colors:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    if with_colors:
        colors = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
        ])
        data = np.zeros(num_points, dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['r'] = colors[:, 0]
        data['g'] = colors[:, 1]
        data['b'] = colors[:, 2]
    else:
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        data = np.zeros(num_points, dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]

    return header.encode('ascii') + data.tobytes()


class TestPLYPreprocessorInit:
    """PLYPreprocessor 초기화 테스트"""

    def test_default_init(self):
        """기본 초기화 테스트"""
        preprocessor = PLYPreprocessor()

        assert preprocessor.max_points == 50000
        assert preprocessor.convert_to_yup is True
        assert preprocessor.enable_alignment is True
        assert preprocessor.enable_scaling is True
        assert preprocessor.enable_downsampling is True

    def test_custom_init(self):
        """커스텀 파라미터 초기화 테스트"""
        preprocessor = PLYPreprocessor(
            max_points=10000,
            convert_to_yup=False,
            enable_alignment=False,
            enable_scaling=False,
            enable_downsampling=False
        )

        assert preprocessor.max_points == 10000
        assert preprocessor.convert_to_yup is False
        assert preprocessor.enable_alignment is False
        assert preprocessor.enable_scaling is False
        assert preprocessor.enable_downsampling is False


class TestPLYPreprocessorDownsampling:
    """다운샘플링 테스트 (Open3D 없이 numpy 기반)"""

    def test_no_downsampling_small_cloud(self):
        """작은 포인트 클라우드는 다운샘플링하지 않음"""
        ply_bytes = create_test_ply_binary(100)
        ply_b64 = base64.b64encode(ply_bytes).decode('utf-8')

        preprocessor = PLYPreprocessor(
            max_points=1000,  # 100 포인트는 1000보다 작음
            enable_alignment=False,
            enable_scaling=False
        )

        _, result = preprocessor.process(ply_b64, 1000, 500, 500)

        assert result.original_points == 100
        # Open3D 가용성에 따라 포인트 수가 달라질 수 있음
        assert result.processed_points == 100 or result.downsampled is False

    def test_downsampling_large_cloud(self):
        """큰 포인트 클라우드 다운샘플링"""
        ply_bytes = create_test_ply_binary(10000)
        ply_b64 = base64.b64encode(ply_bytes).decode('utf-8')

        preprocessor = PLYPreprocessor(
            max_points=1000,
            enable_alignment=False,
            enable_scaling=False
        )

        processed_b64, result = preprocessor.process(ply_b64, 1000, 500, 500)

        assert result.success is True
        assert result.original_points == 10000
        assert result.processed_points <= 1000
        assert result.downsampled is True

    def test_stride_calculation(self):
        """Stride 계산 로직 테스트"""
        preprocessor = PLYPreprocessor(max_points=100)

        # 1000 포인트 → 100 포인트: stride = ceil(1000/100) = 10
        points = np.random.randn(1000, 3)
        colors = np.random.rand(1000, 3)

        new_points, new_colors = preprocessor._downsample_numpy(points, colors)

        expected_stride = math.ceil(1000 / 100)  # 10
        expected_count = len(range(0, 1000, expected_stride))  # 100

        assert len(new_points) == expected_count
        assert len(new_colors) == expected_count


class TestPLYPreprocessorParsing:
    """PLY 파싱 테스트"""

    def test_parse_ascii_ply(self):
        """ASCII PLY 파싱"""
        ply_bytes = create_test_ply_ascii(50, with_colors=True)

        preprocessor = PLYPreprocessor()
        points, colors, header = preprocessor._parse_ply(ply_bytes)

        assert len(points) == 50
        assert colors is not None
        assert len(colors) == 50

    def test_parse_binary_ply(self):
        """Binary PLY 파싱"""
        ply_bytes = create_test_ply_binary(50, with_colors=True)

        preprocessor = PLYPreprocessor()
        points, colors, header = preprocessor._parse_ply(ply_bytes)

        assert len(points) == 50
        assert colors is not None
        assert len(colors) == 50

    def test_parse_ply_without_colors(self):
        """색상 없는 PLY 파싱"""
        ply_bytes = create_test_ply_binary(50, with_colors=False)

        preprocessor = PLYPreprocessor()
        points, colors, header = preprocessor._parse_ply(ply_bytes)

        assert len(points) == 50
        # 색상이 없거나 None
        assert colors is None or len(colors) == 0


class TestPLYPreprocessorBuild:
    """PLY 재구성 테스트"""

    def test_build_ply_with_colors(self):
        """색상 포함 PLY 재구성"""
        preprocessor = PLYPreprocessor()

        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        ply_bytes = preprocessor._build_ply(points, colors)

        # 헤더 검증
        assert b"ply" in ply_bytes
        assert b"format binary_little_endian 1.0" in ply_bytes
        assert b"element vertex 3" in ply_bytes
        assert b"property uchar red" in ply_bytes

    def test_build_ply_without_colors(self):
        """색상 없는 PLY 재구성"""
        preprocessor = PLYPreprocessor()

        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

        ply_bytes = preprocessor._build_ply(points, None)

        assert b"ply" in ply_bytes
        assert b"element vertex 2" in ply_bytes
        assert b"property uchar red" not in ply_bytes


class TestPLYPreprocessorFullPipeline:
    """전체 파이프라인 테스트"""

    def test_full_pipeline_disabled(self):
        """모든 처리 비활성화 시 원본 반환"""
        ply_bytes = create_test_ply_binary(100)
        ply_b64 = base64.b64encode(ply_bytes).decode('utf-8')

        preprocessor = PLYPreprocessor(
            enable_alignment=False,
            enable_scaling=False,
            enable_downsampling=False
        )

        processed_b64, result = preprocessor.process(ply_b64, 1000, 500, 500)

        # 처리가 비활성화되면 원본과 유사한 크기
        assert result.aligned is False
        assert result.scaled is False
        assert result.downsampled is False

    def test_invalid_base64(self):
        """잘못된 base64 입력 처리"""
        preprocessor = PLYPreprocessor()

        _, result = preprocessor.process("invalid_base64!!!", 1000, 500, 500)

        assert result.success is False
        assert "Base64 decode failed" in result.message


class TestConvenienceFunction:
    """편의 함수 테스트"""

    def test_preprocess_ply_function(self):
        """preprocess_ply 편의 함수 테스트"""
        ply_bytes = create_test_ply_binary(1000)
        ply_b64 = base64.b64encode(ply_bytes).decode('utf-8')

        processed_b64, result = preprocess_ply(
            ply_b64=ply_b64,
            target_width_mm=1000,
            target_depth_mm=500,
            target_height_mm=500,
            max_points=100
        )

        # 결과는 유효한 base64
        assert len(processed_b64) > 0
        decoded = base64.b64decode(processed_b64)
        assert decoded.startswith(b"ply")


class TestPreprocessResult:
    """PreprocessResult 데이터클래스 테스트"""

    def test_preprocess_result_fields(self):
        """PreprocessResult 필드 테스트"""
        result = PreprocessResult(
            success=True,
            original_points=1000,
            processed_points=100,
            original_size_bytes=10000,
            processed_size_bytes=1000,
            aligned=True,
            scaled=True,
            downsampled=True,
            final_width_m=1.0,
            final_depth_m=0.5,
            final_height_m=0.8,
            message="Test message"
        )

        assert result.success is True
        assert result.original_points == 1000
        assert result.processed_points == 100
        assert result.message == "Test message"
