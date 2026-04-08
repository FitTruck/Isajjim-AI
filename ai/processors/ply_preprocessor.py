"""
PLY Preprocessor

GCS 업로드 전 PLY 파일 전처리:
1. 축 정렬 (OBB 기반)
2. 스케일링 (절대 치수에 맞게)
3. Stride 다운샘플링 (용량 최소화)
"""

import base64
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Open3D lazy import
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not installed. PLY alignment/scaling will be disabled.")


@dataclass
class PreprocessResult:
    """PLY 전처리 결과"""
    success: bool
    original_points: int
    processed_points: int
    original_size_bytes: int
    processed_size_bytes: int
    aligned: bool
    scaled: bool
    downsampled: bool
    final_width_m: float
    final_depth_m: float
    final_height_m: float
    message: str = ""


class PLYPreprocessor:
    """
    PLY 전처리 파이프라인.

    Usage:
        preprocessor = PLYPreprocessor(max_points=72000)
        processed_b64, result = preprocessor.process(
            ply_b64=original_b64,
            target_width_mm=1000,
            target_depth_mm=3000,
            target_height_mm=900
        )
    """

    def __init__(
        self,
        max_points: int = 72000,
        convert_to_yup: bool = True,
        enable_alignment: bool = True,
        enable_scaling: bool = True,
        enable_downsampling: bool = True
    ):
        """
        Args:
            max_points: 다운샘플링 최대 포인트 수
            convert_to_yup: Y-up 좌표계로 변환 (Three.js 호환)
            enable_alignment: 축 정렬 활성화
            enable_scaling: 스케일링 활성화
            enable_downsampling: 다운샘플링 활성화
        """
        self.max_points = max_points
        self.convert_to_yup = convert_to_yup
        self.enable_alignment = enable_alignment
        self.enable_scaling = enable_scaling
        self.enable_downsampling = enable_downsampling

    def process(
        self,
        ply_b64: str,
        target_width_mm: float,
        target_depth_mm: float,
        target_height_mm: float
    ) -> Tuple[str, PreprocessResult]:
        """
        PLY 전처리 파이프라인 실행.

        Args:
            ply_b64: Base64 인코딩된 PLY 데이터
            target_width_mm: 목표 너비 (mm)
            target_depth_mm: 목표 깊이 (mm)
            target_height_mm: 목표 높이 (mm)

        Returns:
            (처리된 PLY Base64, PreprocessResult)
        """
        # Base64 디코딩
        try:
            ply_bytes = base64.b64decode(ply_b64)
            original_size = len(ply_bytes)
        except Exception as e:
            logger.error(f"[PLYPreprocessor] Base64 decode failed: {e}")
            return ply_b64, PreprocessResult(
                success=False,
                original_points=0, processed_points=0,
                original_size_bytes=0, processed_size_bytes=0,
                aligned=False, scaled=False, downsampled=False,
                final_width_m=0, final_depth_m=0, final_height_m=0,
                message=f"Base64 decode failed: {e}"
            )

        # Open3D 사용 가능 여부 확인
        if not OPEN3D_AVAILABLE:
            # Open3D 없으면 다운샘플링만 적용 (numpy 기반)
            logger.warning("[PLYPreprocessor] Open3D not available, skipping alignment/scaling")
            processed_bytes, result = self._process_without_open3d(
                ply_bytes, original_size
            )
            processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
            return processed_b64, result

        # Open3D로 전체 파이프라인 실행
        return self._process_with_open3d(
            ply_bytes, original_size,
            target_width_mm, target_depth_mm, target_height_mm
        )

    def _process_with_open3d(
        self,
        ply_bytes: bytes,
        original_size: int,
        target_width_mm: float,
        target_depth_mm: float,
        target_height_mm: float
    ) -> Tuple[str, PreprocessResult]:
        """Open3D를 사용한 전체 전처리 파이프라인."""
        # 임시 파일로 저장 후 로드
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp.write(ply_bytes)
            tmp_path = tmp.name

        try:
            pcd = o3d.io.read_point_cloud(tmp_path)
            original_points = len(pcd.points)

            if pcd.is_empty():
                return base64.b64encode(ply_bytes).decode('utf-8'), PreprocessResult(
                    success=False,
                    original_points=0, processed_points=0,
                    original_size_bytes=original_size, processed_size_bytes=original_size,
                    aligned=False, scaled=False, downsampled=False,
                    final_width_m=0, final_depth_m=0, final_height_m=0,
                    message="Empty point cloud"
                )

            aligned = False
            scaled = False
            downsampled = False

            # 1. 축 정렬
            if self.enable_alignment:
                pcd = self._align_to_floor(pcd)
                if self.convert_to_yup:
                    pcd = self._convert_to_yup(pcd)
                aligned = True
                logger.debug(f"[PLYPreprocessor] Aligned: {original_points} points")

            # 2. 스케일링
            if self.enable_scaling and target_width_mm > 0 and target_height_mm > 0:
                pcd = self._scale_to_absolute_size(
                    pcd, target_width_mm, target_depth_mm, target_height_mm
                )
                scaled = True
                logger.debug(f"[PLYPreprocessor] Scaled to target dimensions")

            # 3. 다운샘플링
            if self.enable_downsampling and len(pcd.points) > self.max_points:
                pcd = self._downsample_open3d(pcd)
                downsampled = True
                logger.debug(f"[PLYPreprocessor] Downsampled: {original_points} -> {len(pcd.points)}")

            # 최종 치수 계산
            aabb = pcd.get_axis_aligned_bounding_box()
            final_size = aabb.get_max_bound() - aabb.get_min_bound()

            # Y-up 좌표계에서: width=X, height=Y, depth=Z
            if self.convert_to_yup:
                final_w, final_h, final_d = final_size[0], final_size[1], final_size[2]
            else:
                final_w, final_d, final_h = final_size[0], final_size[1], final_size[2]

            # 저장
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name

            o3d.io.write_point_cloud(tmp_out_path, pcd, write_ascii=False)

            with open(tmp_out_path, "rb") as f:
                processed_bytes = f.read()

            Path(tmp_out_path).unlink(missing_ok=True)

            processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
            processed_size = len(processed_bytes)

            logger.info(
                f"[PLYPreprocessor] Complete: {original_points} -> {len(pcd.points)} points, "
                f"{original_size} -> {processed_size} bytes "
                f"({100 - processed_size * 100 // original_size}% reduction)"
            )

            return processed_b64, PreprocessResult(
                success=True,
                original_points=original_points,
                processed_points=len(pcd.points),
                original_size_bytes=original_size,
                processed_size_bytes=processed_size,
                aligned=aligned,
                scaled=scaled,
                downsampled=downsampled,
                final_width_m=float(final_w),
                final_depth_m=float(final_d),
                final_height_m=float(final_h),
                message="Preprocessing successful"
            )

        except Exception as e:
            logger.error(f"[PLYPreprocessor] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return base64.b64encode(ply_bytes).decode('utf-8'), PreprocessResult(
                success=False,
                original_points=0, processed_points=0,
                original_size_bytes=original_size, processed_size_bytes=original_size,
                aligned=False, scaled=False, downsampled=False,
                final_width_m=0, final_depth_m=0, final_height_m=0,
                message=str(e)
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _process_without_open3d(
        self,
        ply_bytes: bytes,
        original_size: int
    ) -> Tuple[bytes, PreprocessResult]:
        """Open3D 없이 numpy로 다운샘플링만 수행."""
        try:
            # PLY 파싱 (간단한 ASCII/Binary 지원)
            points, colors, header = self._parse_ply(ply_bytes)
            original_points = len(points)

            if original_points == 0:
                return ply_bytes, PreprocessResult(
                    success=False,
                    original_points=0, processed_points=0,
                    original_size_bytes=original_size, processed_size_bytes=original_size,
                    aligned=False, scaled=False, downsampled=False,
                    final_width_m=0, final_depth_m=0, final_height_m=0,
                    message="Empty point cloud"
                )

            # 다운샘플링
            downsampled = False
            if self.enable_downsampling and original_points > self.max_points:
                points, colors = self._downsample_numpy(points, colors)
                downsampled = True

            # PLY 재구성
            processed_bytes = self._build_ply(points, colors)
            processed_size = len(processed_bytes)

            # 치수 계산
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            size = max_bound - min_bound

            logger.info(
                f"[PLYPreprocessor] numpy-only: {original_points} -> {len(points)} points, "
                f"{original_size} -> {processed_size} bytes"
            )

            return processed_bytes, PreprocessResult(
                success=True,
                original_points=original_points,
                processed_points=len(points),
                original_size_bytes=original_size,
                processed_size_bytes=processed_size,
                aligned=False,  # Open3D 없으면 정렬 불가
                scaled=False,   # Open3D 없으면 스케일링 불가
                downsampled=downsampled,
                final_width_m=float(size[0]),
                final_depth_m=float(size[1]),
                final_height_m=float(size[2]),
                message="Downsampling only (Open3D not available)"
            )

        except Exception as e:
            logger.error(f"[PLYPreprocessor] numpy processing failed: {e}")
            return ply_bytes, PreprocessResult(
                success=False,
                original_points=0, processed_points=0,
                original_size_bytes=original_size, processed_size_bytes=original_size,
                aligned=False, scaled=False, downsampled=False,
                final_width_m=0, final_depth_m=0, final_height_m=0,
                message=str(e)
            )

    def _align_to_floor(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """
        OBB 기반 축 정렬 + 바닥 배치.

        ply_alignment.py의 로직 재사용:
        1. OBB 계산
        2. OBB.R.T (역회전)으로 축 정렬
        3. Z-min = 0 으로 이동
        """
        obb = pcd.get_oriented_bounding_box()
        center = pcd.get_center()
        pcd.rotate(obb.R.T, center=center)

        aabb = pcd.get_axis_aligned_bounding_box()
        z_min = aabb.get_min_bound()[2]
        pcd.translate((0, 0, -z_min))

        return pcd

    def _convert_to_yup(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """
        Z-up → Y-up 좌표계 변환.

        변환: (x, y, z) -> (x, z, -y)
        """
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)

        # 포인트 변환
        points = np.asarray(pcd.points)
        transformed_points = points @ rotation_matrix.T
        pcd.points = o3d.utility.Vector3dVector(transformed_points)

        # 법선 변환 (있는 경우)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            transformed_normals = normals @ rotation_matrix.T
            pcd.normals = o3d.utility.Vector3dVector(transformed_normals)

        # Y-min = 0 으로 재조정 (바닥)
        aabb = pcd.get_axis_aligned_bounding_box()
        y_min = aabb.get_min_bound()[1]
        pcd.translate((0, -y_min, 0))

        return pcd

    def _scale_to_absolute_size(
        self,
        pcd: "o3d.geometry.PointCloud",
        target_width_mm: float,
        target_depth_mm: float,
        target_height_mm: float
    ) -> "o3d.geometry.PointCloud":
        """
        절대 치수에 맞게 스케일링 (균일 스케일 유지).

        Args:
            pcd: 포인트 클라우드
            target_*_mm: 목표 치수 (밀리미터)

        Returns:
            스케일링된 포인트 클라우드
        """
        aabb = pcd.get_axis_aligned_bounding_box()
        current_size = aabb.get_max_bound() - aabb.get_min_bound()

        # mm → m 변환
        target_w = target_width_mm / 1000.0
        target_d = target_depth_mm / 1000.0
        target_h = target_height_mm / 1000.0

        # 현재 좌표계에 따른 치수 매핑
        # Y-up: X=width, Y=height, Z=depth
        if self.convert_to_yup:
            scale_factors = [
                target_w / max(current_size[0], 1e-6),  # width -> X
                target_h / max(current_size[1], 1e-6),  # height -> Y
                target_d / max(current_size[2], 1e-6)   # depth -> Z
            ]
        else:
            # Z-up: X=width, Y=depth, Z=height
            scale_factors = [
                target_w / max(current_size[0], 1e-6),
                target_d / max(current_size[1], 1e-6),
                target_h / max(current_size[2], 1e-6)
            ]

        # 균일 스케일 (비율 유지, 가장 작은 팩터 사용)
        scale = min(scale_factors)

        if scale <= 0 or not np.isfinite(scale):
            logger.warning(f"[PLYPreprocessor] Invalid scale factor: {scale}, skipping scaling")
            return pcd

        center = pcd.get_center()
        pcd.scale(scale, center=center)

        # 바닥 재조정
        aabb = pcd.get_axis_aligned_bounding_box()
        if self.convert_to_yup:
            y_min = aabb.get_min_bound()[1]
            pcd.translate((0, -y_min, 0))
        else:
            z_min = aabb.get_min_bound()[2]
            pcd.translate((0, 0, -z_min))

        logger.debug(f"[PLYPreprocessor] Scaled with factor: {scale:.4f}")
        return pcd

    def _downsample_open3d(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """Open3D를 사용한 stride 다운샘플링."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        total_points = len(points)
        stride = math.ceil(total_points / self.max_points)
        indices = np.arange(0, total_points, stride)

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points[indices])

        if colors is not None and len(colors) > 0:
            new_pcd.colors = o3d.utility.Vector3dVector(colors[indices])

        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            new_pcd.normals = o3d.utility.Vector3dVector(normals[indices])

        return new_pcd

    def _downsample_numpy(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Stride 기반 포인트 다운샘플링 (numpy만 사용).

        simulator.html의 downsampleGeometry() 로직 포팅.
        """
        total_points = len(points)
        if total_points <= self.max_points:
            return points, colors

        stride = math.ceil(total_points / self.max_points)
        indices = np.arange(0, total_points, stride)

        new_points = points[indices]
        new_colors = colors[indices] if colors is not None else None

        return new_points, new_colors

    def _parse_ply(self, ply_bytes: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], bytes]:
        """
        간단한 PLY 파서 (ASCII/Binary Little Endian 지원).

        Returns:
            (points, colors, header_bytes)
        """
        # 헤더 파싱
        content = ply_bytes
        header_end = content.find(b"end_header")
        if header_end == -1:
            raise ValueError("Invalid PLY: no end_header found")

        header = content[:header_end + len(b"end_header\n")]
        body = content[len(header):]

        # 헤더에서 정보 추출
        header_str = header.decode('ascii', errors='ignore')
        vertex_count = 0
        is_binary = False
        has_color = False

        for line in header_str.split('\n'):
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("format binary"):
                is_binary = True
            elif "red" in line or "r " in line:
                has_color = True

        if vertex_count == 0:
            return np.array([]), None, header

        if is_binary:
            # Binary PLY (little endian, float32 xyz + optional uint8 rgb)
            if has_color:
                dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                ])
            else:
                dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

            try:
                data = np.frombuffer(body[:vertex_count * dtype.itemsize], dtype=dtype)
                points = np.column_stack([data['x'], data['y'], data['z']])
                if has_color:
                    colors = np.column_stack([data['r'], data['g'], data['b']]) / 255.0
                else:
                    colors = None
            except Exception:
                # Fallback: 단순 float32 배열로 시도
                points = np.frombuffer(body[:vertex_count * 12], dtype='<f4').reshape(-1, 3)
                colors = None
        else:
            # ASCII PLY
            lines = body.decode('ascii', errors='ignore').strip().split('\n')
            points = []
            colors = []
            for line in lines[:vertex_count]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(parts) >= 6:
                        colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])

            points = np.array(points)
            colors = np.array(colors) if len(colors) == len(points) else None

        return points, colors, header

    def _build_ply(self, points: np.ndarray, colors: Optional[np.ndarray]) -> bytes:
        """
        다운샘플링된 포인트로 Binary PLY 재구성.
        """
        vertex_count = len(points)

        # 헤더 생성
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if colors is not None and len(colors) == vertex_count:
            header_lines.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        # 바이너리 데이터 생성
        if colors is not None and len(colors) == vertex_count:
            # xyz + rgb
            colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
            ])
            data = np.zeros(vertex_count, dtype=dtype)
            data['x'] = points[:, 0].astype(np.float32)
            data['y'] = points[:, 1].astype(np.float32)
            data['z'] = points[:, 2].astype(np.float32)
            data['r'] = colors_uint8[:, 0]
            data['g'] = colors_uint8[:, 1]
            data['b'] = colors_uint8[:, 2]
        else:
            # xyz only
            dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
            data = np.zeros(vertex_count, dtype=dtype)
            data['x'] = points[:, 0].astype(np.float32)
            data['y'] = points[:, 1].astype(np.float32)
            data['z'] = points[:, 2].astype(np.float32)

        return header.encode('ascii') + data.tobytes()


# 편의 함수
def preprocess_ply(
    ply_b64: str,
    target_width_mm: float,
    target_depth_mm: float,
    target_height_mm: float,
    max_points: int = 72000
) -> Tuple[str, PreprocessResult]:
    """
    PLY 전처리 편의 함수.

    Args:
        ply_b64: Base64 인코딩된 PLY
        target_*_mm: 목표 치수 (밀리미터)
        max_points: 최대 포인트 수

    Returns:
        (처리된 PLY Base64, PreprocessResult)
    """
    preprocessor = PLYPreprocessor(max_points=max_points)
    return preprocessor.process(ply_b64, target_width_mm, target_depth_mm, target_height_mm)
