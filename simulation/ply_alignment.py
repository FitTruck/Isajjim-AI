"""
PLY 객체 정렬 서비스

OBB(Oriented Bounding Box) 기반으로 PLY 객체를 정렬하여
바닥에 수평으로 놓이게 함.

Features:
- OBB.R.T 역회전으로 축 정렬
- Z-up (Open3D) → Y-up (Three.js) 좌표계 변환
- AABB 치수 계산
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import base64
import io

import numpy as np

logger = logging.getLogger(__name__)

# Open3D lazy import (설치 안 된 환경 대응)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not installed. PLY alignment will be disabled.")


@dataclass
class AlignmentResult:
    """PLY 정렬 결과"""
    success: bool
    width: float  # X축 (정렬 후)
    depth: float  # Y축 또는 Z축 (좌표계에 따라)
    height: float  # Z축 또는 Y축
    center: Tuple[float, float, float]
    min_bound: Tuple[float, float, float]
    max_bound: Tuple[float, float, float]
    point_count: int
    message: str = ""


def convert_zup_to_yup(points: np.ndarray) -> np.ndarray:
    """
    Z-up 좌표계를 Y-up으로 변환 (X축 기준 -90도 회전)

    Open3D (Z-up): X=right, Y=forward, Z=up
    Three.js (Y-up): X=right, Y=up, Z=forward

    변환: (x, y, z) -> (x, z, -y)
    """
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float64)
    return points @ rotation_matrix.T


def convert_yup_to_zup(points: np.ndarray) -> np.ndarray:
    """
    Y-up 좌표계를 Z-up으로 변환 (역변환)

    변환: (x, y, z) -> (x, -z, y)
    """
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float64)
    return points @ rotation_matrix.T


class PLYAlignmentService:
    """PLY 객체 정렬 서비스"""

    def __init__(self, convert_to_yup: bool = True):
        """
        Args:
            convert_to_yup: True면 정렬 후 Y-up 좌표계로 변환 (Three.js 호환)
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for PLY alignment. Install: pip install open3d")

        self.convert_to_yup = convert_to_yup

    def align_single(
        self,
        ply_path: str,
        output_path: Optional[str] = None
    ) -> AlignmentResult:
        """
        단일 PLY 파일 정렬

        Args:
            ply_path: 입력 PLY 파일 경로
            output_path: 출력 경로 (None이면 저장 안 함)

        Returns:
            AlignmentResult
        """
        try:
            # PLY 로드
            pcd = o3d.io.read_point_cloud(ply_path)

            if pcd.is_empty():
                return AlignmentResult(
                    success=False,
                    width=0, depth=0, height=0,
                    center=(0, 0, 0),
                    min_bound=(0, 0, 0),
                    max_bound=(0, 0, 0),
                    point_count=0,
                    message=f"Empty point cloud: {ply_path}"
                )

            # 정렬 수행
            aligned_pcd = self._align_to_floor(pcd)

            # 좌표계 변환 (선택적)
            if self.convert_to_yup:
                aligned_pcd = self._convert_to_yup(aligned_pcd)

            # AABB 계산
            aabb = aligned_pcd.get_axis_aligned_bounding_box()
            min_b = aabb.get_min_bound()
            max_b = aabb.get_max_bound()
            size = max_b - min_b
            center = aligned_pcd.get_center()

            # 저장
            if output_path:
                o3d.io.write_point_cloud(output_path, aligned_pcd)
                logger.info(f"Saved aligned PLY: {output_path}")

            # Y-up 좌표계에서: width=X, height=Y, depth=Z
            if self.convert_to_yup:
                width, height, depth = size[0], size[1], size[2]
            else:
                # Z-up 좌표계에서: width=X, depth=Y, height=Z
                width, depth, height = size[0], size[1], size[2]

            return AlignmentResult(
                success=True,
                width=float(width),
                depth=float(depth),
                height=float(height),
                center=tuple(center),
                min_bound=tuple(min_b),
                max_bound=tuple(max_b),
                point_count=len(aligned_pcd.points),
                message="Alignment successful"
            )

        except Exception as e:
            logger.error(f"Alignment failed for {ply_path}: {e}")
            return AlignmentResult(
                success=False,
                width=0, depth=0, height=0,
                center=(0, 0, 0),
                min_bound=(0, 0, 0),
                max_bound=(0, 0, 0),
                point_count=0,
                message=str(e)
            )

    def align_from_bytes(self, ply_data: bytes) -> Tuple[bytes, AlignmentResult]:
        """
        메모리 내 PLY 데이터 정렬 (base64 전송용)

        Args:
            ply_data: PLY 파일 바이너리 데이터

        Returns:
            (정렬된 PLY 바이너리, AlignmentResult)
        """
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_in:
            tmp_in.write(ply_data)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            result = self.align_single(tmp_in_path, tmp_out_path)

            if result.success:
                with open(tmp_out_path, "rb") as f:
                    aligned_data = f.read()
                return aligned_data, result
            else:
                return ply_data, result
        finally:
            # 임시 파일 정리
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(tmp_out_path).unlink(missing_ok=True)

    def align_from_base64(self, ply_base64: str) -> Tuple[str, AlignmentResult]:
        """
        Base64 인코딩된 PLY 데이터 정렬

        Args:
            ply_base64: Base64 인코딩된 PLY 데이터

        Returns:
            (정렬된 PLY Base64, AlignmentResult)
        """
        ply_data = base64.b64decode(ply_base64)
        aligned_data, result = self.align_from_bytes(ply_data)
        aligned_base64 = base64.b64encode(aligned_data).decode('utf-8')
        return aligned_base64, result

    def get_dimensions(self, ply_path: str) -> AlignmentResult:
        """
        정렬 후 AABB 치수만 반환 (파일 저장 없음)

        Args:
            ply_path: PLY 파일 경로

        Returns:
            AlignmentResult (치수 정보 포함)
        """
        return self.align_single(ply_path, output_path=None)

    def _align_to_floor(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """
        OBB 기반 축 정렬 + 바닥 배치

        1. OBB 계산 (내부적으로 PCA 사용)
        2. OBB.R.T (역회전)으로 축 정렬
        3. Z-min = 0 으로 이동 (바닥에 놓기)
        """
        # OBB 계산
        obb = pcd.get_oriented_bounding_box()

        # OBB 회전 행렬의 역행렬(전치)로 회전 - 축 정렬
        center = pcd.get_center()
        pcd.rotate(obb.R.T, center=center)

        # 정렬 후 AABB 계산
        aabb = pcd.get_axis_aligned_bounding_box()

        # Z 최소값이 0이 되도록 이동 (바닥에 놓기)
        z_min = aabb.get_min_bound()[2]
        pcd.translate((0, 0, -z_min))

        return pcd

    def _convert_to_yup(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """
        Z-up → Y-up 좌표계 변환
        """
        # 포인트 변환
        points = np.asarray(pcd.points)
        transformed_points = convert_zup_to_yup(points)
        pcd.points = o3d.utility.Vector3dVector(transformed_points)

        # 법선 변환 (있는 경우)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            transformed_normals = convert_zup_to_yup(normals)
            pcd.normals = o3d.utility.Vector3dVector(transformed_normals)

        # Y-min = 0 으로 재조정 (바닥)
        aabb = pcd.get_axis_aligned_bounding_box()
        y_min = aabb.get_min_bound()[1]
        pcd.translate((0, -y_min, 0))

        return pcd


def process_directory(
    input_dir: str,
    output_dir: str,
    convert_to_yup: bool = True
) -> dict:
    """
    디렉토리 내 모든 PLY 파일을 정렬하여 저장

    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        convert_to_yup: Y-up 좌표계로 변환 여부

    Returns:
        처리 결과 딕셔너리
    """
    service = PLYAlignmentService(convert_to_yup=convert_to_yup)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ply_files = list(input_path.glob("*.ply"))
    logger.info(f"Found {len(ply_files)} PLY files in {input_dir}")

    results = {
        "total": len(ply_files),
        "success": 0,
        "failed": 0,
        "files": []
    }

    for ply_file in ply_files:
        output_file = output_path / ply_file.name
        result = service.align_single(str(ply_file), str(output_file))

        if result.success:
            results["success"] += 1
            logger.info(
                f"Aligned: {ply_file.name} -> "
                f"{result.width:.2f} x {result.depth:.2f} x {result.height:.2f}"
            )
        else:
            results["failed"] += 1
            logger.error(f"Failed: {ply_file.name} - {result.message}")

        results["files"].append({
            "name": ply_file.name,
            "result": result
        })

    return results


# CLI 실행
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    script_dir = Path(__file__).parent
    input_dir = script_dir / "assets"
    output_dir = script_dir / "assets" / "aligned"

    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    results = process_directory(str(input_dir), str(output_dir))
    print(f"\nResults: {results['success']}/{results['total']} succeeded")
