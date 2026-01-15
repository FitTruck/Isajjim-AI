"""
Volume Calculator Module

SAM-3D에서 생성된 3D 메시/Gaussian Splat에서 부피와 치수를 계산합니다.
trimesh 라이브러리를 사용하여 메시 분석을 수행합니다.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import os

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[Warning] trimesh not available - mesh volume calculation disabled")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VolumeCalculator:
    """
    3D 객체의 부피와 치수를 계산하는 클래스
    """

    def __init__(self):
        if not HAS_TRIMESH:
            print("[VolumeCalculator] Warning: trimesh not installed")

    def calculate_from_ply(self, ply_path: str) -> Optional[Dict]:
        """
        PLY 파일에서 부피와 치수를 계산합니다.

        Args:
            ply_path: PLY 파일 경로

        Returns:
            {
                "volume": float (mm^3),
                "bounding_box": {"width": float, "depth": float, "height": float},
                "ratio": {"w": float, "h": float, "d": float},
                "centroid": [x, y, z],
                "surface_area": float
            }
        """
        if not HAS_TRIMESH:
            return None

        if not os.path.exists(ply_path):
            print(f"[VolumeCalculator] PLY file not found: {ply_path}")
            return None

        try:
            # PLY 파일 로드
            mesh = trimesh.load(ply_path)

            # Point cloud인 경우 convex hull로 메시 생성
            if isinstance(mesh, trimesh.PointCloud):
                points = mesh.vertices
                if len(points) < 4:
                    print("[VolumeCalculator] Not enough points for convex hull")
                    return self._calculate_from_points(points)
                mesh = trimesh.convex.convex_hull(points)

            return self._analyze_mesh(mesh)

        except Exception as e:
            print(f"[VolumeCalculator] Error loading PLY: {e}")
            return None

    def calculate_from_glb(self, glb_path: str) -> Optional[Dict]:
        """
        GLB 파일에서 부피와 치수를 계산합니다.

        Args:
            glb_path: GLB 파일 경로

        Returns:
            치수 정보 딕셔너리
        """
        if not HAS_TRIMESH:
            return None

        if not os.path.exists(glb_path):
            print(f"[VolumeCalculator] GLB file not found: {glb_path}")
            return None

        try:
            # GLB 파일 로드 (Scene으로 로드될 수 있음)
            scene_or_mesh = trimesh.load(glb_path)

            if isinstance(scene_or_mesh, trimesh.Scene):
                # Scene인 경우 모든 geometry를 하나로 합침
                meshes = []
                for name, geometry in scene_or_mesh.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        meshes.append(geometry)

                if not meshes:
                    print("[VolumeCalculator] No meshes found in GLB scene")
                    return None

                # 모든 메시를 하나로 합침
                combined = trimesh.util.concatenate(meshes)
                return self._analyze_mesh(combined)
            else:
                return self._analyze_mesh(scene_or_mesh)

        except Exception as e:
            print(f"[VolumeCalculator] Error loading GLB: {e}")
            return None

    def calculate_from_gaussian_splat(self, gaussian_splat) -> Optional[Dict]:
        """
        Gaussian Splat 객체에서 부피와 치수를 계산합니다.

        Args:
            gaussian_splat: SAM-3D의 Gaussian Splat 객체

        Returns:
            치수 정보 딕셔너리
        """
        if not HAS_TORCH:
            return None

        try:
            # Gaussian의 중심점 추출
            xyz = gaussian_splat.get_xyz

            if HAS_TORCH and torch.is_tensor(xyz):
                points = xyz.detach().cpu().numpy()
            else:
                points = np.array(xyz)

            if len(points) < 4:
                return self._calculate_from_points(points)

            # Convex hull로 메시 생성
            if HAS_TRIMESH:
                mesh = trimesh.convex.convex_hull(points)
                return self._analyze_mesh(mesh)
            else:
                return self._calculate_from_points(points)

        except Exception as e:
            print(f"[VolumeCalculator] Error processing Gaussian Splat: {e}")
            return None

    def _analyze_mesh(self, mesh: 'trimesh.Trimesh') -> Dict:
        """
        trimesh 객체를 분석하여 부피와 치수를 계산합니다.
        """
        # Bounding box 계산
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        dimensions = bounds[1] - bounds[0]

        width = float(dimensions[0])
        depth = float(dimensions[1])
        height = float(dimensions[2])

        # 부피 계산 (mesh가 watertight인 경우)
        try:
            if mesh.is_watertight:
                volume = float(mesh.volume)
            else:
                # watertight가 아닌 경우 convex hull 부피 사용
                volume = float(mesh.convex_hull.volume)
        except Exception:
            # fallback: bounding box 부피
            volume = width * depth * height

        # 비율 계산 (가장 큰 값을 1로 정규화)
        max_dim = max(width, height, depth)
        if max_dim == 0:
            max_dim = 1

        ratio = {
            "w": width / max_dim,
            "h": height / max_dim,
            "d": depth / max_dim
        }

        # 중심점 계산
        centroid = mesh.centroid.tolist() if hasattr(mesh, 'centroid') else [0, 0, 0]

        # 표면적 계산
        try:
            surface_area = float(mesh.area)
        except Exception:
            surface_area = 0.0

        return {
            "volume": volume,
            "bounding_box": {
                "width": width,
                "depth": depth,
                "height": height
            },
            "ratio": ratio,
            "centroid": centroid,
            "surface_area": surface_area
        }

    def _calculate_from_points(self, points: np.ndarray) -> Dict:
        """
        점군에서 bounding box 기반 치수를 계산합니다.
        """
        if len(points) == 0:
            return {
                "volume": 0,
                "bounding_box": {"width": 0, "depth": 0, "height": 0},
                "ratio": {"w": 1, "h": 1, "d": 1},
                "centroid": [0, 0, 0],
                "surface_area": 0
            }

        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        dimensions = max_bounds - min_bounds

        width = float(dimensions[0])
        depth = float(dimensions[1])
        height = float(dimensions[2])

        # Bounding box 부피
        volume = width * depth * height

        # 비율 정규화
        max_dim = max(width, height, depth)
        if max_dim == 0:
            max_dim = 1

        ratio = {
            "w": width / max_dim,
            "h": height / max_dim,
            "d": depth / max_dim
        }

        centroid = points.mean(axis=0).tolist()

        return {
            "volume": volume,
            "bounding_box": {"width": width, "depth": depth, "height": height},
            "ratio": ratio,
            "centroid": centroid,
            "surface_area": 0
        }

    def scale_to_absolute(
        self,
        relative_dims: Dict,
        reference_dims: Dict
    ) -> Dict:
        """
        SAM-3D의 상대적 치수를 DB의 절대적 치수로 스케일링합니다.

        Args:
            relative_dims: SAM-3D에서 계산된 상대적 치수 (비율)
            reference_dims: DB에서 가져온 참조 치수 (mm)

        Returns:
            절대 치수 (mm)
        """
        if not relative_dims or not reference_dims:
            return None

        rel_ratio = relative_dims.get("ratio", {})
        ref_box = reference_dims

        # 참조 치수에서 가장 큰 값으로 스케일 팩터 계산
        ref_max = max(
            ref_box.get("width", 0),
            ref_box.get("depth", 0),
            ref_box.get("height", 0)
        )

        if ref_max == 0:
            return None

        # 상대 비율의 최대값 (이미 1로 정규화됨)
        rel_max = max(
            rel_ratio.get("w", 0),
            rel_ratio.get("h", 0),
            rel_ratio.get("d", 0)
        )

        if rel_max == 0:
            return reference_dims.copy()

        # 스케일 팩터: 참조 최대값 / 상대 최대값
        scale_factor = ref_max / rel_max

        # 절대 치수 계산
        width = rel_ratio.get("w", 0) * scale_factor
        depth = rel_ratio.get("d", 0) * scale_factor
        height = rel_ratio.get("h", 0) * scale_factor

        # 부피 계산 (mm^3 -> cm^3로도 제공)
        volume_mm3 = width * depth * height
        volume_cm3 = volume_mm3 / 1000  # mm^3 to cm^3
        volume_liters = volume_cm3 / 1000  # cm^3 to liters

        return {
            "width": round(width, 1),
            "depth": round(depth, 1),
            "height": round(height, 1),
            "volume_mm3": round(volume_mm3, 1),
            "volume_cm3": round(volume_cm3, 2),
            "volume_liters": round(volume_liters, 3),
            "ratio": {
                "w": round(rel_ratio.get("w", 0), 3),
                "h": round(rel_ratio.get("h", 0), 3),
                "d": round(rel_ratio.get("d", 0), 3)
            }
        }


def estimate_dimensions_from_aspect_ratio(
    db_key: str,
    subtype_name: str,
    aspect_ratio: Dict,
    knowledge_base: Dict
) -> Optional[Dict]:
    """
    CLIP으로 분류된 가구 유형과 SAM-3D 비율을 사용하여 절대 치수를 추정합니다.

    Args:
        db_key: FURNITURE_DB의 키 (예: "bed")
        subtype_name: 서브타입 이름 (예: "퀸 사이즈 침대")
        aspect_ratio: SAM-3D에서 계산된 비율 {"w": float, "h": float, "d": float}
        knowledge_base: FURNITURE_DB 딕셔너리

    Returns:
        절대 치수 딕셔너리
    """
    if db_key not in knowledge_base:
        return None

    db_info = knowledge_base[db_key]

    # 서브타입에서 dimensions 찾기
    ref_dims = None
    if 'subtypes' in db_info:
        for subtype in db_info['subtypes']:
            if subtype['name'] == subtype_name:
                dims = subtype.get('dimensions', {})
                if 'variants' in dims:
                    # variants가 있는 경우 비율로 가장 유사한 것 선택
                    from DeCl.data.knowledge_base import estimate_size_variant
                    _, ref_dims = estimate_size_variant(db_key, subtype_name, aspect_ratio)
                else:
                    ref_dims = dims
                break

    if ref_dims is None:
        # 기본 dimensions 사용
        dims = db_info.get('dimensions', {})
        if 'variants' in dims:
            from DeCl.data.knowledge_base import estimate_size_variant
            _, ref_dims = estimate_size_variant(db_key, subtype_name, aspect_ratio)
        else:
            ref_dims = dims

    if not ref_dims:
        return None

    # 절대 치수 계산
    calculator = VolumeCalculator()
    relative_dims = {"ratio": aspect_ratio}

    return calculator.scale_to_absolute(relative_dims, ref_dims)
