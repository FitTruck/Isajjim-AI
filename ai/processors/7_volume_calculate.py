"""
Volume Calculator Module

SAM-3D에서 생성된 3D 메시/Gaussian Splat에서 부피와 치수를 계산합니다.
trimesh 라이브러리를 사용하여 메시 분석을 수행합니다.
"""

import numpy as np
from typing import Dict, Optional
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
                "volume": float,
                "bounding_box": {"width": float, "depth": float, "height": float},
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

        # 좌표계: X=이미지 가로, Y=이미지 세로(높이), Z=3D 추론 깊이
        width = float(dimensions[0])   # X축 = 이미지 가로
        height = float(dimensions[1])  # Y축 = 이미지 세로 (실제 높이)
        depth = float(dimensions[2])   # Z축 = 3D 추론 깊이

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
                "centroid": [0, 0, 0],
                "surface_area": 0
            }

        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        dimensions = max_bounds - min_bounds

        # 좌표계: X=이미지 가로, Y=이미지 세로(높이), Z=3D 추론 깊이
        width = float(dimensions[0])   # X축 = 이미지 가로
        height = float(dimensions[1])  # Y축 = 이미지 세로 (실제 높이)
        depth = float(dimensions[2])   # Z축 = 3D 추론 깊이

        # Bounding box 부피
        volume = width * depth * height

        centroid = points.mean(axis=0).tolist()

        return {
            "volume": volume,
            "bounding_box": {"width": width, "depth": depth, "height": height},
            "centroid": centroid,
            "surface_area": 0
        }
