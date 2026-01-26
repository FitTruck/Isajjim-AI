"""
Volume Calculator Module

SAM-3D에서 생성된 3D 메시/Gaussian Splat에서 부피와 치수를 계산합니다.
trimesh 라이브러리를 사용하여 메시 분석을 수행합니다.

2026-01 Update: AABB → OBB (Oriented Bounding Box) 변경
- PLY(Point Cloud)가 회전되어 있어서 AABB가 부정확한 치수 반환
- OBB를 사용하면 객체의 실제 방향에 맞춘 정확한 치수 계산
- trimesh.bounding_box_oriented 사용
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

    OBB (Oriented Bounding Box) 사용:
    - 객체가 회전되어 있어도 정확한 치수 계산
    - AABB 대비 최대 300%+ 정확도 향상 (특히 회전된 객체)
    """

    def __init__(self):
        if not HAS_TRIMESH:
            print("[VolumeCalculator] Warning: trimesh not installed")

    def calculate_from_ply(self, ply_path: str) -> Optional[Dict]:
        """
        PLY 파일에서 부피와 치수를 계산합니다.
        OBB (Oriented Bounding Box)를 사용하여 회전된 객체도 정확히 측정.

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

            # Point cloud인 경우 OBB 직접 계산
            if isinstance(mesh, trimesh.PointCloud):
                points = mesh.vertices
                if len(points) < 4:
                    print("[VolumeCalculator] Not enough points for OBB")
                    return self._calculate_from_points(points)
                return self._analyze_pointcloud_obb(mesh)

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
        OBB (Oriented Bounding Box) 사용.

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

            # trimesh PointCloud 생성 후 OBB 계산
            if HAS_TRIMESH:
                pointcloud = trimesh.PointCloud(points)
                return self._analyze_pointcloud_obb(pointcloud)
            else:
                return self._calculate_from_points(points)

        except Exception as e:
            print(f"[VolumeCalculator] Error processing Gaussian Splat: {e}")
            return None

    def _map_obb_to_dimensions(self, obb) -> tuple:
        """
        OBB 축을 SAM-3D 좌표계에 맞춰 width/depth/height로 매핑합니다.

        SAM-3D 좌표계:
        - X축 = 이미지 가로 = width (가로)
        - Y축 = 이미지 세로 = height (높이)
        - Z축 = 깊이 방향 = depth (깊이)

        Greedy 매핑: 각 OBB 축을 고유한 좌표축에 매핑 (중복 방지)

        Returns:
            (width, depth, height)
        """
        extents = obb.primitive.extents
        rotation = obb.primitive.transform[:3, :3]

        # 각 OBB 축(0,1,2)과 좌표축(X,Y,Z) 간의 유사도 계산
        # similarity[i][j] = OBB 축 i가 좌표축 j와 얼마나 정렬되어 있는지
        similarity = np.abs(rotation.T)  # (3, 3) - [obb_axis, coord_axis]

        # Greedy 매핑: 가장 높은 유사도부터 매핑
        obb_to_coord = {}  # OBB 축 인덱스 → 좌표축 인덱스 (0=X, 1=Y, 2=Z)
        used_coords = set()

        # 유사도가 높은 순서로 정렬
        pairs = []
        for i in range(3):
            for j in range(3):
                pairs.append((similarity[i, j], i, j))
        pairs.sort(reverse=True)

        for sim, obb_idx, coord_idx in pairs:
            if obb_idx not in obb_to_coord and coord_idx not in used_coords:
                obb_to_coord[obb_idx] = coord_idx
                used_coords.add(coord_idx)

        # 좌표축별 extent 할당
        # coord 0 (X) → width, coord 1 (Y) → height, coord 2 (Z) → depth
        dims = [0.0, 0.0, 0.0]  # [X, Y, Z] 순서
        for obb_idx, coord_idx in obb_to_coord.items():
            dims[coord_idx] = float(extents[obb_idx])

        width = dims[0]   # X
        height = dims[1]  # Y
        depth = dims[2]   # Z

        return width, depth, height

    def _analyze_pointcloud_obb(self, pointcloud: 'trimesh.PointCloud') -> Dict:
        """
        PointCloud의 OBB를 분석하여 치수를 계산합니다.
        trimesh의 bounding_box_oriented 사용.

        좌표계 기반 매핑:
        - X 방향 축 → width (가로)
        - Y 방향 축 → height (높이)
        - Z 방향 축 → depth (깊이)
        """
        try:
            # OBB 계산 (trimesh가 내부적으로 PCA 사용)
            obb = pointcloud.bounding_box_oriented

            # 좌표계 기반으로 width/depth/height 매핑
            width, depth, height = self._map_obb_to_dimensions(obb)

            # 부피 계산 (OBB 부피)
            volume = float(np.prod(obb.primitive.extents))

            # 중심점 계산
            centroid = pointcloud.centroid.tolist() if hasattr(pointcloud, 'centroid') else [0, 0, 0]

            return {
                "volume": volume,
                "bounding_box": {
                    "width": width,
                    "depth": depth,
                    "height": height
                },
                "centroid": centroid,
                "surface_area": 0.0  # PointCloud는 표면적 없음
            }

        except Exception as e:
            print(f"[VolumeCalculator] OBB calculation failed, fallback to convex hull: {e}")
            # Fallback: convex hull로 mesh 생성 후 분석
            hull = trimesh.convex.convex_hull(pointcloud.vertices)
            return self._analyze_mesh(hull)

    def _analyze_mesh(self, mesh: 'trimesh.Trimesh') -> Dict:
        """
        trimesh 객체를 분석하여 부피와 치수를 계산합니다.
        OBB (Oriented Bounding Box) + 좌표계 기반 매핑 사용.
        """
        # OBB 계산 (회전된 객체도 정확히 측정)
        try:
            obb = mesh.bounding_box_oriented

            # 좌표계 기반으로 width/depth/height 매핑
            width, depth, height = self._map_obb_to_dimensions(obb)

            obb_volume = float(np.prod(obb.primitive.extents))

        except Exception as e:
            print(f"[VolumeCalculator] OBB failed, using AABB: {e}")
            # Fallback to AABB (좌표계 기반)
            bounds = mesh.bounds
            dimensions = bounds[1] - bounds[0]
            width = float(dimensions[0])   # X → width
            depth = float(dimensions[2])   # Z → depth
            height = float(dimensions[1])  # Y → height
            obb_volume = width * depth * height

        # 실제 부피 계산 (mesh가 watertight인 경우)
        try:
            if mesh.is_watertight:
                volume = float(mesh.volume)
            else:
                # watertight가 아닌 경우 OBB 부피 사용
                volume = obb_volume
        except Exception:
            volume = obb_volume

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
        점군에서 OBB 기반 치수를 계산합니다.
        좌표계 기반 매핑 사용 (X→width, Y→height, Z→depth).
        점이 4개 미만인 경우 AABB fallback.
        """
        if len(points) == 0:
            return {
                "volume": 0,
                "bounding_box": {"width": 0, "depth": 0, "height": 0},
                "centroid": [0, 0, 0],
                "surface_area": 0
            }

        # 점이 충분하면 OBB 계산 (PCA 기반)
        if len(points) >= 4:
            try:
                # PCA로 주축 찾기
                centroid = points.mean(axis=0)
                centered = points - centroid
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # 점들을 주축으로 회전
                rotated = centered @ eigenvectors
                obb_min = rotated.min(axis=0)
                obb_max = rotated.max(axis=0)
                obb_dims = obb_max - obb_min

                # 좌표계 기반 Greedy 매핑 (중복 방지)
                similarity = np.abs(eigenvectors.T)  # (3, 3)

                pairs = []
                for i in range(3):
                    for j in range(3):
                        pairs.append((similarity[i, j], i, j))
                pairs.sort(reverse=True)

                obb_to_coord = {}
                used_coords = set()
                for sim, obb_idx, coord_idx in pairs:
                    if obb_idx not in obb_to_coord and coord_idx not in used_coords:
                        obb_to_coord[obb_idx] = coord_idx
                        used_coords.add(coord_idx)

                dims = [0.0, 0.0, 0.0]
                for obb_idx, coord_idx in obb_to_coord.items():
                    dims[coord_idx] = float(obb_dims[obb_idx])

                width = dims[0]   # X
                height = dims[1]  # Y
                depth = dims[2]   # Z

                volume = float(np.prod(obb_dims))

                return {
                    "volume": volume,
                    "bounding_box": {"width": width, "depth": depth, "height": height},
                    "centroid": centroid.tolist(),
                    "surface_area": 0
                }
            except Exception:
                pass  # Fallback to AABB below

        # Fallback: AABB (좌표계 기반)
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        dimensions = max_bounds - min_bounds

        width = float(dimensions[0])   # X → width
        depth = float(dimensions[2])   # Z → depth
        height = float(dimensions[1])  # Y → height
        volume = width * depth * height

        centroid = points.mean(axis=0).tolist()

        return {
            "volume": volume,
            "bounding_box": {"width": width, "depth": depth, "height": height},
            "centroid": centroid,
            "surface_area": 0
        }
