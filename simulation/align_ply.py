#!/usr/bin/env python3
"""
PLY 객체를 OBB 기반으로 정렬하여 바닥에 수평으로 놓이게 함.
Open3D의 Oriented Bounding Box를 사용하여 객체의 주축을 정렬.
"""

import open3d as o3d
import numpy as np
import os
from pathlib import Path


def align_object_to_floor(ply_path: str) -> o3d.geometry.PointCloud:
    """
    PLY 객체를 바닥에 정렬.

    1. OBB(Oriented Bounding Box)를 계산
    2. OBB의 회전 행렬의 역행렬로 회전하여 축 정렬
    3. Z 최소값이 0이 되도록 이동 (바닥에 놓기)

    Args:
        ply_path: PLY 파일 경로

    Returns:
        정렬된 포인트 클라우드
    """
    # PLY 파일 로드
    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        print(f"Warning: Empty point cloud: {ply_path}")
        return pcd

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


def process_all_ply_files(input_dir: str, output_dir: str):
    """
    디렉토리 내 모든 PLY 파일을 정렬하여 저장.

    Args:
        input_dir: 입력 PLY 파일 디렉토리
        output_dir: 출력 디렉토리
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # PLY 파일 목록
    ply_files = list(input_path.glob("*.ply"))
    print(f"Found {len(ply_files)} PLY files in {input_dir}")

    for ply_file in ply_files:
        print(f"Processing: {ply_file.name}")
        try:
            # 정렬
            aligned_pcd = align_object_to_floor(str(ply_file))

            # 저장
            output_file = output_path / ply_file.name
            o3d.io.write_point_cloud(str(output_file), aligned_pcd)

            # 정렬 결과 출력
            aabb = aligned_pcd.get_axis_aligned_bounding_box()
            min_b = aabb.get_min_bound()
            max_b = aabb.get_max_bound()
            size = max_b - min_b
            print(f"  -> Saved: {output_file.name}")
            print(f"     Size: {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")
            print(f"     Min Z: {min_b[2]:.6f} (should be ~0)")

        except Exception as e:
            print(f"  Error processing {ply_file.name}: {e}")

    print(f"\nDone! Aligned files saved to: {output_dir}")


if __name__ == "__main__":
    # 경로 설정
    script_dir = Path(__file__).parent
    input_dir = script_dir / "assets"
    output_dir = script_dir / "assets" / "aligned"

    process_all_ply_files(str(input_dir), str(output_dir))
