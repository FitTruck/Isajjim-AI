"""3D Bin Packing 최적화 서비스 (py3dbp 기반)"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# py3dbp 설치 체크
try:
    from py3dbp import Packer, Bin, Item
    PY3DBP_AVAILABLE = True
except ImportError:
    PY3DBP_AVAILABLE = False
    logger.warning("py3dbp not installed. Run: pip install py3dbp")


@dataclass
class PlacementResult:
    """배치 결과"""
    id: str
    x: float  # 중심 좌표
    y: float
    z: float
    width: float  # 회전 적용된 치수
    depth: float
    height: float
    rotated: bool


@dataclass
class OptimizationResult:
    """최적화 결과"""
    success: bool
    placements: list[PlacementResult]
    unplaced_ids: list[str]
    load_percent: float
    algorithm: str
    message: str


def optimize_placement(
    truck_width: float,
    truck_depth: float,
    truck_height: float,
    items: list[dict],
    algorithm: str = "py3dbp"
) -> OptimizationResult:
    """
    3D Bin Packing 최적화 실행

    Args:
        truck_width: 트럭 너비 (m)
        truck_depth: 트럭 깊이 (m)
        truck_height: 트럭 높이 (m)
        items: 가구 리스트 [{"id", "width", "depth", "height"}, ...]
        algorithm: 알고리즘 선택 ("py3dbp" | "blf")

    Returns:
        OptimizationResult
    """
    if algorithm == "py3dbp" and PY3DBP_AVAILABLE:
        return _optimize_py3dbp(truck_width, truck_depth, truck_height, items)
    else:
        return _optimize_blf(truck_width, truck_depth, truck_height, items)


def _optimize_py3dbp(
    truck_width: float,
    truck_depth: float,
    truck_height: float,
    items: list[dict]
) -> OptimizationResult:
    """py3dbp 라이브러리를 사용한 최적화"""
    packer = Packer()

    # 트럭을 Bin으로 추가 (mm 단위로 변환하여 정밀도 향상)
    SCALE = 1000  # m → mm
    packer.add_bin(Bin(
        "truck",
        truck_width * SCALE,
        truck_height * SCALE,  # py3dbp는 WHD 순서 (Width, Height, Depth)
        truck_depth * SCALE,
        999999999  # max_weight (py3dbp doesn't handle inf)
    ))

    # 가구를 Item으로 추가
    for item in items:
        packer.add_item(Item(
            item["id"],
            item["width"] * SCALE,
            item["height"] * SCALE,
            item["depth"] * SCALE,
            1.0  # weight (무시)
        ))

    # 패킹 실행
    packer.pack(
        bigger_first=True,      # 큰 것부터
        distribute_items=False  # 단일 bin만 사용
    )

    # 결과 추출
    placements = []
    placed_volume = 0
    truck_volume = truck_width * truck_depth * truck_height

    truck_bin = packer.bins[0]
    for fitted_item in truck_bin.items:
        # py3dbp 좌표를 Three.js 좌표로 변환
        # py3dbp: 좌하단 기준 (x, y, z) where y is height
        # Three.js: 중심 기준, Y-up

        # 회전 상태 확인 (py3dbp는 rotation_type으로 표현)
        # rotation_type: 0=WHD, 1=HWD, 2=HDW, 3=DHW, 4=DWH, 5=WDH
        rot = fitted_item.rotation_type
        rotated = rot in [1, 2, 3, 4, 5]  # 0이 아니면 회전됨

        # 실제 배치된 치수 (rotation 적용) - Decimal → float 변환
        w = float(fitted_item.width) / SCALE
        h = float(fitted_item.height) / SCALE
        d = float(fitted_item.depth) / SCALE

        # py3dbp 위치 (좌하단 기준) → Three.js 중심 좌표
        px = float(fitted_item.position[0]) / SCALE
        py = float(fitted_item.position[1]) / SCALE  # height
        pz = float(fitted_item.position[2]) / SCALE

        # Three.js 좌표계로 변환 (트럭 중심 기준)
        center_x = px + w / 2 - truck_width / 2
        center_y = py + h / 2  # 바닥 기준
        center_z = pz + d / 2 - truck_depth / 2

        placements.append(PlacementResult(
            id=fitted_item.name,
            x=center_x,
            y=center_y,
            z=center_z,
            width=w,
            depth=d,
            height=h,
            rotated=rotated
        ))

        placed_volume += w * d * h

    # 미배치 아이템
    unplaced_ids = [item.name for item in truck_bin.unfitted_items]

    load_percent = (placed_volume / truck_volume) * 100 if truck_volume > 0 else 0

    return OptimizationResult(
        success=True,
        placements=placements,
        unplaced_ids=unplaced_ids,
        load_percent=round(load_percent, 1),
        algorithm="py3dbp",
        message=f"{len(placements)}개 배치, {len(unplaced_ids)}개 미배치"
    )


def _optimize_blf(
    truck_width: float,
    truck_depth: float,
    truck_height: float,
    items: list[dict]
) -> OptimizationResult:
    """BLF(Bottom-Left-Fill) 알고리즘 (py3dbp 없을 때 fallback)"""

    # 부피 기준 내림차순 정렬
    sorted_items = sorted(
        items,
        key=lambda x: x["width"] * x["depth"] * x["height"],
        reverse=True
    )

    # Height Map 기반 BLF
    GRID_SIZE = 0.05  # 5cm
    grid_w = int(truck_width / GRID_SIZE) + 1
    grid_d = int(truck_depth / GRID_SIZE) + 1
    height_map = [[0.0] * grid_d for _ in range(grid_w)]

    placements = []
    unplaced_ids = []
    placed_volume = 0
    truck_volume = truck_width * truck_depth * truck_height

    for item in sorted_items:
        best_pos = _find_blf_position(
            item["width"], item["depth"], item["height"],
            height_map, GRID_SIZE, truck_width, truck_depth, truck_height
        )

        if best_pos:
            placements.append(PlacementResult(
                id=item["id"],
                x=best_pos["x"],
                y=best_pos["y"],
                z=best_pos["z"],
                width=best_pos["width"],
                depth=best_pos["depth"],
                height=item["height"],
                rotated=best_pos["rotated"]
            ))

            # Height map 업데이트
            _update_height_map(
                height_map,
                best_pos["x"], best_pos["z"],
                best_pos["width"], best_pos["depth"],
                best_pos["y"] + item["height"],
                GRID_SIZE, truck_width, truck_depth
            )

            placed_volume += best_pos["width"] * best_pos["depth"] * item["height"]
        else:
            unplaced_ids.append(item["id"])

    load_percent = (placed_volume / truck_volume) * 100 if truck_volume > 0 else 0

    return OptimizationResult(
        success=True,
        placements=placements,
        unplaced_ids=unplaced_ids,
        load_percent=round(load_percent, 1),
        algorithm="blf",
        message=f"{len(placements)}개 배치, {len(unplaced_ids)}개 미배치 (BLF fallback)"
    )


def _find_blf_position(
    item_w: float, item_d: float, item_h: float,
    height_map: list[list[float]], grid_size: float,
    truck_w: float, truck_d: float, truck_h: float
) -> Optional[dict]:
    """BLF 위치 찾기 (회전 포함)"""
    pos0 = _find_blf_single(item_w, item_d, item_h, height_map, grid_size, truck_w, truck_d, truck_h)
    pos90 = _find_blf_single(item_d, item_w, item_h, height_map, grid_size, truck_w, truck_d, truck_h)

    if not pos0 and not pos90:
        return None
    if not pos0:
        return {**pos90, "rotated": True, "width": item_d, "depth": item_w}
    if not pos90:
        return {**pos0, "rotated": False, "width": item_w, "depth": item_d}

    # BLF 우선순위: Y → Z → X
    if pos0["y"] < pos90["y"]:
        return {**pos0, "rotated": False, "width": item_w, "depth": item_d}
    if pos90["y"] < pos0["y"]:
        return {**pos90, "rotated": True, "width": item_d, "depth": item_w}
    if pos0["z"] < pos90["z"]:
        return {**pos0, "rotated": False, "width": item_w, "depth": item_d}
    if pos90["z"] < pos0["z"]:
        return {**pos90, "rotated": True, "width": item_d, "depth": item_w}
    return {**pos0, "rotated": False, "width": item_w, "depth": item_d}


def _find_blf_single(
    item_w: float, item_d: float, item_h: float,
    height_map: list[list[float]], grid_size: float,
    truck_w: float, truck_d: float, truck_h: float
) -> Optional[dict]:
    """단일 방향 BLF 위치 찾기"""
    grid_w = len(height_map)
    grid_d = len(height_map[0])
    item_gw = int(item_w / grid_size) + 1
    item_gd = int(item_d / grid_size) + 1

    best = None
    best_y, best_z, best_x = float('inf'), float('inf'), float('inf')

    for gz in range(grid_d - item_gd + 1):
        for gx in range(grid_w - item_gw + 1):
            max_h = 0
            for dx in range(item_gw):
                for dz in range(item_gd):
                    max_h = max(max_h, height_map[gx + dx][gz + dz])

            if max_h + item_h > truck_h:
                continue

            if (max_h < best_y or
                (max_h == best_y and gz < best_z) or
                (max_h == best_y and gz == best_z and gx < best_x)):
                best_y, best_z, best_x = max_h, gz, gx
                best = {
                    "x": -truck_w / 2 + gx * grid_size + item_w / 2,
                    "y": max_h,
                    "z": -truck_d / 2 + gz * grid_size + item_d / 2
                }

    return best


def _update_height_map(
    height_map: list[list[float]],
    cx: float, cz: float,
    item_w: float, item_d: float,
    new_height: float,
    grid_size: float,
    truck_w: float, truck_d: float
):
    """Height map 업데이트"""
    grid_w = len(height_map)
    grid_d = len(height_map[0])

    start_gx = int((cx - item_w / 2 + truck_w / 2) / grid_size)
    start_gz = int((cz - item_d / 2 + truck_d / 2) / grid_size)
    end_gx = int((cx + item_w / 2 + truck_w / 2) / grid_size) + 1
    end_gz = int((cz + item_d / 2 + truck_d / 2) / grid_size) + 1

    for gx in range(max(0, start_gx), min(grid_w, end_gx)):
        for gz in range(max(0, start_gz), min(grid_d, end_gz)):
            height_map[gx][gz] = max(height_map[gx][gz], new_height)
