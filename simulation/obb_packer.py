"""OBB 기반 3D Bin Packing - py3dbp 라이브러리 사용

트럭 컨테이너에 객체를 최적 배치하는 패커.
- py3dbp 라이브러리 기반 (검증된 알고리즘)
- 큰 물건부터 뒤쪽에서 위로 적재
- 6방향 회전 지원
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

# py3dbp 설치 체크
try:
    from py3dbp import Packer, Bin, Item
    PY3DBP_AVAILABLE = True
except ImportError:
    PY3DBP_AVAILABLE = False
    logger.warning("py3dbp not installed. Run: pip install py3dbp")


class Orientation(IntEnum):
    """6방향 회전 (L=길이, W=너비, H=높이)"""
    LWH = 0  # 기본 (L, W, H) - 원래 방향
    LHW = 1  # 옆으로 눕힘 (L, H, W) - 세움
    WLH = 2  # 90도 회전 (W, L, H) - 수평 회전
    WHL = 3  # (W, H, L) - 세움
    HLW = 4  # 완전히 눕힘 (H, L, W) - 세움
    HWL = 5  # (H, W, L) - 세움


# 수평 회전만 허용 (가구를 세우지 않음)
HORIZONTAL_ORIENTATIONS = [Orientation.LWH, Orientation.WLH]


# Corner 우선순위 (뒤쪽부터)
class CornerPosition(IntEnum):
    """4개 바닥 코너 (배치 우선순위 순서)"""
    LEFT_REAR = 0    # 뒤쪽 왼쪽 (1순위)
    RIGHT_REAR = 1   # 뒤쪽 오른쪽 (2순위)
    LEFT_FRONT = 2   # 앞쪽 왼쪽 (3순위)
    RIGHT_FRONT = 3  # 앞쪽 오른쪽 (4순위)


def get_rotated_dims(
    length: float, width: float, height: float, orientation: Orientation
) -> tuple[float, float, float]:
    """회전 적용된 치수 반환 (width, depth, height) - Three.js 좌표계"""
    L, W, H = length, width, height
    dims_map = {
        Orientation.LWH: (L, W, H),
        Orientation.LHW: (L, H, W),
        Orientation.WLH: (W, L, H),
        Orientation.WHL: (W, H, L),
        Orientation.HLW: (H, L, W),
        Orientation.HWL: (H, W, L),
    }
    return dims_map[orientation]


@dataclass
class OBBItem:
    """OBB 기반 아이템"""
    id: str
    original_dims: tuple[float, float, float]  # 원본 (width, depth, height)
    normalized_dims: tuple[float, float, float] = field(default_factory=tuple)  # 바닥 보정됨

    def __post_init__(self):
        if not self.normalized_dims:
            self.normalized_dims = normalize_to_floor(self.original_dims)

    @property
    def volume(self) -> float:
        w, d, h = self.normalized_dims
        return w * d * h


@dataclass
class ExtremePoint:
    """Extreme Point (EP) - 배치 후보 위치"""
    x: float
    y: float
    z: float

    def __lt__(self, other: "ExtremePoint") -> bool:
        # Z → X → Y 우선순위 (뒤쪽 → 왼쪽 → 바닥)
        # 뒤쪽(Z 작은 값)부터, 왼쪽부터 오른쪽으로, 마지막으로 위로 쌓기
        if self.z != other.z:
            return self.z < other.z
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y


@dataclass
class PlacedBox:
    """배치된 박스"""
    item_id: str
    x: float  # 중심 좌표
    y: float  # 바닥 좌표 (중심 아님)
    z: float  # 중심 좌표
    width: float  # X축
    depth: float  # Z축
    height: float  # Y축
    orientation: int

    @property
    def x_min(self) -> float:
        return self.x - self.width / 2

    @property
    def x_max(self) -> float:
        return self.x + self.width / 2

    @property
    def y_min(self) -> float:
        return self.y

    @property
    def y_max(self) -> float:
        return self.y + self.height

    @property
    def z_min(self) -> float:
        return self.z - self.depth / 2

    @property
    def z_max(self) -> float:
        return self.z + self.depth / 2

    @property
    def base_area(self) -> float:
        return self.width * self.depth


@dataclass
class PackingResult:
    """패킹 결과"""
    success: bool
    truck_type: str
    placed_items: list[PlacedBox]
    unplaced_items: list[str]
    volume_utilization: float
    message: str = ""


# 트럭 크기 (cm 단위) - models.py TRUCK_PRESETS와 일치
TRUCK_PRESETS_CM = {
    "1ton": {"width": 170, "depth": 280, "height": 170},
    "2.5ton": {"width": 200, "depth": 430, "height": 190},
    "5ton": {"width": 230, "depth": 620, "height": 240},
}


def normalize_to_floor(dims: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    원본 치수 유지 (width, depth, height)

    가구의 원본 방향을 유지하고, 회전은 Orientation으로 처리
    """
    # 원본 치수 그대로 반환 (L=width, W=depth, H=height)
    return dims


def check_boundary(
    x: float, y: float, z: float,
    w: float, d: float, h: float,
    truck_w: float, truck_d: float, truck_h: float
) -> bool:
    """경계 내 배치 가능 여부"""
    # 중심 좌표 기준 체크
    half_w, half_d = w / 2, d / 2

    if x - half_w < -truck_w / 2 - 0.001:
        return False
    if x + half_w > truck_w / 2 + 0.001:
        return False
    if z - half_d < -truck_d / 2 - 0.001:
        return False
    if z + half_d > truck_d / 2 + 0.001:
        return False
    if y < -0.001:
        return False
    if y + h > truck_h + 0.001:
        return False

    return True


def check_overlap(
    x: float, y: float, z: float,
    w: float, d: float, h: float,
    placed: list[PlacedBox],
    tolerance: float = 0.01
) -> bool:
    """충돌 검사 (AABB) - tolerance로 경계 접촉 허용"""
    # 새 박스 경계 (tolerance 만큼 축소하여 경계 접촉 허용)
    new_x_min = x - w / 2 + tolerance
    new_x_max = x + w / 2 - tolerance
    new_y_min = y + tolerance
    new_y_max = y + h - tolerance
    new_z_min = z - d / 2 + tolerance
    new_z_max = z + d / 2 - tolerance

    for box in placed:
        # 기존 박스 경계 (tolerance 만큼 축소)
        box_x_min = box.x_min + tolerance
        box_x_max = box.x_max - tolerance
        box_y_min = box.y_min + tolerance
        box_y_max = box.y_max - tolerance
        box_z_min = box.z_min + tolerance
        box_z_max = box.z_max - tolerance

        # AABB 충돌 검사 (양쪽 모두 축소된 경계로 검사)
        if (new_x_min < box_x_max and new_x_max > box_x_min and
            new_y_min < box_y_max and new_y_max > box_y_min and
            new_z_min < box_z_max and new_z_max > box_z_min):
            return True

    return False


def calculate_2d_overlap(
    x: float, z: float, w: float, d: float,
    box: PlacedBox
) -> float:
    """X-Z 평면에서 겹침 영역 계산"""
    x_overlap = max(0, min(x + w/2, box.x_max) - max(x - w/2, box.x_min))
    z_overlap = max(0, min(z + d/2, box.z_max) - max(z - d/2, box.z_min))
    return x_overlap * z_overlap


def check_support(
    x: float, y: float, z: float,
    w: float, d: float,
    placed: list[PlacedBox],
    min_ratio: float = 0.7
) -> bool:
    """
    70% 지지 규칙 검사

    바닥 배치(y=0)는 항상 지지됨.
    그 외에는 아래 박스들의 윗면과 겹치는 영역이 70% 이상이어야 함.
    """
    if y < 0.001:  # 바닥 배치
        return True

    base_area = w * d
    supported_area = 0.0

    for box in placed:
        # 아래 박스의 윗면이 현재 위치와 맞닿는지 확인
        if abs(box.y_max - y) < 0.01:
            overlap = calculate_2d_overlap(x, z, w, d, box)
            supported_area += overlap

    support_ratio = supported_area / base_area if base_area > 0 else 0
    return support_ratio >= min_ratio


def get_corner_position(
    corner: CornerPosition,
    w: float, d: float,
    truck_w: float, truck_d: float
) -> tuple[float, float, float]:
    """
    코너에 배치할 때의 중심 좌표 계산

    Args:
        corner: 코너 위치
        w, d: 객체의 회전 적용된 치수 (width, depth)
        truck_w, truck_d: 트럭 치수

    Returns:
        (cx, cy, cz) 중심 좌표
    """
    # 뒤쪽 (Z 음수 방향)
    z_rear = -truck_d / 2 + d / 2
    # 앞쪽 (Z 양수 방향)
    z_front = truck_d / 2 - d / 2
    # 왼쪽 (X 음수 방향)
    x_left = -truck_w / 2 + w / 2
    # 오른쪽 (X 양수 방향)
    x_right = truck_w / 2 - w / 2

    positions = {
        CornerPosition.LEFT_REAR: (x_left, 0, z_rear),
        CornerPosition.RIGHT_REAR: (x_right, 0, z_rear),
        CornerPosition.LEFT_FRONT: (x_left, 0, z_front),
        CornerPosition.RIGHT_FRONT: (x_right, 0, z_front),
    }
    return positions[corner]


def try_corner_placement(
    item: "OBBItem",
    corner: CornerPosition,
    placed: list[PlacedBox],
    truck_w: float, truck_d: float, truck_h: float,
    allow_tilt: bool = False
) -> Optional[tuple[CornerPosition, Orientation, tuple[float, float, float], tuple[float, float, float]]]:
    """
    특정 코너에 아이템 배치 시도

    Args:
        item: 배치할 아이템
        corner: 시도할 코너
        placed: 이미 배치된 박스들
        truck_w, truck_d, truck_h: 트럭 치수
        allow_tilt: 6방향 회전 허용 여부

    Returns:
        (corner, orientation, (w, d, h), (cx, cy, cz)) 또는 None
    """
    L, W, H = item.normalized_dims
    orientations = list(Orientation) if allow_tilt else HORIZONTAL_ORIENTATIONS

    # 유효한 배치 후보들 수집
    valid_placements = []

    for orientation in orientations:
        w, d, h = get_rotated_dims(L, W, H, orientation)
        cx, cy, cz = get_corner_position(corner, w, d, truck_w, truck_d)

        # 1. 경계 검사
        if not check_boundary(cx, cy, cz, w, d, h, truck_w, truck_d, truck_h):
            continue

        # 2. 충돌 검사
        if check_overlap(cx, cy, cz, w, d, h, placed):
            continue

        # 바닥 배치이므로 지지 검사는 불필요 (항상 통과)
        valid_placements.append((corner, orientation, (w, d, h), (cx, cy, cz)))

    if not valid_placements:
        return None

    # 긴 쪽이 트럭 길이 방향(Z축, depth)으로 향하는 방향 우선 선택
    # d >= w인 방향을 선호 (자연스러운 배치)
    for placement in valid_placements:
        _, _, (w, d, h), _ = placement
        if d >= w:  # depth가 width 이상이면 자연스러운 방향
            return placement

    # 모두 w > d이면 첫 번째 유효한 배치 반환
    return valid_placements[0]


def find_corner_placement(
    item: "OBBItem",
    available_corners: list[CornerPosition],
    placed: list[PlacedBox],
    truck_w: float, truck_d: float, truck_h: float,
    allow_tilt: bool = False
) -> Optional[tuple[CornerPosition, Orientation, tuple[float, float, float], tuple[float, float, float]]]:
    """
    가용한 코너들 중 첫 번째 유효한 배치 위치 찾기 (first-match)

    뒤쪽 코너부터 순서대로 시도: LEFT_REAR → RIGHT_REAR → LEFT_FRONT → RIGHT_FRONT

    Returns:
        (corner, orientation, (w, d, h), (cx, cy, cz)) 또는 None
    """
    for corner in available_corners:
        result = try_corner_placement(
            item, corner, placed, truck_w, truck_d, truck_h, allow_tilt
        )
        if result:
            return result
    return None


def generate_new_extreme_points(
    placed_box: PlacedBox,
    truck_w: float, truck_d: float, truck_h: float
) -> list[ExtremePoint]:
    """
    배치 후 새 EP 생성

    - EP1: +X 방향 (오른쪽)
    - EP2: +Z 방향 (앞쪽)
    - EP3: +Y 방향 (위쪽)
    """
    new_eps = []

    # EP1: 오른쪽 (+X)
    ep_x = placed_box.x_max
    if ep_x < truck_w / 2:
        new_eps.append(ExtremePoint(
            x=ep_x,
            y=placed_box.y_min,
            z=placed_box.z - placed_box.depth / 2
        ))

    # EP2: 앞쪽 (+Z)
    ep_z = placed_box.z_max
    if ep_z < truck_d / 2:
        new_eps.append(ExtremePoint(
            x=placed_box.x - placed_box.width / 2,
            y=placed_box.y_min,
            z=ep_z
        ))

    # EP3: 위쪽 (+Y)
    ep_y = placed_box.y_max
    if ep_y < truck_h:
        new_eps.append(ExtremePoint(
            x=placed_box.x - placed_box.width / 2,
            y=ep_y,
            z=placed_box.z - placed_box.depth / 2
        ))

    return new_eps


def is_ep_valid(
    ep: ExtremePoint,
    placed: list[PlacedBox],
    tolerance: float = 0.01
) -> bool:
    """EP가 다른 박스 내부에 있는지 확인 (경계는 유효)"""
    for box in placed:
        # EP가 박스 내부에 완전히 들어가 있는 경우만 무효
        # 경계(edge)에 있는 EP는 유효
        if (box.x_min + tolerance < ep.x < box.x_max - tolerance and
            box.y_min + tolerance < ep.y < box.y_max - tolerance and
            box.z_min + tolerance < ep.z < box.z_max - tolerance):
            return False
    return True


def find_best_placement(
    item: OBBItem,
    extreme_points: list[ExtremePoint],
    placed: list[PlacedBox],
    truck_w: float, truck_d: float, truck_h: float,
    support_ratio: float = 0.7,
    allow_tilt: bool = False
) -> Optional[tuple[ExtremePoint, Orientation, tuple[float, float, float]]]:
    """
    최적 배치 위치 찾기

    모든 EP에서 회전 시도 → 제약 조건 검사 → Z→Y→X 우선순위로 최적 선택
    (뒤쪽(안쪽)부터 채우고, 같은 깊이면 바닥부터, 같은 높이면 왼쪽부터)
    동일 점수일 경우 긴 쪽이 트럭 길이 방향(depth)으로 향하는 방향 우선

    Args:
        allow_tilt: True면 6방향 모두 허용, False면 수평 회전만 (기본)
    """
    best_placement = None
    best_score = (float('inf'), float('inf'), float('inf'), 0)  # (Z, Y, X, natural_orientation)

    L, W, H = item.normalized_dims

    # 허용할 회전 방향
    orientations = list(Orientation) if allow_tilt else HORIZONTAL_ORIENTATIONS

    for ep in extreme_points:
        for orientation in orientations:
            w, d, h = get_rotated_dims(L, W, H, orientation)

            # EP에서 중심 좌표 계산
            cx = ep.x + w / 2
            cy = ep.y
            cz = ep.z + d / 2

            # 1. 경계 검사
            if not check_boundary(cx, cy, cz, w, d, h, truck_w, truck_d, truck_h):
                continue

            # 2. 충돌 검사
            if check_overlap(cx, cy, cz, w, d, h, placed):
                continue

            # 3. 지지 검사 (70% 규칙)
            if not check_support(cx, cy, cz, w, d, placed, support_ratio):
                continue

            # 점수 계산 (Z → X → Y 우선순위) - 뒤쪽부터, 왼쪽에서 오른쪽으로 차곡차곡
            # Z: 뒤쪽(음수)부터, X: 왼쪽부터, Y: 바닥부터 (쌓기는 마지막)
            # 마지막 요소: d >= w면 0 (자연스러운 방향 우선), 아니면 1
            natural = 0 if d >= w else 1
            score = (cz, cx, cy, natural)

            if score < best_score:
                best_score = score
                best_placement = (ep, orientation, (w, d, h))

    return best_placement


def extreme_points_pack(
    items: list[OBBItem],
    truck_dims: dict,
    support_ratio: float = 0.7,
    allow_tilt: bool = False,
    corner_first: bool = True
) -> PackingResult:
    """
    Extreme Points 알고리즘으로 3D Bin Packing 수행

    Args:
        items: 배치할 아이템 리스트
        truck_dims: {"width", "depth", "height"} (cm)
        support_ratio: 지지 비율 (기본 0.7 = 70%)
        allow_tilt: True면 6방향 회전 허용, False면 수평 회전만 (기본)
        corner_first: True면 Corner-first 배치 사용 (기본 True)

    Returns:
        PackingResult
    """
    truck_w = truck_dims["width"]
    truck_d = truck_dims["depth"]
    truck_h = truck_dims["height"]
    truck_volume = truck_w * truck_d * truck_h

    # 부피 순 정렬 (큰 것 먼저)
    sorted_items = sorted(items, key=lambda x: x.volume, reverse=True)

    # 초기 EP: 뒤쪽-왼쪽-바닥 코너
    extreme_points = [ExtremePoint(x=-truck_w / 2, y=0, z=-truck_d / 2)]

    placed: list[PlacedBox] = []
    unplaced: list[str] = []
    placed_volume = 0.0

    # Corner-first 배치: 왼쪽 뒤 코너에서만 시작
    # (오른쪽 코너 제거 - 왼쪽에서 오른쪽으로 차곡차곡 채우기 위해)
    available_corners = [
        CornerPosition.LEFT_REAR,    # 뒤쪽 왼쪽 코너만 사용
    ]

    # Corner-first 단계 완료 여부
    corner_phase_done = not corner_first or len(available_corners) == 0

    for item in sorted_items:
        placement = None
        used_corner = None

        # Phase 1: Corner-first 배치 (4개 코너가 모두 채워질 때까지)
        if not corner_phase_done and available_corners:
            corner_result = find_corner_placement(
                item, available_corners, placed,
                truck_w, truck_d, truck_h, allow_tilt
            )

            if corner_result:
                corner, orientation, (w, d, h), (cx, cy, cz) = corner_result
                used_corner = corner

                # PlacedBox 생성
                box = PlacedBox(
                    item_id=item.id,
                    x=cx,
                    y=cy,
                    z=cz,
                    width=w,
                    depth=d,
                    height=h,
                    orientation=int(orientation)
                )
                placed.append(box)
                placed_volume += w * d * h

                # 사용된 코너 제거
                available_corners.remove(corner)
                logger.debug(
                    f"코너 배치 성공: {item.id} at corner {corner.name} "
                    f"({cx:.1f}, {cy:.1f}, {cz:.1f}), orientation={orientation}"
                )

                # 새 EP 생성 (코너 배치 후에도 EP 업데이트)
                new_eps = generate_new_extreme_points(box, truck_w, truck_d, truck_h)
                extreme_points.extend(new_eps)
                extreme_points = [ep for ep in extreme_points if is_ep_valid(ep, placed)]
                extreme_points.sort()

                # 뒤쪽 코너 모두 채워졌는지 확인
                if len(available_corners) == 0:
                    corner_phase_done = True
                    logger.info("Corner-first 단계 완료: 뒤쪽 코너 모두 채움")

                continue  # 다음 아이템으로

        # Phase 2: EP 기반 일반 배치 (코너 배치 실패 또는 코너 단계 완료 후)
        placement = find_best_placement(
            item, extreme_points, placed,
            truck_w, truck_d, truck_h, support_ratio, allow_tilt
        )

        if placement is None:
            unplaced.append(item.id)
            logger.debug(f"배치 실패: {item.id}")
            continue

        ep, orientation, (w, d, h) = placement

        # 중심 좌표 계산
        cx = ep.x + w / 2
        cy = ep.y
        cz = ep.z + d / 2

        # PlacedBox 생성
        box = PlacedBox(
            item_id=item.id,
            x=cx,
            y=cy,
            z=cz,
            width=w,
            depth=d,
            height=h,
            orientation=int(orientation)
        )
        placed.append(box)
        placed_volume += w * d * h

        # 새 EP 생성
        new_eps = generate_new_extreme_points(box, truck_w, truck_d, truck_h)
        extreme_points.extend(new_eps)

        # 무효 EP 제거 (다른 박스 내부에 있는 EP)
        extreme_points = [ep for ep in extreme_points if is_ep_valid(ep, placed)]

        # EP 정렬 (Y → Z → X)
        extreme_points.sort()

        logger.debug(f"EP 배치 성공: {item.id} at ({cx:.1f}, {cy:.1f}, {cz:.1f}), orientation={orientation}")

    volume_utilization = (placed_volume / truck_volume) * 100 if truck_volume > 0 else 0

    # 코너 배치 통계 메시지 (왼쪽 뒤 코너만 사용)
    corners_filled = 1 - len(available_corners) if corner_first else 0
    corner_msg = f" (코너시작)" if corner_first and corners_filled > 0 else ""

    return PackingResult(
        success=len(unplaced) == 0,
        truck_type="",  # 호출자가 설정
        placed_items=placed,
        unplaced_items=unplaced,
        volume_utilization=round(volume_utilization, 1),
        message=f"{len(placed)}개 배치{corner_msg}, {len(unplaced)}개 미배치"
    )


def select_smallest_fitting_truck(
    items: list[OBBItem],
    support_ratio: float = 0.7,
    allow_tilt: bool = False,
    corner_first: bool = True
) -> tuple[str, PackingResult]:
    """
    모든 아이템을 배치할 수 있는 가장 작은 트럭 선택

    Args:
        items: 배치할 아이템 리스트
        support_ratio: 지지 비율 (기본 0.7)
        allow_tilt: 6방향 회전 허용 여부
        corner_first: Corner-first 배치 사용 여부 (기본 True)

    Returns:
        (truck_type, PackingResult)
    """
    truck_order = ["1ton", "2.5ton", "5ton"]

    for truck_type in truck_order:
        truck_dims = TRUCK_PRESETS_CM[truck_type]
        result = extreme_points_pack(items, truck_dims, support_ratio, allow_tilt, corner_first)
        result.truck_type = truck_type

        if result.success:
            logger.info(f"트럭 선택: {truck_type} (적재율 {result.volume_utilization}%)")
            return truck_type, result

    # 모든 트럭에서 실패 → 최대 트럭 결과 반환
    result = extreme_points_pack(items, TRUCK_PRESETS_CM["5ton"], support_ratio, allow_tilt, corner_first)
    result.truck_type = "5ton"
    logger.warning(f"5톤 트럭에서도 {len(result.unplaced_items)}개 미배치")

    return "5ton", result


def _pack_with_py3dbp(
    items: list[dict],
    truck_dims: dict,
    scale: float = 1.0
) -> PackingResult:
    """
    py3dbp 라이브러리를 사용한 3D bin packing

    전략:
    1. 부피 + 깊이 기준 내림차순 정렬 (Big-to-Small, Deep-first)
    2. 가구 세움 방지 (height가 가장 작은 축이 되도록)
    3. 벽면부터 채우기 (py3dbp가 (0,0,0)부터 채움)

    Args:
        items: [{"id", "width", "depth", "height"}, ...]
        truck_dims: {"width", "depth", "height"} (cm)
        scale: 단위 변환 스케일

    Returns:
        PackingResult
    """
    if not PY3DBP_AVAILABLE:
        raise ImportError("py3dbp not installed. Run: pip install py3dbp")

    packer = Packer()

    # 트럭(컨테이너) 추가
    truck_w = truck_dims["width"]
    truck_h = truck_dims["height"]
    truck_d = truck_dims["depth"]
    truck_volume = truck_w * truck_h * truck_d

    # py3dbp Bin: (name, width, height, depth, max_weight)
    packer.add_bin(Bin('truck', truck_w, truck_h, truck_d, 100000))

    # 아이템 정렬: 무게 → 바닥면적 → 높이 (Strict Sorting)
    # 1. 무거운 것: 바닥에 깔려야 함
    # 2. 바닥면적 넓은 것: 위에 뭘 쌓기 좋음
    # 3. 높이 높은 것: 안쪽 벽을 형성
    sorted_items = sorted(
        items,
        key=lambda x: (
            x.get("weight", x["width"] * x["depth"] * x["height"] * 100),  # 무게 (없으면 부피 기반 추정)
            x["width"] * x["depth"] * scale**2,  # 바닥 면적
            x["height"] * scale  # 높이
        ),
        reverse=True
    )

    # 원본 치수 저장 (후처리용)
    original_dims = {}

    for item in sorted_items:
        w = item["width"] * scale
        d = item["depth"] * scale
        h = item["height"] * scale
        original_dims[item["id"]] = (w, d, h)  # (width, depth, height)

        # 회전 제약: height가 가장 작은 축이 되도록 치수 재배열
        # py3dbp는 (width, height, depth) 순서
        # 가구를 세우지 않으려면 height를 실제 높이로 고정
        #
        # 전략: width와 depth 중 큰 값을 py3dbp의 depth로 전달
        # (py3dbp가 depth 방향으로 먼저 채우므로)
        if d >= w:
            # depth가 더 김 - 그대로 전달
            item_w, item_h, item_d = w, h, d
        else:
            # width가 더 김 - 90도 회전해서 전달
            item_w, item_h, item_d = d, h, w

        # py3dbp Item: (name, width, height, depth, weight)
        packer.add_item(Item(item["id"], item_w, item_h, item_d, 1))

    # 패킹 실행
    packer.pack()

    # 결과 추출
    placed_items = []
    placed_volume = 0.0

    for b in packer.bins:
        for packed_item in b.items:
            pos = packed_item.position
            dims = packed_item.get_dimension()

            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            pw, ph, pd = float(dims[0]), float(dims[1]), float(dims[2])

            # 원본 치수 가져오기
            orig_w, orig_d, orig_h = original_dims.get(packed_item.name, (pw, pd, ph))

            # 가구가 세워졌는지 확인
            tolerance = 1.0  # 1cm 허용오차

            # 배치된 height가 원래 height와 비슷한지 확인
            height_ok = abs(ph - orig_h) < tolerance

            if height_ok:
                # 높이가 맞음 - width/depth 회전만 확인
                final_h = orig_h
                final_y = py

                # 90도 수평 회전 여부 판단
                if abs(pw - orig_w) < tolerance:
                    # 회전 없음
                    final_w = orig_w
                    final_d = orig_d
                    orientation = 0
                else:
                    # 90도 회전됨
                    final_w = orig_d
                    final_d = orig_w
                    orientation = 2
            else:
                # 높이가 다름 - 세워졌을 수 있음, 원래 방향으로 복원
                logger.warning(f"{packed_item.name}: 세움 감지, 복원 (ph={ph:.1f}, orig_h={orig_h:.1f})")
                final_w = orig_w
                final_d = orig_d
                final_h = orig_h
                final_y = py
                orientation = 0

            # 중심 좌표로 변환 (트럭 중심이 원점)
            cx = px + final_w / 2 - truck_w / 2
            cy = final_y  # 바닥 좌표
            cz = pz + final_d / 2 - truck_d / 2

            box = PlacedBox(
                item_id=packed_item.name,
                x=cx,
                y=cy,
                z=cz,
                width=final_w,
                depth=final_d,
                height=final_h,
                orientation=orientation
            )
            placed_items.append(box)
            placed_volume += final_w * final_d * final_h

    # 미배치 아이템
    unplaced_ids = []
    for b in packer.bins:
        for item in b.unfitted_items:
            unplaced_ids.append(item.name)

    # 중력 패스 비활성화 - py3dbp 결과 그대로 사용
    # placed_items = _apply_gravity_pass(placed_items, truck_w, truck_d, truck_h)

    volume_utilization = (placed_volume / truck_volume) * 100 if truck_volume > 0 else 0

    return PackingResult(
        success=len(unplaced_ids) == 0,
        truck_type="",
        placed_items=placed_items,
        unplaced_items=unplaced_ids,
        volume_utilization=round(volume_utilization, 1),
        message=f"{len(placed_items)}개 배치, {len(unplaced_ids)}개 미배치"
    )


def _apply_gravity_pass(
    placed_items: list[PlacedBox],
    truck_w: float, truck_d: float, truck_h: float
) -> list[PlacedBox]:
    """
    레이어드 중력 알고리즘: 완전 재배치

    py3dbp 결과(치수 정보만 사용)를 받아서 처음부터 재배치합니다.

    전략:
    1. 부피가 큰 순서로 처리
    2. 각 아이템을 뒤쪽-왼쪽-바닥 코너에서 시작
    3. 충돌 시 오른쪽 → 앞쪽 → 위쪽 순으로 이동
    4. 항상 바닥 또는 지지대 위에 배치 (공중 X)

    Args:
        placed_items: 배치된 아이템 리스트 (치수 정보용)
        truck_w, truck_d, truck_h: 트럭 치수 (cm)

    Returns:
        재배치된 아이템 리스트
    """
    # 부피가 큰 순서로 정렬
    items_by_volume = sorted(
        placed_items,
        key=lambda x: x.width * x.depth * x.height,
        reverse=True
    )

    adjusted_items = []

    for item in items_by_volume:
        w, d, h = item.width, item.depth, item.height

        # 가능한 Y 레벨 (바닥 + 기존 아이템 윗면)
        y_levels = [0]
        for other in adjusted_items:
            y_levels.append(other.y + other.height)
        y_levels = sorted(set(y_levels))

        placed = False

        # 각 Y 레벨에서 배치 시도 (아래부터)
        for y_level in y_levels:
            if placed:
                break

            # Z 스캔 (뒤쪽부터)
            z_start = -truck_d / 2 + d / 2
            z_end = truck_d / 2 - d / 2
            z_step = d  # 아이템 깊이만큼 스캔0

            z = z_start
            while z <= z_end and not placed:
                # X 스캔 (왼쪽부터)
                x_start = -truck_w / 2 + w / 2
                x_end = truck_w / 2 - w / 2
                x_step = w / 2  # 더 세밀하게 스캔

                x = x_start
                while x <= x_end and not placed:
                    # 이 위치에서 지지되는지 확인
                    if y_level > 0.01:
                        # 바닥이 아니면 지지대가 있어야 함
                        has_support = False
                        for other in adjusted_items:
                            if abs(other.y + other.height - y_level) < 0.01:
                                if _overlaps_xz(x, w, z, d, other):
                                    has_support = True
                                    break
                        if not has_support:
                            x += x_step
                            continue

                    # 충돌 체크
                    collision = False
                    for other in adjusted_items:
                        if _check_collision(x, y_level, z, w, d, h, other):
                            collision = True
                            break

                    # 높이 체크
                    if y_level + h > truck_h:
                        collision = True

                    if not collision:
                        # 배치 성공!
                        adjusted_items.append(PlacedBox(
                            item_id=item.item_id,
                            x=x,
                            y=y_level,
                            z=z,
                            width=w,
                            depth=d,
                            height=h,
                            orientation=item.orientation
                        ))
                        placed = True
                        break

                    x += x_step
                z += z_step

        # 배치 실패 시 원래 위치 사용
        if not placed:
            adjusted_items.append(PlacedBox(
                item_id=item.item_id,
                x=item.x,
                y=item.y,
                z=item.z,
                width=w,
                depth=d,
                height=h,
                orientation=item.orientation
            ))

    return adjusted_items


def _overlaps_xz(x: float, w: float, z: float, d: float, other: PlacedBox) -> bool:
    """XZ 평면에서 겹치는지 확인 (Y 무시) - 지지 체크용"""
    x_overlap = (x - w / 2 < other.x + other.width / 2) and (x + w / 2 > other.x - other.width / 2)
    z_overlap = (z - d / 2 < other.z + other.depth / 2) and (z + d / 2 > other.z - other.depth / 2)
    return x_overlap and z_overlap


def _overlaps_xy(x: float, w: float, y: float, h: float, other: PlacedBox) -> bool:
    """XY 평면에서 겹치는지 확인 (Z 무시) - Z축 밀착용"""
    x_overlap = (x - w / 2 < other.x + other.width / 2) and (x + w / 2 > other.x - other.width / 2)
    y_overlap = (y < other.y + other.height) and (y + h > other.y)
    return x_overlap and y_overlap


def _overlaps_yz(y: float, h: float, z: float, d: float, other: PlacedBox) -> bool:
    """YZ 평면에서 겹치는지 확인 (X 무시) - X축 밀착용"""
    y_overlap = (y < other.y + other.height) and (y + h > other.y)
    z_overlap = (z - d / 2 < other.z + other.depth / 2) and (z + d / 2 > other.z - other.depth / 2)
    return y_overlap and z_overlap


def _check_collision(
    x: float, y: float, z: float,
    w: float, d: float, h: float,
    other: PlacedBox
) -> bool:
    """3D 충돌 체크"""
    x_overlap = (x - w / 2 < other.x + other.width / 2) and (x + w / 2 > other.x - other.width / 2)
    y_overlap = (y < other.y + other.height) and (y + h > other.y)
    z_overlap = (z - d / 2 < other.z + other.depth / 2) and (z + d / 2 > other.z - other.depth / 2)
    return x_overlap and y_overlap and z_overlap


def _boxes_overlap_xz(item: PlacedBox, other: PlacedBox, item_x: float, item_z: float) -> bool:
    """X-Z 평면에서 겹치는지 확인 (Y 무시)"""
    item_x_min = item_x - item.width / 2
    item_x_max = item_x + item.width / 2
    item_z_min = item_z - item.depth / 2
    item_z_max = item_z + item.depth / 2

    other_x_min = other.x - other.width / 2
    other_x_max = other.x + other.width / 2
    other_z_min = other.z - other.depth / 2
    other_z_max = other.z + other.depth / 2

    x_overlap = item_x_min < other_x_max and item_x_max > other_x_min
    z_overlap = item_z_min < other_z_max and item_z_max > other_z_min

    return x_overlap and z_overlap


def _boxes_overlap_xy(item: PlacedBox, other: PlacedBox, item_x: float, item_z: float) -> bool:
    """X-Z 평면에서 겹치는지 확인 (지지 체크용)"""
    return _boxes_overlap_xz(item, other, item_x, item_z)


def _boxes_overlap_yz(item: PlacedBox, other: PlacedBox, item_y: float, item_z: float) -> bool:
    """Y-Z 평면에서 겹치는지 확인 (X 무시)"""
    item_y_min = item_y
    item_y_max = item_y + item.height
    item_z_min = item_z - item.depth / 2
    item_z_max = item_z + item.depth / 2

    other_y_min = other.y
    other_y_max = other.y + other.height
    other_z_min = other.z - other.depth / 2
    other_z_max = other.z + other.depth / 2

    y_overlap = item_y_min < other_y_max and item_y_max > other_y_min
    z_overlap = item_z_min < other_z_max and item_z_max > other_z_min

    return y_overlap and z_overlap


def _find_best_truck_py3dbp(items: list[dict], scale: float = 1.0) -> tuple[str, PackingResult]:
    """
    py3dbp로 모든 아이템이 들어가는 가장 작은 트럭 선택

    Args:
        items: 아이템 리스트
        scale: 단위 변환 스케일

    Returns:
        (truck_type, PackingResult)
    """
    truck_order = ["1ton", "2.5ton", "5ton"]

    for truck_type in truck_order:
        truck_dims = TRUCK_PRESETS_CM[truck_type]
        result = _pack_with_py3dbp(items, truck_dims, scale)
        result.truck_type = truck_type

        if result.success:
            logger.info(f"py3dbp 트럭 선택: {truck_type} (적재율 {result.volume_utilization}%)")
            return truck_type, result

    # 모든 트럭 실패 - 5톤 결과 반환
    result = _pack_with_py3dbp(items, TRUCK_PRESETS_CM["5ton"], scale)
    result.truck_type = "5ton"
    logger.warning(f"5톤 트럭에서도 {len(result.unplaced_items)}개 미배치")

    return "5ton", result


def optimize_obb(
    items: list[dict],
    truck_type: Optional[str] = None,
    unit: str = "cm",
    support_ratio: float = 0.7,
    allow_tilt: bool = False,
    corner_first: bool = True
) -> PackingResult:
    """
    OBB 최적화 메인 함수 (py3dbp 사용)

    Args:
        items: [{"id", "width", "depth", "height"}, ...]
        truck_type: "1ton" | "2.5ton" | "5ton" | None (자동 선택)
        unit: "cm" | "m"
        support_ratio: 지지 비율 (현재 미사용, py3dbp 기본값 사용)
        allow_tilt: 회전 허용 (현재 미사용, py3dbp 기본값 사용)
        corner_first: 코너 우선 (현재 미사용, py3dbp 기본값 사용)

    Returns:
        PackingResult
    """
    # 단위 변환 (m → cm)
    scale = 100 if unit == "m" else 1

    # EP 알고리즘 사용 (py3dbp보다 적재율 높음)
    obb_items = []
    for item in items:
        dims = (
            item["width"] * scale,
            item["depth"] * scale,
            item["height"] * scale
        )
        obb_items.append(OBBItem(id=item["id"], original_dims=dims))

    if truck_type and truck_type in TRUCK_PRESETS_CM:
        result = extreme_points_pack(
            obb_items, TRUCK_PRESETS_CM[truck_type], support_ratio, allow_tilt, corner_first
        )
        result.truck_type = truck_type
    else:
        truck_type, result = select_smallest_fitting_truck(
            obb_items, support_ratio, allow_tilt, corner_first
        )

    # 단위 역변환 (cm → m) 필요시
    if unit == "m":
        for box in result.placed_items:
            box.x /= 100
            box.y /= 100
            box.z /= 100
            box.width /= 100
            box.depth /= 100
            box.height /= 100

    return result
