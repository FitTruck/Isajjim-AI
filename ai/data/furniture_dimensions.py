"""
Furniture Standard Dimensions Database

Backend의 FurnitureType.java와 FurnitureLabel.java를 Python으로 포팅한 데이터.
절대 부피 계산에 사용되는 표준 치수 정보를 제공합니다.

Ported from:
- Isajjim-Backend/src/main/java/kr/co/isajjim/domains/furniture/domain/constant/FurnitureType.java
- Isajjim-Backend/src/main/java/kr/co/isajjim/domains/furniture/domain/constant/FurnitureLabel.java
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FurnitureTypeDimension:
    """
    가구 타입별 표준 치수 정보

    Attributes:
        name: 가구 타입 이름 (예: "SINGLE_BED")
        width: 가로 길이 (mm), -1 = 가변
        depth: 세로 길이 (mm), -1 = 가변
        height: 높이 (mm), -1 = 가변
    """
    name: str
    width: float
    depth: float
    height: float

    def get_dimension_ratio(self) -> float:
        """
        가로/세로 비율 계산 (short/long)

        Returns:
            비율 값 (0.0 ~ 1.0), width나 depth가 -1이면 0.0 반환
        """
        if self.width <= 0 or self.depth <= 0:
            return 0.0
        return min(self.width, self.depth) / max(self.width, self.depth)

    @property
    def is_variable_height(self) -> bool:
        """높이가 가변인지 여부"""
        return self.height == -1

    @property
    def is_fully_variable(self) -> bool:
        """모든 치수가 가변인지 여부 (DEFAULT_DINING_TABLE 등)"""
        return self.width == -1 and self.depth == -1 and self.height == -1


# =============================================================================
# Furniture Types - 51개 타입 (FurnitureType.java 기준)
# =============================================================================
FURNITURE_TYPES: Dict[str, FurnitureTypeDimension] = {
    # 에어컨
    "WALL_MOUNTED_AIR_CONDITIONER": FurnitureTypeDimension(
        name="WALL_MOUNTED_AIR_CONDITIONER",
        width=850, depth=250, height=300
    ),
    "STANDING_AIR_CONDITIONER": FurnitureTypeDimension(
        name="STANDING_AIR_CONDITIONER",
        width=400, depth=350, height=1900
    ),

    # 테이블/수납
    "DEFAULT_COFFEE_TABLE": FurnitureTypeDimension(
        name="DEFAULT_COFFEE_TABLE",
        width=1230, depth=630, height=450
    ),
    "DEFAULT_MIRROR": FurnitureTypeDimension(
        name="DEFAULT_MIRROR",
        width=600, depth=50, height=1700
    ),
    "DEFAULT_STORAGE_BOX": FurnitureTypeDimension(
        name="DEFAULT_STORAGE_BOX",
        width=650, depth=500, height=400
    ),
    "DEFAULT_FAN": FurnitureTypeDimension(
        name="DEFAULT_FAN",
        width=400, depth=400, height=1100
    ),
    "DEFAULT_CABINET": FurnitureTypeDimension(
        name="DEFAULT_CABINET",
        width=700, depth=800, height=1200
    ),
    "DEFAULT_DRAWER": FurnitureTypeDimension(
        name="DEFAULT_DRAWER",
        width=1000, depth=500, height=1400
    ),
    "DEFAULT_NIGHTSTAND": FurnitureTypeDimension(
        name="DEFAULT_NIGHTSTAND",
        width=500, depth=400, height=500
    ),
    "DEFAULT_BOOKSHELF": FurnitureTypeDimension(
        name="DEFAULT_BOOKSHELF",
        width=1200, depth=300, height=2000
    ),
    "DEFAULT_DISPLAY_SHELF": FurnitureTypeDimension(
        name="DEFAULT_DISPLAY_SHELF",
        width=1200, depth=400, height=1800
    ),

    # 냉장고
    "TOP_BOTTOM_REFRIGERATOR": FurnitureTypeDimension(
        name="TOP_BOTTOM_REFRIGERATOR",
        width=500, depth=600, height=1250
    ),
    "SIDE_BY_SIDE_REFRIGERATOR": FurnitureTypeDimension(
        name="SIDE_BY_SIDE_REFRIGERATOR",
        width=920, depth=930, height=1800
    ),
    "FOUR_DOOR_REFRIGERATOR": FurnitureTypeDimension(
        name="FOUR_DOOR_REFRIGERATOR",
        width=920, depth=930, height=1860
    ),

    # 옷장
    "MOVABLE_WARDROBE": FurnitureTypeDimension(
        name="MOVABLE_WARDROBE",
        width=900, depth=600, height=1900
    ),
    "SYSTEM_HANGER": FurnitureTypeDimension(
        name="SYSTEM_HANGER",
        width=850, depth=400, height=2400
    ),

    # 소파
    "SINGLE_SOFA": FurnitureTypeDimension(
        name="SINGLE_SOFA",
        width=1000, depth=1000, height=900
    ),
    "TWIN_SOFA": FurnitureTypeDimension(
        name="TWIN_SOFA",
        width=2000, depth=1000, height=900
    ),
    "THREE_SEATER_SOFA": FurnitureTypeDimension(
        name="THREE_SEATER_SOFA",
        width=3000, depth=1000, height=900
    ),
    "L_SHAPED_SOFA": FurnitureTypeDimension(
        name="L_SHAPED_SOFA",
        width=3200, depth=1800, height=900
    ),

    # 침대 (높이 가변)
    "SINGLE_BED": FurnitureTypeDimension(
        name="SINGLE_BED",
        width=1000, depth=2000, height=-1
    ),
    "SUPER_SINGLE_BED": FurnitureTypeDimension(
        name="SUPER_SINGLE_BED",
        width=1100, depth=2000, height=-1
    ),
    "DOUBLE_BED": FurnitureTypeDimension(
        name="DOUBLE_BED",
        width=1400, depth=2000, height=-1
    ),
    "QUEEN_SIZE_BED": FurnitureTypeDimension(
        name="QUEEN_SIZE_BED",
        width=1500, depth=2000, height=-1
    ),
    "KING_SIZE_BED": FurnitureTypeDimension(
        name="KING_SIZE_BED",
        width=1600, depth=2000, height=-1
    ),
    "BUNK_BED": FurnitureTypeDimension(
        name="BUNK_BED",
        width=1800, depth=2000, height=-1
    ),

    # 식탁 (모든 치수 가변)
    "DEFAULT_DINING_TABLE": FurnitureTypeDimension(
        name="DEFAULT_DINING_TABLE",
        width=-1, depth=-1, height=-1
    ),

    # TV/모니터 (65인치 기준)
    "DEFAULT_MONITOR_TV": FurnitureTypeDimension(
        name="DEFAULT_MONITOR_TV",
        width=1450, depth=80, height=840
    ),

    # 책상
    "L_SHAPED_DESK": FurnitureTypeDimension(
        name="L_SHAPED_DESK",
        width=1600, depth=1200, height=720
    ),
    "DESK_NO_DRAWER": FurnitureTypeDimension(
        name="DESK_NO_DRAWER",
        width=1400, depth=800, height=720
    ),
    "DESK_SINGLE_PEDESTAL": FurnitureTypeDimension(
        name="DESK_SINGLE_PEDESTAL",
        width=1400, depth=800, height=720
    ),
    "DESK_DOUBLE_PEDESTAL": FurnitureTypeDimension(
        name="DESK_DOUBLE_PEDESTAL",
        width=1600, depth=800, height=720
    ),

    # 의자
    "STANDARD_CHAIR": FurnitureTypeDimension(
        name="STANDARD_CHAIR",
        width=600, depth=600, height=1200
    ),
    "ROUND_STOOL": FurnitureTypeDimension(
        name="ROUND_STOOL",
        width=350, depth=350, height=450
    ),
    "ROLLING_OFFICE_CHAIR": FurnitureTypeDimension(
        name="ROLLING_OFFICE_CHAIR",
        width=700, depth=700, height=1350
    ),

    # 세탁기/건조기
    "DRUM_WASHING_MACHINE": FurnitureTypeDimension(
        name="DRUM_WASHING_MACHINE",
        width=700, depth=850, height=1000
    ),
    "TOP_LOADING_WASHING_MACHINE": FurnitureTypeDimension(
        name="TOP_LOADING_WASHING_MACHINE",
        width=700, depth=850, height=1100
    ),
    "DEFAULT_DRYER": FurnitureTypeDimension(
        name="DEFAULT_DRYER",
        width=750, depth=800, height=1000
    ),

    # 화장대
    "VANITY_TABLE_WITH_ATTACHED_MIRROR": FurnitureTypeDimension(
        name="VANITY_TABLE_WITH_ATTACHED_MIRROR",
        width=1000, depth=450, height=1400
    ),
    "CONSOLE_VANITY_TABLE": FurnitureTypeDimension(
        name="CONSOLE_VANITY_TABLE",
        width=1000, depth=450, height=750
    ),

    # TV 거치대
    "TV_ENTERTAINMENT_CENTER_WITH_STORAGE": FurnitureTypeDimension(
        name="TV_ENTERTAINMENT_CENTER_WITH_STORAGE",
        width=2400, depth=450, height=450
    ),
    "LOW_TV_STAND": FurnitureTypeDimension(
        name="LOW_TV_STAND",
        width=2400, depth=450, height=450
    ),

    # 피아노
    "UPRIGHT_VERTICAL_PIANO": FurnitureTypeDimension(
        name="UPRIGHT_VERTICAL_PIANO",
        width=1500, depth=650, height=1320
    ),
    "GRAND_PIANO": FurnitureTypeDimension(
        name="GRAND_PIANO",
        width=1500, depth=1600, height=1020
    ),
    "DIGITAL_PIANO": FurnitureTypeDimension(
        name="DIGITAL_PIANO",
        width=1400, depth=500, height=900
    ),

    # 기타
    "DEFAULT_MASSAGE_CHAIR": FurnitureTypeDimension(
        name="DEFAULT_MASSAGE_CHAIR",
        width=850, depth=1700, height=1200
    ),
    "DEFAULT_TREADMILL": FurnitureTypeDimension(
        name="DEFAULT_TREADMILL",
        width=800, depth=1800, height=1300
    ),
    "DEFAULT_EXERCISE_BIKE": FurnitureTypeDimension(
        name="DEFAULT_EXERCISE_BIKE",
        width=600, depth=1100, height=1200
    ),
    "DEFAULT_MICROWAVE_OVEN": FurnitureTypeDimension(
        name="DEFAULT_MICROWAVE_OVEN",
        width=520, depth=420, height=350
    ),
    "DEFAULT_DISH_CABINET": FurnitureTypeDimension(
        name="DEFAULT_DISH_CABINET",
        width=800, depth=400, height=1800
    ),
    "DEFAULT_KIMCHI_REFRIGERATOR": FurnitureTypeDimension(
        name="DEFAULT_KIMCHI_REFRIGERATOR",
        width=920, depth=800, height=1800
    ),
    "DEFAULT_POTTED_PLANT": FurnitureTypeDimension(
        name="DEFAULT_POTTED_PLANT",
        width=1, depth=1, height=1
    ),
}


# =============================================================================
# Furniture Labels - 라벨 → 서브타입 매핑 (FurnitureLabel.java 기준)
# =============================================================================
FURNITURE_LABELS: Dict[str, List[str]] = {
    "AIR_CONDITIONER": [
        "WALL_MOUNTED_AIR_CONDITIONER",
        "STANDING_AIR_CONDITIONER"
    ],
    "COFFEE_TABLE": ["DEFAULT_COFFEE_TABLE"],
    "MIRROR": ["DEFAULT_MIRROR"],
    "STORAGE_BOX": ["DEFAULT_STORAGE_BOX"],
    "FAN": ["DEFAULT_FAN"],
    "CABINET": ["DEFAULT_CABINET"],
    "DRAWER": ["DEFAULT_DRAWER"],
    "NIGHTSTAND": ["DEFAULT_NIGHTSTAND"],
    "BOOKSHELF": ["DEFAULT_BOOKSHELF"],
    "DISPLAY_SHELF": ["DEFAULT_DISPLAY_SHELF"],
    "REFRIGERATOR": [
        "TOP_BOTTOM_REFRIGERATOR",
        "SIDE_BY_SIDE_REFRIGERATOR",
        "FOUR_DOOR_REFRIGERATOR"
    ],
    "WARDROBE": [
        "MOVABLE_WARDROBE",
        "SYSTEM_HANGER"
    ],
    "SOFA": [
        "SINGLE_SOFA",
        "TWIN_SOFA",
        "THREE_SEATER_SOFA",
        "L_SHAPED_SOFA"
    ],
    "BED": [
        "SINGLE_BED",
        "SUPER_SINGLE_BED",
        "DOUBLE_BED",
        "QUEEN_SIZE_BED",
        "KING_SIZE_BED",
        "BUNK_BED"
    ],
    "DINING_TABLE": ["DEFAULT_DINING_TABLE"],
    "MONITOR_TV": ["DEFAULT_MONITOR_TV"],
    "DESK": [
        "L_SHAPED_DESK",
        "DESK_NO_DRAWER",
        "DESK_SINGLE_PEDESTAL",
        "DESK_DOUBLE_PEDESTAL"
    ],
    "CHAIR_STOOL": [
        "STANDARD_CHAIR",
        "ROUND_STOOL",
        "ROLLING_OFFICE_CHAIR"
    ],
    "WASHING_MACHINE": [
        "DRUM_WASHING_MACHINE",
        "TOP_LOADING_WASHING_MACHINE"
    ],
    "DRYER": ["DEFAULT_DRYER"],
    "VANITY_TABLE": [
        "VANITY_TABLE_WITH_ATTACHED_MIRROR",
        "CONSOLE_VANITY_TABLE"
    ],
    "TV_STAND": [
        "TV_ENTERTAINMENT_CENTER_WITH_STORAGE",
        "LOW_TV_STAND"
    ],
    "PIANO": [
        "UPRIGHT_VERTICAL_PIANO",
        "GRAND_PIANO",
        "DIGITAL_PIANO"
    ],
    "MASSAGE_CHAIR": ["DEFAULT_MASSAGE_CHAIR"],
    "TREADMILL": ["DEFAULT_TREADMILL"],
    "EXERCISE_BIKE": ["DEFAULT_EXERCISE_BIKE"],
    "MICROWAVE_OVEN": ["DEFAULT_MICROWAVE_OVEN"],
    "DISH_CABINET": ["DEFAULT_DISH_CABINET"],
    "POTTED_PLANT": ["DEFAULT_POTTED_PLANT"],
}


# =============================================================================
# Backend Valid FurnitureType enum (33개)
# 백엔드에 정의된 FurnitureType만 API 응답에서 type으로 반환됩니다.
# 이 목록에 없는 타입은 null로 반환됩니다.
# =============================================================================
BACKEND_VALID_TYPES: set = {
    # 에어컨
    "WALL_MOUNTED_AIR_CONDITIONER",
    "STANDING_AIR_CONDITIONER",
    # 냉장고
    "TOP_BOTTOM_REFRIGERATOR",
    "SIDE_BY_SIDE_REFRIGERATOR",
    "FOUR_DOOR_REFRIGERATOR",
    # 옷장
    "MOVABLE_WARDROBE",
    "SYSTEM_HANGER",
    # 소파
    "SINGLE_SOFA",
    "TWIN_SOFA",
    "THREE_SEATER_SOFA",
    "L_SHAPED_SOFA",
    # 침대
    "SINGLE_BED",
    "SUPER_SINGLE_BED",
    "DOUBLE_BED",
    "QUEEN_SIZE_BED",
    "KING_SIZE_BED",
    "BUNK_BED",
    # 책상
    "L_SHAPED_DESK",
    "DESK_NO_DRAWER",
    "DESK_SINGLE_PEDESTAL",
    "DESK_DOUBLE_PEDESTAL",
    # 의자
    "STANDARD_CHAIR",
    "ROUND_STOOL",
    "ROLLING_OFFICE_CHAIR",
    # 세탁기
    "DRUM_WASHING_MACHINE",
    "TOP_LOADING_WASHING_MACHINE",
    # 화장대
    "VANITY_TABLE_WITH_ATTACHED_MIRROR",
    "CONSOLE_VANITY_TABLE",
    # TV 거치대
    "TV_ENTERTAINMENT_CENTER_WITH_STORAGE",
    "LOW_TV_STAND",
    # 피아노
    "UPRIGHT_VERTICAL_PIANO",
    "GRAND_PIANO",
    "DIGITAL_PIANO",
}


def is_valid_backend_type(type_name: str) -> bool:
    """
    백엔드 FurnitureType enum에 존재하는 타입인지 확인

    Args:
        type_name: 타입 이름 (예: "THREE_SEATER_SOFA")

    Returns:
        백엔드에 유효한 타입이면 True, 아니면 False
    """
    return type_name in BACKEND_VALID_TYPES


def get_valid_type_or_none(type_name: Optional[str]) -> Optional[str]:
    """
    백엔드에 유효한 타입이면 반환, 아니면 None 반환

    Args:
        type_name: 타입 이름

    Returns:
        유효한 타입이면 그대로, 아니면 None
    """
    if type_name is None:
        return None
    return type_name if is_valid_backend_type(type_name) else None


# =============================================================================
# Helper Functions
# =============================================================================

def get_furniture_type(type_name: str) -> Optional[FurnitureTypeDimension]:
    """
    타입 이름으로 FurnitureTypeDimension 조회

    Args:
        type_name: 타입 이름 (예: "SINGLE_BED")

    Returns:
        FurnitureTypeDimension 또는 None
    """
    return FURNITURE_TYPES.get(type_name)


def get_subtypes_for_label(label: str) -> List[str]:
    """
    라벨에 해당하는 서브타입 목록 조회

    Args:
        label: 라벨 이름 (예: "BED")

    Returns:
        서브타입 이름 리스트
    """
    return FURNITURE_LABELS.get(label, [])


def get_first_subtype_for_label(label: str) -> Optional[str]:
    """
    라벨의 첫 번째 (기본) 서브타입 조회

    Args:
        label: 라벨 이름

    Returns:
        첫 번째 서브타입 이름 또는 None
    """
    subtypes = get_subtypes_for_label(label)
    return subtypes[0] if subtypes else None


def get_dimension_for_label(label: str) -> Optional[FurnitureTypeDimension]:
    """
    라벨의 기본 서브타입 치수 조회

    Args:
        label: 라벨 이름

    Returns:
        기본 서브타입의 FurnitureTypeDimension 또는 None
    """
    first_subtype = get_first_subtype_for_label(label)
    if first_subtype:
        return get_furniture_type(first_subtype)
    return None
