"""
Furniture Knowledge Base

YOLO 클래스 매핑 + 한국어 라벨 + 프롬프트 저장용 정적 DB

DeCl_v2 기반으로 업데이트됨:
- min_confidence: 최소 신뢰도 임계값 (일부 카테고리)
- subtypes: 세부 유형 프롬프트
"""

from typing import Optional, List, Dict


# =============================================================================
# Furniture Database
# =============================================================================

FURNITURE_DB = {
    "air conditioner": {
        "synonyms": ["air conditioner", "ac unit", "climate control unit"],
        "base_name": "AIR_CONDITIONER",
        "subtypes": [
            {
                "name": "CEILING_MOUNTED_AIR_CONDITIONER_VENT",
                "prompt": "a ceiling mounted cassette air conditioner vent",
                "exclude_from_output": True
            },
            {
                "name": "WALL_MOUNTED_AIR_CONDITIONER",
                "prompt": "a split air conditioner unit mounted high on a wall"
            },
            {
                "name": "STANDING_AIR_CONDITIONER",
                "prompt": "a tall floor standing tower air conditioner"
            }
        ]
    },
    "coffee table": {
        "synonyms": ["coffee table", "living room table", "center table", "low table"],
        "base_name": "COFFEE_TABLE",
    },
    "microwave": {
        "synonyms": ["microwave", "microwave oven", "countertop microwave"],
        "base_name": "MICROWAVE",
    },
    "oven": {
        "synonyms": ["oven", "kitchen oven", "built-in oven", "wall oven"],
        "base_name": "OVEN",
    },
    "mirror": {
        "synonyms": ["mirror", "wall mirror", "standing mirror", "full length mirror", "vanity mirror"],
        "base_name": "MIRROR",
    },
    "storage box": {
        "synonyms": ["storage box", "storage container", "plastic box", "moving box"],
        "base_name": "STORAGE_BOX",
    },
    "fan": {
        "synonyms": ["fan", "electric fan", "standing fan", "floor fan", "ceiling fan", "desk fan"],
        "base_name": "FAN",
    },
    "cabinet": {
        "synonyms": ["cabinet", "cabinet/shelf", "storage cabinet", "display cabinet"],
        "base_name": "CABINET",
    },
    "kitchen cabinet": {
        "synonyms": ["cupboard", "dish cupboard", "dish cabinet", "kitchen cupboard", "pantry cabinet", "wall cabinet", "overhead storage", "kitchen cabinet", "kitchen storage", "white cabinet", "black cabinet", "gray cabinet", "brown cabinet"],
        "base_name": "KITCHEN_CABINET",
    },
    "drawer": {
        "synonyms": ["drawer", "chest of drawers", "dresser", "drawer unit", "brown drawer", "wooden drawer", "low chest of drawers", "low chest"],
        "base_name": "DRAWER",
    },
    "nightstand": {
        "synonyms": ["nightstand", "bedside table", "bedside cabinet", "night table", "bedside drawer", "wooden nightstand", "brown nightstand", "brown cabinet", "oak nightstand", "oak cabinet", "bedside chest"],
        "base_name": "NIGHTSTAND",
        "min_confidence": 0.5
    },
    "bookshelf": {
        "synonyms": ["bookshelf", "bookcase", "library shelf"],
        "base_name": "BOOKSHELF",
    },
    "display shelf": {
        "synonyms": ["close shelf", "wooden shelf", "display shelf", "open shelf", "minimalist rack", "metal frame shelf", "open display stand", "tiered display stand", "slim metal rack", "thin display rack", "floor shelf", "metal shelf", "minimalist shelf"],
        "base_name": "DISPLAY_SHELF",
        "subtypes": [
            {
                "name": "DISPLAY_SHELF",
                "prompt": "a minimalist open display shelf with thin frames and no back panel"
            }
        ]
    },
    "refrigerator": {
        "synonyms": ["refrigerator", "fridge", "freezer"],
        "base_name": "REFRIGERATOR",
        "subtypes": [
            {
                "name": "TOP_BOTTOM_REFRIGERATOR",
                "prompt": "a small compact refrigerator with top and bottom doors divided by a horizontal line"
            },
            {
                "name": "SIDE_BY_SIDE_REFRIGERATOR",
                "prompt": "a large side-by-side refrigerator with left and right doors divided by a vertical line"
            },
            {
                "name": "FOUR_DOOR_REFRIGERATOR",
                "prompt": "a large family refrigerator with four doors in a grid layout"
            },
            {
                "name": "BUILT_IN_REFRIGERATOR",
                "prompt": "a built-in refrigerator flush with kitchen cabinets",
                "exclude_from_output": True
            }
        ]
    },
    "wardrobe": {
        "synonyms": ["wardrobe", "closet", "armoire", "clothes closet"],
        "base_name": "WARDROBE",
        "subtypes": [
            {
                "name": "BUILT_IN_WARDROBE",
                "prompt": "a built-in wardrobe integrated into the wall without gaps",
                "exclude_from_output": True
            },
            {
                "name": "MOVABLE_WARDROBE",
                "prompt": "a free-standing wooden wardrobe furniture"
            },
            {
                "name": "SYSTEM_HANGER",
                "prompt": "an open walk-in closet system hanger with metal poles"
            }
        ]
    },
    "sofa": {
        "synonyms": ["sofa", "couch", "settee"],
        "base_name": "SOFA",
        "subtypes": [
            {
                "name": "SINGLE_SOFA",
                "prompt": "a single seater armchair sofa"
            },
            {
                "name": "TWIN_SOFA",
                "prompt": "a two seater loveseat sofa"
            },
            {
                "name": "THREE_SEATER_SOFA",
                "prompt": "a three seater long sofa"
            },
            {
                "name": "L_SHAPED_SOFA",
                "prompt": "an L-shaped sectional corner sofa"
            }
        ]
    },
    "bed": {
        "synonyms": ["bed", "bed frame", "bunk bed"],
        "base_name": "BED"
    },
    "dining table": {
        "synonyms": ["dining table", "kitchen table"],
        "base_name": "DINING_TABLE",
    },
    "monitor": {
        "synonyms": ["monitor", "monitor/tv", "computer monitor", "pc monitor", "desktop monitor", "lcd monitor", "computer screen", "display monitor", "television", "tv", "flat screen tv", "wall mounted tv", "large screen tv"],
        "base_name": "MONITOR_TV"
    },
    "desk": {
        "synonyms": ["desk", "office desk", "computer desk", "writing desk"],
        "base_name": "DESK",
        "subtypes": [
            {
                "name": "DESK_NO_DRAWER",
                "prompt": "a simple desk without any drawers or pedestal"
            },
            {
                "name": "DESK_SINGLE_PEDESTAL",
                "prompt": "a desk with drawers on one side only"
            },
            {
                "name": "DESK_DOUBLE_PEDESTAL",
                "prompt": "a desk with drawers on both left and right sides"
            },
            {
                "name": "L_SHAPED_DESK",
                "prompt": "an L-shaped corner desk"
            }
        ]
    },
    "chair": {
        "synonyms": ["chair", "office chair", "dining chair", "armchair", "stool", "round stool", "circular stool"],
        "base_name": "CHAIR_STOOL",
        "subtypes": [
            {
                "name": "STANDARD_CHAIR",
                "prompt": "a standard dining or office chair with a backrest"
            },
            {
                "name": "ROUND_STOOL",
                "prompt": "a round stool with a circular seat and thin slim legs"
            }
        ]
    },
    "washing machine": {
        "synonyms": ["washing machine", "washer", "laundry machine"],
        "base_name": "WASHING_MACHINE",
        "subtypes": [
            {
                "name": "DRUM_WASHING_MACHINE",
                "prompt": "a front loading drum washing machine"
            },
            {
                "name": "TOP_LOADING_WASHING_MACHINE",
                "prompt": "a top loading washing machine"
            }
        ]
    },
    "floor": {
        "synonyms": ["floor", "wood grain floor", "hardwood flooring with patterns", "tiled floor", "solid floor texture"],
        "base_name": "FLOOR",
        "exclude_from_output": True,
    },
    "potted plant": {
        "synonyms": ["potted plant", "plant", "vase", "flower pot", "houseplant", "indoor plant", "vase with flowers"],
        "base_name": "POTTED_PLANT"
    },
    "kimchi refrigerator": {
        "synonyms": ["kimchi refrigerator", "kimchi fridge", "secondary fridge", "small refrigerator"],
        "base_name": "KIMCHI_REFRIGERATOR"
    },
    "vanity table": {
        "synonyms": ["vanity table", "dressing table", "makeup table", "vanity desk", "vanity mirror table"],
        "base_name": "VANITY_TABLE",
        "subtypes": [
            {
                "name": "VANITY_TABLE_WITH_ATTACHED_MIRROR",
                "prompt": "a vanity table with attached mirror"
            },
            {
                "name": "CONSOLE_VANITY_TABLE",
                "prompt": "a simple console vanity table without mirror"
            }
        ]
    },
    "tv stand": {
        "synonyms": ["tv stand", "tv console", "media console", "entertainment center", "tv cabinet", "tv unit"],
        "base_name": "TV_STAND",
        "subtypes": [
            {
                "name": "TV_ENTERTAINMENT_CENTER_WITH_STORAGE",
                "prompt": "a large tv entertainment center with storage"
            },
            {
                "name": "LOW_TV_STAND",
                "prompt": "a low tv stand console table"
            }
        ]
    },
    "piano": {
        "synonyms": ["piano", "upright piano", "grand piano", "digital piano", "keyboard piano"],
        "base_name": "PIANO",
        "subtypes": [
            {
                "name": "UPRIGHT_VERTICAL_PIANO",
                "prompt": "an upright vertical piano against a wall"
            },
            {
                "name": "GRAND_PIANO",
                "prompt": "a grand piano with horizontal strings"
            },
            {
                "name": "DIGITAL_PIANO",
                "prompt": "a digital electric piano on a stand"
            }
        ]
    },
    "massage chair": {
        "synonyms": ["massage chair", "recliner massage chair", "electric massage chair"],
        "base_name": "MASSAGE_CHAIR"
    },
    "treadmill": {
        "synonyms": ["treadmill", "running machine", "exercise treadmill"],
        "base_name": "TREADMILL"
    },
    "exercise bike": {
        "synonyms": ["exercise bike", "stationary bike", "spin bike", "indoor cycling bike"],
        "base_name": "EXERCISE_BIKE"
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_db_key_from_label(label: str) -> Optional[str]:
    """
    YOLO 라벨에서 DB 키를 찾습니다.

    Args:
        label: YOLO 탐지 라벨 (예: "Bed", "Sofa")

    Returns:
        DB 키 또는 None
    """
    label_lower = label.lower()
    for key, info in FURNITURE_DB.items():
        if label_lower in [s.lower() for s in info.get("synonyms", [])]:
            return key
    return None


def get_base_name(db_key: str) -> str:
    """
    DB 키에서 영어 라벨을 가져옵니다.

    Args:
        db_key: DB 키 (예: "bed", "sofa")

    Returns:
        영어 라벨 (예: "BED", "SOFA")
    """
    if db_key in FURNITURE_DB:
        return FURNITURE_DB[db_key].get("base_name", db_key)
    return db_key


def get_subtypes(db_key: str) -> List[Dict]:
    """
    특정 가구의 서브타입 목록을 가져옵니다.

    Args:
        db_key: DB 키

    Returns:
        서브타입 리스트 [{"name": str, "prompt": str}, ...]
    """
    if db_key in FURNITURE_DB:
        return FURNITURE_DB[db_key].get("subtypes", [])
    return []


def get_min_confidence(db_key: str) -> Optional[float]:
    """
    DB 키에서 최소 신뢰도 임계값을 가져옵니다.

    Args:
        db_key: DB 키

    Returns:
        min_confidence 값 또는 None (설정되지 않은 경우)
    """
    if db_key in FURNITURE_DB:
        return FURNITURE_DB[db_key].get("min_confidence")
    return None


def get_all_synonyms() -> List[str]:
    """
    모든 카테고리의 동의어 목록을 반환합니다.

    Returns:
        모든 동의어 리스트
    """
    synonyms = []
    for info in FURNITURE_DB.values():
        synonyms.extend(info.get("synonyms", []))
    return synonyms


def get_excluded_base_names() -> set:
    """
    exclude_from_output: True인 항목의 base_name 목록을 반환합니다.

    Returns:
        제외할 base_name 세트 (예: {"KITCHEN_ISLAND", "FLOOR"})
    """
    excluded = set()
    for info in FURNITURE_DB.values():
        if info.get("exclude_from_output", False):
            excluded.add(info.get("base_name", ""))
    return excluded


def get_excluded_subtype_names() -> set:
    """
    subtype 중 exclude_from_output: True인 항목의 name 목록을 반환합니다.

    Returns:
        제외할 subtype name 세트 (예: {"CEILING_MOUNTED_AIR_CONDITIONER_VENT"})
    """
    excluded = set()
    for info in FURNITURE_DB.values():
        for subtype in info.get("subtypes", []):
            if subtype.get("exclude_from_output", False):
                excluded.add(subtype.get("name", ""))
    return excluded


# =============================================================================
# Deprecated Functions (하위 호환성 유지)
# =============================================================================

def get_content_labels(db_key: str) -> list:
    """
    [DEPRECATED] 내용물 탐지용 라벨 목록을 가져옵니다.

    V2 파이프라인에서는 content_labels가 제거되었습니다.
    이 함수는 하위 호환성을 위해 유지되며 빈 리스트를 반환합니다.
    """
    return []


def is_movable(db_key: str) -> bool:
    """
    [DEPRECATED] 모든 탐지 객체는 이동 대상으로 간주합니다.

    V2 파이프라인에서 is_movable은 제거되었습니다.
    이 함수는 하위 호환성을 위해 유지되며 항상 True를 반환합니다.
    """
    return True


def get_dimensions(db_key: str) -> Optional[dict]:
    """
    [DEPRECATED] dimensions는 V2 파이프라인에서 제거되었습니다.

    절대 부피는 백엔드에서 계산합니다.
    이 함수는 하위 호환성을 위해 유지되며 항상 None을 반환합니다.
    """
    return None


def get_dimensions_for_subtype(db_key: str, subtype_name: str) -> Optional[dict]:  # noqa: ARG001
    """
    [DEPRECATED] dimensions는 V2 파이프라인에서 제거되었습니다.

    절대 부피는 백엔드에서 계산합니다.
    이 함수는 하위 호환성을 위해 유지되며 항상 None을 반환합니다.
    """
    _ = db_key, subtype_name  # 하위 호환성 시그니처 유지
    return None


def estimate_size_variant(db_key: str, aspect_ratio: dict) -> str:  # noqa: ARG001
    """
    [DEPRECATED] variants는 V2 파이프라인에서 제거되었습니다.

    이 함수는 하위 호환성을 위해 유지되며 항상 "medium"을 반환합니다.
    """
    _ = db_key, aspect_ratio  # 하위 호환성 시그니처 유지
    return "medium"
