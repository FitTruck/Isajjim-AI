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
        "base_name": "에어컨",
        "subtypes": [
            {
                "name": "천장형 에어컨 (시스템)",
                "prompt": "a ceiling mounted cassette air conditioner vent"
            },
            {
                "name": "벽걸이 에어컨",
                "prompt": "a split air conditioner unit mounted high on a wall"
            },
            {
                "name": "스탠드 에어컨",
                "prompt": "a tall floor standing tower air conditioner"
            }
        ]
    },
    "coffee table": {
        "synonyms": ["coffee table", "living room table", "center table", "low table"],
        "base_name": "커피테이블",
    },
    "microwave": {
        "synonyms": ["microwave", "microwave oven", "countertop microwave"],
        "base_name": "전자레인지",
    },
    "oven": {
        "synonyms": ["oven", "kitchen oven", "built-in oven", "wall oven"],
        "base_name": "오븐",
    },
    "mirror": {
        "synonyms": ["mirror", "wall mirror", "standing mirror", "full length mirror", "vanity mirror"],
        "base_name": "거울",
    },
    "storage box": {
        "synonyms": ["storage box", "storage container", "plastic box", "moving box"],
        "base_name": "박스/수납함",
    },
    "bench": {
        "synonyms": ["bench", "wooden bench", "sitting bench", "entryway bench"],
        "base_name": "벤치",
    },
    "toilet": {
        "synonyms": ["toilet", "toilet bowl", "bathroom toilet", "commode"],
        "base_name": "변기",
    },
    "sink": {
        "synonyms": ["sink", "kitchen sink", "bathroom sink", "wash basin"],
        "base_name": "싱크대",
    },
    "bathtub": {
        "synonyms": ["bathtub", "bath tub", "tub", "soaking tub"],
        "base_name": "욕조",
    },
    "bicycle": {
        "synonyms": ["bicycle", "bike", "cycle", "mountain bike", "road bike"],
        "base_name": "자전거",
    },
    "ladder": {
        "synonyms": ["ladder", "step ladder", "folding ladder", "extension ladder"],
        "base_name": "사다리",
    },
    "fan": {
        "synonyms": ["fan", "electric fan", "standing fan", "floor fan", "ceiling fan", "desk fan"],
        "base_name": "선풍기",
    },
    "box": {
        "synonyms": ["box", "cardboard box", "packing box"],
        "base_name": "박스",
    },
    "kitchen island": {
        "synonyms": ["kitchen island", "island counter", "island table", "center island"],
        "base_name": "아일랜드",
        "exclude_from_output": True,
    },
    "cabinet": {
        "synonyms": ["cabinet", "cabinet/shelf", "storage cabinet", "display cabinet"],
        "base_name": "캐비닛",
    },
    "kitchen cabinet": {
        "synonyms": ["cupboard", "dish cupboard", "dish cabinet", "kitchen cupboard", "pantry cabinet", "wall cabinet", "overhead storage", "kitchen cabinet", "kitchen storage", "white cabinet", "black cabinet", "gray cabinet", "brown cabinet"],
        "base_name": "찬장",
    },
    "drawer": {
        "synonyms": ["drawer", "chest of drawers", "dresser", "drawer unit", "brown drawer", "wooden drawer", "low chest of drawers", "low chest"],
        "base_name": "서랍장",
    },
    "nightstand": {
        "synonyms": ["nightstand", "bedside table", "bedside cabinet", "night table", "bedside drawer", "wooden nightstand", "brown nightstand", "brown cabinet", "oak nightstand", "oak cabinet", "bedside chest"],
        "base_name": "협탁",
        "min_confidence": 0.5
    },
    "bookshelf": {
        "synonyms": ["bookshelf", "bookcase", "library shelf"],
        "base_name": "책장",
    },
    "display shelf": {
        "synonyms": ["close shelf", "wooden shelf", "display shelf", "open shelf", "minimalist rack", "metal frame shelf", "open display stand", "tiered display stand", "slim metal rack", "thin display rack", "floor shelf", "metal shelf", "minimalist shelf"],
        "base_name": "전시대/선반",
        "subtypes": [
            {
                "name": "전시대",
                "prompt": "a minimalist open display shelf with thin frames and no back panel"
            }
        ]
    },
    "refrigerator": {
        "synonyms": ["refrigerator", "fridge", "freezer"],
        "base_name": "냉장고",
        "subtypes": [
            {
                "name": "일반 냉장고",
                "prompt": "a standard free-standing refrigerator"
            },
            {
                "name": "빌트인 냉장고",
                "prompt": "a built-in refrigerator flush with kitchen cabinets"
            }
        ]
    },
    "wardrobe": {
        "synonyms": ["wardrobe", "closet", "armoire", "clothes closet"],
        "base_name": "장롱/수납장",
        "subtypes": [
            {
                "name": "붙박이장 (매립형)",
                "prompt": "a built-in wardrobe integrated into the wall without gaps"
            },
            {
                "name": "일반 옷장 (이동식)",
                "prompt": "a free-standing wooden wardrobe furniture"
            },
            {
                "name": "시스템 행거",
                "prompt": "an open walk-in closet system hanger with metal poles"
            }
        ]
    },
    "sofa": {
        "synonyms": ["sofa", "couch", "settee"],
        "base_name": "소파",
        "subtypes": [
            {
                "name": "1인용 소파",
                "prompt": "a single seater armchair sofa"
            },
            {
                "name": "2인용 소파",
                "prompt": "a two seater loveseat sofa"
            },
            {
                "name": "3인용 소파",
                "prompt": "a three seater long sofa"
            },
            {
                "name": "L자형 소파",
                "prompt": "an L-shaped sectional corner sofa"
            }
        ]
    },
    "bed": {
        "synonyms": ["bed", "bed frame", "bunk bed"],
        "base_name": "침대",
        "subtypes": [
            {
                "name": "싱글 침대",
                "prompt": "a single bed with a narrow mattress"
            },
            {
                "name": "슈퍼싱글 침대",
                "prompt": "a super single bed slightly wider than single"
            },
            {
                "name": "더블 침대",
                "prompt": "a double bed with full size mattress"
            },
            {
                "name": "퀸 사이즈 침대",
                "prompt": "a queen size bed with wide mattress"
            },
            {
                "name": "킹 사이즈 침대",
                "prompt": "a king size bed with extra wide mattress"
            },
            {
                "name": "2층 침대",
                "prompt": "a bunk bed with two levels stacked"
            }
        ]
    },
    "dining table": {
        "synonyms": ["dining table", "kitchen table"],
        "base_name": "식탁",
        "subtypes": [
            {
                "name": "2인용 식탁",
                "prompt": "a small dining table for two people"
            },
            {
                "name": "4인용 식탁",
                "prompt": "a rectangular dining table for four people"
            },
            {
                "name": "6인용 식탁",
                "prompt": "a large dining table for six people"
            }
        ]
    },
    "monitor": {
        "synonyms": ["monitor", "monitor/tv", "computer monitor", "pc monitor", "desktop monitor", "lcd monitor", "computer screen", "display monitor", "television", "tv", "flat screen tv", "wall mounted tv", "large screen tv"],
        "base_name": "모니터/TV"
    },
    "desk": {
        "synonyms": ["desk", "office desk", "computer desk", "writing desk"],
        "base_name": "책상",
        "subtypes": [
            {
                "name": "일반 책상",
                "prompt": "a standard writing desk"
            },
            {
                "name": "L자형 책상",
                "prompt": "an L-shaped corner desk"
            },
            {
                "name": "컴퓨터 책상",
                "prompt": "a computer desk with keyboard tray"
            }
        ]
    },
    "chair": {
        "synonyms": ["chair", "office chair", "dining chair", "armchair", "stool", "round stool", "circular stool"],
        "base_name": "의자/스툴",
        "subtypes": [
            {
                "name": "일반 의자",
                "prompt": "a standard dining or office chair with a backrest"
            },
            {
                "name": "원형 의자",
                "prompt": "a round stool with a circular seat and thin slim legs"
            }
        ]
    },
    "washing machine": {
        "synonyms": ["washing machine", "washer", "laundry machine"],
        "base_name": "세탁기",
        "subtypes": [
            {
                "name": "드럼 세탁기",
                "prompt": "a front loading drum washing machine"
            },
            {
                "name": "통돌이 세탁기",
                "prompt": "a top loading washing machine"
            }
        ]
    },
    "dryer": {
        "synonyms": ["dryer", "clothes dryer", "tumble dryer"],
        "base_name": "건조기"
    },
    "floor": {
        "synonyms": ["floor", "wood grain floor", "hardwood flooring with patterns", "tiled floor", "solid floor texture"],
        "base_name": "바닥",
        "exclude_from_output": True,
    },
    "potted plant": {
        "synonyms": ["potted plant", "plant", "vase", "flower pot", "houseplant", "indoor plant", "vase with flowers"],
        "base_name": "화분/식물"
    },
    "kimchi refrigerator": {
        "synonyms": ["kimchi refrigerator", "kimchi fridge", "secondary fridge", "small refrigerator"],
        "base_name": "김치냉장고"
    },
    "vanity table": {
        "synonyms": ["vanity table", "dressing table", "makeup table", "vanity desk", "vanity mirror table"],
        "base_name": "화장대",
        "subtypes": [
            {
                "name": "거울 화장대",
                "prompt": "a vanity table with attached mirror"
            },
            {
                "name": "콘솔 화장대",
                "prompt": "a simple console vanity table without mirror"
            }
        ]
    },
    "tv stand": {
        "synonyms": ["tv stand", "tv console", "media console", "entertainment center", "tv cabinet", "tv unit"],
        "base_name": "TV 거치대",
        "subtypes": [
            {
                "name": "벽걸이 TV 거치대",
                "prompt": "a wall mounted tv bracket"
            },
            {
                "name": "TV 장식장",
                "prompt": "a large tv entertainment center with storage"
            },
            {
                "name": "낮은 TV 받침대",
                "prompt": "a low tv stand console table"
            }
        ]
    },
    "piano": {
        "synonyms": ["piano", "upright piano", "grand piano", "digital piano", "keyboard piano"],
        "base_name": "피아노",
        "subtypes": [
            {
                "name": "업라이트 피아노",
                "prompt": "an upright vertical piano against a wall"
            },
            {
                "name": "그랜드 피아노",
                "prompt": "a grand piano with horizontal strings"
            },
            {
                "name": "디지털 피아노",
                "prompt": "a digital electric piano on a stand"
            }
        ]
    },
    "massage chair": {
        "synonyms": ["massage chair", "recliner massage chair", "electric massage chair"],
        "base_name": "안마의자"
    },
    "treadmill": {
        "synonyms": ["treadmill", "running machine", "exercise treadmill"],
        "base_name": "러닝머신"
    },
    "exercise bike": {
        "synonyms": ["exercise bike", "stationary bike", "spin bike", "indoor cycling bike"],
        "base_name": "실내자전거"
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
    DB 키에서 한국어 라벨을 가져옵니다.

    Args:
        db_key: DB 키 (예: "bed", "sofa")

    Returns:
        한국어 라벨 (예: "침대", "소파")
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
