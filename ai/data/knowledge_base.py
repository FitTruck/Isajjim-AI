"""
Furniture Knowledge Base

YOLO 클래스 매핑 + 한국어 라벨 + 프롬프트 저장용 정적 DB

V2 파이프라인에서 단순화:
- is_movable 제거 (모든 탐지 객체는 이동 대상)
- dimensions 제거 (절대 부피는 백엔드에서 계산)
- check_strategy 제거
"""

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
    "kitchen cabinet": {
        "synonyms": ["cupboard", "dish cupboard", "dish cabinet", "kitchen cupboard", "pantry cabinet", "wall cabinet", "overhead storage", "kitchen cabinet", "kitchen storage", "white cabinet", "black cabinet", "gray cabinet", "brown cabinet"],
        "base_name": "찬장/수납장",
        "content_labels": ["dish", "bowl", "plate", "cup", "wine glass", "coffee mug", "dish", "pot", "bottle"]
    },
    "bookshelf": {
        "synonyms": ["bookshelf", "bookcase", "library shelf"],
        "base_name": "책장",
        "content_labels": ["book", "magazine collection", "file binder"]
    },
    "display shelf": {
        "synonyms": ["display shelf", "open shelf", "minimalist rack", "metal frame shelf", "open display stand", "tiered display stand", "slim metal rack", "thin display rack", "floor shelf", "metal shelf", "minimalist shelf"],
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
    "tv": {
        "synonyms": ["television", "tv", "flat screen tv"],
        "base_name": "TV",
        "subtypes": [
            {
                "name": "32인치 TV",
                "prompt": "a small 32 inch flat screen television"
            },
            {
                "name": "43인치 TV",
                "prompt": "a medium 43 inch flat screen television"
            },
            {
                "name": "55인치 TV",
                "prompt": "a large 55 inch flat screen television"
            },
            {
                "name": "65인치 TV",
                "prompt": "an extra large 65 inch flat screen television"
            },
            {
                "name": "75인치 TV",
                "prompt": "a huge 75 inch flat screen television"
            }
        ]
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
    "box": {
        "synonyms": ["box", "cardboard box", "moving box", "carton"],
        "base_name": "박스"
    },
    "drawer": {
        "synonyms": ["drawer", "chest of drawers", "dresser", "drawer unit"],
        "base_name": "서랍장"
    },
    "mirror": {
        "synonyms": ["mirror", "full length mirror", "wall mirror", "standing mirror"],
        "base_name": "거울"
    },
    "microwave": {
        "synonyms": ["microwave", "microwave oven"],
        "base_name": "전자레인지"
    },
    "picture frame": {
        "synonyms": ["picture frame", "painting on the wall", "framed art", "walldecor", "artwork"],
        "base_name": "그림/액자"
    },
    "rug": {
        "synonyms": ["rug", "carpet", "floor mat", "fabric rug", "textile floor covering"],
        "base_name": "카펫/러그"
    },
    "cushion": {
        "synonyms": ["sofa cushion", "decorative cushion", "scatter cushion", "backrest cushion", "sofa pillow"],
        "base_name": "쿠션",
        "subtypes": [
            {
                "name": "소파 쿠션",
                "prompt": "a square-shaped decorative cushion or pillow on a sofa or armchair"
            }
        ]
    },
    "pillow": {
        "synonyms": ["sleeping pillow", "bed pillow", "head pillow"],
        "base_name": "베개",
        "subtypes": [
            {
                "name": "취침용 베개",
                "prompt": "a rectangular sleeping pillow on a bed with a pillowcase"
            }
        ]
    },
    "floor": {
        "synonyms": ["wood grain floor", "hardwood flooring with patterns", "tiled floor", "solid floor texture"],
        "base_name": "바닥"
    },
    "candlestick": {
        "synonyms": ["candlestick", "candle holder", "candelabra"],
        "base_name": "촛대"
    },
    "potted plant": {
        "synonyms": ["potted plant", "flower pot", "houseplant", "indoor plant", "vase with flowers"],
        "base_name": "화분/식물"
    },
    "curtain": {
        "synonyms": ["curtain", "drapes", "window blind", "window cover", "sheer fabric"],
        "base_name": "커튼",
        "subtypes": [
            {
                "name": "일반 커튼",
                "prompt": "a standard fabric window curtain"
            },
            {
                "name": "얇은 커튼",
                "prompt": "a thin translucent sheer white curtain with light shining through"
            },
            {
                "name": "무늬 있는 커튼",
                "prompt": "a decorative window curtain with floral, stripes, or geometric patterns, colorful printed fabric drapes"
            }
        ]
    },
    "seasoning container": {
        "synonyms": ["seasoning container", "spice jar", "spice container", "condiment container", "seasoning jar", "spice rack container", "kitchen container", "food storage container", "airtight container", "clear container", "plastic container", "storage jar", "pantry container", "cereal container", "grain container", "dry food container"],
        "base_name": "조미료통/반찬통"
    },
    "side dish container": {
        "synonyms": ["side dish container", "banchan container", "food storage box", "rectangular food container", "meal prep container", "leftover container", "refrigerator container", "stackable container", "glass container", "plastic food box"],
        "base_name": "반찬통"
    },
    "pot": {
        "synonyms": ["pot", "cooking pot", "stock pot", "sauce pot", "stew pot", "soup pot", "dutch oven", "casserole pot", "ceramic pot", "enamel pot", "white pot", "kitchen pot", "two handle pot", "double handle pot", "lidded pot"],
        "base_name": "냄비",
        "subtypes": [
            {
                "name": "양수 냄비",
                "prompt": "a cooking pot with two side handles and a lid"
            },
            {
                "name": "편수 냄비",
                "prompt": "a sauce pot with a single long handle"
            },
            {
                "name": "압력솥",
                "prompt": "a pressure cooker pot with locking lid"
            }
        ]
    },
    "frying pan": {
        "synonyms": ["frying pan", "skillet", "saute pan", "wok", "griddle pan", "non-stick pan", "cast iron pan", "cooking pan"],
        "base_name": "프라이팬"
    },
    "tissue box": {
        "synonyms": ["tissue box", "tissue paper box", "kleenex box", "facial tissue box", "paper tissue holder", "tissue dispenser", "rectangular tissue box", "tissue container"],
        "base_name": "휴지곽"
    },
    "toilet paper": {
        "synonyms": ["toilet paper", "toilet roll", "bathroom tissue", "toilet tissue roll", "paper roll", "rolled paper"],
        "base_name": "두루마리 휴지"
    },
    "paper towel": {
        "synonyms": ["paper towel", "kitchen towel", "kitchen paper", "paper towel roll", "kitchen roll", "absorbent paper"],
        "base_name": "키친타올"
    },
    "plate": {
        "synonyms": ["plate", "dinner plate", "serving plate", "dish plate", "ceramic plate", "flat plate", "round plate", "white plate", "porcelain plate", "food plate", "meal plate"],
        "base_name": "접시"
    },
    "bowl": {
        "synonyms": ["bowl", "soup bowl", "rice bowl", "cereal bowl", "salad bowl", "mixing bowl", "serving bowl", "ceramic bowl", "deep bowl", "round bowl"],
        "base_name": "그릇/볼"
    },
    "cup": {
        "synonyms": ["cup", "mug", "coffee cup", "tea cup", "drinking cup", "ceramic cup", "glass cup", "tumbler"],
        "base_name": "컵/머그"
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_db_key_from_label(label: str) -> str | None:
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


def get_subtypes(db_key: str) -> list:
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


def get_content_labels(db_key: str) -> list:
    """
    내용물 탐지용 라벨 목록을 가져옵니다.

    Args:
        db_key: DB 키

    Returns:
        내용물 라벨 리스트
    """
    if db_key in FURNITURE_DB:
        return FURNITURE_DB[db_key].get("content_labels", [])
    return []


# =============================================================================
# Deprecated Functions (하위 호환성 유지)
# =============================================================================

def is_movable(db_key: str) -> bool:
    """
    [DEPRECATED] 모든 탐지 객체는 이동 대상으로 간주합니다.

    V2 파이프라인에서 is_movable은 제거되었습니다.
    이 함수는 하위 호환성을 위해 유지되며 항상 True를 반환합니다.
    """
    return True


def get_dimensions(db_key: str) -> dict | None:
    """
    [DEPRECATED] dimensions는 V2 파이프라인에서 제거되었습니다.

    절대 부피는 백엔드에서 계산합니다.
    이 함수는 하위 호환성을 위해 유지되며 항상 None을 반환합니다.
    """
    return None


def get_dimensions_for_subtype(db_key: str, subtype_name: str) -> dict | None:  # noqa: ARG001
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
