"""
가구 및 내부 물품 Knowledge Base

Objects365 데이터셋 기반 클래스 매핑
- synonyms: YOLOE 모델이 탐지하는 Objects365 클래스명과 매핑 (입력용)
- base_name: 사용자에게 보여줄 한글 이름 (출력용)
- is_movable: 이사 시 이동 가능 여부
- dimensions: 표준 규격 (mm 단위) - 절대 부피 계산용

CLIP 제거 - prompt 필드 삭제
SAHI 제거 - 단순화된 탐지
"""

FURNITURE_DB = {
    # ===========================================================================
    # 침실 가구
    # ===========================================================================
    "bed": {
        "synonyms": ["bed", "bed frame", "bunk bed"],
        "base_name": "침대",
        "is_movable": True,
        "dimensions": {"width": 1500, "depth": 2000, "height": 450}  # 퀸 사이즈 기준
    },
    "wardrobe": {
        "synonyms": ["wardrobe", "closet", "armoire", "clothes closet"],
        "base_name": "옷장",
        "is_movable": True,
        "dimensions": {"width": 1200, "depth": 600, "height": 2000}
    },
    "drawer": {
        "synonyms": ["drawer", "chest of drawers", "dresser", "drawer unit"],
        "base_name": "서랍장",
        "is_movable": True,
        "dimensions": {"width": 800, "depth": 450, "height": 1000}
    },
    "nightstand": {
        "synonyms": ["nightstand", "bedside table", "night table"],
        "base_name": "협탁",
        "is_movable": True,
        "dimensions": {"width": 450, "depth": 400, "height": 550}
    },
    "mirror": {
        "synonyms": ["mirror", "full length mirror", "wall mirror", "standing mirror"],
        "base_name": "거울",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 50, "height": 1500}
    },

    # ===========================================================================
    # 거실 가구
    # ===========================================================================
    "sofa": {
        "synonyms": ["sofa", "couch", "settee"],
        "base_name": "소파",
        "is_movable": True,
        "dimensions": {"width": 2100, "depth": 900, "height": 900}  # 3인용 기준
    },
    "coffee table": {
        "synonyms": ["coffee table", "center table", "tea table"],
        "base_name": "커피테이블",
        "is_movable": True,
        "dimensions": {"width": 1200, "depth": 600, "height": 450}
    },
    "tv": {
        "synonyms": ["television", "tv", "flat screen tv", "monitor/tv"],
        "base_name": "TV",
        "is_movable": True,
        "dimensions": {"width": 1230, "depth": 70, "height": 710}  # 55인치 기준
    },
    "bookshelf": {
        "synonyms": ["bookshelf", "bookcase", "library shelf"],
        "base_name": "책장",
        "is_movable": True,
        "dimensions": {"width": 800, "depth": 300, "height": 1800}
    },

    # ===========================================================================
    # 식당/주방 가구
    # ===========================================================================
    "dining table": {
        "synonyms": ["dining table", "kitchen table"],
        "base_name": "식탁",
        "is_movable": True,
        "dimensions": {"width": 1200, "depth": 800, "height": 750}  # 4인용 기준
    },
    "chair": {
        "synonyms": ["chair", "office chair", "dining chair", "armchair"],
        "base_name": "의자",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 500, "height": 900}
    },
    "stool": {
        "synonyms": ["stool", "bar stool"],
        "base_name": "스툴",
        "is_movable": True,
        "dimensions": {"width": 400, "depth": 400, "height": 650}
    },
    "bench": {
        "synonyms": ["bench"],
        "base_name": "벤치",
        "is_movable": True,
        "dimensions": {"width": 1200, "depth": 400, "height": 450}
    },
    "kitchen cabinet": {
        "synonyms": ["kitchen cupboard", "pantry cabinet", "wall cabinet", "overhead storage", "kitchen cabinet", "cabinet/shelf"],
        "base_name": "찬장/수납장",
        "is_movable": False,
        "dimensions": {"width": 600, "depth": 350, "height": 800}
    },

    # ===========================================================================
    # 가전제품
    # ===========================================================================
    "refrigerator": {
        "synonyms": ["refrigerator", "fridge", "freezer"],
        "base_name": "냉장고",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 700, "height": 1700}
    },
    "washing machine": {
        "synonyms": ["washing machine", "washer", "laundry machine"],
        "base_name": "세탁기",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 600, "height": 850}
    },
    "dryer": {
        "synonyms": ["dryer", "clothes dryer", "tumble dryer", "drying machine"],
        "base_name": "건조기",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 600, "height": 850}
    },
    "air conditioner": {
        "synonyms": ["air conditioner", "ac unit", "climate control unit"],
        "base_name": "에어컨",
        "is_movable": True,  # 스탠드형 기준 (벽걸이/천장형은 False)
        "dimensions": {"width": 400, "depth": 400, "height": 1800}  # 스탠드형 기준
    },
    "microwave": {
        "synonyms": ["microwave", "microwave oven"],
        "base_name": "전자레인지",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 400, "height": 300}
    },
    "oven": {
        "synonyms": ["oven", "gas oven", "electric oven"],
        "base_name": "오븐",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 600, "height": 600}
    },

    # ===========================================================================
    # 사무/학습 가구
    # ===========================================================================
    "desk": {
        "synonyms": ["desk", "office desk", "computer desk", "writing desk"],
        "base_name": "책상",
        "is_movable": True,
        "dimensions": {"width": 1200, "depth": 600, "height": 750}
    },
    "computer": {
        "synonyms": ["computer", "desktop computer", "pc"],
        "base_name": "컴퓨터",
        "is_movable": True,
        "dimensions": {"width": 200, "depth": 450, "height": 400}
    },
    "laptop": {
        "synonyms": ["laptop", "notebook computer"],
        "base_name": "노트북",
        "is_movable": True,
        "dimensions": {"width": 350, "depth": 250, "height": 25}
    },
    "printer": {
        "synonyms": ["printer"],
        "base_name": "프린터",
        "is_movable": True,
        "dimensions": {"width": 450, "depth": 350, "height": 200}
    },

    # ===========================================================================
    # 욕실 (대부분 고정식)
    # ===========================================================================
    "toilet": {
        "synonyms": ["toilet"],
        "base_name": "변기",
        "is_movable": False,
        "dimensions": {"width": 400, "depth": 700, "height": 400}
    },
    "sink": {
        "synonyms": ["sink", "bathroom sink", "washbasin"],
        "base_name": "싱크대",
        "is_movable": False,
        "dimensions": {"width": 600, "depth": 500, "height": 850}
    },
    "bathtub": {
        "synonyms": ["bathtub", "bath"],
        "base_name": "욕조",
        "is_movable": False,
        "dimensions": {"width": 700, "depth": 1500, "height": 600}
    },

    # ===========================================================================
    # 이동/수납
    # ===========================================================================
    "box": {
        "synonyms": ["box", "cardboard box", "moving box", "carton", "storage box"],
        "base_name": "박스",
        "is_movable": True,
        "dimensions": {"width": 400, "depth": 300, "height": 300}  # 중간 사이즈 기준
    },
    "luggage": {
        "synonyms": ["luggage", "suitcase"],
        "base_name": "캐리어",
        "is_movable": True,
        "dimensions": {"width": 450, "depth": 280, "height": 700}
    },
    "backpack": {
        "synonyms": ["backpack", "bag"],
        "base_name": "배낭",
        "is_movable": True,
        "dimensions": {"width": 350, "depth": 200, "height": 500}
    },
    "basket": {
        "synonyms": ["basket"],
        "base_name": "바구니",
        "is_movable": True,
        "dimensions": {"width": 400, "depth": 300, "height": 250}
    },

    # ===========================================================================
    # 기타 가전/소품
    # ===========================================================================
    "fan": {
        "synonyms": ["fan", "electric fan", "standing fan"],
        "base_name": "선풍기",
        "is_movable": True,
        "dimensions": {"width": 400, "depth": 400, "height": 1200}
    },
    "heater": {
        "synonyms": ["heater", "space heater"],
        "base_name": "히터",
        "is_movable": True,
        "dimensions": {"width": 300, "depth": 200, "height": 600}
    },
    "lamp": {
        "synonyms": ["lamp", "table lamp", "floor lamp", "desk lamp"],
        "base_name": "램프",
        "is_movable": True,
        "dimensions": {"width": 300, "depth": 300, "height": 500}
    },
    "clock": {
        "synonyms": ["clock", "wall clock"],
        "base_name": "시계",
        "is_movable": True,
        "dimensions": {"width": 300, "depth": 50, "height": 300}
    },
    "vase": {
        "synonyms": ["vase", "flower vase"],
        "base_name": "꽃병",
        "is_movable": True,
        "dimensions": {"width": 150, "depth": 150, "height": 300}
    },
    "plant": {
        "synonyms": ["plant", "potted plant", "houseplant"],
        "base_name": "화분",
        "is_movable": True,
        "dimensions": {"width": 300, "depth": 300, "height": 500}
    },
    "picture frame": {
        "synonyms": ["picture frame", "photo frame", "painting"],
        "base_name": "액자",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 30, "height": 800}
    },
    "curtain": {
        "synonyms": ["curtain", "drape"],
        "base_name": "커튼",
        "is_movable": True,
        "dimensions": {"width": 2000, "depth": 50, "height": 2500}
    },
    "rug": {
        "synonyms": ["rug", "carpet", "mat"],
        "base_name": "러그",
        "is_movable": True,
        "dimensions": {"width": 2000, "depth": 3000, "height": 20}
    },
    "pillow": {
        "synonyms": ["pillow", "cushion"],
        "base_name": "베개",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 400, "height": 150}
    },
    "blanket": {
        "synonyms": ["blanket", "comforter", "quilt"],
        "base_name": "담요",
        "is_movable": True,
        "dimensions": {"width": 2000, "depth": 2000, "height": 100}
    },
    "bicycle": {
        "synonyms": ["bicycle", "bike"],
        "base_name": "자전거",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 1800, "height": 1100}
    },
    "ladder": {
        "synonyms": ["ladder", "step ladder"],
        "base_name": "사다리",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 100, "height": 2000}
    },
    "trash can": {
        "synonyms": ["trash can", "garbage bin", "waste bin"],
        "base_name": "쓰레기통",
        "is_movable": True,
        "dimensions": {"width": 300, "depth": 300, "height": 500}
    },
}


def get_db_key_from_label(label: str) -> str:
    """
    YOLO 탐지 라벨에서 DB 키를 찾습니다.

    Args:
        label: YOLO가 탐지한 라벨 (예: "Bed", "Sofa")

    Returns:
        DB 키 (예: "bed", "sofa") 또는 None
    """
    label_lower = label.lower()

    for db_key, info in FURNITURE_DB.items():
        for syn in info.get('synonyms', []):
            if syn.lower() == label_lower:
                return db_key

    return None


def get_furniture_info(db_key: str) -> dict:
    """
    DB 키로 가구 정보를 가져옵니다.

    Args:
        db_key: FURNITURE_DB의 키 (예: "bed", "sofa")

    Returns:
        가구 정보 dict 또는 빈 dict
    """
    return FURNITURE_DB.get(db_key, {})


def get_dimensions(db_key: str) -> dict:
    """
    DB 키로 치수 정보를 가져옵니다.

    Args:
        db_key: FURNITURE_DB의 키 (예: "bed", "sofa")

    Returns:
        dimensions dict with width, depth, height (mm) 또는 None
    """
    info = FURNITURE_DB.get(db_key)
    if info is None:
        return None

    return info.get('dimensions')


def is_movable(db_key: str) -> bool:
    """
    가구가 이동 가능한지 확인합니다.

    Args:
        db_key: FURNITURE_DB의 키

    Returns:
        is_movable 값 (기본값 True)
    """
    info = FURNITURE_DB.get(db_key, {})
    return info.get('is_movable', True)


def get_base_name(db_key: str) -> str:
    """
    가구의 한글 이름을 가져옵니다.

    Args:
        db_key: FURNITURE_DB의 키

    Returns:
        한글 이름 (예: "침대", "소파") 또는 db_key
    """
    info = FURNITURE_DB.get(db_key, {})
    return info.get('base_name', db_key)


def get_all_synonyms() -> list:
    """
    모든 동의어 목록을 반환합니다.
    YOLO 검색 클래스로 사용할 수 있습니다.

    Returns:
        동의어 리스트
    """
    synonyms = []
    for info in FURNITURE_DB.values():
        synonyms.extend(info.get('synonyms', []))
    return list(set(synonyms))


# 하위 호환성을 위한 함수 (CLIP 제거 후)
def get_dimensions_for_subtype(db_key: str, subtype_name: str = None) -> dict:
    """
    가구 유형으로 치수 정보를 가져옵니다.
    (subtype_name은 무시됨 - CLIP 제거로 서브타입 분류 없음)

    Args:
        db_key: FURNITURE_DB의 키 (예: "bed", "sofa")
        subtype_name: 무시됨 (하위 호환성용)

    Returns:
        dimensions dict with width, depth, height (mm)
    """
    return get_dimensions(db_key)


def estimate_size_variant(db_key: str, subtype_name: str, aspect_ratio: dict) -> tuple:
    """
    SAM-3D로 계산된 비율과 DB의 규격을 비교합니다.
    (CLIP 제거로 단순화됨)

    Args:
        db_key: FURNITURE_DB의 키
        subtype_name: 무시됨 (하위 호환성용)
        aspect_ratio: SAM-3D에서 계산된 비율 {"w": float, "h": float, "d": float}

    Returns:
        (None, dimensions) 튜플
    """
    dims = get_dimensions(db_key)
    return None, dims
