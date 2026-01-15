# 가구 및 내부 물품 정의
# - synonyms: YOLO가 탐지할 영문 동의어 (입력용)
# - base_name: 사용자에게 보여줄 한글 이름 (출력용)
# - dimensions: 표준 규격 (mm 단위) - 절대 부피 계산용

# check_strategy: 판단 기준
# subtypes: 분류가 필요한 가구들

'''
인식할 가구의 추가는 아래 형식을 참조하면 됨
DB 연결 시 참고용

dimensions 필드:
- width: 가로 (mm)
- depth: 깊이 (mm)
- height: 높이 (mm)
- variants: 사이즈별 규격 (예: 침대 퀸/킹)
'''

FURNITURE_DB = {
    "air conditioner": {
        "synonyms": ["air conditioner", "ac unit", "climate control unit"],
        "check_strategy": "AC_STRATEGY",
        "base_name": "에어컨",
        "subtypes": [
            {
                "name": "천장형 에어컨 (시스템)",
                "prompt": "a ceiling mounted cassette air conditioner vent",
                "is_movable": False,
                "dimensions": {"width": 600, "depth": 600, "height": 250}
            },
            {
                "name": "벽걸이 에어컨",
                "prompt": "a split air conditioner unit mounted high on a wall",
                "is_movable": True,
                "dimensions": {"width": 800, "depth": 250, "height": 300}
            },
            {
                "name": "스탠드 에어컨",
                "prompt": "a tall floor standing tower air conditioner",
                "is_movable": True,
                "dimensions": {"width": 400, "depth": 400, "height": 1800}
            }
        ]
    },
    "kitchen cabinet": {
        "synonyms": ["kitchen cupboard", "pantry cabinet", "wall cabinet", "overhead storage", "kitchen cabinet"],
        "check_strategy": "CHECK_CONTENTS",
        "base_name": "찬장/수납장",
        "content_labels": ["bowl", "plate", "cup", "wine glass", "coffee mug", "dish", "pot", "bottle"],
        "is_movable": False,
        "dimensions": {"width": 600, "depth": 350, "height": 800}
    },
    "bookshelf": {
        "synonyms": ["bookshelf", "bookcase", "library shelf"],
        "check_strategy": "CHECK_CONTENTS",
        "base_name": "책장",
        "content_labels": ["book", "magazine collection", "file binder"],
        "is_movable": True,
        "dimensions": {"width": 800, "depth": 300, "height": 1800}
    },
    "refrigerator": {
        "synonyms": ["refrigerator", "fridge", "freezer"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "냉장고",
        "subtypes": [
            {
                "name": "일반 냉장고",
                "prompt": "a standard free-standing refrigerator",
                "is_movable": True,
                "dimensions": {
                    "variants": {
                        "small": {"width": 550, "depth": 600, "height": 1400},
                        "medium": {"width": 600, "depth": 700, "height": 1700},
                        "large": {"width": 900, "depth": 700, "height": 1800}
                    }
                }
            },
            {
                "name": "빌트인 냉장고",
                "prompt": "a built-in refrigerator flush with kitchen cabinets",
                "is_movable": False,
                "dimensions": {"width": 600, "depth": 600, "height": 1800}
            }
        ]
    },
    "wardrobe": {
        "synonyms": ["wardrobe", "closet", "armoire", "clothes closet"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "장롱/수납장",
        "subtypes": [
            {
                "name": "붙박이장 (매립형)",
                "prompt": "a built-in wardrobe integrated into the wall without gaps",
                "is_movable": False,
                "dimensions": {"width": 2400, "depth": 600, "height": 2400}
            },
            {
                "name": "일반 옷장 (이동식)",
                "prompt": "a free-standing wooden wardrobe furniture",
                "is_movable": True,
                "dimensions": {
                    "variants": {
                        "single": {"width": 800, "depth": 600, "height": 2000},
                        "double": {"width": 1200, "depth": 600, "height": 2000},
                        "triple": {"width": 1800, "depth": 600, "height": 2000}
                    }
                }
            },
            {
                "name": "시스템 행거",
                "prompt": "an open walk-in closet system hanger with metal poles",
                "is_movable": True,
                "dimensions": {"width": 1200, "depth": 500, "height": 1800}
            }
        ]
    },
    "sofa": {
        "synonyms": ["sofa", "couch", "settee"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "소파",
        "is_movable": True,
        "subtypes": [
            {
                "name": "1인용 소파",
                "prompt": "a single seater armchair sofa",
                "is_movable": True,
                "dimensions": {"width": 900, "depth": 900, "height": 900}
            },
            {
                "name": "2인용 소파",
                "prompt": "a two seater loveseat sofa",
                "is_movable": True,
                "dimensions": {"width": 1400, "depth": 900, "height": 900}
            },
            {
                "name": "3인용 소파",
                "prompt": "a three seater long sofa",
                "is_movable": True,
                "dimensions": {"width": 2100, "depth": 900, "height": 900}
            },
            {
                "name": "L자형 소파",
                "prompt": "an L-shaped sectional corner sofa",
                "is_movable": True,
                "dimensions": {"width": 2800, "depth": 1800, "height": 900}
            }
        ]
    },
    "bed": {
        "synonyms": ["bed", "bed frame", "bunk bed"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "침대",
        "is_movable": True,
        "subtypes": [
            {
                "name": "싱글 침대",
                "prompt": "a single bed with a narrow mattress",
                "is_movable": True,
                "dimensions": {"width": 1000, "depth": 2000, "height": 450}
            },
            {
                "name": "슈퍼싱글 침대",
                "prompt": "a super single bed slightly wider than single",
                "is_movable": True,
                "dimensions": {"width": 1100, "depth": 2000, "height": 450}
            },
            {
                "name": "더블 침대",
                "prompt": "a double bed with full size mattress",
                "is_movable": True,
                "dimensions": {"width": 1400, "depth": 2000, "height": 450}
            },
            {
                "name": "퀸 사이즈 침대",
                "prompt": "a queen size bed with wide mattress",
                "is_movable": True,
                "dimensions": {"width": 1500, "depth": 2000, "height": 450}
            },
            {
                "name": "킹 사이즈 침대",
                "prompt": "a king size bed with extra wide mattress",
                "is_movable": True,
                "dimensions": {"width": 1800, "depth": 2000, "height": 450}
            },
            {
                "name": "2층 침대",
                "prompt": "a bunk bed with two levels stacked",
                "is_movable": True,
                "dimensions": {"width": 1000, "depth": 2000, "height": 1700}
            }
        ]
    },
    "dining table": {
        "synonyms": ["dining table", "kitchen table"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "식탁",
        "is_movable": True,
        "subtypes": [
            {
                "name": "2인용 식탁",
                "prompt": "a small dining table for two people",
                "is_movable": True,
                "dimensions": {"width": 700, "depth": 700, "height": 750}
            },
            {
                "name": "4인용 식탁",
                "prompt": "a rectangular dining table for four people",
                "is_movable": True,
                "dimensions": {"width": 1200, "depth": 800, "height": 750}
            },
            {
                "name": "6인용 식탁",
                "prompt": "a large dining table for six people",
                "is_movable": True,
                "dimensions": {"width": 1800, "depth": 900, "height": 750}
            }
        ]
    },
    "tv": {
        "synonyms": ["television", "tv", "flat screen tv"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "TV",
        "is_movable": True,
        "subtypes": [
            {
                "name": "32인치 TV",
                "prompt": "a small 32 inch flat screen television",
                "is_movable": True,
                "dimensions": {"width": 730, "depth": 50, "height": 430}
            },
            {
                "name": "43인치 TV",
                "prompt": "a medium 43 inch flat screen television",
                "is_movable": True,
                "dimensions": {"width": 970, "depth": 60, "height": 570}
            },
            {
                "name": "55인치 TV",
                "prompt": "a large 55 inch flat screen television",
                "is_movable": True,
                "dimensions": {"width": 1230, "depth": 70, "height": 710}
            },
            {
                "name": "65인치 TV",
                "prompt": "an extra large 65 inch flat screen television",
                "is_movable": True,
                "dimensions": {"width": 1450, "depth": 80, "height": 840}
            },
            {
                "name": "75인치 TV",
                "prompt": "a huge 75 inch flat screen television",
                "is_movable": True,
                "dimensions": {"width": 1680, "depth": 80, "height": 970}
            }
        ]
    },
    "desk": {
        "synonyms": ["desk", "office desk", "computer desk", "writing desk"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "책상",
        "is_movable": True,
        "subtypes": [
            {
                "name": "일반 책상",
                "prompt": "a standard writing desk",
                "is_movable": True,
                "dimensions": {"width": 1200, "depth": 600, "height": 750}
            },
            {
                "name": "L자형 책상",
                "prompt": "an L-shaped corner desk",
                "is_movable": True,
                "dimensions": {"width": 1600, "depth": 1200, "height": 750}
            },
            {
                "name": "컴퓨터 책상",
                "prompt": "a computer desk with keyboard tray",
                "is_movable": True,
                "dimensions": {"width": 1000, "depth": 600, "height": 750}
            }
        ]
    },
    "chair": {
        "synonyms": ["chair", "office chair", "dining chair", "armchair"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "의자",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 500, "height": 900}
    },
    "washing machine": {
        "synonyms": ["washing machine", "washer", "laundry machine"],
        "check_strategy": "SUBTYPE_MATCH",
        "base_name": "세탁기",
        "is_movable": True,
        "subtypes": [
            {
                "name": "드럼 세탁기",
                "prompt": "a front loading drum washing machine",
                "is_movable": True,
                "dimensions": {"width": 600, "depth": 600, "height": 850}
            },
            {
                "name": "통돌이 세탁기",
                "prompt": "a top loading washing machine",
                "is_movable": True,
                "dimensions": {"width": 600, "depth": 600, "height": 1000}
            }
        ]
    },
    "dryer": {
        "synonyms": ["dryer", "clothes dryer", "tumble dryer"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "건조기",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 600, "height": 850}
    },
    "box": {
        "synonyms": ["box", "cardboard box", "moving box", "carton"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "박스",
        "is_movable": True,
        "dimensions": {
            "variants": {
                "small": {"width": 300, "depth": 200, "height": 200},
                "medium": {"width": 400, "depth": 300, "height": 300},
                "large": {"width": 500, "depth": 400, "height": 400},
                "extra_large": {"width": 600, "depth": 500, "height": 500}
            }
        }
    },
    "drawer": {
        "synonyms": ["drawer", "chest of drawers", "dresser", "drawer unit"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "서랍장",
        "is_movable": True,
        "dimensions": {"width": 800, "depth": 450, "height": 1000}
    },
    "mirror": {
        "synonyms": ["mirror", "full length mirror", "wall mirror", "standing mirror"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "거울",
        "is_movable": True,
        "dimensions": {"width": 600, "depth": 50, "height": 1500}
    },
    "microwave": {
        "synonyms": ["microwave", "microwave oven"],
        "check_strategy": "ALWAYS_MOVABLE",
        "base_name": "전자레인지",
        "is_movable": True,
        "dimensions": {"width": 500, "depth": 400, "height": 300}
    }
}


def get_dimensions_for_subtype(db_key: str, subtype_name: str) -> dict:
    """
    가구 유형과 서브타입 이름으로 치수 정보를 가져옵니다.

    Args:
        db_key: FURNITURE_DB의 키 (예: "bed", "sofa")
        subtype_name: 서브타입 이름 (예: "퀸 사이즈 침대")

    Returns:
        dimensions dict with width, depth, height (mm)
    """
    if db_key not in FURNITURE_DB:
        return None

    db_info = FURNITURE_DB[db_key]

    # subtypes가 있는 경우
    if 'subtypes' in db_info:
        for subtype in db_info['subtypes']:
            if subtype['name'] == subtype_name:
                dims = subtype.get('dimensions', {})
                # variants가 있는 경우 기본값 반환
                if 'variants' in dims:
                    # 가장 일반적인 사이즈 반환 (medium 또는 첫 번째)
                    variants = dims['variants']
                    if 'medium' in variants:
                        return variants['medium']
                    return list(variants.values())[0]
                return dims

    # 기본 dimensions 반환
    dims = db_info.get('dimensions', {})
    if 'variants' in dims:
        variants = dims['variants']
        if 'medium' in variants:
            return variants['medium']
        return list(variants.values())[0]

    return dims


def estimate_size_variant(db_key: str, subtype_name: str, aspect_ratio: dict) -> tuple:
    """
    SAM-3D로 계산된 비율과 DB의 규격을 비교하여 가장 적합한 사이즈 변형을 추정합니다.

    Args:
        db_key: FURNITURE_DB의 키
        subtype_name: 서브타입 이름
        aspect_ratio: SAM-3D에서 계산된 비율 {"w": float, "h": float, "d": float}

    Returns:
        (variant_name, dimensions) 튜플
    """
    if db_key not in FURNITURE_DB:
        return None, None

    db_info = FURNITURE_DB[db_key]

    # subtypes에서 dimensions 찾기
    dims = None
    if 'subtypes' in db_info:
        for subtype in db_info['subtypes']:
            if subtype['name'] == subtype_name:
                dims = subtype.get('dimensions', {})
                break

    if dims is None:
        dims = db_info.get('dimensions', {})

    # variants가 없으면 기본 반환
    if 'variants' not in dims:
        return None, dims

    variants = dims['variants']

    # 각 variant와 비율 비교
    best_match = None
    best_score = float('inf')

    # 입력 비율 정규화 (가장 큰 값을 1로)
    max_ratio = max(aspect_ratio.values())
    if max_ratio == 0:
        max_ratio = 1
    input_normalized = {
        'w': aspect_ratio['w'] / max_ratio,
        'h': aspect_ratio['h'] / max_ratio,
        'd': aspect_ratio['d'] / max_ratio
    }

    for variant_name, variant_dims in variants.items():
        # variant 비율 정규화
        max_dim = max(variant_dims['width'], variant_dims['height'], variant_dims['depth'])
        variant_normalized = {
            'w': variant_dims['width'] / max_dim,
            'h': variant_dims['height'] / max_dim,
            'd': variant_dims['depth'] / max_dim
        }

        # 유클리드 거리로 유사도 계산
        score = sum((input_normalized[k] - variant_normalized[k]) ** 2 for k in ['w', 'h', 'd'])

        if score < best_score:
            best_score = score
            best_match = (variant_name, variant_dims)

    return best_match if best_match else (None, list(variants.values())[0])
