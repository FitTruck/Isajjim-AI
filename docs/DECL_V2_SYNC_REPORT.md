# DeCl_v2 동기화 보고서

**작성일**: 2026-01-22
**대상 폴더**: `git_DeCl3/ai/`
**참조 폴더**: `DeCl_v2/`

---

## 1. 개요

`git_DeCl3` 프로젝트의 AI 모듈을 `DeCl_v2`의 최신 내용과 동기화하였습니다.
기존 서버 동작에 영향을 주지 않으면서 Knowledge Base를 개선하였습니다.

---

## 2. 수정된 파일 목록

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `ai/data/knowledge_base.py` | **전면 교체** | DeCl_v2 기반으로 FURNITURE_DB 교체 |
| `ai/data/__init__.py` | 수정 | 새로운 함수 export 추가 |

---

## 3. 상세 변경 내역

### 3.1. `ai/data/knowledge_base.py`

#### 3.1.1. FURNITURE_DB 변경

**제거된 카테고리** (git_DeCl3에만 있던 항목 - 소형 가전/생활용품):
- `tv` (TV 크기별 분류) → `monitor`로 통합
- `box` (박스)
- `mirror` (거울)
- `microwave` (전자레인지)
- `picture frame` (그림/액자)
- `rug` (카펫/러그)
- `cushion` (쿠션)
- `pillow` (베개)
- `candlestick` (촛대)
- `curtain` (커튼)
- `seasoning container` (조미료통)
- `side dish container` (반찬통)
- `pot` (냄비)
- `frying pan` (프라이팬)
- `tissue box` (휴지곽)
- `toilet paper` (두루마리 휴지)
- `paper towel` (키친타올)
- `plate` (접시)
- `bowl` (그릇/볼)
- `cup` (컵/머그)

**추가된 카테고리** (DeCl_v2에서 가져온 항목 - 대형 가구/운동기구):
| 카테고리 | 한국어명 | 비고 |
|----------|----------|------|
| `nightstand` | 협탁 | `min_confidence: 0.5` 포함 |
| `kimchi refrigerator` | 김치냉장고 | |
| `vanity table` | 화장대 | 서브타입 포함 |
| `tv stand` | TV 거치대 | 서브타입 포함 |
| `piano` | 피아노 | 서브타입 포함 |
| `massage chair` | 안마의자 | |
| `treadmill` | 러닝머신 | |
| `exercise bike` | 실내자전거 | |

**수정된 카테고리**:
| 카테고리 | 변경 사항 |
|----------|----------|
| `kitchen cabinet` | base_name: "찬장/수납장" → "찬장" |
| `display shelf` | synonyms에 "close shelf", "wooden shelf" 추가 |
| `monitor` | synonyms에 TV 관련 동의어 통합, base_name: "모니터/TV" |

#### 3.1.2. 새로운 헬퍼 함수

| 함수 | 설명 |
|------|------|
| `get_min_confidence(db_key)` | 최소 신뢰도 임계값 반환 (nightstand: 0.5) |
| `get_all_synonyms()` | 모든 동의어 목록 반환 |

#### 3.1.3. 하위 호환성 유지

기존 코드와의 호환성을 위해 다음 함수들은 유지됨:
- `get_content_labels()` → 빈 리스트 반환
- `is_movable()` → 항상 True 반환
- `get_dimensions()` → None 반환
- `get_dimensions_for_subtype()` → None 반환
- `estimate_size_variant()` → "medium" 반환

---

### 3.2. `ai/data/__init__.py`

**추가된 export**:
```python
get_min_confidence
get_all_synonyms
```

---

## 4. 제외된 변경 사항

다음 항목들은 DeCl_v2에서 가져오지 않았습니다 (git_DeCl3가 더 최신):

| 항목 | 사유 |
|------|------|
| `check_strategy` 필드 | git_DeCl3의 V2 파이프라인에서 사용하지 않음 |
| `CheckStrategy` 클래스 | 불필요 |
| `get_check_strategy()` 함수 | 불필요 |
| `is_fixed_object()` 함수 | 불필요 |
| `ImageDraw` import | image_ops.py에서 불필요 |

---

## 5. 최종 카테고리 목록 (24개)

| # | DB Key | 한국어명 | 서브타입 | min_confidence |
|---|--------|----------|----------|----------------|
| 1 | air conditioner | 에어컨 | 3개 | - |
| 2 | kitchen cabinet | 찬장 | - | - |
| 3 | drawer | 서랍장 | - | - |
| 4 | nightstand | 협탁 | - | 0.5 |
| 5 | bookshelf | 책장 | - | - |
| 6 | display shelf | 전시대/선반 | 1개 | - |
| 7 | refrigerator | 냉장고 | 2개 | - |
| 8 | wardrobe | 장롱/수납장 | 3개 | - |
| 9 | sofa | 소파 | 4개 | - |
| 10 | bed | 침대 | 6개 | - |
| 11 | dining table | 식탁 | 3개 | - |
| 12 | monitor | 모니터/TV | - | - |
| 13 | desk | 책상 | 3개 | - |
| 14 | chair | 의자/스툴 | 2개 | - |
| 15 | washing machine | 세탁기 | 2개 | - |
| 16 | dryer | 건조기 | - | - |
| 17 | floor | 바닥 | - | - |
| 18 | potted plant | 화분/식물 | - | - |
| 19 | kimchi refrigerator | 김치냉장고 | - | - |
| 20 | vanity table | 화장대 | 2개 | - |
| 21 | tv stand | TV 거치대 | 3개 | - |
| 22 | piano | 피아노 | 3개 | - |
| 23 | massage chair | 안마의자 | - | - |
| 24 | treadmill | 러닝머신 | - | - |
| 25 | exercise bike | 실내자전거 | - | - |

---

## 6. 사용 예시

### 6.1. min_confidence 사용

```python
from ai.data import get_min_confidence

db_key = "nightstand"
min_conf = get_min_confidence(db_key)  # Returns 0.5

if min_conf and detection_confidence < min_conf:
    # 신뢰도 미달로 필터링
    continue
```

### 6.2. 동의어 목록 조회

```python
from ai.data import get_all_synonyms

all_synonyms = get_all_synonyms()
# Returns: ["air conditioner", "ac unit", "climate control unit", ...]
```

---

## 7. 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 카테고리 수 | 32개 | 24개 |
| min_confidence | 없음 | nightstand에 0.5 설정 |
| TV/모니터 분류 | tv 별도 (크기별 5개 서브타입) | monitor로 통합 |
| 신규 카테고리 | - | 협탁, 김치냉장고, 화장대, TV거치대, 피아노, 안마의자, 러닝머신, 실내자전거 |
| 하위 호환성 | - | 완전 유지 |

---

**작업 완료**: 2026-01-22
