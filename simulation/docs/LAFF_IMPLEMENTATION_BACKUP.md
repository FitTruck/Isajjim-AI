# LAFF 알고리즘 구현 백업 (2026-01-29)

## 개요

트럭 적재 시뮬레이션에 LAFF (Largest Area Fit First) 알고리즘을 구현했으나,
프로젝트 방향 변경으로 제거됨. 추후 참고용으로 문서화.

## 구현된 기능

### 1. LAFF 알고리즘 (`optimizer.py`)

레벨(층) 기반 3D Bin Packing 알고리즘으로, 실제 이삿짐 적재 방식과 유사.

#### 핵심 데이터 구조

```python
@dataclass
class LevelSpace:
    """레벨 내 빈 공간 (2D 바닥 면적 + 높이 제한)"""
    x: float          # 좌하단 x 좌표
    z: float          # 좌하단 z 좌표
    width: float      # 너비 (X축)
    depth: float      # 깊이 (Z축)
    max_height: float # 사용 가능한 최대 높이

@dataclass
class Level:
    """트럭 내 하나의 레벨 (수평 층)"""
    y: float                              # 레벨 시작 Y 좌표
    height: float                         # 레벨 높이 (첫 아이템으로 결정)
    spaces: list[LevelSpace]              # 빈 공간들
    placed_items: list                    # 배치된 아이템들
```

#### 핵심 함수

```python
def split_level_space(space, item_x, item_z, item_w, item_d, level_height):
    """Guillotine 2D 분할 - L자 형태로 공간 분할"""

def find_best_space_laff(level, item_w, item_d, item_h):
    """LAFF 방식 최적 공간 탐색 (tight fit + 위치 우선순위)"""

def merge_level_spaces(spaces):
    """중복 공간 제거"""

def _optimize_laff(truck_width, truck_depth, truck_height, items):
    """LAFF 메인 함수"""
```

#### 알고리즘 흐름

```
1. 바닥 면적 기준 내림차순 정렬 (큰 가구 먼저)
2. 첫 아이템으로 Level 0 생성 (높이 결정)
3. 현재 레벨에 맞는 공간 찾기
4. 공간 없으면 새 레벨 생성
5. 레벨 내 Guillotine 2D 분할로 효율적 공간 활용
```

#### 배치 시각화

```
┌─────────────────────────────────────┐
│            Level 2 (y=1.0m)         │  ← 작은 물건들
│  ┌────┐  ┌────┐                     │
│  │화분│  │화분│                     │
│  └────┘  └────┘                     │
├─────────────────────────────────────┤
│            Level 1 (y=0.5m)         │  ← 중간 크기
│  ┌────────────┐  ┌────┐             │
│  │    협탁    │  │ TV │             │
│  └────────────┘  └────┘             │
├─────────────────────────────────────┤
│            Level 0 (y=0)            │  ← 큰 가구 (바닥)
│  ┌────────────────────────────────┐ │
│  │         싱글침대               │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 2. EMS 알고리즘 (기존)

Empty Maximal Spaces 기반 3D Bin Packing.

- Phase 1: 기본 EMS (Space 클래스, 공간 분할)
- Phase 2: Guillotine 분할, Corner placement, Weight stability
- Phase 3: Block building (동일 아이템 스태킹)

### 3. 프론트엔드 변경 (`simulator.html`)

- LAFF/EMS 버튼 분리
- `serverOptimize(algorithm)` 함수에 알고리즘 파라미터 추가
- 회전 정렬 버튼 (─ 수평, │ 수직)

### 4. API 변경 (`routes.py`)

- `algorithm` 파라미터: "laff" | "ems" | "py3dbp" | "blf"
- 기본값: "laff"

## 테스트 결과

7개 가구 (침대 3, 협탁 1, TV 1, 화분 2) 테스트:

| 알고리즘 | 배치 | 적재율 | 특징 |
|----------|------|--------|------|
| LAFF | 6/7 | 26.6% | 레벨 기반, 직관적 |
| EMS | 7/7 | 28.1% | 빈 공간 최적화 |

## 제거 이유

프로젝트 방향 변경으로 복잡한 적재 최적화 기능 불필요.

## 복원 방법

이 문서의 코드를 참고하여 `optimizer.py`에 재구현 가능.
Git 히스토리에서도 복원 가능: `git log --oneline simulation/optimizer.py`
