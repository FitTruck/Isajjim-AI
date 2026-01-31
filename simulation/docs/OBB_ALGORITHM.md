# OBB 기반 3D Bin Packing 알고리즘

## 개요

트럭 적재함에 가구/물품을 최적으로 배치하는 3D Bin Packing 알고리즘입니다.
**Extreme Points (EP)** 알고리즘을 기반으로, 실제 이삿짐 적재 방식을 모사합니다.

### 핵심 특징

- **뒤쪽부터 차곡차곡**: 트럭 안쪽(뒤)부터 바깥쪽(앞)으로 채움
- **왼쪽에서 오른쪽으로**: 같은 깊이에서는 왼쪽부터 오른쪽으로 채움
- **70% 지지 규칙**: 쌓을 때 바닥 면적의 70% 이상이 지지되어야 함
- **수평 회전만**: 가구를 눕히지 않고 수평으로만 회전 (90도)

## 좌표계

```
        Y (높이)
        │
        │    Z (깊이: 뒤 → 앞)
        │   /
        │  /
        │ /
        └──────── X (폭: 왼쪽 → 오른쪽)

트럭 범위:
  X: [-width/2, +width/2]  (왼쪽 → 오른쪽)
  Y: [0, height]           (바닥 → 천장)
  Z: [-depth/2, +depth/2]  (뒤쪽 → 앞쪽)
```

## 알고리즘 흐름

### 1. 정렬 (Sorting)

```python
# 부피가 큰 것부터 배치 (Greedy)
sorted_items = sorted(items, key=lambda x: x.volume, reverse=True)
```

큰 물건을 먼저 배치해야 작은 물건이 빈 공간을 채울 수 있습니다.

### 2. 초기 Extreme Point

```python
# 시작점: 뒤쪽-왼쪽-바닥 코너
initial_ep = ExtremePoint(x=-truck_w/2, y=0, z=-truck_d/2)
```

### 3. 배치 우선순위 (Scoring)

각 후보 위치에 대해 점수를 계산하고, 가장 낮은 점수를 선택합니다:

```python
score = (cz, cx, cy, natural)
```

| 우선순위 | 축 | 의미 |
|---------|-----|------|
| 1순위 | Z | 뒤쪽(음수)부터 채움 |
| 2순위 | X | 왼쪽(음수)부터 채움 |
| 3순위 | Y | 바닥부터 채움 (쌓기는 마지막) |
| 4순위 | natural | 긴 쪽이 깊이 방향이면 우선 |

### 4. Extreme Points 생성

물건 배치 후 3개의 새로운 EP 생성:

```
          ┌─────────┐
          │  배치된  │
          │   박스   │──→ EP1 (+X: 오른쪽)
          └─────────┘
               │
               ↓
          EP2 (+Z: 앞쪽)

               ↑
          EP3 (+Y: 위쪽)
```

```python
def generate_new_extreme_points(placed_box):
    # EP1: 오른쪽 (+X)
    ep1 = (box.x_max, box.y_min, box.z_min)

    # EP2: 앞쪽 (+Z)
    ep2 = (box.x_min, box.y_min, box.z_max)

    # EP3: 위쪽 (+Y)
    ep3 = (box.x_min, box.y_max, box.z_min)
```

### 5. 지지 검사 (Support Check)

```python
def check_support(x, y, z, width, depth, placed_boxes):
    if y == 0:  # 바닥은 항상 지지됨
        return True

    # 아래 박스들과의 겹침 면적 계산
    supported_area = sum(overlap_area(box) for box in boxes_below)
    base_area = width * depth

    return supported_area / base_area >= 0.7  # 70% 이상 지지
```

## 회전 (Orientation)

수평 회전만 허용 (allow_tilt=False):

| Orientation | 출력 (w, d, h) | 설명 |
|-------------|---------------|------|
| LWH (0) | (L, W, H) | 원래 방향 |
| WLH (2) | (W, L, H) | 90도 수평 회전 |

프론트엔드에서 orientation=2일 때 `mesh.rotation.y = Math.PI/2` 적용.

## 트럭 프리셋

| 타입 | 폭(cm) | 깊이(cm) | 높이(cm) |
|------|--------|----------|----------|
| 1ton | 170 | 280 | 170 |
| 2.5ton | 200 | 430 | 190 |
| 5ton | 230 | 620 | 240 |

## API

### 요청

```http
POST /simulation/optimize-obb
Content-Type: application/json

{
  "truck_type": "2.5ton",
  "items": [
    {"id": "sofa", "width": 2.0, "depth": 0.9, "height": 0.85},
    {"id": "table", "width": 1.2, "depth": 0.8, "height": 0.75}
  ],
  "unit": "m",
  "support_ratio": 0.7
}
```

### 응답

```json
{
  "success": true,
  "truck_type": "2.5ton",
  "placements": [
    {
      "id": "sofa",
      "x": -0.6,
      "y": 0.0,
      "z": -1.7,
      "width": 0.9,
      "depth": 2.0,
      "height": 0.85,
      "orientation": 2
    }
  ],
  "unplaced_ids": [],
  "volume_utilization": 15.3,
  "message": "2개 배치 (코너시작), 0개 미배치"
}
```

### 좌표 설명

- `x, z`: 객체 중심의 수평 좌표
- `y`: 객체 **바닥**의 수직 좌표 (중심 아님!)
- `width, depth, height`: 회전 적용된 최종 치수

## 프론트엔드 통합

### Three.js 위치 설정

```javascript
// y는 바닥 좌표이므로 중심으로 변환
const posY = placement.y + placement.height / 2;
mesh.position.set(placement.x, posY, placement.z);

// 회전 적용 (orientation 2, 3, 5일 때)
if (placement.orientation === 2) {
  mesh.rotation.y = Math.PI / 2;
}
```

### PLY 스케일링

PLY 포인트클라우드는 **균일 스케일링**으로 원본 비율을 유지하고, 히트박스는 스케일된 PLY에 딱 맞게 생성합니다.

```javascript
// 1. 균일 스케일 계산 (원본 비율 유지, 왜곡 없음)
const scale = Math.min(width / plySize.x, height / plySize.y, depth / plySize.z);

// 2. 스케일 적용 후 실제 PLY 크기 계산
const scaledWidth = plySize.x * scale;
const scaledHeight = plySize.y * scale;
const scaledDepth = plySize.z * scale;

// 3. PLY에 균일 스케일 적용
points.scale.setScalar(scale);

// 4. 히트박스는 스케일된 PLY 크기에 맞게 생성
const hitboxGeometry = new THREE.BoxGeometry(scaledWidth, scaledHeight, scaledDepth);

// 5. userData에 실제 크기 저장 (OBB API에서 사용)
mesh.userData.width = scaledWidth;
mesh.userData.height = scaledHeight;
mesh.userData.depth = scaledDepth;
```

**왜 균일 스케일링인가?**
- 비균일 스케일링은 객체를 찌그러뜨려 왜곡시킴
- 균일 스케일링은 원본 3D 모델의 비율을 유지
- 히트박스가 PLY에 정확히 맞아서 충돌 감지가 정확함

### 히트박스 시각화

디버깅 및 배치 확인을 위해 히트박스를 반투명하게 표시합니다.

```javascript
// 반투명 히트박스 (배치 영역 확인용)
const hitboxMaterial = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0.15,           // 15% 불투명
  depthWrite: false,
  color: color || 0x4488ff // 객체별 색상
});
const hitbox = new THREE.Mesh(hitboxGeometry, hitboxMaterial);

// 히트박스 엣지 (흰색 경계선)
const hitboxEdges = new THREE.EdgesGeometry(hitboxGeometry);
const hitboxLine = new THREE.LineSegments(
  hitboxEdges,
  new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.5, transparent: true })
);
group.add(hitboxLine);
```

**시각화 구성요소:**

| 요소 | 설명 |
|------|------|
| PLY 포인트클라우드 | 실제 3D 스캔 데이터 (색상 포함) |
| 반투명 히트박스 | 충돌 감지 영역 (15% 불투명) |
| 흰색 엣지 라인 | 히트박스 경계선 (50% 불투명) |

```
┌─────────────────┐
│ ░░░░░░░░░░░░░░░ │  ← 흰색 엣지 라인
│ ░  ·····  ░░░░░ │
│ ░ ·······  ░░░░ │  ← PLY 포인트클라우드
│ ░  ·····  ░░░░░ │
│ ░░░░░░░░░░░░░░░ │  ← 반투명 히트박스
└─────────────────┘
```

## 제약 조건

1. **경계 검사**: 트럭 범위 내에만 배치
2. **충돌 검사**: 다른 박스와 겹치지 않음
3. **지지 검사**: 70% 이상 면적이 아래에서 지지됨
4. **수평 유지**: 가구를 눕히지 않음 (allow_tilt=False)

## 파일 구조

```
simulation/
├── obb_packer.py          # OBB 알고리즘 구현
├── routes.py              # /optimize-obb API 엔드포인트
├── models.py              # 데이터 모델 정의
└── static/
    └── simulator.html     # 3D 시각화 (Three.js)
```

## 성능

| 항목 | 값 |
|------|-----|
| 알고리즘 복잡도 | O(n² × m) (n=아이템, m=EP 수) |
| 일반적인 적재율 | 50-70% |
| 처리 시간 | < 100ms (20개 아이템 기준) |

## 향후 개선 사항

- [ ] 무게 중심 최적화
- [ ] 취급주의 물품 상단 배치
- [ ] 다중 트럭 분배
- [ ] 언로딩 순서 고려
