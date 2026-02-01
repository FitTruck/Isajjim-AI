# 시뮬레이션 결과 JSON Format

## 개요

AI 가구 분석 결과를 기반으로 OBB 패킹 시뮬레이션을 수행한 후, 백엔드로 전송되는 최종 JSON 포맷입니다.

### 데이터 플로우

```
1. AI 분석 → 절대 치수 (label, type, width/depth/height mm, volume m³)
2. Simulation → OBB 패킹 (위치, 회전, 트럭 선택)
3. 최종 결과 → 백엔드 전송
```

## JSON Format

```json
{
  "estimate_id": 123,

  "simulation": {
    "success": true,
    "total_trucks": 2,
    "total_items": 15,
    "total_volume_m3": 5.43,

    "trucks": [
      {
        "truck_index": 0,
        "type": "5ton",
        "spec": {
          "name": "5톤 트럭",
          "width": 2.3,
          "depth": 6.2,
          "height": 2.4,
          "max_weight": 5000
        },
        "utilization": 72.5,
        "items_count": 10,

        "items": [
          {
            "label": "SOFA",
            "type": "THREE_SEATER_SOFA",
            "ply_url": "https://storage.example.com/ply/sofa_001.ply",
            "width": 2000.0,
            "depth": 900.0,
            "height": 850.0,
            "volume": 1.53,
            "placement": {
              "x": -0.6,
              "y": 0.0,
              "z": -1.7,
              "orientation": 2
            },
            "order": 1
          },
          {
            "label": "BED",
            "type": "QUEEN_BED",
            "ply_url": "https://storage.example.com/ply/bed_002.ply",
            "width": 1600.0,
            "depth": 2000.0,
            "height": 500.0,
            "volume": 1.6,
            "placement": {
              "x": 0.5,
              "y": 0.0,
              "z": -2.5,
              "orientation": 0
            },
            "order": 2
          }
        ]
      },
      {
        "truck_index": 1,
        "type": "1ton",
        "spec": {
          "name": "1톤 트럭",
          "width": 1.7,
          "depth": 2.8,
          "height": 1.7,
          "max_weight": 1000
        },
        "utilization": 45.3,
        "items_count": 5,

        "items": [
          {
            "label": "CHAIR",
            "type": "OFFICE_CHAIR",
            "ply_url": "https://storage.example.com/ply/chair_001.ply",
            "width": 600.0,
            "depth": 600.0,
            "height": 1000.0,
            "volume": 0.36,
            "placement": {
              "x": -0.3,
              "y": 0.0,
              "z": -1.0,
              "orientation": 0
            },
            "order": 1
          }
        ]
      }
    ],

    "unplaced_items": [
      {
        "label": "WARDROBE",
        "type": "THREE_DOOR_WARDROBE",
        "width": 1800.0,
        "depth": 600.0,
        "height": 2200.0,
        "volume": 2.38,
        "reason": "트럭 높이 초과"
      }
    ]
  }
}
```

## 필드 명세

### 최상위

| 필드 | 타입 | 설명 |
|------|------|------|
| `estimate_id` | int | 견적 ID |

### simulation

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `success` | bool | - | 시뮬레이션 성공 여부 |
| `total_trucks` | int | - | 사용된 트럭 대수 |
| `total_items` | int | - | 배치된 총 아이템 수 |
| `total_volume_m3` | float | m³ | 총 적재 부피 |

### trucks[]

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `truck_index` | int | - | 트럭 인덱스 (0부터 시작) |
| `type` | string | - | 트럭 유형 ("1ton", "2.5ton", "5ton") |
| `spec.name` | string | - | 트럭 이름 ("5톤 트럭") |
| `spec.width` | float | **m** | 트럭 너비 |
| `spec.depth` | float | **m** | 트럭 깊이 |
| `spec.height` | float | **m** | 트럭 높이 |
| `spec.max_weight` | int | **kg** | 최대 적재 중량 |
| `utilization` | float | **%** | 적재율 |
| `items_count` | int | - | 해당 트럭 아이템 수 |

### items[]

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `label` | string | - | 영어 라벨 (SOFA, BED 등) |
| `type` | string | - | 세부 유형 (THREE_SEATER_SOFA 등) |
| `ply_url` | string | - | PLY 파일 URL |
| `width` | float | **mm** | 절대 너비 |
| `depth` | float | **mm** | 절대 깊이 |
| `height` | float | **mm** | 절대 높이 |
| `volume` | float | **m³** | 부피 |
| `placement.x` | float | **m** | X 좌표 (객체 중심) |
| `placement.y` | float | **m** | Y 좌표 (객체 **바닥**) |
| `placement.z` | float | **m** | Z 좌표 (객체 중심) |
| `placement.orientation` | int | - | 0: 원래, 2: 90도 회전 |
| `order` | int | - | 애니메이션 순서 (1부터) |

### unplaced_items[]

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `label` | string | - | 영어 라벨 |
| `type` | string | - | 세부 유형 |
| `width` | float | **mm** | 절대 너비 |
| `depth` | float | **mm** | 절대 깊이 |
| `height` | float | **mm** | 절대 높이 |
| `volume` | float | **m³** | 부피 |
| `reason` | string | - | 미배치 사유 |

## 단위 요약

| 항목 | 단위 |
|------|------|
| 가구 치수 (width, depth, height) | **mm** |
| 트럭 규격 (spec.width/depth/height) | **m** |
| 배치 좌표 (placement.x/y/z) | **m** |
| 부피 (volume, total_volume_m3) | **m³** |
| 적재율 (utilization) | **%** |

## placement.orientation 상세

PLY 파일은 OBB 기반으로 축 정렬된 상태로 저장되므로, 수평 회전만 사용합니다.

| 값 | 설명 | Three.js |
|----|------|----------|
| **0** | 원래 방향 (회전 없음) | `rotation.y = 0` |
| **2** | 90도 수평 회전 | `rotation.y = Math.PI / 2` |

### 시각화

```
orientation = 0 (원래 방향)
     ┌───────────┐
     │           │  depth (Z)
     │   소파    │    ↑
     │           │    │
     └───────────┘    └──→ width (X)
        2000mm           900mm


orientation = 2 (90도 회전)
     ┌─────┐
     │     │
     │소파 │  depth (Z)
     │     │    ↑
     │     │    │
     └─────┘    └──→ width (X)
      900mm         2000mm
```

## 좌표계

Three.js Y-up 좌표계를 사용합니다.

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

## 프론트엔드 사용 예시

```javascript
// 시뮬레이션 결과 렌더링
for (const truck of data.simulation.trucks) {
  // 트럭 생성
  createTruck(truck.spec, truck.truck_index);

  // 아이템 순서대로 애니메이션
  const sortedItems = truck.items.sort((a, b) => a.order - b.order);

  for (const item of sortedItems) {
    // PLY 로드 (mm → m 변환)
    const mesh = await loadPLY(item.ply_url, {
      width: item.width / 1000,
      depth: item.depth / 1000,
      height: item.height / 1000
    });

    // 회전 적용
    if (item.placement.orientation === 2) {
      mesh.rotation.y = Math.PI / 2;
    }

    // 배치 (y는 바닥 좌표이므로 높이/2 더함)
    const targetY = item.placement.y + (item.height / 1000) / 2;

    await animatePlacement(mesh, {
      x: item.placement.x,
      y: targetY,
      z: item.placement.z
    });
  }
}
```

## 관련 파일

- `simulation/obb_packer.py` - OBB 패킹 알고리즘
- `simulation/routes.py` - API 엔드포인트
- `simulation/ply_alignment.py` - PLY 축 정렬 서비스
- `simulation/static/simulator.html` - Three.js 시뮬레이터

## 구현 상태

### 현재 상태

- [x] JSON 포맷 설계 완료
- [ ] 백엔드 전송 API 구현
- [ ] 프론트엔드 시각화 연동

### 추후 구현 계획

#### 1단계: 백엔드 API 연동

- [ ] `/simulation/result` POST 엔드포인트 추가
  - AI 분석 결과 + OBB 패킹 결과를 이 포맷으로 조합
  - 백엔드 callback URL로 전송
- [ ] AI 분석 callback과 시뮬레이션 결과 통합
  - `estimate_id` 기반으로 AI 결과 조회
  - OBB 패킹 실행 후 최종 JSON 생성

#### 2단계: 프론트엔드 연동

- [ ] 백엔드에서 저장된 시뮬레이션 결과 조회 API
- [ ] 프론트엔드에서 JSON 기반 시각화 재현
  - `order` 필드 기반 애니메이션 순서
  - `placement` 필드 기반 위치/회전 적용
  - 멀티 트럭 시각화

#### 3단계: PLY 스토리지 연동

- [ ] AI 서버에서 생성된 PLY를 GCS에 업로드
- [ ] `ply_url` 필드에 GCS URL 저장
- [ ] 프론트엔드에서 PLY 로드 시 GCS URL 사용

#### 4단계: 히트박스 제거

- [ ] PLY 포인트클라우드만으로 시각화
- [ ] 충돌 감지 로직 수정 (필요시)
