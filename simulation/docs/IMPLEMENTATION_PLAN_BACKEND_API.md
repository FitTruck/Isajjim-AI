# 백엔드 전송 API 구현 계획

> 작성일: 2026-02-02
> 상태: 계획 완료, 구현 대기

## 개요

AI 가구 분석 결과와 OBB 패킹 시뮬레이션을 통합하여 백엔드로 전송하는 기능 구현.
`/analyze-furniture` 호출 시 AI 분석 → 시뮬레이션까지 **자동으로 실행**됨.

## 확정 사항

| 항목 | 결정 |
|------|------|
| Callback URL | 기존과 동일 (`/api/v1/estimates/{estimateId}/callback`) |
| PLY 저장 | GCS 업로드 필요 |
| 호출 타이밍 | **자동** (AI 분석 완료 후 자동으로 시뮬레이션 실행) |

## 아키텍처

### 통합 플로우 (자동 호출)

```
┌─────────────────────────────────────────────────────────────────┐
│              /analyze-furniture 통합 플로우                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 이미지 다운로드 (Firebase)                                   │
│  2. AI 분석 (YOLOE → SAM-3D → 절대 부피)                        │
│  3. PLY 파일 GCS 업로드 ← 신규                                   │
│  4. OBB 패킹 시뮬레이션 ← 신규                                   │
│  5. Callback 전송 (SIMULATION_API_FORMAT.md 포맷)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 플로우

```
POST /analyze-furniture
    │
    ▼
┌──────────────────┐
│ AI Pipeline      │
│ - YOLOE-seg      │
│ - SAM-3D         │
│ - Abs Volume     │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ GCS Upload       │
│ - PLY 파일들     │
│ - URL 생성       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ OBB Packing      │
│ - optimize_obb   │
│   _multi()       │
│ - 멀티 트럭 자동 │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Callback 전송    │
│ (통합 JSON)      │
│ → Backend        │
└──────────────────┘
```

## 파일 변경 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `api/routes/furniture.py` | 수정 | 시뮬레이션 단계 추가, 응답 포맷 변경 |
| `api/services/gcs_upload.py` | **신규** | PLY → GCS 업로드 서비스 |
| `simulation/result_builder.py` | **신규** | AI 결과 + OBB → JSON 변환 |
| `simulation/integration.py` | 수정 | AI 결과 → OBBItem 변환 개선 |

## 구현 단계

### Phase 1: GCS 업로드 서비스

**파일**: `api/services/gcs_upload.py`

```python
# 예상 인터페이스
class GCSUploader:
    def __init__(self, bucket_name: str, credentials_path: str):
        ...

    async def upload_ply(self, ply_data: bytes, object_id: str) -> str:
        """PLY 파일 업로드 후 public URL 반환"""
        ...
```

**필요 정보 (구현 시 확인)**:
- GCS 버킷 이름
- 서비스 계정 JSON 경로
- PLY 파일 네이밍 규칙

### Phase 2: 결과 변환 서비스

**파일**: `simulation/result_builder.py`

```python
def build_simulation_result(
    estimate_id: int,
    ai_results: list[dict],  # AI 파이프라인 결과
    obb_results: MultiTruckResult,  # OBB 패킹 결과
    ply_urls: dict[str, str]  # object_id → GCS URL 매핑
) -> dict:
    """
    SIMULATION_API_FORMAT.md 포맷으로 변환

    단위 변환 주의:
    - 가구 치수: mm
    - 트럭 spec: m
    - placement: m
    - volume: m³
    """
```

### Phase 3: furniture.py 수정

**파일**: `api/routes/furniture.py`

`process_furniture_analysis_background()` 함수 수정:

```python
async def process_furniture_analysis_background(...):
    # 기존: AI 분석
    results = await pipeline.process_multiple_images(...)

    # 신규 1: PLY GCS 업로드
    ply_urls = await upload_all_plys_to_gcs(results)

    # 신규 2: OBB 패킹 시뮬레이션
    obb_items = convert_to_obb_items(results)
    obb_result = optimize_obb_multi(obb_items)

    # 신규 3: 통합 JSON 생성
    response = build_simulation_result(
        estimate_id, results, obb_result, ply_urls
    )

    # 기존: Callback 전송 (포맷만 변경됨)
    await send_callback(estimate_id, result_data=response)
```

## Callback 응답 포맷 변경

### 기존 (AI 결과만)

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "SOFA",
          "type": "THREE_SEATER_SOFA",
          "width": 2000.0,
          "depth": 900.0,
          "height": 850.0,
          "volume": 1.53
        }
      ]
    }
  ]
}
```

### 변경 (SIMULATION_API_FORMAT.md)

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
        "spec": {...},
        "utilization": 72.5,
        "items_count": 10,
        "items": [
          {
            "label": "SOFA",
            "type": "THREE_SEATER_SOFA",
            "ply_url": "https://storage.googleapis.com/...",
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
          }
        ]
      }
    ],
    "unplaced_items": [...]
  }
}
```

## 리스크 및 대응

| 리스크 | 수준 | 대응 |
|--------|------|------|
| 단위 변환 오류 (mm/m/cm) | HIGH | 단위 주석 명확화, 변환 함수 단위 테스트 |
| GCS 업로드 실패 | MEDIUM | 재시도 로직, 실패 시 ply_url = null 허용 |
| OBB 패킹 실패 | LOW | unplaced_items에 실패 항목 포함 |
| 기존 백엔드 호환성 | MEDIUM | 백엔드 팀과 포맷 변경 사전 협의 필요 |

## 구현 전 확인 필요

- [ ] GCS 버킷 이름
- [ ] GCS 서비스 계정 JSON 경로
- [ ] 백엔드 팀 새 포맷 수용 준비 확인

## 테스트 계획

### 단위 테스트
- `tests/test_gcs_upload.py` - GCS 업로드 mock 테스트
- `tests/test_result_builder.py` - JSON 변환 테스트 (단위 변환 검증)

### 통합 테스트
- `tests/test_furniture_with_simulation.py` - 전체 플로우 테스트

## 관련 문서

- `simulation/docs/SIMULATION_API_FORMAT.md` - 최종 JSON 포맷 명세
- `simulation/docs/OBB_ALGORITHM.md` - OBB 패킹 알고리즘 설명
