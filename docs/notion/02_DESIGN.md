# Isajjim AI - 설계 문서

## 1. API 스펙

### 1.1 가구 분석 (주요 엔드포인트)

| 메서드 | 경로 | 방식 | 설명 |
|--------|------|------|------|
| POST | `/analyze-furniture` | 비동기 (Callback) | 다중 이미지 가구 분석 |
| POST | `/analyze-furniture-single` | 동기 | 단일 이미지 가구 분석 |
| POST | `/analyze-furniture-base64` | 동기 | Base64 이미지 가구 분석 |
| POST | `/detect-furniture` | 동기 | 탐지만 (3D 없음, 빠른 응답) |

### 1.2 시스템

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| GET | `/gpu-status` | GPU 풀 상태 |
| GET | `/assets-list` | 저장된 에셋 목록 |
| GET | `/assets/{filename}` | 에셋 다운로드 (PLY, GLB, GIF) |

---

## 2. 엔드포인트 상세

### 2.1 POST /analyze-furniture

비동기 방식. 즉시 응답 후 백그라운드에서 처리하고, 완료 시 Callback URL로 결과 전송.

**요청:**

```json
{
  "estimate_id": 123,
  "image_urls": [
    {"id": 101, "url": "https://firebase-storage-url-1.jpg"},
    {"id": 102, "url": "https://firebase-storage-url-2.jpg"}
  ],
  "enable_mask": true,
  "enable_3d": true,
  "max_concurrent": 3
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `estimate_id` | int | O | - | 견적 ID (Callback URL에 사용) |
| `image_urls` | array | O | - | 이미지 URL 객체 배열 (1~20개) |
| `image_urls[].id` | int | O | - | 사용자 지정 이미지 ID |
| `image_urls[].url` | string | O | - | Firebase Storage URL |
| `enable_mask` | bool | X | true | 마스크 생성 여부 |
| `enable_3d` | bool | X | true | 3D 생성 여부 |
| `max_concurrent` | int | X | 3 | 최대 동시 처리 수 |

**즉시 응답 (HTTP 200):**

```json
{
  "success": true,
  "estimate_id": 123,
  "status": "processing"
}
```

**Callback 전송 (성공):**

URL: `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback`

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "type": "THREE_SEATER_SOFA",
          "width": 200.0,
          "depth": 90.0,
          "height": 85.0
        },
        {
          "label": "desk",
          "type": null,
          "width": 120.0,
          "depth": 60.0,
          "height": 75.0
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [
        {
          "label": "chair",
          "type": "STANDARD_CHAIR",
          "width": 45.0,
          "depth": 50.0,
          "height": 90.0
        }
      ]
    }
  ]
}
```

**Callback 전송 (실패):**

```json
{
  "error": "Furniture analysis failed: 에러 메시지"
}
```

**응답 필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `label` | string | 탐지된 객체 라벨 (영어, Knowledge Base의 base_name) |
| `type` | string/null | 세부 유형 (예: `"THREE_SEATER_SOFA"`) 또는 null |
| `width` | float | OBB X축 상대 길이 (단위 없음) |
| `depth` | float | OBB Z축 상대 길이 (단위 없음) |
| `height` | float | OBB Y축 상대 길이 (단위 없음) |

> 절대 치수/부피는 백엔드에서 Knowledge Base의 실제 치수와 비율을 조합하여 계산합니다.

---

### 2.2 POST /analyze-furniture-single

단일 이미지 동기 분석 (테스트/디버깅용).

**요청:**

```json
{
  "image_url": "https://firebase-storage-url.jpg",
  "enable_mask": true,
  "enable_3d": true
}
```

**응답:** JSON (분석 결과 직접 반환, Callback 없음)

---

### 2.3 POST /analyze-furniture-base64

Base64 이미지 입력 (Firebase URL 없이 직접 전송).

**요청:**

```json
{
  "image": "<BASE64_ENCODED_IMAGE>",
  "enable_mask": true,
  "enable_3d": true,
  "return_ply": false
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `image` | string | O | - | Base64 인코딩된 이미지 |
| `enable_mask` | bool | X | true | 마스크 생성 여부 |
| `enable_3d` | bool | X | true | 3D 생성 여부 |
| `return_ply` | bool | X | false | PLY base64 데이터 반환 여부 (테스트용) |

**응답:**

```json
{
  "objects": [
    {
      "label": "sofa",
      "subtype": "THREE_SEATER_SOFA",
      "width": 200.0,
      "depth": 90.0,
      "height": 85.0,
      "volume": 0.72
    }
  ]
}
```

> 이 엔드포인트는 `/analyze-furniture`와 다른 응답 형식을 사용합니다 (`subtype`, `volume` 포함).

---

### 2.4 POST /detect-furniture

탐지만 수행 (3D 없음). 가장 빠른 응답.

**요청:** `/analyze-furniture-base64`와 동일 (image 필드만 사용)

**응답:**

```json
{
  "success": true,
  "objects": [
    {
      "id": 0,
      "label": "sofa",
      "db_key": "sofa",
      "subtype": null,
      "bbox": [100, 200, 400, 500],
      "center_point": [250.0, 350.0],
      "confidence": 0.95
    }
  ],
  "total_objects": 1,
  "processing_time_seconds": 0.5
}
```

---

### 2.5 GET /health

```json
{
  "status": "healthy",
  "device": "cuda"
}
```

---

### 2.6 GET /gpu-status

```json
{
  "total_gpus": 4,
  "available_gpus": 3,
  "pipelines_initialized": 4,
  "gpus": {
    "0": {
      "available": true,
      "task_id": null,
      "memory_used_mb": 1024.0,
      "memory_total_mb": 40441.38,
      "error_count": 0,
      "has_pipeline": true
    },
    "1": {
      "available": false,
      "task_id": "image_processing",
      "memory_used_mb": 2048.0,
      "memory_total_mb": 40441.38,
      "error_count": 0,
      "has_pipeline": true
    }
  }
}
```

---

## 3. 파이프라인 설계

### 3.1 처리 단계

| 단계 | 파일 | 입력 | 출력 |
|------|------|------|------|
| 1. 이미지 다운로드 | `1_firebase_images_fetch.py` | Firebase URL | PIL Image |
| 2. 객체 탐지 | `2_YOLO_detect.py` | PIL Image | bbox, label, score, mask |
| 3. DB 매칭 | `4_DB_movability_check.py` | YOLO label | base_name (영어 라벨) |
| 4. 3D 생성 | `persistent_3d_worker.py` | image + mask (base64) | PLY (Gaussian Splat) |
| 5. 치수 계산 | `7_volume_calculate.py` | PLY 파일 | width, depth, height (OBB) |

### 3.2 YOLOE-seg 탐지

- **모델**: `yoloe-26x-seg.pt` (Objects365 기반, 365 classes)
- **출력**: bbox, class label, confidence score, segmentation mask
- **마스크 형식**: `np.ndarray`, shape `(H, W)`, dtype `uint8`, 값 `0`/`255`
- **CLAHE 전처리**: 저조도 이미지 대비 향상 (선택적)

### 3.3 Knowledge Base (DB 매칭)

- **24개 카테고리**: 에어컨, 냉장고, 소파, 침대, 식탁, 책상 등
- **동의어 매핑**: YOLO 365 클래스 → DB key (예: "Couch" → "sofa")
- **base_name**: 백엔드에 전달하는 영어 라벨
- **서브타입**: 일부 카테고리에 세부 유형 (예: sofa → THREE_SEATER_SOFA)

### 3.4 SAM-3D 3D 생성

- **방식**: Persistent Worker Pool (GPU당 1개 워커)
- **통신**: JSON stdin/stdout (TaskMessage → ResultMessage)
- **출력**: PLY (Gaussian Splat, Binary 형식)
- **Gaussian-only 모드**: GLB/Mesh 생성 스킵 (속도 37.4% 향상)

### 3.5 OBB 치수 계산

PLY 포인트 클라우드에서 Oriented Bounding Box를 계산:

1. PCA로 주축(eigenvectors) 추출
2. 점들을 주축으로 회전
3. 좌표계 기반 Greedy 매핑: X→width, Y→height, Z→depth

> AABB(축 정렬) 대신 OBB(방향 정렬)를 사용하는 이유: PLY가 회전되어 있어 AABB는 최대 300%+ 부정확

---

## 4. Callback 규약

### 4.1 Callback URL

```
POST https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback
```

- `{estimateId}`: 요청 시 전달받은 `estimate_id`
- URL은 `api/config.py`에 하드코딩

### 4.2 Callback 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| Timeout | 30초 | HTTP 요청 타임아웃 |
| Retry | 1회 | 실패 시 재시도 횟수 |

### 4.3 Callback Payload 구조

**성공:**

```json
{
  "results": [
    {
      "image_id": <int>,
      "objects": [
        {
          "label": <string>,
          "type": <string|null>,
          "width": <float>,
          "depth": <float>,
          "height": <float>
        }
      ]
    }
  ]
}
```

**실패:**

```json
{
  "error": "<에러 메시지>"
}
```

### 4.4 주의사항

- Callback은 비동기로 전송되며, 전송 실패 시 1회 재시도
- 모든 재시도 실패 시 로그에 기록되지만 별도 알림 없음
- `results` 배열의 순서는 요청의 `image_urls` 순서와 동일하지 않을 수 있음
- `image_id`로 매칭해야 함

---

## 5. 데이터 모델

### 5.1 DetectedObject (내부)

```python
@dataclass
class DetectedObject:
    id: int
    label: str                    # 영어 라벨 (base_name)
    db_key: str                   # Knowledge Base 키
    subtype_name: str | None      # 세부 유형 또는 None
    bbox: list[int]               # [x1, y1, x2, y2]
    center_point: list[float]     # [cx, cy]
    confidence: float
    crop_image: Image | None      # 크롭된 이미지
    mask_base64: str | None       # 마스크 Base64 PNG
    yolo_mask: np.ndarray | None  # 원본 세그멘테이션 마스크
    relative_dimensions: dict | None  # {bounding_box: {width, depth, height}}
```

### 5.2 PipelineResult (내부)

```python
@dataclass
class PipelineResult:
    image_id: str
    image_url: str
    objects: list[DetectedObject]
    processing_time_seconds: float
    status: str                   # "pending" | "completed" | "failed"
    error: str | None
    user_image_id: int | None     # 사용자 지정 이미지 ID
```

### 5.3 Worker Protocol

| 메시지 타입 | 방향 | 설명 |
|------------|------|------|
| `INIT` | Worker → Pool | 워커 초기화 완료 |
| `TASK` | Pool → Worker | 3D 생성 작업 요청 |
| `RESULT` | Worker → Pool | 작업 결과 (PLY base64) |
| `HEARTBEAT` | Worker → Pool | 상태 확인 |
| `SHUTDOWN` | Pool → Worker | 워커 종료 |

---

## 6. 성능 최적화 설정

`ai/subprocess/persistent_3d_worker.py` 상단에서 관리:

| 설정 | 현재 값 | 효과 |
|------|---------|------|
| `MAX_IMAGE_SIZE` | None (비활성화) | 부피 정확도 유지 |
| `STAGE1_INFERENCE_STEPS` | 14 | 속도/정확도 균형 (12~16 사이 최적값) |
| `STAGE2_INFERENCE_STEPS` | 4 | 치수 오차 0.5% 이내, 30% 속도 향상 |
| `USE_BINARY_PLY` | True | ~70% 파일 크기 감소, ~50% I/O 향상 |
| `GAUSSIAN_ONLY_MODE` | True | 37.4% 속도 향상 (GLB/Mesh 스킵) |
| `ENABLE_COMPILE` | True | torch.compile, 10-20% 추론 향상 |

---

## 7. Knowledge Base 카테고리 (24개)

| # | DB Key | 한국어명 | 서브타입 수 |
|---|--------|---------|-----------|
| 1 | air conditioner | 에어컨 | 3 |
| 2 | kitchen cabinet | 찬장 | - |
| 3 | drawer | 서랍장 | - |
| 4 | nightstand | 협탁 | - |
| 5 | bookshelf | 책장 | - |
| 6 | display shelf | 전시대/선반 | 1 |
| 7 | refrigerator | 냉장고 | 2 |
| 8 | wardrobe | 장롱/수납장 | 3 |
| 9 | sofa | 소파 | 4 |
| 10 | bed | 침대 | 6 |
| 11 | dining table | 식탁 | 3 |
| 12 | monitor | 모니터/TV | - |
| 13 | desk | 책상 | 3 |
| 14 | chair | 의자/스툴 | 2 |
| 15 | washing machine | 세탁기 | 2 |
| 16 | dryer | 건조기 | - |
| 17 | floor | 바닥 | - |
| 18 | potted plant | 화분/식물 | - |
| 19 | kimchi refrigerator | 김치냉장고 | - |
| 20 | vanity table | 화장대 | 2 |
| 21 | tv stand | TV 거치대 | 3 |
| 22 | piano | 피아노 | 3 |
| 23 | massage chair | 안마의자 | - |
| 24 | treadmill | 러닝머신 | - |
| 25 | exercise bike | 실내자전거 | - |
