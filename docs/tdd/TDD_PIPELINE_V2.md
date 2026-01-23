# Technical Design Document: Furniture Analysis Pipeline V2

## Document Info

| 항목 | 내용 |
|------|------|
| Version | 2.3 |
| Last Updated | 2026-01-23 |
| Author | AI Team |
| Status | Implemented |

---

## 1. Overview

### 1.1 Purpose

가구 분석 파이프라인 V2는 2D 이미지에서 가구를 탐지하고, 3D 모델을 생성하여 부피를 계산하는 AI 시스템입니다.

### 1.2 Key Changes from V1

| 항목 | V1 | V2 |
|------|----|----|
| 탐지 모델 | yolov8l-world.pt | yoloe-26x-seg.pt |
| 마스크 생성 | SAM2 (center point prompt) | YOLOE-seg (직접 사용) |
| 분류 | CLIP 분류 후 DB 매칭 | YOLO 클래스로 직접 DB 매칭 |
| 탐지 | SAHI 타일링 + YOLO-World | YOLOE-seg 단일 추론 |
| API 호출 | 3회 (YOLO → SAM2 → SAM-3D) | 2회 (YOLO → SAM-3D) |

### 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pipeline V2 Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Firebase   │     │   YOLOE-seg  │     │   SAM-3D     │        │
│  │   Storage    │────▶│   Detection  │────▶│   Convert    │        │
│  │              │     │   + Mask     │     │              │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  PIL Image   │     │  bbox, label │     │  PLY, GLB    │        │
│  │              │     │  mask (seg)  │     │  GIF preview │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                              │                    │                 │
│                              ▼                    ▼                 │
│                       ┌──────────────┐     ┌──────────────┐        │
│                       │  DB Matching │     │   Volume     │        │
│                       │  (한국어 라벨) │     │   Calculate  │        │
│                       └──────────────┘     └──────────────┘        │
│                                                   │                 │
│                                                   ▼                 │
│                                            ┌──────────────┐        │
│                                            │ JSON Response│        │
│                                            │ (dimensions) │        │
│                                            └──────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Input

| Field | Type | Description |
|-------|------|-------------|
| image_url | string | Firebase Storage URL |
| image_b64 | string | Base64 encoded image (alternative) |

### 2.2 Processing Stages

#### Stage 1: Image Fetch
```python
ImageFetcher.fetch_async(url) → PIL.Image
```

#### Stage 2: YOLOE-seg Detection
```python
YoloDetector.detect_smart(image, return_masks=True) → {
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": ["Bed", "Sofa", ...],
    "scores": [0.95, 0.87, ...],
    "masks": [np.ndarray, ...]  # (H, W), uint8, 0/255
}
```

#### Stage 3: DB Matching
```python
MovabilityChecker.check_from_label(label, score) → MovabilityResult {
    label: str,       # Korean label (한국어 라벨)
    db_key: str,      # FURNITURE_DB key
    confidence: float
}
```

#### Stage 4: Mask to Base64 (V2 신규)
```python
FurniturePipeline._yolo_mask_to_base64(mask) → str (base64 PNG)
```

#### Stage 5: SAM-3D Conversion
```python
SAM3DConverter.convert(image_path, mask_path) → SAM3DResult {
    success: bool,
    ply_b64: str,
    ply_size_bytes: int,
    gif_b64: str,
    mesh_url: str
}
```

#### Stage 6: Volume Calculation
```python
VolumeCalculator.calculate_from_ply(ply_path) → {
    "relative_width": float,
    "relative_height": float,
    "relative_depth": float
}
```

### 2.3 Output

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "width": 200.0,
          "depth": 90.0,
          "height": 85.0,
          "volume": 1.53
        },
        {
          "label": "table",
          "width": 120.0,
          "depth": 60.0,
          "height": 45.0,
          "volume": 0.324
        },
        {
          "label": "lamp",
          "width": 30.0,
          "depth": 30.0,
          "height": 150.0,
          "volume": 0.135
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [
        {
          "label": "chair",
          "width": 45.0,
          "depth": 50.0,
          "height": 90.0,
          "volume": 0.2025
        }
      ]
    }
  ]
}
```

### 2.4 Output Field Description

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| label | string | - | 탐지된 객체 라벨 (YOLO 클래스명) |
| width | float | 상대 길이 | 3D 메시 bounding box 너비 (모델 좌표계) |
| depth | float | 상대 길이 | 3D 메시 bounding box 깊이 (모델 좌표계) |
| height | float | 상대 길이 | 3D 메시 bounding box 높이 (모델 좌표계) |
| volume | float | 상대 부피 | bounding box 부피 (모델 좌표계, 절대 부피는 백엔드 계산) |

> **Note**: SAM-3D가 생성하는 3D 모델은 실제 물리적 크기 정보가 없습니다.
> 절대 부피/치수는 백엔드에서 Knowledge Base의 실제 치수와 비율을 조합하여 계산합니다.

---

## 3. Component Details

### 3.1 YOLOE-seg Detector

**File:** `ai/processors/2_YOLO_detect.py`

**Model:** `yoloe-26x-seg.pt` (Objects365 기반, 365 classes)

**Key Methods:**
- `detect_smart(image, return_masks=True)`: 통합 탐지 (bbox + mask)
- `_resize_mask(mask, target_size)`: 마스크 리사이징

**Mask Output Format:**
- Type: `np.ndarray`
- Shape: `(H, W)` - 원본 이미지 크기
- dtype: `uint8`
- Values: `0` (background), `255` (foreground)

### 3.2 SAM-3D Converter

**File:** `ai/processors/6_SAM3D_convert.py`

**Architecture:**
- Subprocess isolation (spconv GPU 충돌 방지)
- Worker script: `ai/subprocess/generate_3d_worker.py`

**Input Requirements:**
- Image: PNG file path
- Mask: Grayscale PNG file path (0/255)

**Output Files:**
- PLY: Gaussian splat point cloud (ASCII format)
- GLB: Textured 3D mesh
- GIF: 360° rotating preview

### 3.3 Furniture Pipeline

**File:** `ai/pipeline/furniture_pipeline.py`

**V2 Changes:**
```python
# V2: YOLOE-seg 마스크 직접 사용
if obj.yolo_mask is not None:
    mask_b64 = self._yolo_mask_to_base64(obj.yolo_mask)
    # SAM-3D에 직접 전달
    result = await self.generate_3d(image, mask_b64)
```

---

## 4. API Endpoints

### 4.1 POST /analyze-furniture

**Description:** Multi-image furniture analysis (Multi-GPU parallel, Callback 지원)

**Request:**
```json
{
  "estimate_id": 123,
  "image_urls": [
    {
      "id": 101,
      "url": "https://firebase-storage-url-1.jpg/"
    },
    {
      "id": 102,
      "url": "https://firebase-storage-url-2.jpg/"
    }
  ],
  "callback_url": "http://api.example.com/api/v1/estimates/{estimateId}/callback"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| estimate_id | int | Yes | 백엔드 견적 ID (callback URL 경로에 삽입됨) |
| image_urls | array[object] | Yes | Firebase Storage URL 객체 리스트 (1-20개) |
| image_urls[].id | int | Yes | 사용자 지정 이미지 ID |
| image_urls[].url | string | Yes | Firebase Storage URL |
| callback_url | string | No | 비동기 처리 완료 후 결과 POST할 URL (`{estimateId}` placeholder 지원) |

**동기 모드 (callback_url 없음):**
- 처리 완료 후 결과 직접 반환

**비동기 모드 (callback_url 있음):**
- 즉시 202 Accepted 반환
- 처리 완료 후 callback_url로 결과 POST

**Response:**
```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "width": 200.0,
          "depth": 90.0,
          "height": 85.0,
          "volume": 1.53
        },
        {
          "label": "table",
          "width": 120.0,
          "depth": 60.0,
          "height": 45.0,
          "volume": 0.324
        },
        {
          "label": "lamp",
          "width": 30.0,
          "depth": 30.0,
          "height": 150.0,
          "volume": 0.135
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [
        {
          "label": "chair",
          "width": 45.0,
          "depth": 50.0,
          "height": 90.0,
          "volume": 0.2025
        }
      ]
    }
  ]
}
```

> Response 필드 상세는 **Section 2.4** 참조

**비동기 모드 응답 (202 Accepted):**
```json
{
  "task_id": "uuid-...",
  "message": "Processing started",
  "status_url": "/analyze-furniture/status/uuid-..."
}
```

**Callback URL 변환:**
- 요청: `callback_url: "http://api.example.com/api/v1/estimates/{estimateId}/callback"`, `estimate_id: 123`
- 실제 전송: `POST http://api.example.com/api/v1/estimates/123/callback`

**Callback 페이로드:**
```json
{
  "task_id": "uuid-...",
  "status": "completed",
  "results": [...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| task_id | string | AI 서버에서 생성한 작업 ID |
| status | string | "completed" 또는 "failed" |
| results | array | 분석 결과 (위 Response와 동일 형식) |
| error | string | 실패 시 에러 메시지 (status=failed일 때만) |

### 4.1.1 GET /analyze-furniture/status/{task_id}

**Description:** 비동기 작업 상태 조회 (callback 실패 시 fallback)

**Response:**
```json
{
  "status": "completed",
  "created_at": "2026-01-23T12:00:00Z",
  "updated_at": "2026-01-23T12:05:00Z",
  "callback_url": "http://api.example.com/callback",
  "callback_sent": true,
  "results": [...]
}
```

### 4.2 POST /analyze-furniture-single

**Description:** Single image analysis

**Request:**
```json
{
  "image_url": "https://firebase-storage-url.jpg"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image_url | string | Yes | Firebase Storage URL (단일) |

**Response:** `/analyze-furniture`와 동일

### 4.3 POST /analyze-furniture-base64

**Description:** Base64 encoded image input (Firebase URL 없이 직접 이미지 전송)

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | string | Yes | Base64 인코딩된 이미지 |

**Response:** `/analyze-furniture`와 동일

### 4.4 POST /detect-furniture

**Description:** Detection only (no 3D, fast response)

**Request:**
```json
{
  "image_url": "https://firebase-storage-url.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "objects": [
    {
      "label": "box",
      "bbox": [100, 200, 300, 400],
      "center_point": [200, 300],
      "confidence": 0.95
    }
  ],
  "total_objects": 1,
  "processing_time_seconds": 0.5
}
```

> 3D 변환 없이 탐지만 수행하므로 `width`, `depth`, `height`, `volume` 필드 없음

### 4.5 GET /health

**Description:** 서버 상태 및 모델 로드 여부 확인

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "model": "facebook/sam2.1-hiera-large"
}
```

### 4.6 GET /gpu-status

**Description:** GPU 풀 상태 조회 (Multi-GPU 환경)

**Response:**
```json
{
  "total_gpus": 4,
  "available_gpus": 3,
  "pipelines_initialized": 4,
  "gpus": {
    "0": {
      "available": true,
      "task_id": null,
      "memory_used_mb": 1024,
      "has_pipeline": true
    },
    "1": {
      "available": false,
      "task_id": "image_processing",
      "memory_used_mb": 2048,
      "has_pipeline": true
    }
  }
}
```

### 4.7 POST /generate-3d

**Description:** 3D Gaussian Splat 생성 (비동기, task_id 반환)

**Request:**
```json
{
  "image": "base64_encoded_rgb_image",
  "mask": "base64_encoded_binary_mask",
  "seed": 42
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64 인코딩된 RGB 이미지 (PNG/JPEG) |
| mask | string | Yes | - | Base64 인코딩된 바이너리 마스크 (0-1 grayscale) |
| seed | integer | No | 42 | 랜덤 시드 (재현성) |

**Response:**
```json
{
  "success": true,
  "task_id": "uuid-task-id",
  "status": "queued"
}
```

### 4.8 GET /generate-3d-status/{task_id}

**Description:** 3D 생성 작업 상태 조회 (폴링)

**Path Parameter:**
| Field | Type | Description |
|-------|------|-------------|
| task_id | string | /generate-3d에서 반환받은 task_id |

**Response (Processing):**
```json
{
  "task_id": "uuid-task-id",
  "status": "processing",
  "progress": 50
}
```

**Response (Completed):**
```json
{
  "task_id": "uuid-task-id",
  "status": "completed",
  "progress": 100,
  "ply_b64": "base64_encoded_ply",
  "ply_size_bytes": 1234567,
  "gif_b64": "base64_encoded_gif",
  "gif_size_bytes": 234567,
  "mesh_url": "/assets/mesh_abc123.glb",
  "mesh_b64": "base64_encoded_glb",
  "mesh_size_bytes": 345678,
  "mesh_format": "glb"
}
```

**Response (Failed):**
```json
{
  "task_id": "uuid-task-id",
  "status": "failed",
  "progress": 0,
  "error": "Error message"
}
```

| Status | Description |
|--------|-------------|
| queued | 대기 중 |
| processing | 처리 중 |
| completed | 완료 |
| failed | 실패 |

### 4.9 GET /assets-list

**Description:** 저장된 에셋 파일 목록 조회 (최신순 정렬)

**Response:**
```json
{
  "files": [
    {
      "name": "mesh_abc123.glb",
      "size_bytes": 345678,
      "url": "/assets/mesh_abc123.glb",
      "created_at": "2026-01-18T12:34:56"
    }
  ],
  "total_files": 10,
  "total_size_bytes": 12345678
}
```

### 4.10 GET /assets/{filename}

**Description:** 정적 에셋 파일 다운로드 (PLY, GLB, GIF)

**Path Parameter:**
| Field | Type | Description |
|-------|------|-------------|
| filename | string | 파일명 (예: mesh_abc123.glb) |

**Response:** Binary file download

**Supported Formats:**
- `.ply` - Gaussian Splat Point Cloud
- `.glb` - 3D Mesh (GLTF Binary)
- `.gif` - 360° 회전 프리뷰

---

## 5. Multi-GPU Support

### 5.1 GPU Pool Manager

**File:** `ai/gpu/gpu_pool_manager.py`

**Features:**
- Round-robin GPU allocation
- Pipeline pre-initialization per GPU
- Thread-safe acquire/release

**Usage:**
```python
async with pool.pipeline_context(task_id="img_001") as (gpu_id, pipeline):
    result = await pipeline.process_single_image(url)
```

### 5.2 Pipeline Pre-initialization

```python
# At server startup
await pool.initialize_pipelines(
    lambda gpu_id: FurniturePipeline(device_id=gpu_id),
    skip_on_error=True
)
```

---

## 6. Error Handling

### 6.1 Detection Errors

| Error | Handling |
|-------|----------|
| No objects detected | Return empty list |
| Invalid bbox | Skip object |
| Mask generation failed | Skip 3D for this object |

### 6.2 SAM-3D Errors

| Error | Handling |
|-------|----------|
| Subprocess timeout | Return error status |
| Empty mask | Skip object |
| GLB export failed | Continue with PLY only |

### 6.3 Volume Calculation Errors

| Error | Handling |
|-------|----------|
| Invalid PLY | Return None dimensions |
| Zero volume | Use relative dimensions only |

---

## 7. Performance Metrics

### 7.1 V1 vs V2 Comparison

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| API Calls | 3 | 2 | 33% reduction |
| Mask Quality | Partial coverage | Full coverage | Improved |
| Processing Time | Baseline | -SAM2 time | Reduced |

### 7.2 Benchmarks (Single Image)

| Stage | Duration |
|-------|----------|
| Image Fetch | ~0.5s |
| YOLOE-seg Detection | ~1.0s |
| DB Matching | <0.1s |
| SAM-3D (per object) | ~20-30s |
| Volume Calculation | ~0.1s (워커 내부) |

### 7.3 Phase 4 최적화 (워커 내부 부피 계산)

| 항목 | 이전 | 최적화 후 | 개선 |
|------|------|----------|------|
| 부피 계산 위치 | 메인 프로세스 | 워커 내부 | - |
| 전송 데이터 | PLY base64 (10-20MB) | dimensions JSON (수 KB) | 99% 감소 |
| Base64 인코딩/디코딩 | ~2-3초/객체 | 0초 | 제거 |
| 메인 프로세스 I/O | 2회 | 0회 | 제거 |

---

## 8. Testing

### 8.1 Unit Tests

```bash
# YOLOE vs SAM2 mask comparison
python test_yoloe_vs_sam2_masks.py

# Pipeline V2 QA
python test_pipeline_qa.py
```

### 8.2 Integration Tests

```bash
# API server
python api.py &

# Test endpoint
curl -X POST http://localhost:8000/analyze-furniture-single \
  -H "Content-Type: application/json" \
  -d '{"image_url": "..."}'
```

### 8.3 QA Checklist

- [ ] YOLOE detection returns masks
- [ ] Masks are valid (>100 pixels, correct dimensions)
- [ ] DB matching works for all detected classes
- [ ] SAM-3D generates PLY/GLB/GIF
- [ ] Volume calculation returns valid dimensions

---

## 9. Dependencies

### 9.1 Python Packages

```
ultralytics>=8.0.0      # YOLOE-seg
torch>=2.0.0            # PyTorch
trimesh                 # Volume calculation
pillow                  # Image processing
aiohttp                 # Async HTTP client
```

### 9.2 External Services

- Firebase Storage (image hosting)
- sam-3d-objects (3D generation)

---

## 10. Changelog

### V2.4 (2026-01-23)

**Changes:**
- Phase 4 최적화: 워커 내부 부피 계산 구현
- PLY base64 전송 제거, dimensions JSON만 전송 (99% 데이터 감소)
- `ResultMessage.dimensions` 필드 추가
- `calculate_volume_from_ply()` 함수 추가 (워커 내부)
- ~20-30% 속도 향상 예상

**Files Modified:**
- `ai/subprocess/worker_protocol.py` - dimensions 필드 추가
- `ai/subprocess/persistent_3d_worker.py` - 워커 내부 부피 계산 구현
- `ai/pipeline/furniture_pipeline.py` - 워커 결과에서 dimensions 직접 사용

### V2.3 (2026-01-23)

**Changes:**
- `estimate_id` 필드 추가 (필수)
- Callback URL에 `{estimateId}` placeholder 지원
- YOLOE + SAM3D 병렬 초기화 (서버 시작 시 ~57초)

### V2.0 (2026-01-18)

**Major Changes:**
- YOLOE-seg 마스크를 SAM-3D에 직접 전달 (SAM2 제거)
- CLIP/SAHI 완전 제거
- is_movable/dimensions 필드 제거 (백엔드 계산)
- `_yolo_mask_to_base64()` 메서드 추가

**Rationale:**
- YOLOE-seg 마스크가 객체 전체를 더 정확하게 커버 (테스트 결과)
- API 호출 3회 → 2회로 감소
- 파이프라인 아키텍처 단순화

### V1.0 (Initial)

- YOLO-World + SAHI 타일링
- CLIP 분류
- SAM2 마스크 생성
- SAM-3D 변환
