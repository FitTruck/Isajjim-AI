# Technical Design Document: Furniture Analysis Pipeline V2

## Document Info

| 항목 | 내용 |
|------|------|
| Version | 2.0 |
| Last Updated | 2026-01-18 |
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
│                       │  is_movable  │     │   Calculate  │        │
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
    label: str,       # Korean label
    db_key: str,      # FURNITURE_DB key
    is_movable: bool,
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
    "relative_depth": float,
    "ratio": {"w": float, "h": float, "d": float}
}
```

### 2.3 Output

```json
{
  "objects": [
    {
      "label": "침대",
      "is_movable": true,
      "confidence": 0.95,
      "bbox": [100, 200, 500, 600],
      "center_point": [300, 400],
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "ratio": {"w": 0.75, "h": 1.0, "d": 0.225},
      "mesh_url": "/assets/mesh_abc123.glb"
    }
  ]
}
```

### 2.4 Output Field Description

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| label | string | - | 가구 한글 라벨 (예: "침대", "소파") |
| is_movable | boolean | - | 이동 가능 여부 |
| confidence | float | 0-1 | 탐지 신뢰도 |
| bbox | array[4] | px | 바운딩 박스 [x1, y1, x2, y2] |
| center_point | array[2] | px | 객체 중심점 [x, y] |
| width | float | mm | 가구 너비 |
| depth | float | mm | 가구 깊이 |
| height | float | mm | 가구 높이 |
| volume | float | liters | 부피 (리터) |
| ratio | object | - | 정규화된 비율 {"w", "h", "d"} |
| mesh_url | string | - | 3D 메쉬 파일 URL (GLB 형식) |

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
else:
    # Fallback: SAM2 (deprecated)
    mask_b64 = await self.generate_mask(image, obj.center_point)
```

---

## 4. API Endpoints

### 4.1 POST /analyze-furniture

**Description:** Multi-image furniture analysis (Multi-GPU parallel)

**Request:**
```json
{
  "image_urls": ["https://firebase-storage-url-1.jpg", "https://firebase-storage-url-2.jpg"],
  "enable_mask": true,
  "enable_3d": true,
  "max_concurrent": 4
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image_urls | array[string] | Yes | - | Firebase Storage URL 리스트 (1-20개) |
| enable_mask | boolean | No | true | 마스크 생성 여부 |
| enable_3d | boolean | No | true | 3D 모델 생성 여부 |
| max_concurrent | integer | No | null | 최대 동시 처리 이미지 수 |

**Response:**
```json
{
  "success": true,
  "objects": [
    {
      "label": "침대",
      "is_movable": true,
      "confidence": 0.95,
      "bbox": [100, 200, 500, 600],
      "center_point": [300, 400],
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "ratio": {"w": 0.75, "h": 1.0, "d": 0.225},
      "mesh_url": "/assets/mesh_abc123.glb"
    }
  ]
}
```

> Response 필드 상세는 **Section 2.4** 참조

### 4.2 POST /analyze-furniture-single

**Description:** Single image analysis

**Request:**
```json
{
  "image_url": "https://firebase-storage-url.jpg",
  "enable_mask": true,
  "enable_3d": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image_url | string | Yes | - | Firebase Storage URL (단일) |
| enable_mask | boolean | No | true | 마스크 생성 여부 |
| enable_3d | boolean | No | true | 3D 모델 생성 여부 |

**Response:** `/analyze-furniture`와 동일

### 4.3 POST /analyze-furniture-base64

**Description:** Base64 encoded image input (Firebase URL 없이 직접 이미지 전송)

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "enable_mask": true,
  "enable_3d": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64 인코딩된 이미지 |
| enable_mask | boolean | No | true | 마스크 생성 여부 |
| enable_3d | boolean | No | true | 3D 모델 생성 여부 |

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
      "label": "침대",
      "bbox": [100, 200, 500, 600],
      "confidence": 0.95,
      "is_movable": true
    }
  ]
}
```

> 3D 변환 없이 탐지만 수행하므로 `width`, `depth`, `height`, `volume`, `mesh_url` 필드 없음

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

### 4.7 POST /segment

**Description:** 단일 포인트 기반 세그멘테이션 (SAM2)

**Request:**
```json
{
  "image": "base64_encoded_image",
  "x": 300.0,
  "y": 400.0,
  "multimask_output": true,
  "mask_threshold": 0.0,
  "invert_mask": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64 인코딩된 이미지 |
| x | float | Yes | - | 포인트 X 좌표 |
| y | float | Yes | - | 포인트 Y 좌표 |
| multimask_output | boolean | No | true | 다중 마스크 반환 여부 |
| mask_threshold | float | No | 0.0 | 마스크 임계값 |
| invert_mask | boolean | No | false | 마스크 반전 여부 |

**Response:**
```json
{
  "masks": [[[0, 0, 255, ...]]],
  "scores": [0.95, 0.87, 0.72],
  "input_point": [300.0, 400.0],
  "image_shape": [1080, 1920]
}
```

### 4.8 POST /segment-binary

**Description:** 멀티 포인트 기반 세그멘테이션, 마스킹된 이미지 반환 (SAM2)

**Request:**
```json
{
  "image": "base64_encoded_image",
  "points": [
    {"x": 300.0, "y": 400.0},
    {"x": 350.0, "y": 450.0}
  ],
  "previous_mask": "base64_encoded_mask_optional",
  "mask_threshold": 0.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64 인코딩된 이미지 |
| points | array | Yes | - | 포인트 좌표 배열 [{x, y}, ...] |
| previous_mask | string | No | null | 이전 마스크 (누적 마스킹용) |
| mask_threshold | float | No | 0.0 | 마스크 임계값 |

**Response:**
```json
{
  "success": true,
  "mask_b64": "base64_encoded_masked_image_png",
  "score": 0.95
}
```

### 4.9 POST /generate-3d

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

### 4.10 GET /generate-3d-status/{task_id}

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

### 4.11 GET /assets-list

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

### 4.12 GET /assets/{filename}

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
| SAM-3D (per object) | ~30-60s |
| Volume Calculation | ~0.5s |

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
transformers            # SAM2 (deprecated in V2)
trimesh                 # Volume calculation
pillow                  # Image processing
aiohttp                 # Async HTTP client
```

### 9.2 External Services

- Firebase Storage (image hosting)
- sam-3d-objects (3D generation)

---

## 10. Changelog

### V2.0 (2026-01-16)

**Major Changes:**
- Removed SAM2 from main pipeline
- YOLOE-seg masks used directly for SAM-3D
- Added `_yolo_mask_to_base64()` method
- Deprecated `generate_mask()` method

**Rationale:**
- YOLOE-seg masks provide better object coverage (테스트 결과)
- Reduced API call overhead
- Simplified pipeline architecture

### V1.0 (Initial)

- YOLO-World + SAHI tiling
- CLIP classification
- SAM2 mask generation
- SAM-3D conversion
