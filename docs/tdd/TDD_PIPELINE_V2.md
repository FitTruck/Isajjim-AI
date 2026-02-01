# Technical Design Document: Furniture Analysis Pipeline V2

## Document Info

| 항목 | 내용 |
|------|------|
| Version | 2.5 |
| Last Updated | 2026-02-01 |
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
| 부피 계산 | 백엔드에서 계산 | **AI 서버에서 절대 부피 계산** (V2.5) |

### 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Pipeline V2.5 Architecture                       │
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
│  │  PIL Image   │     │  bbox, label │     │  PLY (3D)    │        │
│  │              │     │  mask (seg)  │     │              │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                              │                    │                 │
│                              ▼                    ▼                 │
│                       ┌──────────────┐     ┌──────────────┐        │
│                       │  DB Matching │     │  Dimension   │        │
│                       │ (base_name)  │     │  (OBB-based) │        │
│                       └──────────────┘     └──────────────┘        │
│                              │                    │                 │
│                              └────────┬──────────┘                 │
│                                       ▼                            │
│                              ┌──────────────────┐                  │
│                              │ Absolute Volume  │  ← V2.5 신규     │
│                              │   Calculator     │                  │
│                              │ (상대→절대 변환) │                  │
│                              └──────────────────┘                  │
│                                       │                            │
│                                       ▼                            │
│                              ┌──────────────────┐                  │
│                              │  JSON Response   │                  │
│                              │ (절대 치수+부피) │                  │
│                              └──────────────────┘                  │
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
    label: str,       # English label (base_name from Knowledge Base)
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

#### Stage 6: Dimension Calculation (OBB-based)
```python
DimensionCalculator.calculate_from_ply(ply_path) → {
    "bounding_box": {
        "width": float,        # OBB X-axis extent (이미지 가로) - 상대 치수
        "depth": float,        # OBB Z-axis extent (깊이) - 상대 치수
        "height": float        # OBB Y-axis extent (이미지 세로) - 상대 치수
    },
    "centroid": [x, y, z],
    "surface_area": float
}
```

**OBB (Oriented Bounding Box) 사용 이유:**
- PLY(Point Cloud)가 회전되어 있어서 AABB가 부정확한 치수 반환
- OBB는 객체의 실제 방향에 맞춘 정확한 치수 계산 (AABB 대비 최대 300%+ 정확도 향상)
- 좌표계 기반 Greedy 매핑: X→width, Y→height, Z→depth

#### Stage 7: Absolute Volume Calculation (V2.5 신규)
```python
AbsoluteVolumeCalculator.calculate_absolute_volume(
    label: str,           # "SOFA"
    type_name: str,       # "THREE_SEATER_SOFA" 또는 None
    rel_width: float,     # 상대 가로 (OBB 결과)
    rel_depth: float,     # 상대 세로
    rel_height: float     # 상대 높이
) → AbsoluteVolumeResult {
    matched_type: str,    # "THREE_SEATER_SOFA"
    width_mm: float,      # 1000.0 (mm)
    depth_mm: float,      # 3000.0 (mm)
    height_mm: float,     # 900.0 (mm)
    volume_m3: float      # 2.7 (m³)
}
```

**알고리즘 (Backend FurnitureDimensionConverter.java 포팅):**
1. 상대 치수 정렬: `[l1, l2, l3]` (l1 < l2 < l3)
2. 탐지 비율 계산: `detected_ratio = l2 / l3`
3. `type_name`이 없으면 비율로 best match 서브타입 찾기
4. 표준 장단변 추출: `long = max(width, depth)`, `short = min(width, depth)`
5. 높이 계산:
   - 고정 높이: `actual_height = standard_height`
   - 가변 높이 (height=-1): `scale_factor = long / l3`, `actual_height = l1 * scale_factor`
6. 부피 계산: `volume = short * long * height * 1e-9` (mm³ → m³)

**표준 치수 데이터:** `ai/data/furniture_dimensions.py`
- 52개 가구 타입 표준 치수 (FurnitureType.java 기반)
- 29개 라벨 → 서브타입 매핑 (FurnitureLabel.java 기반)

### 2.3 Output

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "SOFA",
          "type": "THREE_SEATER_SOFA",
          "width": 1000.0,
          "depth": 3000.0,
          "height": 900.0,
          "volume": 2.7
        },
        {
          "label": "DINING_TABLE",
          "type": "DEFAULT_DINING_TABLE",
          "width": 800.0,
          "depth": 1200.0,
          "height": 750.0,
          "volume": 0.72
        },
        {
          "label": "BED",
          "type": "SINGLE_BED",
          "width": 1000.0,
          "depth": 2000.0,
          "height": 500.0,
          "volume": 1.0
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [
        {
          "label": "CHAIR_STOOL",
          "type": "STANDARD_CHAIR",
          "width": 600.0,
          "depth": 600.0,
          "height": 1200.0,
          "volume": 0.432
        }
      ]
    }
  ]
}
```

### 2.4 Output Field Description

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| label | string | - | 탐지된 객체 라벨 (YOLO 클래스명 → base_name) |
| type | string | - | 매칭된 세부 유형 (예: "THREE_SEATER_SOFA") |
| width | float | mm | **절대 가로 길이** (표준 치수 기반) |
| depth | float | mm | **절대 세로 길이** (표준 치수 기반) |
| height | float | mm | **절대 높이** (고정 또는 스케일 팩터로 계산) |
| volume | float | m³ | **절대 부피** (width × depth × height × 1e-9) |

> **Note (V2.5 변경사항)**:
> - 절대 치수와 부피가 **AI 서버에서 계산**됩니다 (이전에는 백엔드).
> - `AbsoluteVolumeCalculator`가 상대 치수를 표준 가구 치수와 매칭하여 절대값으로 변환합니다.
> - 백엔드에서는 AI 응답의 `volume` 값을 바로 사용하며, fallback 로직만 유지합니다.
> - 가변 높이 가구 (침대 등)는 스케일 팩터로 높이를 역산합니다.

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

### 3.2 SAM-3D Worker Pool

**File:** `ai/gpu/sam3d_worker_pool.py`

**Architecture:**
- Persistent Worker Pool 패턴 (GPU당 1개 워커)
- Worker script: `ai/subprocess/persistent_3d_worker.py`
- JSON stdin/stdout 통신: `ai/subprocess/worker_protocol.py`

**Input Requirements:**
- Image: Base64 encoded PNG
- Mask: Base64 encoded grayscale PNG (0/255)

**Output Files:**
- PLY: Gaussian splat point cloud (Binary format, USE_BINARY_PLY=True)
- GLB/Mesh: 비활성화 (GAUSSIAN_ONLY_MODE=True)

**최적화 설정 (persistent_3d_worker.py):**
- `STAGE1_INFERENCE_STEPS=15`: 47% 속도 향상
- `STAGE2_INFERENCE_STEPS=8`: 15-20% 속도 향상
- `GAUSSIAN_ONLY_MODE=True`: 37.4% 속도 향상
- `compile=True`: 10-20% 추론 속도 향상

### 3.3 Absolute Volume Calculator (V2.5 신규)

**File:** `ai/processors/8_absolute_volume_calculate.py`

**Data File:** `ai/data/furniture_dimensions.py`

**Classes:**
- `AbsoluteVolumeResult`: 계산 결과 dataclass
- `AbsoluteVolumeCalculator`: 메인 계산기 클래스

**Key Methods:**
```python
class AbsoluteVolumeCalculator:
    def find_best_match(self, label, rel_w, rel_d, rel_h) -> str:
        """비율로 최적 서브타입 매칭"""

    def calculate_absolute_volume(self, label, type_name, rel_w, rel_d, rel_h) -> AbsoluteVolumeResult:
        """절대 치수 및 부피 계산"""
```

**Furniture Dimensions Data:**
- `FURNITURE_TYPES`: 52개 타입의 표준 치수 (width, depth, height in mm)
- `FURNITURE_LABELS`: 29개 라벨의 서브타입 리스트

**가변 높이 타입 (height=-1):**
- `SINGLE_BED`, `SUPER_SINGLE_BED`, `DOUBLE_BED`, `QUEEN_SIZE_BED`, `KING_SIZE_BED`, `BUNK_BED`

**전체 가변 타입 (-1,-1,-1):**
- `DEFAULT_DINING_TABLE`: 4인용 식탁 기준으로 스케일링

### 3.4 Furniture Pipeline

**File:** `ai/pipeline/furniture_pipeline.py`

**V2.5 Changes:**
```python
# V2: YOLOE-seg 마스크 직접 사용
if obj.yolo_mask is not None:
    mask_b64 = self._yolo_mask_to_base64(obj.yolo_mask)
    result = await self.generate_3d(image, mask_b64)

# V2.5: 절대 부피 계산
abs_calc = AbsoluteVolumeCalculator()
abs_result = abs_calc.calculate_absolute_volume(
    label=obj.label,
    type_name=obj.subtype_name,
    rel_width=bbox["width"],
    rel_depth=bbox["depth"],
    rel_height=bbox["height"]
)
# 응답에 절대 치수(mm) + 부피(m³) 포함
```

---

## 4. API Endpoints

### 4.1 POST /analyze-furniture

**Description:** Multi-image furniture analysis (Multi-GPU parallel)

**Request:**
```json
{
  "estimate_id": 1,
  "image_urls": [
    {
      "id": 101,
      "url": "https://firebase-storage-url-1.jpg/"
    },
    {
      "id": 102,
      "url": "https://firebase-storage-url-2.jpg/"
    }
  ]
}

```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| estimate_id | int | Yes | 견적 ID (callback URL에 사용) |
| image_urls | array[object] | Yes | 이미지 URL 객체 배열 (1-20개) |
| image_urls[].id | int | Yes | 사용자 지정 이미지 ID |
| image_urls[].url | string | Yes | Firebase Storage URL |

**Immediate Response (비동기 방식):**
```json
{
  "success": true,
  "estimate_id": 1,
  "status": "processing"
}
```

작업은 백그라운드에서 실행되며, 완료 시 callback URL로 결과가 전송됩니다.

**Callback URL:** `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback`

**Callback Payload (성공):**
```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "SOFA",
          "type": "THREE_SEATER_SOFA",
          "width": 1000.0,
          "depth": 3000.0,
          "height": 900.0,
          "volume": 2.7
        },
        {
          "label": "BED",
          "type": "SINGLE_BED",
          "width": 1000.0,
          "depth": 2000.0,
          "height": 500.0,
          "volume": 1.0
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [
        {
          "label": "CHAIR_STOOL",
          "type": "STANDARD_CHAIR",
          "width": 600.0,
          "depth": 600.0,
          "height": 1200.0,
          "volume": 0.432
        }
      ]
    }
  ]
}
```

> **Note**: 모든 치수는 **절대값(mm)**, 부피는 **m³** 단위입니다.

**Callback Payload (실패):**
```json
{
  "error": "Furniture analysis failed: 에러 메시지"
}
```

> Response 필드 상세는 **Section 2.4** 참조

**Callback URL (하드코딩):**
`https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback`

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
| YOLOE-seg Detection | ~0.5-1.0s |
| DB Matching | <0.1s |
| SAM-3D (per object, optimized) | ~7-8s |
| Volume Calculation | ~0.5s |

**최적화 적용 (2026-01-25):**
- `STAGE1_INFERENCE_STEPS=15`: 47% 속도 향상, 1.31% 부피 오차
- `STAGE2_INFERENCE_STEPS=8`: 15-20% 속도 향상
- `GAUSSIAN_ONLY_MODE=True`: 37.4% 속도 향상
- `compile=True`: 10-20% 추론 속도 향상
- `in_place=True`: 5-10% 속도/메모리 향상

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
fastapi                 # API 프레임워크
uvicorn[standard]       # ASGI 서버
pydantic                # 요청/응답 모델
torch>=2.1.0            # PyTorch (DeCl requires >=2.1.0)
torchvision>=0.16.0     # 이미지 처리
ultralytics>=8.3.0      # YOLOE-seg 지원
trimesh                 # 3D 메시/볼륨 계산
pillow>=10.0.0          # 이미지 처리
aiohttp                 # Async HTTP client (Firebase URL)
omegaconf>=2.3.0        # SAM-3D 설정
hydra-core>=1.3.2       # SAM-3D 설정
```

### 9.2 External Services

- Firebase Storage (image hosting)
- sam-3d-objects (3D generation)

---

## 10. Changelog

### V2.5 (2026-02-01)

**Absolute Volume Calculation Migration:**
- 절대 부피 계산 로직을 **백엔드에서 AI 서버로 이전**
- 상대 치수 계산 직후 절대 치수(mm)와 부피(m³)를 함께 계산

**New Files:**
- `ai/data/furniture_dimensions.py`: 52개 가구 타입 표준 치수 데이터
- `ai/processors/8_absolute_volume_calculate.py`: AbsoluteVolumeCalculator 클래스

**API Response Changes:**
- `width`, `depth`, `height`: 상대 치수 → **절대 치수 (mm)**
- `volume`: 새로 추가 → **절대 부피 (m³)**

**Backend Changes:**
- `FurnitureService.java`: AI 응답의 `volume > 0`이면 바로 사용
- `FurnitureDimensionConverter.java`: Fallback용으로 유지

**알고리즘 (Backend FurnitureDimensionConverter.java 포팅):**
1. 상대 치수 정렬 후 탐지 비율 계산
2. `type_name`이 없으면 비율로 best match 서브타입 찾기
3. 표준 장단변 추출
4. 높이 계산: 고정 높이 사용 또는 스케일 팩터로 역산
5. 부피 계산: `volume = short * long * height * 1e-9`

### V2.3 (2026-01-26)

**Volume Calculation Updates:**
- AABB → OBB (Oriented Bounding Box)로 변경
  - 회전된 3D 객체도 정확한 치수 계산 (AABB 대비 최대 300%+ 정확도 향상)
  - 좌표계 기반 Greedy 매핑: X→width, Y→height, Z→depth
  - trimesh.bounding_box_oriented 사용
- 한국어 라벨 제거 → 영어 라벨 (base_name) 사용

### V2.2 (2026-01-25)

**Optimization Updates:**
- `STAGE1_INFERENCE_STEPS`: 25 → 15 (47% 속도 향상, 1.31% 부피 오차)
- `GAUSSIAN_ONLY_MODE`: True 활성화 (37.4% 속도 향상)
- `compile=True`: torch.compile 활성화 (10-20% 추론 속도 향상)
- `in_place=True`: deepcopy 제거 (5-10% 속도/메모리 향상)

**File Structure Updates:**
- `api/routes/generate_3d.py` 제거 (Worker Pool 방식으로 통합)
- `api/services/tasks.py` 제거 (callback 방식으로 대체)
- `ai/processors/6_SAM3D_convert.py` 제거 (Worker Pool 사용)
- `ai/subprocess/generate_3d_worker.py` 제거 (persistent_3d_worker.py로 대체)

### V2.1 (2026-01-21)

**Performance Metrics:**
- 8 GPU, 8 이미지, 101 객체: ~3분 47초 (객체당 2.24초)

### V2.0 (2026-01-18)

**Major Changes:**
- YOLOE-seg 마스크를 SAM-3D에 직접 전달 (SAM2 제거)
- CLIP/SAHI 완전 제거
- is_movable/dimensions 필드 제거 (백엔드 계산)
- `_yolo_mask_to_base64()` 메서드 추가
- Persistent Worker Pool 아키텍처 도입
- 비동기 callback 패턴 도입

**Rationale:**
- YOLOE-seg 마스크가 객체 전체를 더 정확하게 커버 (테스트 결과)
- API 호출 3회 → 2회로 감소
- 파이프라인 아키텍처 단순화

### V1.0 (Initial)

- YOLO-World + SAHI 타일링
- CLIP 분류
- SAM2 마스크 생성
- SAM-3D 변환
