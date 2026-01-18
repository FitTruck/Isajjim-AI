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
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "mesh_url": "/assets/mesh_abc123.glb"
    }
  ],
  "summary": {
    "total_objects": 10,
    "movable_objects": 8,
    "total_volume_liters": 15.5,
    "movable_volume_liters": 12.3
  }
}
```

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
  "image_urls": ["https://...", "https://..."],
  "enable_mask": true,
  "enable_3d": true
}
```

**Response:** See Section 2.3

### 4.2 POST /analyze-furniture-single

**Description:** Single image analysis

**Request:**
```json
{
  "image_url": "https://..."
}
```

### 4.3 POST /detect-furniture

**Description:** Detection only (no 3D, fast response)

**Response:**
```json
{
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
