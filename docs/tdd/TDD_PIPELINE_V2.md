# Technical Design Document: Furniture Analysis Pipeline V2

## Document Info

| 항목 | 내용 |
|------|------|
| Version | 2.6 |
| Last Updated | 2026-04-09 |
| Author | AI Team |
| Status | Implemented |

---

## 목차

1. [Overview](#1-overview)
   - 1.1 Purpose
   - 1.2 Key Changes from V1
   - 1.3 Architecture Diagram
2. [Data Flow](#2-data-flow)
   - 2.1 Input
   - 2.2 Processing Stages (Stage 1 ~ 8)
     - Stage 1: Image Fetch
     - Stage 2: YOLOE-seg Detection
     - Stage 3: DB Matching
     - Stage 4: Mask to Base64
     - Stage 5: SAM-3D 3D 생성 (Persistent Worker Pool)
     - Stage 6: Dimension Calculation (OBB-based)
     - Stage 7: Absolute Volume Calculation (V2.5 신규)
     - Stage 8: PLY 전처리 + GCS 업로드 (V2.5 신규)
   - 2.3 Output (JSON)
   - 2.4 Output Field Description
3. [Component Details](#3-component-details)
   - 3.1 YOLOE-seg Detector
   - 3.2 SAM-3D Worker Pool (Event 기반 work-stealing + VRAM 최적화)
   - 3.3 Absolute Volume Calculator (V2.5 신규)
   - 3.4 Furniture Pipeline
4. [API Endpoints](#4-api-endpoints)
   - 4.1 POST `/analyze-furniture` (비동기 callback)
   - 4.2 POST `/analyze-furniture-single`
   - 4.3 POST `/analyze-furniture-base64` (동기)
   - 4.4 POST `/detect-furniture`
   - 4.5 GET `/health`
   - 4.6 GET `/gpu-status`
   - 4.7 GET `/assets-list`
   - 4.8 GET `/assets/{filename}` (StaticFiles mount)
5. [Multi-GPU Support (2단계 병렬 처리)](#5-multi-gpu-support-2단계-병렬-처리)
   - 5.1 1단계: GPU Pool Manager (YOLOE, 라운드로빈)
   - 5.2 2단계: SAM3D Worker Pool (Event 기반 work-stealing)
   - 5.3 Pipeline Pre-initialization
6. [Error Handling](#6-error-handling)
   - 6.1 Detection Errors
   - 6.2 SAM-3D Errors
   - 6.3 Volume Calculation Errors
7. [Performance Metrics](#7-performance-metrics)
   - 7.1 V1 vs V2 Comparison
   - 7.2 Benchmarks (Single Image, L4 GPU)
8. [Testing](#8-testing)
   - 8.1 Unit Tests
   - 8.2 Integration Tests
   - 8.3 QA Checklist
9. [Dependencies](#9-dependencies)
   - 9.1 Python Packages
   - 9.2 External Services
10. [Changelog](#10-changelog)
    - V2.6 (2026-04-08) — Dead code 제거 및 문서 정합성 개선
    - V2.5 (2026-02-02) — Absolute Volume + PLY GCS 업로드
    - V2.3 (2026-01-26) — AABB → OBB 전환
    - V2.2 (2026-01-25) — SAM-3D 추론 최적화 (Gaussian-only, torch.compile)
    - V2.1 (2026-01-21) — Multi-GPU 벤치마크
    - V2.0 (2026-01-18) — SAM2/CLIP/SAHI 제거, Persistent Worker Pool
    - V1.0 (Initial)

> **관련 문서**: 최적화 기법의 정의/원리와 구현 세부사항은 [`PIPELINE_OPTIMIZATION.md`](../PIPELINE_OPTIMIZATION.md) 참고

---

## 1. Overview

### 1.1 Purpose

가구 분석 파이프라인 V2는 2D 이미지에서 가구를 탐지하고, 3D 모델을 생성하여 부피를 계산하는 AI 시스템입니다.

### 1.2 Key Changes from V1

| 항목 | V1 | V2.5 (현재) |
|------|----|------------|
| 탐지 모델 | yolov8l-world.pt | yoloe-26x-seg.pt |
| 마스크 생성 | SAM2 (center point prompt) | YOLOE-seg (직접 사용) |
| 분류 | CLIP 분류 후 DB 매칭 | YOLO 클래스로 직접 DB 매칭 |
| 탐지 | SAHI 타일링 + YOLO-World | YOLOE-seg 단일 추론 |
| API 호출 | 3회 (YOLO → SAM2 → SAM-3D) | 2회 (YOLO → SAM-3D) |
| 부피 계산 | 백엔드에서 계산 | **AI 서버에서 절대 부피 계산** (V2.5) |
| PLY 저장 | 로컬 파일 | **GCS 업로드 후 `ply_url` 반환** (V2.5) |
| SAM3D 스케줄링 | 매 요청마다 subprocess | **Persistent Worker Pool + Event 기반 work-stealing** |
| 3D 출력 포맷 | PLY + GLB + Mesh + GIF | **PLY만** (Gaussian-only 모드) |
| is_movable / dimensions | AI가 결정 | 제거 (모든 탐지 객체 이동 대상) |

### 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Pipeline V2.5 Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐   │
│  │   Firebase   │     │   GPUPoolManager │     │ SAM3DWorkerPool   │   │
│  │   Storage    │────▶│  (YOLOE-seg,     │────▶│ (Persistent +     │   │
│  │              │     │   round-robin)   │     │  work-stealing)   │   │
│  └──────────────┘     └──────────────────┘     └───────────────────┘   │
│         │                      │                         │              │
│         ▼                      ▼                         ▼              │
│  ┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐   │
│  │  PIL Image   │     │  bbox, label,    │     │  Binary PLY       │   │
│  │              │     │  mask (seg)      │     │  (Gaussian Splat) │   │
│  └──────────────┘     └──────────────────┘     └───────────────────┘   │
│                                │                         │              │
│                                ▼                         ▼              │
│                       ┌──────────────────┐     ┌───────────────────┐   │
│                       │  DB Matching     │     │  OBB Dimension    │   │
│                       │  (base_name)     │     │  (상대 치수)       │   │
│                       └──────────────────┘     └───────────────────┘   │
│                                │                         │              │
│                                └────────┬───────────────┘              │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │ AbsoluteVolume       │  ← V2.5           │
│                              │ Calculator           │                   │
│                              │ (52 furniture DB)    │                   │
│                              └──────────────────────┘                   │
│                                         │                               │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │ PLYPreprocessor      │  ← V2.5           │
│                              │ (OBB 정렬 + Y-up +    │                   │
│                              │  72k 다운샘플링)      │                   │
│                              └──────────────────────┘                   │
│                                         │                               │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │ GCS Upload           │  ← V2.5           │
│                              │ → ply_url            │                   │
│                              └──────────────────────┘                   │
│                                         │                               │
│                                         ▼                               │
│                              ┌──────────────────────┐                   │
│                              │ Callback URL         │                   │
│                              │ (비동기 POST)         │                   │
│                              └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
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

#### Stage 5: SAM-3D 3D 생성 (Persistent Worker Pool)
```python
# SAM3DWorkerPool이 GPU당 하나의 persistent 워커를 관리
# Event 기반 work-stealing으로 여러 객체를 병렬 처리
sam3d_pool = get_sam3d_worker_pool()
tasks = [{"task_id": f"obj_{i}", "image_b64": ..., "mask_b64": ..., "seed": 42}, ...]
results: List[ResultMessage] = await sam3d_pool.submit_tasks_parallel(tasks)

# ResultMessage (Gaussian-only 모드)
ResultMessage {
    task_id: str,
    success: bool,
    ply_b64: str,           # Binary PLY 포맷 (base64)
    ply_size_bytes: int,
    error: Optional[str],
    processing_time_seconds: float
}
```
> **Note**: V2.2부터 GLB/Mesh 생성이 제거되었고, V2.5 현재는 Gaussian-only 모드에서 PLY만 생성합니다.

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

#### Stage 8: PLY 전처리 + GCS 업로드 (V2.5 신규)
```python
# PLYPreprocessor: 프론트엔드 렌더링 + 용량 최적화
preprocessor = PLYPreprocessor(
    max_points=Config.PLY_MAX_POINTS,         # 72000
    convert_to_yup=Config.PLY_CONVERT_TO_YUP, # True (Three.js 호환)
    enable_alignment=True,                    # OBB 기반 축 정렬
    enable_scaling=True,                      # 절대 치수(mm → m) 스케일링
    enable_downsampling=True,                 # Stride 다운샘플링
)
processed_ply_b64, preprocess_result = preprocessor.process(
    ply_b64=raw_ply_b64,
    target_width_mm=abs_result.width_mm,
    target_depth_mm=abs_result.depth_mm,
    target_height_mm=abs_result.height_mm,
)

# GCS 업로드
ply_url = await self.gcs_service.upload_ply_base64(processed_ply_b64, filename)
```

**전처리 파이프라인:**
1. OBB 기반 축 정렬 (`OBB.R.T` 역회전)
2. 바닥 배치 (`Z-min = 0`)
3. Z-up → Y-up 좌표계 변환
4. 절대 치수(mm → m) 스케일링
5. Stride 다운샘플링 (max 72,000 points)

**파일 크기:** ~2MB → ~290KB (85% 감소)

**GCS 경로:** `ply/est{estimate_id}_img{image_id}_{label}_{timestamp}_{uuid}.ply`

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
          "volume": 2.7,
          "ply_url": "https://storage.isajjim.kr/ply/sofa.ply",
          "center_x": 43.2,
          "center_y": 234.2  //이미지에서 객체의 중심 위치 
        },
        {
          "label": "DINING_TABLE",
          "type": "DEFAULT_DINING_TABLE",
          "width": 800.0,
          "depth": 1200.0,
          "height": 750.0,
          "volume": 0.72,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
        },
        {
          "label": "BED",
          "type": "SINGLE_BED",
          "width": 1000.0,
          "depth": 2000.0,
          "height": 500.0,
          "volume": 1.0,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
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
          "volume": 0.432,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
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
| ply_url | string | - | GCS Public URL (전처리 완료된 PLY 파일, V2.5 신규) |
| center_x | float | px | 이미지 내 객체 중심의 X 좌표 |
| center_y | float | px | 이미지 내 객체 중심의 Y 좌표 |

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
- Persistent Worker Pool 패턴 (GPU당 1개 워커, 서버 시작 시 모델 1회 로드)
- Worker script: `ai/subprocess/persistent_3d_worker.py`
- JSON stdin/stdout 통신: `ai/subprocess/worker_protocol.py`
- **Event 기반 work-stealing 스케줄링**: `asyncio.Event` 시그널로 먼저 끝난 워커가
  다음 작업을 즉시 획득. 객체 크기가 불균등할 때 (TV 5초, Bed 17초) GPU 유휴 시간 최소화.
- SAM-3D 전용 conda 환경 Python (`~/miniconda3/envs/sam3d-objects/bin/python`) 사용

**Input Requirements:**
- Image: Base64 encoded PNG
- Mask: Base64 encoded grayscale PNG (0/255)

**Output:**
- PLY: Gaussian splat point cloud (Binary format, `USE_BINARY_PLY=True`)
- GLB/Mesh/GIF/비디오 생성 없음 (Gaussian-only 모드)

**최적화 설정 (`ai/subprocess/persistent_3d_worker.py`):**

| 설정 | 값 | 효과 |
|------|-----|------|
| `MAX_IMAGE_SIZE` | `None` (비활성화) | 부피 정확도 유지 (다운샘플링이 91.7% 영향) |
| `STAGE1_INFERENCE_STEPS` | `14` (기본 25) | ~50% 속도 향상, ~1.5% 부피 오차 |
| `STAGE2_INFERENCE_STEPS` | `4` (기본 12) | ~30% 속도 향상, 치수 오차 0.5% 이내 |
| `USE_BINARY_PLY` | `True` | 파일 크기 ~70% 감소, 쓰기 속도 ~50% 향상 |
| `GAUSSIAN_ONLY_MODE` | `True` | 37.4% 속도 향상, 부피 오차 0.005% |
| `ENABLE_SS_STEP_CACHING` | `True` (stride=3, warmup=2) | SS backbone 호출 42→18 (57% 감소) |
| `ENABLE_SLAT_STEP_CACHING` | `False` | 4 steps에서 얇은 객체 품질 리스크 |
| `ENABLE_COMPILE` | `True` (reduce-overhead) | 핵심 모듈 수동 compile, ~10-20% 속도 향상 |
| `in_place=True` | - | deepcopy 제거, ~5-10% 속도 향상 |

**VRAM 최적화 (Gaussian-only 모드 조건부 언로드):**
- `slat_decoder_mesh` → CPU (~3-4GB 절감)
- `slat_decoder_gs_4` → CPU (~2-3GB 절감)
- `depth_model` (MoGe) → CPU + None (~1-3GB 절감, synthetic pointmap 사용)
- **총 ~10GB 절감** (21GB → 11.25GB, L4 GPU에서 YOLOE + SAM-3D 동일 GPU 탑재 가능)

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
          "volume": 2.7,
          "ply_url": "https://storage.isajjim.kr/ply/sofa.ply",
          "center_x": 43.2,
          "center_y": 234.2
        },
        {
          "label": "DINING_TABLE",
          "type": "DEFAULT_DINING_TABLE",
          "width": 800.0,
          "depth": 1200.0,
          "height": 750.0,
          "volume": 0.72,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
        },
        {
          "label": "BED",
          "type": "SINGLE_BED",
          "width": 1000.0,
          "depth": 2000.0,
          "height": 500.0,
          "volume": 1.0,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
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
          "volume": 0.432,
          "ply_url": "https://storage.isajjim.kr/ply/{label}.ply",
          "center_x": 43.2,
          "center_y": 234.2
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

**Description:** Base64 encoded image input (Firebase URL 없이 직접 이미지 전송, 동기 방식)

**Request:**
```json
{
  "image": "iVBORw0KGgo...",
  "enable_mask": true,
  "enable_3d": true,
  "return_ply": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| image | string | Yes | - | Base64 인코딩된 이미지 |
| enable_mask | bool | No | `true` | YOLOE-seg 마스크 생성 여부 |
| enable_3d | bool | No | `true` | SAM-3D 3D 생성 여부 |
| return_ply | bool | No | `false` | 응답에 PLY base64 데이터 포함 여부 (테스트용) |

**Response:**
```json
{
  "objects": [
    {
      "label": "SOFA",
      "subtype": "THREE_SEATER_SOFA",
      "width": 1.5,
      "depth": 0.8,
      "height": 0.9,
      "volume": 1.08
    }
  ]
}
```

> **Note**: 이 엔드포인트는 동기 방식이며 상대 치수를 반환합니다 (`/analyze-furniture`의 비동기 callback과 다름).
> `return_ply=true`일 때 각 객체에 `ply_b64` 필드가 추가됩니다.

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

**Description:** 서버 상태 확인 (간단한 liveness 체크)

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0"
}
```

> 모델 로드 여부 / GPU 상세 정보는 `/gpu-status` 참고

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

> **Note**: 이전 버전에 존재하던 `POST /generate-3d`와 `GET /generate-3d-status/{task_id}` 폴링 기반
> 엔드포인트는 V2.2에서 제거되었습니다. 3D 생성은 이제 `/analyze-furniture*` 엔드포인트 내부에서
> SAM3D Worker Pool을 통해 동기/callback 방식으로 처리됩니다.

### 4.7 GET /assets-list

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

### 4.8 GET /assets/{filename}

**Description:** 정적 에셋 파일 다운로드 (StaticFiles mount, `api/app.py:33`)

**Path Parameter:**
| Field | Type | Description |
|-------|------|-------------|
| filename | string | 파일명 (예: `obj_abc123.ply`) |

**Response:** Binary file download

**Supported Formats:**
- `.ply` - Gaussian Splat Point Cloud (현재 유일하게 생성되는 포맷)

> **Note**: V2.5 현재는 PLY만 생성하며, 생성된 PLY는 GCS에 업로드되어 `ply_url`로 제공됩니다.
> `/assets/` 는 로컬 static file mount로 남아있지만 실제 파이프라인은 GCS를 사용합니다.

---

## 5. Multi-GPU Support (2단계 병렬 처리)

파이프라인은 **두 단계에서 각각 다른 수준의 병렬성**을 사용합니다.

### 5.1 1단계: GPU Pool Manager (YOLOE 탐지)

**File:** `ai/gpu/gpu_pool_manager.py`

**Features:**
- 라운드로빈 GPU 할당 (`_next_gpu_index` 카운터)
- Pipeline pre-initialization per GPU (서버 시작 시 YOLOE 모델 로드)
- Thread-safe acquire/release (`asyncio.Lock`)
- Health check 및 자동 failover
- 획득 실패 시 `asyncio.sleep(0.5)` 폴링 재시도

**Usage:**
```python
async with pool.pipeline_context(task_id="img_001") as (gpu_id, pipeline):
    result = await pipeline.process_single_image(url)
```

### 5.2 2단계: SAM3D Worker Pool (3D 생성)

**File:** `ai/gpu/sam3d_worker_pool.py`

**Features:**
- GPU당 하나의 persistent 워커 프로세스 (서버 시작 시 1회 spawn)
- **Event 기반 work-stealing 스케줄링** (`asyncio.Event` + `_worker_available.set()`)
- 모델 사전 로드 (매 요청마다 모델 로딩 오버헤드 0)
- JSON stdin/stdout 통신 프로토콜

**Work-Stealing 동작:**
```python
# ai/gpu/sam3d_worker_pool.py:307-348
async def _acquire_worker(self, task_id):
    while time.time() - start_time < self.task_timeout:
        async with self._allocation_lock:
            for _ in range(len(self.gpu_ids)):
                # 빈 워커가 있으면 즉시 획득
                if worker_info.is_ready and not worker_info.is_busy:
                    return worker_info
        # 빈 워커 없으면 Event 시그널 대기 (폴링 X)
        await asyncio.wait_for(self._worker_available.wait(), timeout=remaining)
```
워커가 작업을 끝내면 `_release_worker()`가 `_worker_available.set()`을 호출 → 대기 중인 코루틴이
즉시 깨어나서 다음 작업 획득. 객체 크기가 불균등할 때 GPU 유휴 시간 최소화.

### 5.3 Pipeline Pre-initialization

```python
# At server startup (api/app.py:42-65)
gpu_ids = Config.get_available_gpus()

# 1단계: YOLOE Pipeline 사전 초기화
pool = initialize_gpu_pool(gpu_ids)
await pool.initialize_pipelines(
    lambda gpu_id: FurniturePipeline(device_id=gpu_id),
    skip_on_error=True
)

# 2단계: SAM3D Worker Pool 초기화 (GPU당 1개 워커 spawn + 모델 로드)
sam3d_pool = await initialize_sam3d_worker_pool(gpu_ids)
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
| Worker subprocess timeout (`task_timeout=300s`) | ResultMessage(success=False, error="Task timeout") |
| Worker process died | 다음 요청 시 자동 재시작 시도 |
| Empty mask | ValueError raise → ResultMessage 실패 응답 |
| PLY 저장/후처리 실패 | stderr에 로깅 후 success=False로 반환 |

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

### 7.2 Benchmarks (Single Image, L4 GPU)

| Stage | Duration |
|-------|----------|
| Image Fetch | ~0.5s |
| YOLOE-seg Detection | ~0.5-1.0s |
| DB Matching | <0.1s |
| SAM-3D (객체당, Fast-SAM3D 적용 후) | **~13s** (Bed) / ~5s (TV) |
| Dimension + Absolute Volume | <0.5s |
| PLY 전처리 + GCS 업로드 | ~0.5-1s |

**현재 적용된 최적화 (2026-03-18 기준):**
- `STAGE1_INFERENCE_STEPS=14`: ~50% 속도 향상, ~1.5% 부피 오차
- `STAGE2_INFERENCE_STEPS=4`: ~30% 속도 향상, 치수 오차 0.5% 이내
- `GAUSSIAN_ONLY_MODE=True`: 37.4% 속도 향상, 부피 오차 0.005%
- `ENABLE_SS_STEP_CACHING=True` (stride=3, warmup=2): SS backbone 42→18 호출 (57% 감소)
- `ENABLE_COMPILE=True` (reduce-overhead): Bed 25.6s→18.7s (1.37x), TV 10.0s→7.6s (1.31x)
- `USE_BINARY_PLY=True`: 파일 크기 ~70% 감소, 쓰기 ~50% 빠름
- `in_place=True`: deepcopy 제거, ~5-10% 속도/메모리 향상
- VRAM 모델 언로드: 21GB → 11.25GB (48% 절감)

자세한 내용은 `docs/PIPELINE_OPTIMIZATION.md` 참고.

---

## 8. Testing

### 8.1 Unit Tests

```bash
# 전체 테스트 실행
pytest -v

# 주요 테스트 파일
pytest tests/test_ply_preprocessor.py -v            # PLY 전처리 파이프라인
pytest tests/test_furniture_pipeline_unit.py -v     # Furniture Pipeline 단위 테스트
pytest tests/test_volume_calculator_real.py -v      # 실제 PLY 파일로 부피 계산 검증

# 커버리지 포함
pytest --cov=ai --cov-report=term-missing
```

### 8.2 Integration Tests

```bash
# API 서버 실행
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Base64 이미지 엔드포인트 테스트 (동기)
curl -X POST http://localhost:8000/analyze-furniture-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "enable_mask": true, "enable_3d": true}'

# Firebase URL 엔드포인트 테스트 (비동기 callback)
curl -X POST http://localhost:8000/analyze-furniture \
  -H "Content-Type: application/json" \
  -d '{"estimate_id": 1, "image_urls": [{"id": 101, "url": "https://..."}]}'

# GPU 상태 확인
curl http://localhost:8000/gpu-status
```

### 8.3 QA Checklist

- [ ] YOLOE-seg detection returns masks (`detect_smart()` 앙상블 + CLAHE)
- [ ] Masks are valid (>100 pixels, 원본 이미지 크기)
- [ ] DB matching works for all detected classes (`knowledge_base.py`)
- [ ] SAM-3D Worker Pool generates Binary PLY (`GAUSSIAN_ONLY_MODE=True`)
- [ ] Work-stealing 분배 확인 (`/gpu-status`로 busy 워커 모니터링)
- [ ] Dimension calculation returns OBB 기반 상대 치수
- [ ] AbsoluteVolumeCalculator가 52개 타입 DB로 절대 치수/부피 산출
- [ ] PLY 전처리 (OBB 정렬 + Y-up 변환 + 72k 포인트 다운샘플링)
- [ ] GCS 업로드 성공 → `ply_url` 반환
- [ ] Callback URL로 결과 전송 성공 (`/analyze-furniture` 비동기)

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

### V2.6 (2026-04-08) — Dead code 제거 및 문서 정합성 개선

**Dead code 제거:**
- `worker_protocol.py` TaskMessage에서 `skip_gif`, `volume_only` 제거
- `worker_protocol.py` ResultMessage에서 `gif_b64`, `gif_size_bytes`, `mesh_url` 제거 (항상 None이었음)
- `sam3d_worker_pool.py` `submit_task`/`submit_tasks_parallel`에서 `skip_gif` 파라미터 제거
- `furniture_pipeline.py` generate_3d/_parallel_3d_generation 관련 필드 참조 제거
- `api/models.py` `AnalyzeFurnitureBase64Request`에서 사용되지 않는 `skip_gif`, `max_image_size` 제거

**일관성 개선:**
- `PLYPreprocessor` 기본값 `max_points` 50000 → 72000 (Config와 일치)
- TDD 문서 전체 재검토: SAM3DConverter 참조, stale 엔드포인트(`/generate-3d`), 잘못된 최적화 수치(15/8),
  잘못된 `/health` 응답 등 모두 수정

**문서 업데이트:**
- `PIPELINE_OPTIMIZATION.md` Section 2, 5.2, 7, 3.1 내용 오류 수정 + stale 라인 번호 15건 갱신
- `CLAUDE.md` SAM3D Worker Pool → Event 기반 work-stealing 반영
- `TDD_PIPELINE_V2.md` 전체 재작성 (아키텍처, Stage 설명, 엔드포인트, 벤치마크, 테스트 가이드)

### V2.5 (2026-02-02)

**Absolute Volume Calculation Migration:**
- 절대 부피 계산 로직을 **백엔드에서 AI 서버로 이전**
- 상대 치수 계산 직후 절대 치수(mm)와 부피(m³)를 함께 계산

**PLY 파일 GCS 업로드 (NEW):**
- 생성된 PLY 파일을 Google Cloud Storage에 업로드
- 응답에 `ply_url` 필드 추가 (Public URL)
- GCS 버킷: `isajjim-bucket`
- 파일 경로: `ply/est{estimate_id}_img{image_id}_{label}_{timestamp}_{uuid}.ply`

**New Files:**
- `ai/data/furniture_dimensions.py`: 52개 가구 타입 표준 치수 데이터
- `ai/processors/8_absolute_volume_calculate.py`: AbsoluteVolumeCalculator 클래스
- `api/services/gcs_storage.py`: GCS PLY 업로드 서비스

**API Response Changes:**
- `width`, `depth`, `height`: 상대 치수 → **절대 치수 (mm)**
- `volume`: 새로 추가 → **절대 부피 (m³)**
- `ply_url`: 새로 추가 → **GCS Public URL**

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
