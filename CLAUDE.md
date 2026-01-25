# CLAUDE.md

This guide is for Claude Code (claude.ai/code) when working with this repository.

## Overview

A FastAPI-based service integrating:
1. **YOLOE-seg** - Object detection with instance segmentation (V2 파이프라인)
2. **Sam-3d-objects** - Facebook Research's 2D image to 3D object generation pipeline
3. **AI Module** - Furniture detection and analysis system (Korean language support)
4. **Multi-GPU Parallel Processing** - Round-robin GPU allocation via GPU pool manager

The API accepts 2D images and generates 3D Gaussian Splats, PLY, GIF, and GLB meshes.

### Pipeline Version: V2 (2026-01)

V2 파이프라인에서는 **YOLOE-seg 마스크를 SAM-3D에 직접 전달**합니다 (SAM2 제거).
```
[V1]  YOLO detect → center_point → SAM2 → mask → SAM-3D
[V2]  YOLO-seg detect → mask (직접) → SAM-3D  ← 현재
```

## Architecture

### Process Isolation Pattern

Uses **subprocess isolation** to prevent GPU/spconv state conflicts:
- `api/app.py` - Main FastAPI server (YOLOE + GPU pool + 3D task management)
- `ai/subprocess/persistent_3d_worker.py` - Persistent Worker Process for Sam-3d-objects 3D generation

This isolation is **essential** because spconv maintains persistent GPU state, and loading models in the same process causes conflicts.

### Persistent Worker Pool Architecture (2026-01 Update)

SAM-3D 3D 생성을 위한 **Persistent Worker Pool** 패턴:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SAM3DWorkerPool                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Worker Processes                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
│  │  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker 7 │    │    │
│  │  │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 7   │    │    │
│  │  │ (SAM-3D) │  │ (SAM-3D) │  │ (SAM-3D) │  │ (SAM-3D) │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│              Round-robin task distribution via stdin/stdout          │
│                              ▼                                       │
│         ┌───────────────────────────────────────────┐               │
│         │      submit_tasks_parallel()              │               │
│         │  obj1→Worker0, obj2→Worker1, ...          │               │
│         └───────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Model pre-loading**: 워커 시작 시 SAM-3D 모델 1회 로드
- **JSON protocol**: stdin/stdout으로 JSON 메시지 교환
- **Parallel processing**: 여러 객체를 동시에 다른 GPU에서 처리
- **Auto-restart**: 워커 프로세스 종료 시 자동 재시작

### Multi-GPU Parallel Processing Architecture

Uses **GPU Pool Manager** pattern for parallelizing image processing across multiple GPUs:

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server (api.py)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GPUPoolManager                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │     │   │
│  │  │Pipeline │  │Pipeline │  │Pipeline │  │Pipeline │     │   │
│  │  │(YOLOE)  │  │(YOLOE)  │  │(YOLOE)  │  │(YOLOE)  │     │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    Round-robin GPU allocation                    │
│                              ▼                                   │
│         ┌─────────────────────────────────────────┐             │
│         │      process_multiple_images()          │             │
│         │  img1→GPU0, img2→GPU1, img3→GPU2, ...   │             │
│         └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Round-robin GPU allocation**: Distributes image requests sequentially across different GPUs
- **Pipeline pre-initialization**: Loads YOLOE model on each GPU at server startup
- **Pipeline reuse**: Uses pre-initialized pipelines instead of creating new ones per request
- **Thread-safe acquire/release**: Ensures concurrency safety using asyncio.Lock

### Core Components

1. **Main API (`api/app.py`)**
   - FastAPI server with YOLOE-seg pipelines
   - Initializes GPU pool and pre-initializes pipelines at startup
   - Initializes SAM3D Worker Pool for persistent 3D generation
   - Serves static assets (PLY, GIF, GLB) at /assets endpoint

2. **GPU Pool Manager (`ai/gpu/gpu_pool_manager.py`)**
   - GPUPoolManager class: Manages GPU resource pool for YOLOE detection
   - Round-robin GPU allocation (acquire/release)
   - GPU health check and automatic failover
   - Pipeline pre-initialization and registry management
   - gpu_context/pipeline_context async context managers

3. **SAM3D Worker Pool (`ai/gpu/sam3d_worker_pool.py`)**
   - SAM3DWorkerPool class: GPU당 하나의 persistent 워커 프로세스 관리
   - 모델 1회 로드 후 재사용 (모델 로딩 오버헤드 제거)
   - 라운드로빈 작업 분배
   - JSON 기반 stdin/stdout 통신 프로토콜
   - submit_tasks_parallel(): 여러 작업 동시 제출

4. **Persistent 3D Worker (`ai/subprocess/persistent_3d_worker.py`)**
   - GPU별 독립 프로세스로 SAM-3D 모델 실행
   - 성능 최적화 설정 포함 (다운샘플링, inference steps, binary PLY)
   - Synthetic pinhole pointmap으로 intrinsics 문제 방지
   - SH coefficients에서 RGB 추출하여 PLY 후처리

5. **Worker Protocol (`ai/subprocess/worker_protocol.py`)**
   - TaskMessage, ResultMessage, InitMessage, HeartbeatMessage 정의
   - JSON 기반 메시지 직렬화/역직렬화
   - 워커-풀 매니저 간 통신 규약

6. **Furniture Analysis Pipeline (`ai/pipeline/furniture_pipeline.py`)**
   - FurniturePipeline class: Full AI logic orchestrator
   - _parallel_3d_generation(): Worker Pool을 통한 병렬 3D 생성
   - device_id parameter for running on specific GPU
   - gpu_pool parameter for Multi-GPU parallel processing support
   - **CLIP/SAHI 제거** - YOLOE 클래스로 직접 DB 매칭

7. **AI Processors (`ai/processors/`)**
   - Furniture detection using YOLOE-seg (Objects365 기반 365 classes)
   - Korean language interface for moving services
   - Furniture dimension Knowledge Base for volume calculation
   - **V2 Pipeline**: Firebase URL → YOLOE-seg (mask 포함) → DB → SAM-3D → Volume (SAM2 제거)

### Fixed Path Dependencies

Sam-3d-objects integration uses **fixed relative paths** (not environment variables):
- `./sam-3d-objects/notebook` - Required for inference imports
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml` - Pipeline configuration

These paths are hardcoded in `ai/subprocess/persistent_3d_worker.py`.

### Environment Variable Setup

**Important**: Several environment variables must be set **before importing torch/spconv** to prevent GPU tuning issues:
- `SPCONV_TUNE_DEVICE=0`
- `SPCONV_ALGO_TIME_LIMIT=100` (prevents infinite tuning)
- `OMP_NUM_THREADS=4` (prevents thread explosion)
- `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `VECLIB_MAXIMUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS compatibility)

These are set at the top of `api/app.py` and `ai/subprocess/persistent_3d_worker.py` before any imports.

### Performance Optimization Configuration

`ai/subprocess/persistent_3d_worker.py`에서 성능 최적화 설정:

```python
# Phase 1: 이미지 다운샘플링 (None = 비활성화)
MAX_IMAGE_SIZE = None  # 부피 정확도 유지 (다운샘플링이 91.7% 영향)

# Phase 2: Inference Steps
STAGE1_INFERENCE_STEPS = 14  # 속도/정확도 균형 (12~16 사이 최적값)
STAGE2_INFERENCE_STEPS = 8   # 디테일은 속도 우선

# Phase 3: PLY 형식 (True = Binary)
USE_BINARY_PLY = True  # ~70% 파일 크기 감소, ~50% I/O 속도 향상

# Phase 5: Gaussian-only 모드 (GLB/Mesh 스킵)
GAUSSIAN_ONLY_MODE = True  # 37.4% 속도 향상, 0.005% 부피 오차
```

**추가 최적화:**
- `compile=True`: torch.compile로 CUDA 커널 컴파일 (10-20% 추론 속도 향상)
- `in_place=True`: deepcopy 제거로 메모리/속도 최적화 (5-10% 향상)

**성능 테스트 결과 (8 GPU, 8 이미지, 101 객체):**
- 총 시간: ~3분 47초 (226초)
- 객체당 평균: 2.24초
- 부피 정확도: ~4% 오차 이내

### Async Callback Pattern

`/analyze-furniture` 엔드포인트는 비동기 callback 패턴을 사용합니다:
- POST 즉시 `{"success": true, "estimate_id": X, "status": "processing"}` 응답
- 백그라운드에서 파이프라인 실행
- 완료 시 callback URL로 결과 전송: `http://api.isajjim.kro.kr:8080/api/v1/estimates/{estimate_id}/callback`
- Callback 서비스: `api/services/callback.py`

## Common Commands

### Setup

```bash
# Install Hugging Face CLI and authenticate
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# Run setup script (clones sam-3d-objects, creates conda env, downloads checkpoints)
source setup.sh
```

### Development

```bash
# Run with auto-reload (development)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Run without reload
python api.py
```

### Production

```bash
# Use Uvicorn with multiple workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Use Gunicorn + Uvicorn worker class
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app --log-level info
```

**Production Notes:**
- Adjust `--workers` based on available CPU/memory
- For GPU workloads, be careful not to have too many workers competing for the same GPU
- Use process manager (systemd, docker-compose) and reverse proxy (NGINX) in production
- Ensure Sam-3d-objects paths exist before startup

### AI Module (Furniture Detection)

The AI module is integrated with the main API:

```bash
# Standalone execution (for testing)
cd ai
python main.py  # Analyzes images in imgs/ directory
```

## API Endpoints

### Health & Status (api/routes/health.py)
- `GET /health` - Health check with model status
- `GET /gpu-status` - GPU pool status (available GPUs, pipeline initialization status)
- `GET /assets-list` - List stored PLY/GIF/GLB files
- `GET /assets/{filename}` - Download static assets

### Furniture Analysis Endpoints (api/routes/furniture.py)
- `POST /analyze-furniture` - Full pipeline: Firebase URLs → Detection → 3D → Volume (비동기 callback)
- `POST /analyze-furniture-single` - Single image analysis (동기)
- `POST /analyze-furniture-base64` - Base64 image input (동기)
- `POST /detect-furniture` - Detection only (no 3D, fast response)

All endpoints use base64 encoded images or Firebase Storage URLs.

## Multi-GPU Parallel Processing

### GPU Pool Manager Usage

```python
from ai.gpu import GPUPoolManager, get_gpu_pool, initialize_gpu_pool

# Initialize global GPU pool (at server startup)
pool = initialize_gpu_pool(gpu_ids=[0, 1, 2, 3])

# GPU context for automatic acquire/release
async with pool.gpu_context(task_id="image_1") as gpu_id:
    # gpu_id is exclusively used in this context
    process_on_gpu(gpu_id)
# GPU automatically released

# Use with pre-initialized pipeline
async with pool.pipeline_context(task_id="image_1") as (gpu_id, pipeline):
    result = await pipeline.process_single_image(url)
# GPU automatically released
```

### Pipeline Pre-initialization

Pre-initializes pipelines on each GPU at server startup to eliminate model loading overhead:

```python
# In api.py startup event
await pool.initialize_pipelines(
    lambda gpu_id: FurniturePipeline(device_id=gpu_id),
    skip_on_error=True
)
```

**Benefits:**
- Eliminates 3-5 second model loading time per request
- Independent YOLOE model instances per GPU
- Prevents GPU conflicts during parallel image processing

### GPU Status Check

```bash
# Query GPU pool status
curl http://localhost:8000/gpu-status

# Example response:
{
    "total_gpus": 4,
    "available_gpus": 3,
    "pipelines_initialized": 4,
    "gpus": {
        "0": {"available": true, "task_id": null, "memory_used_mb": 1024, "has_pipeline": true},
        "1": {"available": false, "task_id": "image_processing", "memory_used_mb": 2048, "has_pipeline": true},
        ...
    }
}
```

## 3D Generation Pipeline

### Persistent Worker Pool 방식 (권장, /analyze-furniture 사용)

1. 서버 시작 시 SAM3DWorkerPool 초기화 (GPU당 1개 워커 프로세스)
2. 각 워커는 SAM-3D 모델을 1회 로드하고 대기
3. FurniturePipeline._parallel_3d_generation() 호출:
   - 이미지와 마스크를 base64로 인코딩
   - TaskMessage를 JSON으로 워커에 전송 (stdin)
   - 워커가 3D 생성 후 ResultMessage 반환 (stdout)
   - PLY base64를 받아서 부피 계산
4. 여러 객체가 있으면 다른 워커에 병렬 분배

### Persistent Worker 상세 동작

각 워커의 처리 과정:
1. stdin에서 TaskMessage JSON 수신
2. 이미지와 마스크 base64 디코딩 → 임시 파일로 저장
3. SAM-3D 파이프라인 실행:
   - synthetic pinhole pointmap 생성
   - decode_formats: Gaussian-only 모드에서는 `["gaussian"]`만
   - 후처리 비활성화 (texture_baking, mesh_postprocess, layout_postprocess)
4. PLY 저장 및 RGB 후처리 (SH coefficients에서 RGB 추출)
5. stdout으로 ResultMessage JSON 반환

## Furniture Analysis Pipeline V2 (AI Integration)

The `/analyze-furniture` endpoint implements the V2 AI Logic pipeline:

### V2 Pipeline (CLIP/SAHI/SAM2/is_movable/dimensions 제거)

1. **Image Fetch**: Download images from Firebase Storage URLs (5-10 images)
2. **GPU Allocation**: Round-robin GPU allocation from GPUPoolManager
3. **Object Detection**: YOLOE-seg for object detection (bbox + class + **segmentation mask**)
4. **DB Matching**: YOLOE class directly matches with Knowledge Base → 한국어 라벨 반환
5. **Mask Direct Use**: YOLOE-seg 마스크를 SAM-3D에 직접 전달 (**SAM2 제거**)
6. **3D Generation**: SAM-3D converts masked image to 3D model
7. **Volume Calculation**: trimesh analyzes mesh for **relative dimensions only** (절대 부피는 백엔드에서 계산)

### Key Changes (V1 → V2)
| 항목 | V1 | V2 |
|------|------|------|
| 탐지 모델 | yolov8l-world.pt | yoloe-26x-seg.pt |
| SAHI | 사용 | **완전 제거** |
| CLIP 분류 | 세부 유형 분류 | **완전 제거** |
| **SAM2 마스크** | center point prompt | **완전 제거 (YOLO 마스크 직접 사용)** |
| 분류 단계 | YOLO → CLIP → DB | YOLO → DB (직접) |
| DB 매칭 | CLIP 결과로 서브타입 매칭 | YOLO 클래스로 직접 매칭 |
| API 호출 수 | 3회 (YOLO→SAM2→SAM3D) | 2회 (YOLO→SAM3D) |
| **부피 계산** | AI에서 절대 부피 계산 | **상대 부피만 반환 (절대 부피는 백엔드)** |
| **is_movable** | DB에서 is_movable 결정 | **제거 (모든 탐지 객체는 이동 대상)** |
| **dimensions** | DB에 치수 정보 저장 | **제거 (절대 부피는 백엔드 계산)** |

### V2 변경 이유 (테스트 결과)
- **마스크 품질**: YOLOE-seg가 SAM2보다 객체 전체를 더 정확하게 커버
- **속도**: SAM2 API 호출 제거로 latency 감소
- **단순화**: HTTP 호출 제거, 코드 복잡도 감소
- **is_movable/dimensions 제거**: 백엔드에서 계산하므로 AI API에서는 불필요

### Request Format (TDD 문서 Section 4.1)
```json
{
  "estimate_id": 123,
  "image_urls": [
    {"id": 101, "url": "https://firebase-storage-url-1.jpg/"},
    {"id": 102, "url": "https://firebase-storage-url-2.jpg/"}
  ]
}
```

**필드:**
- `estimate_id`: 견적 ID (정수, 필수) - callback URL에 사용
- `image_urls`: 이미지 URL 객체 배열 (1-20개)
  - `id`: 사용자 지정 이미지 ID (정수)
  - `url`: Firebase Storage URL (문자열)

### Response Format (TDD 문서 Section 4.1)

**비동기 방식**: 즉시 processing 응답 반환, 작업 완료 후 callback URL로 결과 전송

**즉시 응답:**
```json
{
  "success": true,
  "estimate_id": 123,
  "status": "processing"
}
```

**Callback URL:** `http://api.isajjim.kro.kr:8080/api/v1/estimates/{estimate_id}/callback`

**Callback Payload (성공):**
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
        }
      ]
    },
    {
      "image_id": 102,
      "objects": [...]
    }
  ]
}
```

**Callback Payload (실패):**
```json
{
  "error": "에러 메시지"
}
```

**Callback URL (하드코딩):**
- `http://api.isajjim.kro.kr:8080/api/v1/estimates/{estimate_id}/callback`

**단위:**
- `width`, `depth`, `height`: **상대 길이** (3D 메시 bounding box 기준, 단위 없음)
- `volume`: **상대 부피** (bounding box 부피, 단위 없음)

> 절대 부피/치수 계산은 백엔드에서 Knowledge Base 실제 치수와 비율을 조합하여 계산

**Note**: `is_movable`, `dimensions`, `ratio`는 V2 파이프라인에서 제거되었습니다.
절대 부피 계산은 백엔드에서 Knowledge Base를 사용합니다.

### Key Components
- `ai/pipeline/furniture_pipeline.py` - Main pipeline orchestrator
- `ai/gpu/gpu_pool_manager.py` - YOLOE Multi-GPU pool manager
- `ai/gpu/sam3d_worker_pool.py` - SAM-3D Persistent Worker Pool manager
- `ai/subprocess/persistent_3d_worker.py` - Persistent 3D Worker (성능 최적화 설정 포함)
- `ai/subprocess/worker_protocol.py` - 워커-풀 통신 프로토콜
- `ai/processors/2_YOLO_detect.py` - YOLOE-seg detector (Objects365 기반)
- `ai/processors/4_DB_movability_check.py` - 한국어 라벨 매핑 (is_movable 제거됨)
- `ai/processors/7_volume_calculate.py` - Mesh relative volume/dimensions (절대 부피는 백엔드 계산)
- `ai/data/knowledge_base.py` - YOLO 클래스 매핑 + 한국어 라벨 + 프롬프트 저장용 정적 DB (dimensions/is_movable 제거됨)

## Code Modification Guidelines

### When Modifying 3D Generation
- **Never** load Sam-3d-objects in the api/app.py main process - always use subprocess
- Maintain environment variables at the top of persistent_3d_worker.py
- Use synthetic pinhole pointmaps (make_synthetic_pointmap) instead of MoGe/dummy maps
- Check debug markers in subprocess stdout (PLY_URL_START/END, etc.)
- Test GLB export carefully - to_glb() requires mesh data and can raise AttributeError

### When Modifying Performance Settings
- 성능 최적화 설정은 `ai/subprocess/persistent_3d_worker.py` 상단에 위치
- `MAX_IMAGE_SIZE`: None(비활성화 권장) - 다운샘플링은 부피 정확도에 91.7% 영향
- `STAGE1_INFERENCE_STEPS`: 14 권장 (속도/정확도 균형, 12~16 사이 최적값)
- `STAGE2_INFERENCE_STEPS`: 8(속도) / 12(품질) - ~4% 부피 오차
- `GAUSSIAN_ONLY_MODE`: True 권장 (부피 계산만 필요시) - 37.4% 속도 향상
- `USE_BINARY_PLY`: True 권장 - 70% 파일 크기 감소

### When Modifying Worker Pool
- SAM3DWorkerPool은 `ai/gpu/sam3d_worker_pool.py`에 정의
- 워커 통신 프로토콜은 `ai/subprocess/worker_protocol.py` 참조
- JSON 메시지 형식: TaskMessage, ResultMessage, InitMessage, HeartbeatMessage
- 워커 시작 시 init_timeout(120초), 작업 처리 시 task_timeout(300초) 설정 확인

### When Modifying YOLO Detection
- Use YoloDetector class in `ai/processors/2_YOLO_detect.py`
- YOLOE-seg uses Objects365 class names (365 classes)
- DB matching is done via `knowledge_base.py` synonyms
- No more CLIP classification - YOLOE class directly maps to DB

### When Modifying Multi-GPU Processing
- YOLOE용: GPUPoolManager class in `ai/gpu/gpu_pool_manager.py`
- SAM-3D용: SAM3DWorkerPool class in `ai/gpu/sam3d_worker_pool.py`
- Follow acquire/release pattern when adding new GPU features
- Pipeline factory functions must accept device_id
- Use GPU context managers to ensure automatic release
- Check Multi-GPU settings in `ai/config.py` (GPU_IDS, ENABLE_MULTI_GPU, etc.)

### When Adding New Endpoints
- 라우트는 `api/routes/` 디렉토리에 추가
- Use background_tasks for operations taking more than 5 seconds
- Return task_id immediately for async operations
- Store results in generation_tasks dict or persistent storage
- If Multi-GPU is needed, acquire pool via get_gpu_pool() or get_sam3d_worker_pool()

### When Working with PLY Files
- PLY files are post-processed to add RGB from SH coefficients (add_rgb_to_ply)
- Binary format is now default (USE_BINARY_PLY = True) - ~70% smaller, ~50% faster
- ASCII format available for compatibility if needed
- Clients need to handle large base64 payloads when polling results

### When Debugging 3D Generation
- Check subprocess stdout/stderr logs (Worker Pool logs to stderr)
- Verify Sam-3d-objects paths exist: ./sam-3d-objects/notebook and checkpoints
- Check for memory issues - peak GPU memory is logged
- Verify mask has sufficient pixels (100+ recommended, logged in subprocess)
- Worker Pool status: `curl http://localhost:8000/gpu-status`

## File Structure

```
api/                                    # FastAPI 애플리케이션 (모듈화)
  app.py                                # 메인 애플리케이션 및 라우터 등록
  config.py                             # API 설정 (device, ASSETS_DIR)
  models.py                             # Pydantic 요청/응답 모델
  routes/                               # API 라우트
    furniture.py                        # /analyze-furniture 엔드포인트
    health.py                           # /health, /gpu-status, /assets 엔드포인트
  services/                             # 서비스 레이어
    callback.py                         # 비동기 Callback 서비스
requirements.txt                        # Python dependencies
setup.sh                                # Setup script (clones sam-3d-objects, creates conda env)
assets/                                 # Static files served at /assets/ (PLY, GIF, GLB)
docs/                                   # Documentation
  PIPELINE_OPTIMIZATION.md              # 파이프라인 최적화 가이드
  qa/                                   # QA test reports
    QA_MULTI_GPU_TEST_REPORT.md         # Multi-GPU parallel processing test report
    QA_YOLOE_MIGRATION_REPORT.md        # YOLOE migration test report
  tdd/                                  # Technical Design Documents
    TDD_PIPELINE_V2.md                  # Technical Design Document for V2 pipeline
sam-3d-objects/                         # Cloned Facebook Research repo (not in git)
  notebook/inference.py                 # Sam-3d-objects pipeline class
  checkpoints/hf/pipeline.yaml          # Pipeline configuration
ai/                                     # AI module (integrated with main API)
  __init__.py                           # Module entry point (exports FurniturePipeline, processors)
  config.py                             # YOLOE model settings, Multi-GPU settings
  gpu/                                  # GPU pool management module
    __init__.py                         # Exports GPUPoolManager, SAM3DWorkerPool
    gpu_pool_manager.py                 # YOLOE GPU pool manager implementation
    sam3d_worker_pool.py                # SAM-3D Persistent Worker Pool
  processors/                           # AI Logic step-by-step processors
    __init__.py                         # Exports processor classes
    1_firebase_images_fetch.py          # Step 1: Fetch images from Firebase
    2_YOLO_detect.py                    # Step 2: YOLOE-seg object detection (with mask)
    4_DB_movability_check.py            # Step 4: 한국어 라벨 매핑
    7_volume_calculate.py               # Step 7: Relative volume/dimension calculation
  pipeline/
    __init__.py                         # Exports FurniturePipeline
    furniture_pipeline.py               # Pipeline orchestrator V2 (YOLO mask direct use)
  subprocess/
    persistent_3d_worker.py             # Persistent 3D Worker (성능 최적화 설정 포함)
    worker_protocol.py                  # 워커-풀 통신 프로토콜
  data/
    __init__.py                         # Exports FURNITURE_DB
    knowledge_base.py                   # YOLO 클래스 매핑 + 한국어 라벨 정적 DB
  utils/
    __init__.py                         # Exports utilities
    image_ops.py                        # Image processing utilities
  fonts/                                # Korean font files (NanumGothic)
  imgs/                                 # Test images
  outputs/                              # Output results
tests/                                  # Test files
```

## Known Issues and Solutions

1. **spconv float64 error**: Prevented by setting torch.set_default_dtype(torch.float32) before all imports
2. **Intrinsics recovery failure**: Prevented by using synthetic pinhole pointmaps instead of MoGe
3. **GLB export AttributeError**: Occurs when mesh_data is a list of non-mesh objects - subprocess now catches and logs this
4. **Mesh generation CPU spike**: Gaussian-to-mesh conversion is disabled by default (CPU intensive)
5. **Empty mask error**: Subprocess validates mask has >0 pixels and warns if <100 pixels
6. **Multi-GPU pipeline not initialized**: Falls back to creating new pipeline on-demand if pre-initialization fails
7. **GPU allocation timeout**: Raises RuntimeError if wait_timeout (default 300s) is exceeded
