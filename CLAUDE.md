# CLAUDE.md

This guide is for Claude Code (claude.ai/code) when working with this repository.

## Overview

A FastAPI-based service integrating:
1. **YOLOE-seg** - Object detection with instance segmentation (V2 파이프라인)
2. **Sam-3d-objects** - Facebook Research's 2D image to 3D object generation pipeline
3. **AI Module** - Furniture detection and analysis system (Korean language support)
4. **Multi-GPU Parallel Processing** - Round-robin GPU allocation via GPU pool manager
5. **SAM 2 (Segment Anything Model 2)** - Meta's Hugging Face model (deprecated in V2, available for /segment endpoints)

The API accepts 2D images and generates 3D Gaussian Splats, PLY, GIF, and GLB meshes.

### Pipeline Version: V2 (2024-01)

V2 파이프라인에서는 **YOLOE-seg 마스크를 SAM-3D에 직접 전달**합니다 (SAM2 제거).
```
[V1]  YOLO detect → center_point → SAM2 → mask → SAM-3D
[V2]  YOLO-seg detect → mask (직접) → SAM-3D  ← 현재
```

## Architecture

### Process Isolation Pattern

Uses **subprocess isolation** to prevent GPU/spconv state conflicts:
- `api.py` - Main FastAPI server handling SAM 2 segmentation
- `ai/subprocess/generate_3d_worker.py` - Isolated subprocess for Sam-3d-objects 3D generation

This isolation is **essential** because spconv maintains persistent GPU state, and loading models in the same process causes conflicts.

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

1. **Main API (`api.py`)**
   - FastAPI server loading SAM 2 model at startup
   - Initializes GPU pool and pre-initializes pipelines at startup
   - Handles segmentation requests (/segment, /segment-binary)
   - Manages async 3D generation tasks via background workers
   - Serves static assets (PLY, GIF, GLB) at /assets endpoint

2. **GPU Pool Manager (`ai/gpu/gpu_pool_manager.py`)**
   - GPUPoolManager class: Manages GPU resource pool
   - Round-robin GPU allocation (acquire/release)
   - GPU health check and automatic failover
   - Pipeline pre-initialization and registry management
   - gpu_context/pipeline_context async context managers

3. **3D Generation Subprocess (`ai/subprocess/generate_3d_worker.py`)**
   - Fresh Python process for each 3D generation task
   - Loads Sam-3d-objects pipeline independently
   - Gaussian splat generation, rotating GIF rendering, GLB mesh export
   - Uses synthetic pinhole pointmaps to prevent intrinsics recovery failure
   - Post-processes PLY files by adding RGB colors from spherical harmonics

4. **Furniture Analysis Pipeline (`ai/pipeline/furniture_pipeline.py`)**
   - FurniturePipeline class: Full AI logic orchestrator
   - device_id parameter for running on specific GPU
   - gpu_pool parameter for Multi-GPU parallel processing support
   - process_multiple_images: Parallel image processing from GPU pool
   - **CLIP/SAHI 제거** - YOLOE 클래스로 직접 DB 매칭

5. **AI Processors (`ai/processors/`)**
   - Furniture detection using YOLOE-seg (Objects365 기반 365 classes)
   - Korean language interface for moving services
   - Furniture dimension Knowledge Base for volume calculation
   - **V2 Pipeline**: Firebase URL → YOLOE-seg (mask 포함) → DB → SAM-3D → Volume (SAM2 제거)

### Fixed Path Dependencies

Sam-3d-objects integration uses **fixed relative paths** (not environment variables):
- `./sam-3d-objects/notebook` - Required for inference imports
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml` - Pipeline configuration

These paths are hardcoded in both `api.py` and `ai/subprocess/generate_3d_worker.py`.

### Environment Variable Setup

**Important**: Several environment variables must be set **before importing torch/spconv** to prevent GPU tuning issues:
- `SPCONV_TUNE_DEVICE=0`
- `SPCONV_ALGO_TIME_LIMIT=100` (prevents infinite tuning)
- `OMP_NUM_THREADS=4` (prevents thread explosion)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS compatibility)

These are set at the top of `api.py` and `ai/subprocess/generate_3d_worker.py` before any imports.

### Task Storage Pattern

3D generation uses an in-memory dict (`generation_tasks`) to track async tasks:
- POST /generate-3d returns task_id immediately (avoids gateway timeouts)
- Client polls GET /generate-3d-status/{task_id} for results
- Tasks store status (queued/processing/completed/failed), output files, and metadata

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

### Core Endpoints
- `GET /health` - Health check with model status
- `GET /gpu-status` - GPU pool status (available GPUs, pipeline initialization status)
- `POST /segment` - Single point segmentation (returns multiple masks)
- `POST /segment-binary` - Multi-point segmentation (returns masked PNG)
- `POST /generate-3d` - Async 3D generation (returns task_id)
- `GET /generate-3d-status/{task_id}` - Poll 3D generation results
- `GET /assets-list` - List stored PLY/GIF/GLB files
- `GET /assets/{filename}` - Download static assets

### Furniture Analysis Endpoints (AI Integration)
- `POST /analyze-furniture` - Full pipeline: Firebase URLs → Detection → 3D → Volume (Multi-GPU parallel processing)
- `POST /analyze-furniture-single` - Single image analysis
- `POST /analyze-furniture-base64` - Base64 image input (no Firebase URL)
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

1. Client sends image + mask to POST /generate-3d
2. API generates task_id and starts background worker
3. Background worker runs generate_3d_worker.py in a new process
4. Subprocess:
   - Loads Sam-3d-objects pipeline with fixed config path
   - Creates synthetic pinhole pointmap (avoids MoGe intrinsics failure)
   - Runs pipeline with decode_formats=["gaussian", "glb", "mesh"]
   - Renders 360° rotating GIF using render_video()
   - Exports PLY with RGB colors added from SH coefficients
   - Attempts textured GLB export via to_glb() or native pipeline output
   - Saves files with metadata to assets/ folder
   - Returns URLs via stdout markers (GIF_DATA_START/END, MESH_URL_START/END, PLY_URL_START/END)
5. API extracts URLs from subprocess stdout and updates task status
6. Client polls /generate-3d-status/{task_id} to receive base64 encoded files

## Furniture Analysis Pipeline V2 (AI Integration)

The `/analyze-furniture` endpoint implements the V2 AI Logic pipeline:

### V2 Pipeline (CLIP/SAHI/SAM2 제거)

1. **Image Fetch**: Download images from Firebase Storage URLs (5-10 images)
2. **GPU Allocation**: Round-robin GPU allocation from GPUPoolManager
3. **Object Detection**: YOLOE-seg for object detection (bbox + class + **segmentation mask**)
4. **DB Matching**: YOLOE class directly matches with Knowledge Base → is_movable determination
5. **Mask Direct Use**: YOLOE-seg 마스크를 SAM-3D에 직접 전달 (**SAM2 제거**)
6. **3D Generation**: SAM-3D converts masked image to 3D model
7. **Volume Calculation**: trimesh analyzes mesh for relative dimensions
8. **Absolute Dimensions**: Match with DB furniture specifications for real measurements

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

### V2 변경 이유 (테스트 결과)
- **마스크 품질**: YOLOE-seg가 SAM2보다 객체 전체를 더 정확하게 커버
- **속도**: SAM2 API 호출 제거로 latency 감소
- **단순화**: HTTP 호출 제거, 코드 복잡도 감소

### Response Format
```json
{
  "objects": [
    {
      "label": "침대",
      "width": 1500.0,
      "depth": 2000.0,
      "height": 450.0,
      "volume": 1.35,
      "ratio": {"w": 0.75, "h": 1.0, "d": 0.225},
      "is_movable": true
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

### Key Components
- `ai/pipeline/furniture_pipeline.py` - Main pipeline orchestrator
- `ai/gpu/gpu_pool_manager.py` - Multi-GPU pool manager
- `ai/processors/2_YOLO_detect.py` - YOLOE-seg detector (Objects365 기반)
- `ai/processors/7_volume_calculate.py` - Mesh volume analysis
- `ai/data/knowledge_base.py` - Furniture dimensions database (Objects365 매핑)

## Code Modification Guidelines

### When Modifying SAM 2 Integration
- Check existing mask handling in api.py:segment_image and api.py:segment_binary
- Maintain morphological smoothing for mask quality (cv2.morphologyEx)
- Maintain SAM 2's 4D input format: [[[[x, y]]]]

### When Modifying 3D Generation
- **Never** load Sam-3d-objects in the api.py main process - always use subprocess
- Maintain environment variables at the top of generate_3d_worker.py
- Use synthetic pinhole pointmaps (make_synthetic_pointmap) instead of MoGe/dummy maps
- Check debug markers in subprocess stdout (PLY_URL_START/END, etc.)
- Test GLB export carefully - to_glb() requires mesh data and can raise AttributeError

### When Modifying YOLO Detection
- Use YoloDetector class in `ai/processors/2_YOLO_detect.py`
- YOLOE-seg uses Objects365 class names (365 classes)
- DB matching is done via `knowledge_base.py` synonyms
- No more CLIP classification - YOLOE class directly maps to DB

### When Modifying Multi-GPU Processing
- Check GPUPoolManager class in `ai/gpu/gpu_pool_manager.py`
- Follow acquire/release pattern when adding new GPU features
- Pipeline factory functions must accept device_id
- Use GPU context managers to ensure automatic release
- Check Multi-GPU settings in `ai/config.py` (GPU_IDS, ENABLE_MULTI_GPU, etc.)

### When Adding New Endpoints
- Use background_tasks for operations taking more than 5 seconds
- Return task_id immediately for async operations
- Store results in generation_tasks dict or persistent storage
- If Multi-GPU is needed, acquire pool via get_gpu_pool()

### When Working with PLY Files
- PLY files are post-processed to add RGB from SH coefficients (add_rgb_to_ply)
- ASCII format is used for compatibility (files can be large)
- Clients need to handle large base64 payloads when polling results

### When Debugging 3D Generation
- Check subprocess stdout/stderr logs (printed in _generate_3d_background)
- Verify Sam-3d-objects paths exist: ./sam-3d-objects/notebook and checkpoints
- Check for memory issues - peak GPU memory is logged
- Verify mask has sufficient pixels (100+ recommended, logged in subprocess)

## File Structure

```
api.py                                  # Main FastAPI server (SAM 2 + task management + AI integration)
requirements.txt                        # Python dependencies
setup.sh                                # Setup script (clones sam-3d-objects, creates conda env)
assets/                                 # Static files served at /assets/ (PLY, GIF, GLB)
docs/                                   # Documentation
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
  main.py                               # Standalone CLI entry point
  config.py                             # YOLOE model settings, Multi-GPU settings
  gpu/                                  # GPU pool management module
    __init__.py                         # Exports GPUPoolManager, get_gpu_pool
    gpu_pool_manager.py                 # GPU pool manager implementation
  processors/                           # AI Logic step-by-step processors
    __init__.py                         # Exports processor classes
    1_firebase_images_fetch.py          # Step 1: Fetch images from Firebase
    2_YOLO_detect.py                    # Step 2: YOLOE-seg object detection (with mask)
    4_DB_movability_check.py            # Step 4: is_movable determination
    5_SAM2_mask_generate.py             # Step 5: SAM2 mask generation [DEPRECATED in V2]
    6_SAM3D_convert.py                  # Step 6: SAM-3D 3D conversion
    7_volume_calculate.py               # Step 7: Volume/dimension calculation
  pipeline/
    __init__.py                         # Exports FurniturePipeline
    furniture_pipeline.py               # Pipeline orchestrator V2 (YOLO mask direct use)
  subprocess/
    generate_3d_worker.py               # Isolated 3D generation worker
  data/
    __init__.py                         # Exports FURNITURE_DB
    knowledge_base.py                   # Furniture database with dimensions (Objects365 매핑)
  utils/
    __init__.py                         # Exports utilities
    image_ops.py                        # Image processing utilities
  fonts/                                # Korean font files (NanumGothic)
  imgs/                                 # Test images
  outputs/                              # Output results
test_pipeline_qa.py                     # Pipeline V2 QA test script
test_yoloe_vs_sam2_masks.py             # YOLOE vs SAM2 mask comparison test
```

## Known Issues and Solutions

1. **spconv float64 error**: Prevented by setting torch.set_default_dtype(torch.float32) before all imports
2. **Intrinsics recovery failure**: Prevented by using synthetic pinhole pointmaps instead of MoGe
3. **GLB export AttributeError**: Occurs when mesh_data is a list of non-mesh objects - subprocess now catches and logs this
4. **Mesh generation CPU spike**: Gaussian-to-mesh conversion is disabled by default (CPU intensive)
5. **Empty mask error**: Subprocess validates mask has >0 pixels and warns if <100 pixels
6. **Multi-GPU pipeline not initialized**: Falls back to creating new pipeline on-demand if pre-initialization fails
7. **GPU allocation timeout**: Raises RuntimeError if wait_timeout (default 300s) is exceeded
