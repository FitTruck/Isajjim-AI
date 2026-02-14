**English** | [한국어](README.md)

# YOLOE-seg + SAM-3D API (V2)

An integrated API service for furniture detection and 3D model generation for moving services.

> **V2 (2026-01)**: Simplified pipeline by using YOLOE-seg masks directly, removing CLIP/SAHI/SAM2

---

## Key Features

- **YOLOE-seg Detection**: 365-class detection based on Objects365 + instance segmentation masks
- **SAM-3D 3D Generation**: Generates 3D Gaussian Splats, PLY, and GLB meshes from 2D images + YOLO masks
- **Volume Calculation**: Relative volume/dimension calculation based on 3D model bounding boxes
- **Multi-GPU Parallel Processing**: Parallel processing with Persistent Worker Pool across up to 8 GPUs

---

## Performance Metrics

| Test Environment | Images | Objects | Total Time | Per Object |
|-----------------|--------|---------|------------|------------|
| 8 GPUs | 8 | 101 | ~3min 47s | **2.24s** |

### Optimization Settings

| Setting | Value | Effect |
|---------|-------|--------|
| `MAX_IMAGE_SIZE` | None (disabled) | Preserves volume accuracy |
| `STAGE1_INFERENCE_STEPS` | 14 | Speed/accuracy balance (optimal range: 12-16) |
| `STAGE2_INFERENCE_STEPS` | 8 | ~15-20% speed improvement |
| `GAUSSIAN_ONLY_MODE` | True | 37.4% speed improvement, skips GLB/Mesh |
| `USE_BINARY_PLY` | True | ~70% file size reduction, ~50% I/O speed improvement |
| `compile=True` | torch.compile | 10-20% inference speed improvement |
| `in_place=True` | Remove deepcopy | 5-10% speed/memory improvement |

> Configuration file: `ai/subprocess/persistent_3d_worker.py`

---

## Quick Start

### 1. Installation

```bash
# Install and authenticate Hugging Face CLI
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# Run setup script (clones sam-3d-objects, installs dependencies)
source setup.sh
```

### 2. Run Server

```bash
# Development
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

### 3. Health Check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/gpu-status
```

API Documentation: http://localhost:8000/docs

---

## AI Logic Pipeline V2

```
Firebase URL → YOLOE-seg (bbox + mask) → DB Matching → SAM-3D (Persistent Worker Pool) → Volume Calculation
```

| Step | File | Description |
|------|------|-------------|
| 1 | `1_firebase_images_fetch.py` | Download images from Firebase Storage URLs |
| 2 | `2_YOLO_detect.py` | YOLOE-seg object detection (bbox + class + mask) |
| 3 | `4_DB_movability_check.py` | DB matching by YOLO class, returns English label (base_name) |
| 4 | `6_SAM3D_convert.py` | YOLO mask → SAM-3D 3D model generation |
| 5 | `7_volume_calculate.py` | Relative volume/dimension calculation via trimesh OBB |

### V1 → V2 Changes

| Item | V1 | V2 |
|------|------|------|
| Detection Model | yolov8l-world.pt | yoloe-26x-seg.pt |
| SAHI/CLIP | Used | **Removed** |
| Mask Generation | SAM2 (center point) | **Direct YOLO mask usage** |
| API Calls | 3 | 2 |
| Volume Calculation | AABB + absolute | **OBB + relative** (absolute calculated in backend) |
| 3D Worker | Subprocess per request | **Persistent Worker Pool** |
| Parallelism | Sequential per image | **Parallel per object (Multi-GPU)** |

---

## API Endpoints

### Furniture Analysis (Main Endpoints)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze-furniture` | Multi-image furniture analysis (Firebase URLs) |
| POST | `/analyze-furniture-single` | Single image furniture analysis |
| POST | `/analyze-furniture-base64` | Base64 image furniture analysis |
| POST | `/detect-furniture` | Detection only (no 3D, fast response) |

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| GET | `/gpu-status` | GPU pool status |
| POST | `/generate-3d` | 3D generation (returns task_id) |
| GET | `/generate-3d-status/{task_id}` | Poll 3D generation result |
| GET | `/assets-list` | List stored assets |
| GET | `/assets/{filename}` | Download asset |

---

## Request/Response Examples

### Furniture Analysis Request

```bash
curl -X POST http://localhost:8000/analyze-furniture \
  -H 'Content-Type: application/json' \
  -d '{
    "estimate_id": 123,
    "image_urls": [
      {"id": 101, "url": "https://firebase-url-1.jpg"},
      {"id": 102, "url": "https://firebase-url-2.jpg"}
    ]
  }'
```

### Immediate Response (Async)

```json
{
  "success": true,
  "estimate_id": 123,
  "status": "processing"
}
```

### Callback Response

Results are sent to `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback` upon completion:

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "width": 1.5,
          "depth": 0.8,
          "height": 0.6,
          "volume": 0.72
        }
      ]
    }
  ]
}
```

**Unit Description:**
- `width`, `depth`, `height`: **Relative dimensions** (3D mesh bounding box, model coordinate system)
- `volume`: **Relative volume** (bounding box volume, model coordinate system)

> Absolute volume/dimensions are calculated in the backend by combining Knowledge Base actual dimensions with ratios

### Quick Detection (No 3D)

```bash
curl -X POST http://localhost:8000/detect-furniture \
  -H 'Content-Type: application/json' \
  -d '{"image":"<BASE64_IMAGE>"}'
```

```json
{
  "success": true,
  "objects": [
    {
      "label": "sofa",
      "bbox": [100, 200, 400, 500],
      "center_point": [250, 350],
      "confidence": 0.95
    }
  ],
  "total_objects": 1,
  "processing_time_seconds": 0.5
}
```

---

## Directory Structure

```
sam3d-api/
├── api/                        # FastAPI application (modular)
│   ├── app.py                  # Main application & router registration
│   ├── config.py               # API configuration
│   ├── models.py               # Pydantic models
│   ├── routes/                 # API routes
│   │   ├── furniture.py        # /analyze-furniture endpoints
│   │   └── health.py           # /health, /gpu-status, /assets endpoints
│   └── services/               # Service layer
│       └── callback.py         # Async callback service
├── requirements.txt            # Python dependencies
├── setup.sh                    # Environment setup script
├── assets/                     # Generated PLY/GIF/GLB assets
├── docs/                       # Documentation
│   ├── tdd/TDD_PIPELINE_V2.md  # Technical design document
│   └── qa/                     # QA test reports
├── ai/                         # AI module
│   ├── config.py               # Settings (GPU_IDS, model paths)
│   ├── gpu/                    # GPU pool manager
│   │   ├── gpu_pool_manager.py # YOLOE GPU pool
│   │   └── sam3d_worker_pool.py # SAM-3D Persistent Worker Pool
│   ├── processors/             # Pipeline processors
│   ├── pipeline/               # Pipeline orchestrator
│   ├── subprocess/             # SAM-3D worker (GPU isolation)
│   │   ├── persistent_3d_worker.py # Persistent Worker (optimized settings)
│   │   └── worker_protocol.py  # Worker communication protocol
│   ├── data/                   # Knowledge Base
│   └── utils/                  # Utilities
├── sam-3d-objects/             # Facebook Research SAM-3D (cloned via setup.sh)
└── tests/                      # Tests
```

---

## Operations Guide

### Environment Variables

Automatically set **before torch import** in `api.py` and `persistent_3d_worker.py`:

```bash
# GPU settings (prevents spconv tuning issues)
CUDA_HOME=/usr/local/cuda
SPCONV_TUNE_DEVICE=0
SPCONV_ALGO_TIME_LIMIT=100

# Thread limits (prevents thread explosion)
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4
NUMEXPR_NUM_THREADS=4

# macOS compatibility
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Performance Tuning

The following settings can be adjusted in `ai/subprocess/persistent_3d_worker.py`:

```python
# Phase 1: Image downsampling (None = disabled, preserves volume accuracy)
MAX_IMAGE_SIZE = None

# Phase 2: Inference Steps
STAGE1_INFERENCE_STEPS = 14  # Speed/accuracy balance (optimal range: 12-16)
STAGE2_INFERENCE_STEPS = 8   # Default 12 → 8 (~15-20% faster)

# Phase 3: PLY format (True = Binary, 70% smaller files)
USE_BINARY_PLY = True

# Phase 5: Gaussian-only mode (skip GLB/Mesh, volume calculation only)
GAUSSIAN_ONLY_MODE = True    # 37.4% faster, 0.005% volume error
```

**Optimization Test Results:**
- Downsampling: 91.7% impact on volume accuracy (up to 576% difference on small objects)
- Stage1 Steps (25→14): Speed/accuracy balance, ~0.5% volume error (vs 16)
- Stage2 Steps (12→8): ~15-20% speed improvement
- Gaussian-only: 37.4% speed improvement, 0.005% volume error
- torch.compile: 10-20% inference speed improvement
- in_place=True: 5-10% speed/memory improvement
- Binary PLY: ~50% faster I/O, ~70% smaller file size

### Monitoring

```bash
# API status
curl http://localhost:8000/health

# GPU pool status
curl http://localhost:8000/gpu-status
```

**GPU Status Response Example:**

```json
{
  "total_gpus": 4,
  "available_gpus": 3,
  "pipelines_initialized": 4,
  "gpus": {
    "0": {"available": true, "task_id": null, "has_pipeline": true},
    "1": {"available": false, "task_id": "processing", "has_pipeline": true}
  }
}
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| spconv float64 error | dtype not set before torch import | Verify `torch.set_default_dtype(torch.float32)` |
| Intrinsics recovery failure | MoGe pointmap failure | Use synthetic pinhole pointmap (default) |
| GLB export AttributeError | mesh_data is a list | PLY fallback used (automatic) |
| CUDA out of memory | Insufficient GPU memory | Reduce worker count, lower image resolution |
| Empty mask error | Segmentation failure | Verify mask has >100 pixels |
| Pipeline not initialized | Initialization failed at startup | On-demand creation fallback (automatic) |
| Subprocess timeout | 3D generation exceeds 5min | Check GPU performance, verify mask size |
| Worker not ready | Worker initialization failed | Check logs, restart worker |
| Volume accuracy issues | Image downsampling | Verify `MAX_IMAGE_SIZE = None` |

### Rollback Procedure

```bash
# Stop server
pkill -f "uvicorn api:app"

# Rollback to previous version
git checkout HEAD~1

# Reinstall dependencies (if needed)
pip install -r requirements.txt

# Restart server
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Asset Cleanup

```bash
# Delete assets older than 7 days
find assets/ -type f -mtime +7 -delete
```

---

## Docker Deployment

### Prerequisites

```bash
# VM setup
sudo bash scripts/vm-setup-docker.sh
sudo bash scripts/vm-setup-nvidia-toolkit.sh
sudo bash scripts/vm-setup-data.sh
```

### Docker Compose

```bash
export GCP_PROJECT_ID=your-project-id
export IMAGE_TAG=latest

docker compose up -d
docker compose logs -f
```

### Volume Mounts

| Host | Container | Description |
|------|-----------|-------------|
| `/data/sam3d/sam-3d-objects` | `/data/sam3d/sam-3d-objects` | SAM-3D checkpoints |
| `/data/sam3d/models` | `/data/sam3d/models` | YOLO models |
| `/data/sam3d/huggingface` | `/data/sam3d/huggingface` | HuggingFace cache |
| `/data/sam3d/assets` | `/app/assets` | Generated assets |

### CI/CD (GitHub Actions)

Auto-deploy on `main` branch push:
1. Build Docker image
2. Push to GCP Artifact Registry
3. Deploy via VM SSH

Required Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `VM_HOST`, `VM_SSH_KEY`, `VM_USER`

---

## Requirements

- Python 3.10+
- CUDA 11.8+ (GPU recommended)
- 32GB+ VRAM
- 50GB+ disk (for model storage)

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | latest | API framework |
| uvicorn | latest | ASGI server |
| torch | >=2.1.0 | PyTorch |
| ultralytics | >=8.3.0 | YOLOE-seg |
| trimesh | latest | 3D mesh analysis |
| aiohttp | latest | Async HTTP |

---

## Documentation

- [Technical Design Document (TDD)](docs/tdd/TDD_PIPELINE_V2.md) - Architecture, API specs
- [CLAUDE.md](CLAUDE.md) - Claude Code guide, code modification guidelines

---

## License

MIT
