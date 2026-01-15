# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a FastAPI-based service that integrates:
1. **SAM 2 (Segment Anything Model 2)** - Meta's image segmentation model via Hugging Face
2. **Sam-3d-objects** - Facebook Research's 3D object generation pipeline from 2D images
3. **DeCl** - A furniture detection and analysis system (Korean language support)

The API accepts 2D images with point-based segmentation and generates 3D Gaussian splats, PLY files, GIFs, and GLB meshes.

## Architecture

### Process Isolation Pattern

The service uses **subprocess isolation** to avoid GPU/spconv state conflicts:
- `api.py` - Main FastAPI server handling SAM 2 segmentation
- `generate_3d_subprocess.py` - Isolated subprocess for Sam-3d-objects 3D generation

This isolation is **critical** because spconv has persistent GPU state that causes conflicts when models are loaded in the same process.

### Key Components

1. **Main API (`api.py`)**
   - FastAPI server with SAM 2 model loaded at startup
   - Handles segmentation requests (/segment, /segment-binary)
   - Manages async 3D generation tasks via background workers
   - Serves static assets (PLY, GIF, GLB files) from /assets endpoint

2. **3D Generation Subprocess (`generate_3d_subprocess.py`)**
   - Fresh Python process for each 3D generation task
   - Loads Sam-3d-objects pipeline independently
   - Generates Gaussian splats, renders rotating GIFs, exports GLB meshes
   - Uses synthetic pinhole pointmaps to avoid intrinsics recovery failures
   - Post-processes PLY files to add RGB colors from spherical harmonics

3. **DeCl Module (`DeCl/`)** - Now integrated with main API
   - Furniture detection using YOLO-World + SAHI (small object detection)
   - CLIP classification for detailed furniture subtypes
   - Korean language interface for moving services
   - Knowledge base with furniture dimensions for volume calculation
   - Integrated pipeline: Firebase URL → YOLO → CLIP → SAM2 → SAM-3D → Volume

### Fixed Path Dependencies

The Sam-3d-objects integration uses **fixed relative paths** (not environment variables):
- `./sam-3d-objects/notebook` - Required for inference imports
- `./sam-3d-objects/checkpoints/hf/pipeline.yaml` - Pipeline configuration

These paths are hardcoded in both `api.py` and `generate_3d_subprocess.py`. If you need to change them, update both files.

### Environment Variable Configuration

**Critical**: Several environment variables **must be set before importing torch/spconv** to prevent GPU tuning issues:
- `SPCONV_TUNE_DEVICE=0`
- `SPCONV_ALGO_TIME_LIMIT=100` (prevents infinite tuning)
- `OMP_NUM_THREADS=4` (prevents thread explosion)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (macOS compatibility)

These are set at the top of both `api.py` and `generate_3d_subprocess.py` before any imports.

### Task Storage Pattern

3D generation uses an in-memory dict (`generation_tasks`) to track async jobs:
- POST /generate-3d returns a task_id immediately (avoids gateway timeouts)
- Client polls GET /generate-3d-status/{task_id} for results
- Tasks store status (queued/processing/completed/failed), output files, and metadata

## Common Commands

### Setup

```bash
# Install Hugging Face CLI and authenticate
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login

# Run setup script to clone sam-3d-objects, create conda env, download checkpoints
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
# Using Uvicorn with multiple workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Using Gunicorn + Uvicorn worker class
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app --log-level info
```

**Production Notes:**
- Tune `--workers` based on available CPU/memory
- For GPU workloads, avoid too many workers competing for the same GPU
- Use process manager (systemd, docker-compose) and reverse proxy (NGINX) for production
- Ensure Sam-3d-objects paths exist before starting

### DeCl (Furniture Detection)

The DeCl module is a separate tool, not integrated with the main API:

```bash
cd DeCl
python main.py  # Analyzes images in imgs/ directory
```

## API Endpoints

### Core Endpoints
- `GET /health` - Health check with model status
- `POST /segment` - Single-point segmentation (returns multiple masks)
- `POST /segment-binary` - Multi-point segmentation (returns masked PNG)
- `POST /generate-3d` - Async 3D generation (returns task_id)
- `GET /generate-3d-status/{task_id}` - Poll for 3D generation results
- `GET /assets-list` - List saved PLY/GIF/GLB files with metadata
- `GET /assets/{filename}` - Download static assets

### Furniture Analysis Endpoints (DeCl Integration)
- `POST /analyze-furniture` - Full pipeline: Firebase URLs → Detection → 3D → Volume
- `POST /analyze-furniture-single` - Single image analysis
- `POST /analyze-furniture-base64` - Base64 image input (no Firebase URL)
- `POST /detect-furniture` - Detection only (no 3D, fast response)

All endpoints expect/return base64-encoded images or Firebase Storage URLs.

## 3D Generation Pipeline

1. Client sends image + mask to POST /generate-3d
2. API creates task_id and spawns background worker
3. Background worker runs generate_3d_subprocess.py in fresh process
4. Subprocess:
   - Loads Sam-3d-objects pipeline with fixed config path
   - Generates synthetic pinhole pointmap (avoids MoGe intrinsics failures)
   - Runs pipeline with decode_formats=["gaussian", "glb", "mesh"]
   - Renders rotating 360° GIF using render_video()
   - Exports PLY with RGB colors from SH coefficients
   - Attempts to export textured GLB using to_glb() or native pipeline output
   - Saves files to assets/ folder with metadata
   - Returns URLs via stdout markers (GIF_DATA_START/END, MESH_URL_START/END, PLY_URL_START/END)
5. API extracts URLs from subprocess stdout and updates task status
6. Client polls /generate-3d-status/{task_id} to get base64-encoded files

## Furniture Analysis Pipeline (DeCl Integration)

The `/analyze-furniture` endpoint implements the full AI Logic pipeline:

1. **Image Fetching**: Download images from Firebase Storage URLs (5-10 images)
2. **Object Detection**: YOLO-World + SAHI for small object detection
3. **Classification**: CLIP for detailed subtype classification (e.g., queen bed vs king bed)
4. **Movability Check**: Compare with knowledge base to determine is_movable
5. **Mask Generation**: SAM2 creates segmentation masks using center point prompts
6. **3D Generation**: SAM-3D converts masked images to 3D models
7. **Volume Calculation**: trimesh analyzes mesh for relative dimensions
8. **Absolute Dimensions**: Match with DB furniture specs to get real-world measurements

### Response Format
```json
{
  "objects": [
    {
      "label": "퀸 사이즈 침대",
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
- `DeCl/services/furniture_pipeline.py` - Main pipeline orchestrator
- `DeCl/models/sahi_detector.py` - SAHI-enhanced YOLO detector
- `DeCl/utils/volume_calculator.py` - Mesh volume analysis
- `DeCl/data/knowledge_base.py` - Furniture dimensions database

## Code Modification Guidelines

### When Modifying SAM 2 Integration
- Read existing mask processing in api.py:segment_image and api.py:segment_binary
- Maintain morphological smoothing (cv2.morphologyEx) for mask quality
- Preserve the 4D input format for SAM 2: [[[[x, y]]]]

### When Modifying 3D Generation
- **Never** load Sam-3d-objects in api.py main process - always use subprocess
- Maintain environment variables at top of generate_3d_subprocess.py
- Use synthetic pinhole pointmap (make_synthetic_pointmap) instead of MoGe/dummy maps
- Check subprocess stdout for debug markers (PLY_URL_START/END, etc.)
- Test GLB export carefully - to_glb() requires mesh data and can fail with AttributeError

### When Adding New Endpoints
- Use background_tasks for any operation taking >5 seconds
- Return task_id immediately for async operations
- Store results in generation_tasks dict or persistent storage

### When Working with PLY Files
- PLY files are post-processed to add RGB from SH coefficients (add_rgb_to_ply)
- Uses ASCII format for compatibility (can be large files)
- Client must handle large base64 payloads when downloading

### When Debugging 3D Generation
- Check subprocess stdout/stderr logs (printed by _generate_3d_background)
- Verify Sam-3d-objects paths exist: ./sam-3d-objects/notebook and checkpoints
- Look for memory issues - peak GPU memory is logged
- Validate mask has sufficient pixels (>100 recommended, printed in subprocess logs)

## File Structure

```
api.py                          # Main FastAPI server (SAM 2 + task management + DeCl integration)
generate_3d_subprocess.py       # Isolated 3D generation worker
requirements.txt                # Python dependencies
setup.sh                        # Setup script (clones sam-3d-objects, creates conda env)
assets/                         # Static files (PLY, GIF, GLB) served at /assets/
sam-3d-objects/                 # Cloned Facebook Research repo (not in git)
  notebook/inference.py         # Sam-3d-objects pipeline classes
  checkpoints/hf/pipeline.yaml  # Pipeline configuration
DeCl/                           # Furniture detection module (integrated with main API)
  __init__.py                   # Module entry point
  main.py                       # Standalone CLI entry point
  config.py                     # YOLO/CLIP model configuration
  core/
    analyzer.py                 # Main analysis logic (standalone)
  models/
    detector.py                 # Standard YOLO-World detector
    sahi_detector.py            # SAHI-enhanced detector for small objects
    classifier.py               # CLIP classifier
    refiners/
      ac_refiner.py             # AC-specific refinement logic
  data/
    knowledge_base.py           # Furniture database with dimensions
  services/
    furniture_pipeline.py       # Integrated pipeline service (API)
  utils/
    image_ops.py                # Image processing utilities
    volume_calculator.py        # 3D mesh volume calculation
    output_manager.py           # Result serialization
```

## Known Issues & Workarounds

1. **spconv float64 errors**: Prevented by setting torch.set_default_dtype(torch.float32) before all imports
2. **Intrinsics recovery failures**: Avoided by using synthetic pinhole pointmap instead of MoGe
3. **GLB export AttributeError**: Occurs when mesh_data is a list of non-mesh objects - subprocess now catches and logs this
4. **Mesh generation CPU spike**: Gaussian-to-mesh conversion is disabled by default (too CPU intensive)
5. **Empty mask errors**: Subprocess validates mask has >0 pixels and warns if <100 pixels

## Integration Notes

This project is designed to work with the [Sam3D Mobile](https://github.com/andrisgauracs/sam3d-mobile) app.
