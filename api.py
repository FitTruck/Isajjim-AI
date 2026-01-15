"""
SAM 2 Image Segmentation API + Sam-3d-objects 3D Generation

This API provides:
1. Image segmentation using Meta's Segment Anything Model 2 (SAM 2)
2. 3D object generation from masks using Sam-3d-objects

Endpoints:
- /segment: Get segmentation mask from a single point
- /segment-binary: Get segmentation mask with mask context support
- /generate-3d: Generate 3D Gaussian splat from image and mask
"""

import os

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# These must be set BEFORE any imports that use spconv
# ============================================================================
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

# Set spconv environment variables early (before any imports)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"  # Set to 100ms (was 0 = infinite tuning)
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"

# Prevent thread explosion - limit OpenMP threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import io
import base64
import numpy as np
import torch

# ============================================================================
# CRITICAL: Set PyTorch default dtype to float32 IMMEDIATELY
# This MUST be done before any other imports to prevent spconv float64 errors
# ============================================================================
torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import cv2
import json
import tempfile
import sys
import subprocess
import uuid
from typing import List, Dict, Optional
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# SAM 2 imports from transformers
from transformers import Sam2Processor, Sam2Model

# ============================================================================
# PYTORCH CONFIGURATION FOR SPCONV COMPATIBILITY
# Set default float dtype to float32 to prevent algorithm tuning errors
# ============================================================================
torch.set_default_dtype(torch.float32)

# Sam-3d-objects imports (optional - gracefully fail if not available)
try:
    sam3d_notebook_path = "./sam-3d-objects/notebook"
    if os.path.exists(sam3d_notebook_path):
        sys.path.insert(0, sam3d_notebook_path)
        from inference import Inference

        print(f"✓ Sam-3d-objects imported successfully")
    else:
        print(f"⚠ Sam-3d-objects notebook path not found at {sam3d_notebook_path}")
except Exception as e:
    print(f"⚠ Sam-3d-objects import failed: {e}")

# Configure device
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Initialize FastAPI app
app = FastAPI(
    title="SAM 2 Image Segmentation API",
    description="Segment objects in images using Segment Anything Model 2 (Hugging Face)",
    version="1.0.0",
)

# Create assets folder for downloadable files
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Mount assets folder as static files (accessible at /assets/)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Global model and processor instances
model = None
processor = None

# Task storage for async 3D generation
generation_tasks: Dict[str, Dict] = {}


def initialize_model():
    """Initialize SAM 2 model and processor from Hugging Face"""
    global model, processor

    try:
        model_id = "facebook/sam2.1-hiera-large"
        print(f"Loading model from {model_id}...")

        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device)

        print("✓ SAM 2 model and processor initialized successfully")

    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize models and GPU pool on API startup"""
    initialize_model()

    # GPU 풀 초기화
    try:
        from ai.gpu import initialize_gpu_pool, get_gpu_pool
        from ai.config import Config
        gpu_ids = Config.get_available_gpus()
        pool = initialize_gpu_pool(gpu_ids)
        print(f"✓ GPU pool initialized with {len(gpu_ids)} GPUs: {gpu_ids}")

        # GPU별 FurniturePipeline 사전 초기화
        try:
            from ai.pipeline import FurniturePipeline

            def create_pipeline(gpu_id: int) -> FurniturePipeline:
                """GPU별 파이프라인 팩토리"""
                return FurniturePipeline(
                    sam2_api_url="http://localhost:8000",
                    enable_3d_generation=True,
                    device_id=gpu_id,
                    gpu_pool=pool
                )

            await pool.initialize_pipelines(create_pipeline, skip_on_error=True)
            print(f"✓ Furniture pipelines pre-initialized for {len(gpu_ids)} GPUs")
        except Exception as e:
            print(f"⚠ Pipeline pre-initialization failed (will create on-demand): {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"⚠ GPU pool initialization failed (will use default): {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "device": str(device),
        "model": "facebook/sam2.1-hiera-large",
    }


@app.get("/gpu-status")
async def gpu_status():
    """
    GPU 풀 상태를 반환합니다.

    Returns:
        {
            "total_gpus": int,
            "available_gpus": int,
            "pipelines_initialized": int,
            "gpus": {
                "0": {"available": bool, "task_id": str|null, "memory_used_mb": float, "has_pipeline": bool, ...},
                ...
            }
        }
    """
    try:
        from ai.gpu import get_gpu_pool
        pool = get_gpu_pool()
        status = pool.get_status()
        # 파이프라인 상태 추가
        pipelines_status = pool.get_pipelines_status()
        status["pipelines_initialized"] = pipelines_status["initialized_pipelines"]
        # GPU별 파이프라인 여부 추가
        for gpu_id, gpu_info in status["gpus"].items():
            gpu_info["has_pipeline"] = pipelines_status["gpus"].get(int(gpu_id), {}).get("has_pipeline", False)
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "total_gpus": 1 if torch.cuda.is_available() else 0,
            "available_gpus": 1 if torch.cuda.is_available() else 0,
            "pipelines_initialized": 0,
            "gpus": {}
        })


class SegmentRequest(BaseModel):
    image: str  # base64 encoded image
    x: float  # X coordinate
    y: float  # Y coordinate
    multimask_output: bool = True  # Whether to return multiple masks
    mask_threshold: float = (
        0.0  # Threshold for mask logits (default: 0.0, use 0.5 for stricter)
    )
    invert_mask: bool = (
        False  # Whether to invert the mask (0=foreground, 255=background)
    )


@app.post("/segment")
async def segment_image(request: SegmentRequest):
    """
    Segment an object in an image based on a point coordinate.

    Args:
        request: JSON body containing:
            - image: Base64 encoded image string
            - x: X coordinate of the point (horizontal position)
            - y: Y coordinate of the point (vertical position)
            - multimask_output: Whether to return multiple mask predictions (default: True)

    Returns:
        JSON response containing:
        - masks: The segmentation masks as arrays
        - scores: Quality scores for each mask
        - input_point: The input point coordinate
        - image_shape: Dimensions of the input image
    """
    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image_pil)

        # Prepare input points and labels in the format expected by the processor
        # Format: [[[[x, y]]]] - 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
        input_points = [[[[request.x, request.y]]]]
        input_labels = [[[1]]]  # 1 for positive click, 0 for negative click

        # Process inputs
        inputs = processor(
            images=image_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        # Convert masks to list and get scores
        mask_list = []
        scores = (
            outputs.iou_preds[0].cpu().numpy().tolist()
            if hasattr(outputs, "iou_preds")
            else [0.95] * masks.shape[0]
        )

        for i in range(masks.shape[0]):
            mask = masks[i].numpy()
            # Squeeze extra dimensions and ensure 2D
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                mask = mask[0] if mask.ndim > 2 else mask

            # Threshold mask
            mask = (mask > request.mask_threshold).astype(np.uint8) * 255

            # Apply morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 127).astype(np.uint8) * 255

            # Invert if requested
            if request.invert_mask:
                mask = 255 - mask

            mask_image = Image.fromarray(mask, mode="L")
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            buffer.seek(0)
            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mask_list.append(
                {
                    "mask": mask_base64,
                    "mask_shape": mask.shape,
                    "score": float(scores[i]) if i < len(scores) else 0.95,
                }
            )

        return JSONResponse(
            {
                "success": True,
                "masks": mask_list,
                "input_point": [request.x, request.y],
                "image_shape": [image_pil.height, image_pil.width],
            }
        )

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


class SegmentBinaryRequest(BaseModel):
    image: str  # base64 encoded image
    points: List[Dict[str, float]]  # [{"x": float, "y": float}, ...]
    previous_mask: Optional[str] = None  # base64 PNG of previous mask (optional)
    mask_threshold: float = 0.0  # Threshold for mask logits


@app.post("/segment-binary")
async def segment_image_binary(request: SegmentBinaryRequest):
    """
    Segment an image and return the mask as base64 encoded PNG.
    """
    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Validate points
        if not request.points or len(request.points) == 0:
            return JSONResponse(
                status_code=400, content={"error": "At least one point required"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_pil_array = np.array(
            image_pil
        )  # Keep original image for color preservation

        # Convert points to the format expected by SAM 2
        # Process each point SEPARATELY to avoid losing segments when adding new points
        # Format: [[[[x, y]]]] - 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)

        # Collect masks from each point
        all_masks = []

        for point_idx, point in enumerate(request.points):

            # Process single point
            input_points = [[[[point["x"], point["y"]]]]]
            input_labels = [[[1]]]  # Positive point

            # Process inputs
            inputs = processor(
                images=image_pil,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(device)

            # Run inference for this single point
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process masks
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]

            # Get scores
            scores = (
                outputs.iou_preds[0].cpu().numpy()
                if hasattr(outputs, "iou_preds")
                else np.array([0.95] * masks.shape[0])
            )

            # Get best mask for this point
            best_mask_idx = np.argmax(scores)
            point_mask = masks[best_mask_idx].numpy()

            # Squeeze and ensure 2D
            point_mask = np.squeeze(point_mask)
            if point_mask.ndim != 2:
                point_mask = point_mask[0] if point_mask.ndim > 2 else point_mask

            # Apply threshold
            point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255

            all_masks.append(point_mask)

        # Union all masks from all points
        mask = all_masks[0].copy()
        for i in range(1, len(all_masks)):
            mask = np.maximum(mask, all_masks[i])

        # Add previous mask to the union (accumulate)
        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_array = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_array)
            except Exception:
                pass

        mask = (mask > request.mask_threshold).astype(np.uint8) * 255

        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_np = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_np)
            except Exception:
                pass

        # Apply morphological smoothing (less aggressive to preserve thin regions from multiple points)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Only use CLOSE (fill small holes) - skip OPEN which can eliminate thin connections
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Light gaussian blur instead of heavy filtering
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8) * 255

        # Check if mask is mostly white (inverted) - if mean > 127, invert it
        mask_mean = mask.mean()
        if mask_mean > 127:
            mask = 255 - mask

        # Verify dimensions match
        if image_pil_array.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image_pil_array.shape[1], image_pil_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert mask from 0-255 to 0-1 for multiplication
        mask_normalized = mask.astype(np.float32) / 255.0

        # Expand mask to 3 channels (R, G, B)
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Apply mask: foreground keeps original colors, background becomes black
        masked_image = (image_pil_array.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Convert to PNG and encode as base64
        masked_image_pil = Image.fromarray(masked_image, mode="RGB")
        buffer = io.BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        score = float(scores[best_mask_idx])

        return JSONResponse(
            {
                "success": True,
                "mask": mask_base64,
                "score": score,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


class Generate3dRequest(BaseModel):
    image: str  # base64 encoded image
    mask: str  # base64 encoded binary mask
    seed: int = 42


def _generate_3d_background(
    task_id: str, image_temp_path: str, mask_temp_path: str, seed: int
):
    """
    Background task for 3D generation.
    This function updates the generation_tasks dict with status and results.
    """
    ply_temp_path = None
    gif_temp_path = None

    try:
        generation_tasks[task_id]["status"] = "processing"

        # Create temp file for output PLY
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_temp_path = tmp.name
            gif_temp_path = ply_temp_path.replace(".ply", ".gif")

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess_script = os.path.join(script_dir, "ai", "subprocess", "generate_3d_worker.py")

        print(f"[Task {task_id}] Running 3D generation in subprocess...")

        # Run subprocess
        result = subprocess.run(
            [
                sys.executable,
                subprocess_script,
                image_temp_path,
                mask_temp_path,
                str(seed),
                ply_temp_path,
                ASSETS_DIR,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Print subprocess output for debugging
        if result.stdout:
            print(f"[Task {task_id}][Subprocess stdout]:\n{result.stdout}")
        if result.stderr:
            print(f"[Task {task_id}][Subprocess stderr]:\n{result.stderr}")

        # Check if subprocess succeeded
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(
                f"[Task {task_id}] Subprocess failed with return code {result.returncode}"
            )

            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id]["error"] = error_msg
            return

        # Extract GIF data from subprocess output
        gif_b64 = None
        if "GIF_DATA_START" in result.stdout and "GIF_DATA_END" in result.stdout:
            try:
                start_idx = result.stdout.find("GIF_DATA_START") + len("GIF_DATA_START")
                end_idx = result.stdout.find("GIF_DATA_END")
                gif_b64 = result.stdout[start_idx:end_idx].strip()
                print(
                    f"[Task {task_id}] ✓ Extracted GIF: {len(gif_b64)} chars (base64)"
                )
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract GIF data: {e}")

        # Extract mesh URL from subprocess output
        mesh_url = None
        if "MESH_URL_START" in result.stdout and "MESH_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("MESH_URL_START") + len("MESH_URL_START")
                end_idx = result.stdout.find("MESH_URL_END")
                mesh_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] ✓ Extracted mesh URL: {mesh_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract mesh URL: {e}")

        # Extract PLY URL from subprocess output
        ply_url = None
        if "PLY_URL_START" in result.stdout and "PLY_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("PLY_URL_START") + len("PLY_URL_START")
                end_idx = result.stdout.find("PLY_URL_END")
                ply_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] ✓ Extracted PLY URL: {ply_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract PLY URL: {e}")

        # Always read PLY as primary output
        ply_b64 = None
        ply_size_bytes = None

        if os.path.exists(ply_temp_path):
            print(f"[Task {task_id}] Reading PLY from {ply_temp_path}")
            with open(ply_temp_path, "rb") as f:
                ply_bytes = f.read()

            # Validate PLY header
            try:
                header_text = ply_bytes[: min(50000, len(ply_bytes))].decode(
                    "utf-8", errors="ignore"
                )
                if "end_header" not in header_text:
                    print(
                        f"[Task {task_id}] WARNING: PLY missing 'end_header' in first 50KB"
                    )
                    print(
                        f"[Task {task_id}] PLY appears to be binary, checking full file..."
                    )
                    # Check entire file
                    full_text = ply_bytes.decode("utf-8", errors="ignore")
                    if "end_header" not in full_text:
                        print(
                            f"[Task {task_id}] ERROR: PLY file corrupted or not ASCII format"
                        )
                    else:
                        print(
                            f"[Task {task_id}] Found end_header after 50KB - file is large but valid"
                        )
                else:
                    print(f"[Task {task_id}] ✓ PLY header valid (ASCII format)")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not validate PLY header: {e}")

            ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
            ply_size_bytes = len(ply_bytes)
            print(f"[Task {task_id}] ✓ PLY loaded: {ply_size_bytes} bytes")

        # GIF data was already extracted from subprocess stdout above
        gif_size_bytes = len(gif_b64) if gif_b64 else None

        # Determine primary output (for backward compatibility)
        output_b64 = ply_b64 if ply_b64 else gif_b64
        output_type = "ply" if ply_b64 else "gif"
        output_size_bytes = ply_size_bytes if ply_b64 else gif_size_bytes

        if output_b64:
            print(
                f"[Task {task_id}] ✓ 3D generation successful ({output_type}): {output_size_bytes} bytes"
            )
        else:
            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id][
                "error"
            ] = "Neither GIF nor PLY file was generated"
            return

        generation_tasks[task_id]["status"] = "completed"
        generation_tasks[task_id]["output_b64"] = output_b64
        generation_tasks[task_id]["output_type"] = output_type
        generation_tasks[task_id]["output_size_bytes"] = output_size_bytes
        generation_tasks[task_id]["ply_b64"] = ply_b64
        generation_tasks[task_id]["ply_size_bytes"] = ply_size_bytes
        generation_tasks[task_id]["ply_url"] = ply_url
        generation_tasks[task_id]["gif_b64"] = gif_b64
        generation_tasks[task_id]["gif_size_bytes"] = gif_size_bytes
        generation_tasks[task_id]["mesh_url"] = mesh_url
        generation_tasks[task_id]["progress"] = 100

    except subprocess.TimeoutExpired:
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id][
            "error"
        ] = "3D generation timed out (exceeded 10 minutes)"
    except Exception as e:
        print(f"[Task {task_id}] Error in 3D generation: {e}")
        import traceback

        traceback.print_exc()
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary files
        for path in [image_temp_path, mask_temp_path, ply_temp_path, gif_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"[Task {task_id}] Cleaned up temp file: {path}")
                except:
                    pass


@app.post("/generate-3d")
async def generate_3d(request: Generate3dRequest, background_tasks: BackgroundTasks):
    """
    Start 3D Gaussian splat generation (non-blocking, returns task ID).

    Returns immediately with a task_id that can be polled for results.
    This avoids gateway timeouts by returning immediately.

    Args:
        request: JSON body containing:
            - image: Base64 encoded RGB image (PNG or JPEG)
            - mask: Base64 encoded binary mask (0-1 grayscale)
            - seed: Random seed for reproducibility (default: 42)

    Returns:
        JSON response containing:
        - task_id: Unique ID to poll for results
        - status: "queued"
    """
    image_temp_path = None
    mask_temp_path = None

    try:
        # Decode base64 to temporary PNG files
        try:
            image_bytes = base64.b64decode(request.image)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_temp_path = tmp.name
                tmp.write(image_bytes)

            # Save for debugging
            image_pil_save = Image.open(image_temp_path).convert("RGB")
            image_pil_save.save("./test_img.png")
            print(f"✓ Saved incoming image as test_img.png")

            mask_bytes = base64.b64decode(request.mask)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                mask_temp_path = tmp.name
                tmp.write(mask_bytes)

            # Save for debugging
            mask_pil_save = Image.open(mask_temp_path).convert("L")
            mask_pil_save.save("./test_img_mask.png")
            print(f"✓ Saved incoming mask as test_img_mask.png")

        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid image or mask: {str(e)}"}
            )

        # Create unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task in storage
        generation_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": str(np.datetime64("now")),
        }

        # Add background task
        background_tasks.add_task(
            _generate_3d_background,
            task_id,
            image_temp_path,
            mask_temp_path,
            request.seed,
        )

        print(f"[API] Task {task_id} queued for 3D generation")

        return JSONResponse(
            {
                "success": True,
                "task_id": task_id,
                "status": "queued",
            }
        )

    except Exception as e:
        print(f"[API] Error creating 3D generation task: {e}")
        import traceback

        traceback.print_exc()

        # Clean up temp files on error
        for path in [image_temp_path, mask_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to queue 3D generation: {str(e)}"},
        )


@app.get("/generate-3d-status/{task_id}")
async def generate_3d_status(task_id: str):
    """
    Poll for 3D generation task status and results.

    Args:
        task_id: The task ID returned from /generate-3d

    Returns:
        JSON response containing:
        - status: "queued", "processing", "completed", or "failed"
        - progress: 0-100 (if applicable)
        - ply_b64: Base64 encoded PLY file (if completed)
        - error: Error message (if failed)
    """
    if task_id not in generation_tasks:
        return JSONResponse(
            status_code=404,
            content={"error": f"Task {task_id} not found"},
        )

    task = generation_tasks[task_id]

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
    }

    if task["status"] == "completed":
        response["ply_b64"] = task.get("output_b64")
        response["ply_size_bytes"] = task.get("output_size_bytes")
        response["gif_b64"] = task.get("gif_b64")
        response["gif_size_bytes"] = task.get("gif_size_bytes")
        response["mesh_url"] = task.get("mesh_url")

        # Encode mesh file to base64 if URL exists
        mesh_url = task.get("mesh_url")
        if mesh_url:
            mesh_filename = mesh_url.split("/")[-1]
            mesh_path = os.path.join(ASSETS_DIR, mesh_filename)

            # Detect mesh format from file extension
            if mesh_filename.endswith(".glb"):
                response["mesh_format"] = "glb"
            elif mesh_filename.endswith(".ply"):
                response["mesh_format"] = "ply"
            else:
                response["mesh_format"] = "unknown"

            if os.path.exists(mesh_path):
                try:
                    with open(mesh_path, "rb") as f:
                        mesh_bytes = f.read()
                    response["mesh_b64"] = base64.b64encode(mesh_bytes).decode("utf-8")
                    response["mesh_size_bytes"] = len(mesh_bytes)
                except Exception as e:
                    print(f"[API] Warning: Could not encode mesh to base64: {e}")
                    response["mesh_b64"] = None
                    response["mesh_size_bytes"] = 0
            else:
                print(f"[API] Warning: Mesh file not found at {mesh_path}")
                response["mesh_b64"] = None
                response["mesh_size_bytes"] = 0
        else:
            response["mesh_b64"] = None
            response["mesh_size_bytes"] = 0

        # Also include new naming convention
        response["output_b64"] = task.get("output_b64")
        response["output_type"] = task.get("output_type")  # "gif" or "ply"
        response["output_size_bytes"] = task.get("output_size_bytes")
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return JSONResponse(response)


@app.get("/assets-list")
async def list_assets():
    """
    List all available assets in the assets folder, sorted by creation date (newest first).

    Returns:
        JSON response containing:
        - files: List of file objects with name, size_bytes, url, and created_at
        - total_files: Total number of files
        - total_size_bytes: Total size of all files
    """
    if not os.path.exists(ASSETS_DIR):
        return JSONResponse({"files": [], "total_files": 0, "total_size_bytes": 0})

    files = []
    total_size = 0

    try:
        import json
        from datetime import datetime

        for filename in os.listdir(ASSETS_DIR):
            # Skip metadata files
            if filename.endswith(".metadata.json"):
                continue

            filepath = os.path.join(ASSETS_DIR, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)

                # Try to load metadata
                created_at = None
                metadata_path = os.path.join(ASSETS_DIR, f"{filename}.metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            created_at = metadata.get("created_at")
                    except:
                        created_at = None

                # Fallback to file modification time if metadata not available
                if not created_at:
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).isoformat()

                files.append(
                    {
                        "name": filename,
                        "size_bytes": size,
                        "url": f"/assets/{filename}",
                        "created_at": created_at,
                    }
                )
                total_size += size

        # Sort by creation date (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)

    except Exception as e:
        print(f"[API] Error listing assets: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list assets: {str(e)}"},
        )

    return JSONResponse(
        {
            "files": files,
            "total_files": len(files),
            "total_size_bytes": total_size,
        }
    )


# ============================================================================
# FURNITURE ANALYSIS ENDPOINTS (AI Integration)
# ============================================================================

# Add AI module to Python path
ai_path = os.path.join(os.path.dirname(__file__), "ai")
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

# Lazy-load furniture pipeline to avoid startup delays
furniture_pipeline = None


def get_furniture_pipeline(device_id: Optional[int] = None):
    """
    Get furniture pipeline - uses pre-initialized pipeline from GPU pool if available.

    Args:
        device_id: GPU 디바이스 ID (None이면 기본값 또는 풀에서 할당)

    Returns:
        Pre-initialized FurniturePipeline from GPU pool, or creates new one if not available
    """
    global furniture_pipeline

    # 먼저 GPU 풀에서 사전 초기화된 파이프라인 확인
    try:
        from ai.gpu import get_gpu_pool
        gpu_pool = get_gpu_pool()

        # device_id가 지정되지 않으면 기본 GPU (0) 사용
        target_gpu = device_id if device_id is not None else 0

        # 사전 초기화된 파이프라인이 있으면 사용
        if gpu_pool.has_pipeline(target_gpu):
            pre_initialized = gpu_pool.get_pipeline(target_gpu)
            if pre_initialized is not None:
                return pre_initialized
    except Exception as e:
        print(f"[get_furniture_pipeline] Could not get pre-initialized pipeline: {e}")

    # 사전 초기화된 파이프라인이 없으면 새로 생성 (폴백)
    if furniture_pipeline is None:
        try:
            from ai.pipeline import FurniturePipeline
            from ai.gpu import get_gpu_pool

            # GPU 풀 가져오기
            try:
                gpu_pool = get_gpu_pool()
            except Exception:
                gpu_pool = None

            furniture_pipeline = FurniturePipeline(
                sam2_api_url="http://localhost:8000",
                enable_3d_generation=True,
                use_sahi=True,
                device_id=device_id,
                gpu_pool=gpu_pool
            )
            print(f"✓ Furniture pipeline initialized (device_id={device_id}) [fallback - not pre-initialized]")
        except Exception as e:
            print(f"⚠ Failed to initialize furniture pipeline: {e}")
            raise
    return furniture_pipeline


class AnalyzeFurnitureRequest(BaseModel):
    """가구 분석 요청 모델"""
    image_urls: List[str]  # Firebase Storage URLs (5~10개)
    enable_mask: bool = True  # SAM2 마스크 생성
    enable_3d: bool = True  # SAM-3D 3D 생성
    max_concurrent: int = 3  # 최대 동시 처리 수


class AnalyzeFurnitureSingleRequest(BaseModel):
    """단일 이미지 가구 분석 요청 모델"""
    image_url: str  # Firebase Storage URL
    enable_mask: bool = True
    enable_3d: bool = True


@app.post("/analyze-furniture")
async def analyze_furniture(
    request: AnalyzeFurnitureRequest,
    background_tasks: BackgroundTasks
):
    """
    Firebase Storage 이미지 URL들을 받아 가구를 분석합니다.

    AI Logic Pipeline:
    1. Firebase Storage에서 이미지 다운로드 (5~10장)
    2. YOLO-World + SAHI로 객체 탐지 (작은 객체도 검출)
    3. CLIP으로 세부 유형 분류
    4. DB 대조하여 is_movable 결정
    5. SAM2로 마스크 생성
    6. SAM-3D로 3D 모델 생성
    7. trimesh로 부피/치수 계산
    8. DB 규격 대조하여 절대 치수 계산

    Args:
        request: AnalyzeFurnitureRequest with image_urls

    Returns:
        {
            "objects": [
                {
                    "label": "퀸 사이즈 침대",
                    "width": 1500.0 (mm),
                    "depth": 2000.0,
                    "height": 450.0,
                    "volume": 1.35 (liters),
                    "ratio": {"w": 0.75, "h": 1.0, "d": 0.225},
                    "is_movable": true
                },
                ...
            ],
            "summary": {
                "total_objects": 10,
                "movable_objects": 8,
                "total_volume_liters": 15.5,
                "movable_volume_liters": 12.3
            }
        }
    """
    try:
        pipeline = get_furniture_pipeline()

        # 이미지 URL 수 검증
        if len(request.image_urls) < 1:
            return JSONResponse(
                status_code=400,
                content={"error": "At least 1 image URL required"}
            )

        if len(request.image_urls) > 20:
            return JSONResponse(
                status_code=400,
                content={"error": "Maximum 20 image URLs allowed"}
            )

        # 파이프라인 실행
        import asyncio
        results = await pipeline.process_multiple_images(
            request.image_urls,
            enable_mask=request.enable_mask,
            enable_3d=request.enable_3d,
            max_concurrent=request.max_concurrent
        )

        # JSON 응답 생성
        response = pipeline.to_json_response(results)
        response["success"] = True

        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


@app.post("/analyze-furniture-single")
async def analyze_furniture_single(request: AnalyzeFurnitureSingleRequest):
    """
    단일 이미지 가구 분석 (빠른 테스트용)

    Args:
        request: AnalyzeFurnitureSingleRequest with image_url

    Returns:
        Same format as /analyze-furniture
    """
    try:
        pipeline = get_furniture_pipeline()

        result = await pipeline.process_single_image(
            request.image_url,
            enable_mask=request.enable_mask,
            enable_3d=request.enable_3d
        )

        response = pipeline.to_json_response([result])
        response["success"] = True
        response["processing_time_seconds"] = result.processing_time_seconds

        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


class AnalyzeFurnitureBase64Request(BaseModel):
    """Base64 이미지 가구 분석 요청 모델"""
    image: str  # Base64 encoded image
    enable_mask: bool = True
    enable_3d: bool = True


@app.post("/analyze-furniture-base64")
async def analyze_furniture_base64(request: AnalyzeFurnitureBase64Request):
    """
    Base64 이미지를 받아 가구를 분석합니다.
    Firebase URL 없이 직접 이미지를 전송할 때 사용합니다.

    Args:
        request: Base64 encoded image

    Returns:
        {
            "objects": [...],
            "summary": {...}
        }
    """
    try:
        pipeline = get_furniture_pipeline()

        # Base64 디코딩
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid base64 image: {str(e)}"}
            )

        import time
        start_time = time.time()

        # 객체 탐지
        detected_objects = pipeline.detect_objects(image)

        objects_data = []
        for obj in detected_objects:
            # SAM2 마스크 생성
            if request.enable_mask:
                mask_b64 = await pipeline.generate_mask(image, obj.center_point)
                obj.mask_base64 = mask_b64

            # SAM-3D 3D 생성 (선택적)
            if request.enable_3d and obj.mask_base64:
                gen_result = await pipeline.generate_3d(image, obj.mask_base64)
                if gen_result:
                    obj.glb_url = gen_result.get("mesh_url")

                    # 부피 계산
                    if gen_result.get("ply_b64"):
                        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                            tmp.write(base64.b64decode(gen_result["ply_b64"]))
                            ply_path = tmp.name

                        rel_dims, abs_dims = pipeline.calculate_dimensions(obj, ply_path=ply_path)
                        obj.relative_dimensions = rel_dims
                        obj.absolute_dimensions = abs_dims

                        os.unlink(ply_path)

            # 객체 데이터 구성
            obj_data = {
                "label": obj.label,
                "is_movable": obj.is_movable,
                "confidence": round(obj.confidence, 3),
                "bbox": obj.bbox,
                "center_point": obj.center_point
            }

            if obj.absolute_dimensions:
                dims = obj.absolute_dimensions
                obj_data.update({
                    "width": dims.get("width", 0),
                    "depth": dims.get("depth", 0),
                    "height": dims.get("height", 0),
                    "volume": dims.get("volume_liters", 0),
                    "ratio": dims.get("ratio", {"w": 1, "h": 1, "d": 1})
                })

            if obj.mask_base64:
                obj_data["mask"] = obj.mask_base64

            if obj.glb_url:
                obj_data["mesh_url"] = obj.glb_url

            objects_data.append(obj_data)

        processing_time = time.time() - start_time

        # 요약 계산
        total_volume = sum(
            o.get("volume", 0) for o in objects_data
        )
        movable_volume = sum(
            o.get("volume", 0) for o in objects_data if o.get("is_movable", True)
        )

        return JSONResponse({
            "success": True,
            "objects": objects_data,
            "summary": {
                "total_objects": len(objects_data),
                "movable_objects": sum(1 for o in objects_data if o.get("is_movable", True)),
                "fixed_objects": sum(1 for o in objects_data if not o.get("is_movable", True)),
                "total_volume_liters": round(total_volume, 2),
                "movable_volume_liters": round(movable_volume, 2)
            },
            "processing_time_seconds": round(processing_time, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


@app.post("/detect-furniture")
async def detect_furniture_only(request: AnalyzeFurnitureBase64Request):
    """
    가구 탐지만 수행합니다 (3D 생성 없음, 빠른 응답용)

    YOLO-World + SAHI + CLIP 분류만 수행하고
    SAM2/SAM-3D는 건너뜁니다.

    Returns:
        {
            "objects": [
                {
                    "label": "소파",
                    "bbox": [x1, y1, x2, y2],
                    "center_point": [cx, cy],
                    "is_movable": true,
                    "confidence": 0.95
                },
                ...
            ]
        }
    """
    try:
        pipeline = get_furniture_pipeline()

        # Base64 디코딩
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid base64 image: {str(e)}"}
            )

        import time
        start_time = time.time()

        # 객체 탐지만 수행
        detected_objects = pipeline.detect_objects(image)

        objects_data = []
        for obj in detected_objects:
            objects_data.append({
                "id": obj.id,
                "label": obj.label,
                "db_key": obj.db_key,
                "subtype": obj.subtype_name,
                "bbox": obj.bbox,
                "center_point": obj.center_point,
                "is_movable": obj.is_movable,
                "confidence": round(obj.confidence, 3)
            })

        processing_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "objects": objects_data,
            "total_objects": len(objects_data),
            "movable_objects": sum(1 for o in objects_data if o.get("is_movable", True)),
            "processing_time_seconds": round(processing_time, 3)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Detection failed: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
