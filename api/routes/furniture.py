"""
Furniture Analysis Routes

/analyze-furniture, /analyze-furniture-single, /analyze-furniture-base64, /detect-furniture
"""

import os
import io
import base64
import time
import tempfile
from typing import Optional

from PIL import Image
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models import (
    AnalyzeFurnitureRequest,
    AnalyzeFurnitureSingleRequest,
    AnalyzeFurnitureBase64Request,
)

router = APIRouter()

# Lazy-load furniture pipeline
_furniture_pipeline = None


def get_furniture_pipeline(device_id: Optional[int] = None):
    """
    Get furniture pipeline - uses pre-initialized pipeline from GPU pool if available.
    """
    global _furniture_pipeline

    # Try to get pre-initialized pipeline from GPU pool
    try:
        from ai.gpu import get_gpu_pool
        gpu_pool = get_gpu_pool()

        target_gpu = device_id if device_id is not None else 0

        if gpu_pool.has_pipeline(target_gpu):
            pre_initialized = gpu_pool.get_pipeline(target_gpu)
            if pre_initialized is not None:
                return pre_initialized
    except Exception as e:
        print(f"[get_furniture_pipeline] Could not get pre-initialized pipeline: {e}")

    # Fallback: create new pipeline
    if _furniture_pipeline is None:
        try:
            from ai.pipeline import FurniturePipeline
            from ai.gpu import get_gpu_pool

            try:
                gpu_pool = get_gpu_pool()
            except Exception:
                gpu_pool = None

            _furniture_pipeline = FurniturePipeline(
                sam2_api_url="http://localhost:8000",
                enable_3d_generation=True,
                device_id=device_id,
                gpu_pool=gpu_pool
            )
            print(f"Furniture pipeline initialized (device_id={device_id}) [fallback]")
        except Exception as e:
            print(f"Failed to initialize furniture pipeline: {e}")
            raise

    return _furniture_pipeline


@router.post("/analyze-furniture")
async def analyze_furniture(
    request: AnalyzeFurnitureRequest,
    background_tasks: BackgroundTasks
):
    """
    Firebase Storage image URLs to analyze furniture (TDD Section 4.1).

    Pipeline V2:
    1. Download images from Firebase Storage
    2. YOLOE-seg object detection (bbox + class + segmentation)
    3. DB matching for Korean labels
    4. YOLOE-seg mask directly to SAM-3D (no SAM2)
    5. SAM-3D 3D model generation
    6. trimesh relative volume calculation

    Returns:
        {"results": [{"image_id": 101, "objects": [...]}, ...]}
    """
    try:
        pipeline = get_furniture_pipeline()

        # Validate image URL count
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

        # Convert to [(id, url), ...] format
        image_items = [(item.id, item.url) for item in request.image_urls]

        # Run pipeline
        results = await pipeline.process_multiple_images_with_ids(
            image_items,
            enable_mask=request.enable_mask,
            enable_3d=request.enable_3d,
            max_concurrent=request.max_concurrent
        )

        # Generate response (TDD format)
        response = pipeline.to_json_response_v2(results)
        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


@router.post("/analyze-furniture-single")
async def analyze_furniture_single(request: AnalyzeFurnitureSingleRequest):
    """
    Single image furniture analysis (for quick testing).
    """
    try:
        pipeline = get_furniture_pipeline()

        result = await pipeline.process_single_image(
            request.image_url,
            enable_mask=request.enable_mask,
            enable_3d=request.enable_3d
        )

        response = pipeline.to_json_response([result])
        return JSONResponse(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


@router.post("/analyze-furniture-base64")
async def analyze_furniture_base64(request: AnalyzeFurnitureBase64Request):
    """
    Base64 image furniture analysis (no Firebase URL).

    Returns (TDD format):
        {"objects": [{"label": "box", "width": 30.5, ...}]}
    """
    try:
        pipeline = get_furniture_pipeline()

        # Decode base64
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Detect objects
        detected_objects = pipeline.detect_objects(image)

        objects_data = []
        for obj in detected_objects:
            # YOLOE-seg mask
            if request.enable_mask and obj.yolo_mask is not None:
                mask_b64 = pipeline._yolo_mask_to_base64(obj.yolo_mask)
                obj.mask_base64 = mask_b64

            # SAM-3D 3D generation
            if request.enable_3d and obj.mask_base64:
                gen_result = await pipeline.generate_3d(image, obj.mask_base64)
                if gen_result and gen_result.get("ply_b64"):
                    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                        tmp.write(base64.b64decode(gen_result["ply_b64"]))
                        ply_path = tmp.name

                    rel_dims, _ = pipeline.calculate_dimensions(obj, ply_path=ply_path)
                    obj.relative_dimensions = rel_dims

                    os.unlink(ply_path)

            # TDD format: 5 fields only
            if obj.relative_dimensions:
                dims = obj.relative_dimensions
                bbox = dims.get("bounding_box", {})

                # Volume conversion
                volume_raw = dims.get("volume", 0)
                volume_m3 = volume_raw / 1e9 if volume_raw > 1000 else volume_raw

                objects_data.append({
                    "label": obj.label,
                    "width": round(bbox.get("width", 0), 2),
                    "depth": round(bbox.get("depth", 0), 2),
                    "height": round(bbox.get("height", 0), 2),
                    "volume": round(volume_m3, 6)
                })

        return JSONResponse({"objects": objects_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Furniture analysis failed: {str(e)}"}
        )


@router.post("/detect-furniture")
async def detect_furniture_only(request: AnalyzeFurnitureBase64Request):
    """
    Detection only (no 3D generation, fast response).

    Returns:
        {"objects": [{"label": "sofa", "bbox": [...], ...}]}
    """
    try:
        pipeline = get_furniture_pipeline()

        # Decode base64
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid base64 image: {str(e)}"}
            )

        start_time = time.time()

        # Detection only
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
                "confidence": round(obj.confidence, 3)
            })

        processing_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "objects": objects_data,
            "total_objects": len(objects_data),
            "processing_time_seconds": round(processing_time, 3)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Detection failed: {str(e)}"}
        )
