"""
3D Generation Routes

/generate-3d, /generate-3d-status, /assets-list endpoints
"""

import os
import base64
import tempfile
import uuid
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from api.config import ASSETS_DIR
from api.models import Generate3dRequest
from api.services.tasks import generation_tasks, generate_3d_background

router = APIRouter()


@router.post("/generate-3d")
async def generate_3d(request: Generate3dRequest, background_tasks: BackgroundTasks):
    """
    Start 3D Gaussian splat generation (non-blocking, returns task ID).

    Returns immediately with a task_id that can be polled for results.
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
            print("Saved incoming image as test_img.png")

            mask_bytes = base64.b64decode(request.mask)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                mask_temp_path = tmp.name
                tmp.write(mask_bytes)

            # Save for debugging
            mask_pil_save = Image.open(mask_temp_path).convert("L")
            mask_pil_save.save("./test_img_mask.png")
            print("Saved incoming mask as test_img_mask.png")

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

        # Add background task with optimization options
        background_tasks.add_task(
            generate_3d_background,
            task_id,
            image_temp_path,
            mask_temp_path,
            request.seed,
            skip_gif=request.skip_gif,
            max_image_size=request.max_image_size,
        )

        print(f"[API] Task {task_id} queued for 3D generation")

        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "status": "queued",
        })

    except Exception as e:
        print(f"[API] Error creating 3D generation task: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp files on error
        for path in [image_temp_path, mask_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to queue 3D generation: {str(e)}"},
        )


@router.get("/generate-3d-status/{task_id}")
async def generate_3d_status(task_id: str):
    """
    Poll for 3D generation task status and results.
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

            # Detect mesh format
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

        response["output_b64"] = task.get("output_b64")
        response["output_type"] = task.get("output_type")
        response["output_size_bytes"] = task.get("output_size_bytes")

    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return JSONResponse(response)


@router.get("/assets-list")
async def list_assets():
    """
    List all available assets in the assets folder, sorted by creation date.
    """
    if not os.path.exists(ASSETS_DIR):
        return JSONResponse({"files": [], "total_files": 0, "total_size_bytes": 0})

    files = []
    total_size = 0

    try:
        import json

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
                    except Exception:
                        pass

                # Fallback to file modification time
                if not created_at:
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).isoformat()

                files.append({
                    "name": filename,
                    "size_bytes": size,
                    "url": f"/assets/{filename}",
                    "created_at": created_at,
                })
                total_size += size

        # Sort by creation date (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)

    except Exception as e:
        print(f"[API] Error listing assets: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list assets: {str(e)}"},
        )

    return JSONResponse({
        "files": files,
        "total_files": len(files),
        "total_size_bytes": total_size,
    })
