"""
Health Check Routes

/health, /gpu-status, /assets-list endpoints
"""

import os
from datetime import datetime

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import device, ASSETS_DIR

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
    }


@router.get("/gpu-status")
async def gpu_status():
    """
    GPU pool status.

    Returns:
        {
            "total_gpus": int,
            "available_gpus": int,
            "pipelines_initialized": int,
            "gpus": {...}
        }
    """
    try:
        from ai.gpu import get_gpu_pool
        pool = get_gpu_pool()
        status = pool.get_status()

        # Add pipeline status
        pipelines_status = pool.get_pipelines_status()
        status["pipelines_initialized"] = pipelines_status["initialized_pipelines"]

        # Add per-GPU pipeline info
        for gpu_id, gpu_info in status["gpus"].items():
            gpu_info["has_pipeline"] = pipelines_status["gpus"].get(
                int(gpu_id), {}
            ).get("has_pipeline", False)

        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "total_gpus": 1 if torch.cuda.is_available() else 0,
            "available_gpus": 1 if torch.cuda.is_available() else 0,
            "pipelines_initialized": 0,
            "gpus": {}
        })


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
            if filename.endswith(".metadata.json"):
                continue

            filepath = os.path.join(ASSETS_DIR, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)

                created_at = None
                metadata_path = os.path.join(ASSETS_DIR, f"{filename}.metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            created_at = metadata.get("created_at")
                    except Exception:
                        pass

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
