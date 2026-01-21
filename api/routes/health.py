"""
Health Check Routes

/health, /gpu-status endpoints
"""

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import device
from api.services.sam2 import is_model_loaded

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "device": str(device),
        "model": "facebook/sam2.1-hiera-large",
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
