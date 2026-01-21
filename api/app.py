"""
FastAPI Application

Main application entry point with router registration and startup events.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import ASSETS_DIR
from api.services.sam2 import initialize_model
from api.routes import (
    health_router,
    segment_router,
    generate_3d_router,
    furniture_router,
)

# Add AI module to path
ai_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

# Initialize FastAPI app
app = FastAPI(
    title="SAM 2 Image Segmentation API",
    description="Segment objects in images using Segment Anything Model 2 (Hugging Face)",
    version="2.0.0",
)

# Mount assets folder as static files
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(segment_router, tags=["Segmentation"])
app.include_router(generate_3d_router, tags=["3D Generation"])
app.include_router(furniture_router, tags=["Furniture Analysis"])


@app.on_event("startup")
async def startup_event():
    """Initialize models and GPU pool on API startup"""
    initialize_model()

    # Initialize GPU pool
    try:
        from ai.gpu import initialize_gpu_pool
        from ai.config import Config

        gpu_ids = Config.get_available_gpus()
        pool = initialize_gpu_pool(gpu_ids)
        print(f"GPU pool initialized with {len(gpu_ids)} GPUs: {gpu_ids}")

        # Pre-initialize pipelines per GPU
        try:
            from ai.pipeline import FurniturePipeline

            def create_pipeline(gpu_id: int) -> FurniturePipeline:
                return FurniturePipeline(
                    sam2_api_url="http://localhost:8000",
                    enable_3d_generation=True,
                    device_id=gpu_id,
                    gpu_pool=pool
                )

            await pool.initialize_pipelines(create_pipeline, skip_on_error=True)
            print(f"Furniture pipelines pre-initialized for {len(gpu_ids)} GPUs")
        except Exception as e:
            print(f"Pipeline pre-initialization failed (will create on-demand): {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"GPU pool initialization failed (will use default): {e}")
