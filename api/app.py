"""
FastAPI Application

Main application entry point with router registration and startup events.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import ASSETS_DIR
from api.routes import (
    health_router,
    generate_3d_router,
    furniture_router,
)

# Add AI module to path
ai_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai")
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

# Initialize FastAPI app
app = FastAPI(
    title="SAM-3D Furniture Analysis API",
    description="YOLOE-seg detection + SAM-3D 3D generation for furniture analysis",
    version="2.0.0",
)

# Mount assets folder as static files
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(generate_3d_router, tags=["3D Generation"])
app.include_router(furniture_router, tags=["Furniture Analysis"])


@app.on_event("startup")
async def startup_event():
    """Initialize GPU pool and SAM3D Worker Pool on API startup (parallel)"""
    import asyncio
    import time

    start_time = time.time()
    print("=" * 60)
    print("Starting parallel initialization: YOLOE + SAM3D Worker Pool")
    print("=" * 60)

    # Get GPU IDs first
    try:
        from ai.config import Config
        gpu_ids = Config.get_available_gpus()
        print(f"Available GPUs: {gpu_ids}")
    except Exception as e:
        print(f"GPU detection failed, using default [0]: {e}")
        gpu_ids = [0]

    # Define initialization tasks
    async def init_yoloe_pipelines():
        """Initialize YOLOE detection pipelines"""
        yoloe_start = time.time()
        try:
            from ai.gpu import initialize_gpu_pool
            from ai.pipeline import FurniturePipeline

            pool = initialize_gpu_pool(gpu_ids)
            print(f"[YOLOE] GPU pool initialized")

            def create_pipeline(gpu_id: int) -> FurniturePipeline:
                return FurniturePipeline(
                    sam2_api_url="http://localhost:8000",
                    enable_3d_generation=True,
                    device_id=gpu_id,
                    gpu_pool=pool
                )

            await pool.initialize_pipelines(create_pipeline, skip_on_error=True)
            elapsed = time.time() - yoloe_start
            print(f"[YOLOE] Pipelines initialized for {len(gpu_ids)} GPUs in {elapsed:.2f}s")
            return True
        except Exception as e:
            elapsed = time.time() - yoloe_start
            print(f"[YOLOE] Initialization failed in {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def init_sam3d_worker_pool():
        """Initialize SAM3D Worker Pool"""
        sam3d_start = time.time()
        try:
            from ai.gpu.sam3d_worker_pool import initialize_sam3d_worker_pool

            await initialize_sam3d_worker_pool(gpu_ids=gpu_ids)
            elapsed = time.time() - sam3d_start
            print(f"[SAM3D] Worker Pool initialized for {len(gpu_ids)} GPUs in {elapsed:.2f}s")
            return True
        except Exception as e:
            elapsed = time.time() - sam3d_start
            print(f"[SAM3D] Worker Pool initialization failed in {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run both initializations in parallel
    results = await asyncio.gather(
        init_yoloe_pipelines(),
        init_sam3d_worker_pool(),
        return_exceptions=True
    )

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Initialization complete in {total_elapsed:.2f}s")
    print(f"  YOLOE: {'OK' if results[0] is True else 'FAILED'}")
    print(f"  SAM3D: {'OK' if results[1] is True else 'FAILED'}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on API shutdown"""
    # Shutdown SAM3D Worker Pool
    try:
        from ai.gpu import shutdown_sam3d_worker_pool
        await shutdown_sam3d_worker_pool()
        print("SAM3D Worker Pool shutdown complete")
    except Exception as e:
        print(f"SAM3D Worker Pool shutdown error: {e}")
