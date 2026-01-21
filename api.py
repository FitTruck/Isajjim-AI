"""
SAM 2 Image Segmentation API + Sam-3d-objects 3D Generation

This file is a thin wrapper that imports the app from api/ package.
For development: uvicorn api:app --reload
For production: uvicorn api:app --workers 4

Endpoints:
- /health: Health check
- /gpu-status: GPU pool status
- /segment: Single point segmentation
- /segment-binary: Multi-point segmentation
- /generate-3d: 3D generation (async)
- /generate-3d-status/{task_id}: Poll 3D generation results
- /assets-list: List stored assets
- /analyze-furniture: Multi-image furniture analysis
- /analyze-furniture-single: Single image analysis
- /analyze-furniture-base64: Base64 image analysis
- /detect-furniture: Detection only (fast)
"""

# Import app from api package (this triggers config setup)
from api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
