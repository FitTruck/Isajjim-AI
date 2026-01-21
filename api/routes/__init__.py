"""
API Routes
"""

from .health import router as health_router
from .segment import router as segment_router
from .generate_3d import router as generate_3d_router
from .furniture import router as furniture_router

__all__ = [
    "health_router",
    "segment_router",
    "generate_3d_router",
    "furniture_router",
]
