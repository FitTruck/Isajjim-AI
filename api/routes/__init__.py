"""
API Routes
"""

from .health import router as health_router
from .furniture import router as furniture_router

__all__ = [
    "health_router",
    "furniture_router",
]
