"""
API Package

Re-exports the FastAPI app for uvicorn.
"""

from api.app import app

__all__ = ["app"]
