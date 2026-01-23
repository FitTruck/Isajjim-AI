"""
Pydantic Request/Response Models
"""

from typing import List, Optional
from pydantic import BaseModel


# ============================================================================
# Generate 3D Models
# ============================================================================

class Generate3dRequest(BaseModel):
    """3D generation request"""
    image: str  # base64 encoded image
    mask: str  # base64 encoded binary mask
    seed: int = 42
    skip_gif: bool = True  # GIF 렌더링 스킵 (부피 계산 최적화, 기본값: True)
    max_image_size: int = 512  # 최대 이미지 크기 (속도 최적화)


# ============================================================================
# Furniture Analysis Models
# ============================================================================

class ImageUrlItem(BaseModel):
    """Image URL item (TDD Section 4.1)"""
    id: int  # User-specified image ID
    url: str  # Firebase Storage URL


class AnalyzeFurnitureRequest(BaseModel):
    """Multi-image furniture analysis request (TDD Section 4.1)"""
    estimate_id: int  # Backend estimate ID (used in callback URL path)
    image_urls: List[ImageUrlItem]
    callback_url: Optional[str] = None  # Callback URL with {estimateId} placeholder
    enable_mask: bool = True
    enable_3d: bool = True
    max_concurrent: int = 3


# ============================================================================
# Callback Models
# ============================================================================

class CallbackPayload(BaseModel):
    """Callback API payload sent to backend server"""
    task_id: str                            # AI server generated task ID
    status: str                             # "completed" | "failed"
    results: Optional[List[dict]] = None    # Analysis results (TDD format)
    error: Optional[str] = None             # Error message on failure


class AnalyzeFurnitureSingleRequest(BaseModel):
    """Single image furniture analysis request"""
    image_url: str  # Firebase Storage URL
    enable_mask: bool = True
    enable_3d: bool = True


class AnalyzeFurnitureBase64Request(BaseModel):
    """Base64 image furniture analysis request"""
    image: str  # Base64 encoded image
    enable_mask: bool = True
    enable_3d: bool = True
    skip_gif: bool = True  # GIF 렌더링 스킵 (부피 계산 최적화, 기본값: True)
    max_image_size: int = 512  # 3D 생성 시 최대 이미지 크기 (속도 최적화)
