"""
Pydantic Request/Response Models
"""

from typing import List, Dict, Optional
from pydantic import BaseModel


# ============================================================================
# Segment Models
# ============================================================================

class SegmentRequest(BaseModel):
    """Single point segmentation request"""
    image: str  # base64 encoded image
    x: float
    y: float
    multimask_output: bool = True
    mask_threshold: float = 0.0
    invert_mask: bool = False


class SegmentBinaryRequest(BaseModel):
    """Multi-point segmentation request"""
    image: str  # base64 encoded image
    points: List[Dict[str, float]]  # [{"x": float, "y": float}, ...]
    previous_mask: Optional[str] = None  # base64 PNG
    mask_threshold: float = 0.0


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
    image_urls: List[ImageUrlItem]
    enable_mask: bool = True
    enable_3d: bool = True
    max_concurrent: int = 3


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
