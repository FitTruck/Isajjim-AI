"""
Pydantic Request/Response Models
"""

from typing import List
from pydantic import BaseModel


# ============================================================================
# Furniture Analysis Models
# ============================================================================

class ImageUrlItem(BaseModel):
    """Image URL item (TDD Section 4.1)"""
    id: int  # User-specified image ID
    url: str  # Firebase Storage URL


class AnalyzeFurnitureRequest(BaseModel):
    """Multi-image furniture analysis request (TDD Section 4.1)"""
    estimate_id: int  # Required: 견적 ID (callback URL에 사용)
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
    image: str  # Base64 encoded image (or 'image_base64' alias)
    enable_mask: bool = True
    enable_3d: bool = True
    skip_gif: bool = True  # GIF 렌더링 스킵 (부피 계산 최적화, 기본값: True)
    max_image_size: int = 512  # 3D 생성 시 최대 이미지 크기 (속도 최적화)
    return_ply: bool = False  # PLY base64 데이터 반환 여부 (테스트용)
