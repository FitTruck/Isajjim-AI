# DeCl - Furniture Detection and Classification Module
"""
DeCl (Detection & Classification) Module

이사 서비스를 위한 가구 탐지 및 분류 시스템:
- YOLO-World: 객체 탐지
- SAHI: 작은 객체 탐지 향상
- CLIP: 세부 유형 분류
- 지식 베이스: 가구 규격 및 is_movable 정보

Usage:
    from DeCl.services import FurniturePipeline

    pipeline = FurniturePipeline()
    results = await pipeline.process_multiple_images(image_urls)
"""

__version__ = "1.0.0"
