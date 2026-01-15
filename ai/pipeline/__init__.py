"""
AI Furniture Analysis Pipeline

전체 AI 로직을 통합한 파이프라인:
1~4: DeCl processors (탐지, 분류, is_movable)
5~7: SAM2/SAM-3D 연동 (마스크, 3D, 부피 계산)
"""

from .furniture_pipeline import FurniturePipeline, DetectedObject, PipelineResult

__all__ = [
    'FurniturePipeline',
    'DetectedObject',
    'PipelineResult'
]
