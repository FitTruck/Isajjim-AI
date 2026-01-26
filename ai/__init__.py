# AI - Furniture Detection and Classification Module
"""
AI (Artificial Intelligence) Module

이사 서비스를 위한 가구 탐지 및 분류 시스템

V2 Pipeline Processors (CLIP/SAHI/SAM2 제거):
    1. Firebase Storage에서 이미지 가져오기 (1_firebase_images_fetch.py)
    2. YOLOE-seg로 객체 탐지 + 마스크 (2_YOLO_detect.py)
    3. (CLIP 제거됨)
    4. DB 대조하여 한국어 라벨 매핑 (4_DB_movability_check.py)
    5. (SAM2 제거됨 - YOLOE-seg 마스크 직접 사용)
    6. SAM-3D로 3D 변환 (6_SAM3D_convert.py)
    7. 부피/치수 계산 (7_volume_calculate.py)

Directory Structure:
    ai/
    ├── pipeline/           # 통합 파이프라인 오케스트레이터
    ├── processors/         # AI Logic 단계별 모듈
    ├── subprocess/         # GPU 격리된 서브프로세스 워커
    ├── data/               # Knowledge Base (가구 DB)
    ├── utils/              # 유틸리티 (이미지 처리)
    └── config.py           # 설정

Usage:
    from ai.pipeline import FurniturePipeline
    from ai.processors import VolumeCalculator

    pipeline = FurniturePipeline()
    results = await pipeline.process_multiple_images(image_urls)
"""

__version__ = "4.1.0"  # V2 pipeline - SAM2 removed

# 주요 클래스 노출
from .pipeline import FurniturePipeline, DetectedObject, PipelineResult
from .processors import (
    ImageFetcher,
    YoloDetector,
    YoloWorldDetector,  # 하위 호환성 별칭
    MovabilityChecker,
    DimensionCalculator,
    VolumeCalculator  # 하위 호환성 별칭
)

__all__ = [
    # Pipeline
    'FurniturePipeline',
    'DetectedObject',
    'PipelineResult',
    # Processors
    'ImageFetcher',
    'YoloDetector',
    'YoloWorldDetector',  # 하위 호환성 별칭
    'MovabilityChecker',
    'DimensionCalculator',
    'VolumeCalculator'  # 하위 호환성 별칭
]
