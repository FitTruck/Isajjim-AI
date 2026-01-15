# AI - Furniture Detection and Classification Module
"""
AI (Artificial Intelligence) Module

이사 서비스를 위한 가구 탐지 및 분류 시스템

AI Logic Processors:
    1. Firebase Storage에서 이미지 가져오기 (1_firebase_images_fetch.py)
    2. YOLO-World로 객체 탐지 (2_Yolo-World_detect.py)
    3. CLIP으로 세부 분류 (3_CLIP_classify.py)
    4. DB 대조하여 is_movable 결정 (4_DB_movability_check.py)
    5. SAM2로 마스크 생성 (5_SAM2_mask_generate.py)
    6. SAM-3D로 3D 변환 (6_SAM3D_convert.py)
    7. 부피/치수 계산 (7_volume_calculate.py)

Directory Structure:
    ai/
    ├── pipeline/           # 통합 파이프라인 오케스트레이터
    ├── processors/         # AI Logic 단계별 모듈 (1~7)
    ├── subprocess/         # GPU 격리된 서브프로세스 워커
    ├── data/               # Knowledge Base (가구 DB)
    ├── utils/              # 유틸리티 (이미지 처리)
    └── config.py           # 설정

Usage:
    from ai.pipeline import FurniturePipeline
    from ai.processors import SAM2MaskGenerator, SAM3DConverter

    pipeline = FurniturePipeline()
    results = await pipeline.process_multiple_images(image_urls)
"""

__version__ = "3.0.0"

# 주요 클래스 노출
from .pipeline import FurniturePipeline, DetectedObject, PipelineResult
from .processors import (
    ImageFetcher,
    YoloWorldDetector,
    ClipClassifier,
    MovabilityChecker,
    SAM2MaskGenerator,
    SAM3DConverter,
    SAM3DResult,
    VolumeCalculator
)

__all__ = [
    # Pipeline
    'FurniturePipeline',
    'DetectedObject',
    'PipelineResult',
    # Processors
    'ImageFetcher',
    'YoloWorldDetector',
    'ClipClassifier',
    'MovabilityChecker',
    'SAM2MaskGenerator',
    'SAM3DConverter',
    'SAM3DResult',
    'VolumeCalculator'
]
