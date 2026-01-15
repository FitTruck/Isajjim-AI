"""
AI Pipeline Processors

AI Logic 단계별 모듈:
1. Firebase Storage에서 이미지 가져오기
2. YOLO-World로 객체 탐지
3. CLIP으로 세부 분류
4. DB 대조하여 is_movable 결정
5. SAM2로 마스크 생성
6. SAM-3D로 3D 변환
7. 부피/치수 계산
"""

import importlib

# 숫자/하이픈이 포함된 파일명을 위한 동적 import
_stage1 = importlib.import_module('.1_firebase_images_fetch', package='ai.processors')
_stage2 = importlib.import_module('.2_Yolo-World_detect', package='ai.processors')
_stage3 = importlib.import_module('.3_CLIP_classify', package='ai.processors')
_stage4 = importlib.import_module('.4_DB_movability_check', package='ai.processors')
_stage5 = importlib.import_module('.5_SAM2_mask_generate', package='ai.processors')
_stage6 = importlib.import_module('.6_SAM3D_convert', package='ai.processors')
_stage7 = importlib.import_module('.7_volume_calculate', package='ai.processors')

# 클래스 노출
ImageFetcher = _stage1.ImageFetcher
YoloWorldDetector = _stage2.YoloWorldDetector
ClipClassifier = _stage3.ClipClassifier
ACRefiner = _stage3.ACRefiner
MovabilityChecker = _stage4.MovabilityChecker
MovabilityResult = _stage4.MovabilityResult
SAM2MaskGenerator = _stage5.SAM2MaskGenerator
SAM3DConverter = _stage6.SAM3DConverter
SAM3DResult = _stage6.SAM3DResult
VolumeCalculator = _stage7.VolumeCalculator
estimate_dimensions_from_aspect_ratio = _stage7.estimate_dimensions_from_aspect_ratio

__all__ = [
    # Step 1-4: DeCl
    'ImageFetcher',
    'YoloWorldDetector',
    'ClipClassifier',
    'ACRefiner',
    'MovabilityChecker',
    'MovabilityResult',
    # Step 5-7: SAM2/SAM-3D
    'SAM2MaskGenerator',
    'SAM3DConverter',
    'SAM3DResult',
    'VolumeCalculator',
    'estimate_dimensions_from_aspect_ratio'
]
