"""
AI Pipeline Processors

AI Logic 단계별 모듈 (V3 Pipeline):
1. Firebase Storage에서 이미지 가져오기
2. YOLOE-seg로 객체 탐지 (마스크 포함)
3. (CLIP 제거됨)
4. DB 대조하여 한국어 라벨 매핑
5. (SAM2 제거됨 - YOLOE-seg 마스크 직접 사용)
6. SAM-3D Worker Pool로 3D 변환 (Legacy subprocess 방식 제거됨)
7. 치수(width, depth, height) 계산 (부피는 백엔드에서 계산)
8. 절대 부피 계산 (규칙기반)
9. Boxer 절대 치수 추정 (V3)
"""

import importlib

# 숫자/하이픈이 포함된 파일명을 위한 동적 import
_stage1 = importlib.import_module('.1_firebase_images_fetch', package='ai.processors')
_stage2 = importlib.import_module('.2_YOLO_detect', package='ai.processors')
# _stage3 = CLIP 제거됨
_stage4 = importlib.import_module('.4_DB_movability_check', package='ai.processors')
# _stage5 = SAM2 제거됨 (V2 파이프라인에서 YOLOE-seg 마스크 직접 사용)
# _stage6 = Legacy subprocess 방식 제거됨 (Worker Pool 사용)
_stage7 = importlib.import_module('.7_volume_calculate', package='ai.processors')
_stage8 = importlib.import_module('.8_absolute_volume_calculate', package='ai.processors')
_ply_preprocess = importlib.import_module('.ply_preprocessor', package='ai.processors')
_stage9 = importlib.import_module('.9_boxer_dimension', package='ai.processors')
_dim_strategy = importlib.import_module('.dimension_strategy', package='ai.processors')

# 클래스 노출
ImageFetcher = _stage1.ImageFetcher
YoloDetector = _stage2.YoloDetector
YoloWorldDetector = _stage2.YoloWorldDetector  # 하위 호환성 별칭
MovabilityChecker = _stage4.MovabilityChecker
MovabilityResult = _stage4.MovabilityResult
LabelMappingResult = _stage4.LabelMappingResult  # V2 별칭
DimensionCalculator = _stage7.DimensionCalculator
VolumeCalculator = _stage7.VolumeCalculator  # 하위 호환성 별칭
AbsoluteVolumeCalculator = _stage8.AbsoluteVolumeCalculator
AbsoluteVolumeResult = _stage8.AbsoluteVolumeResult
get_calculator = _stage8.get_calculator
PLYPreprocessor = _ply_preprocess.PLYPreprocessor
PreprocessResult = _ply_preprocess.PreprocessResult
preprocess_ply = _ply_preprocess.preprocess_ply
# Step 9: Boxer Dimension Estimation (V3)
BoxerDimensionEstimator = _stage9.BoxerDimensionEstimator
BoxerResult = _stage9.BoxerResult
get_boxer_estimator = _stage9.get_boxer_estimator
is_boxer_available = _stage9.is_boxer_available
# Dimension Strategy (Boxer + Fallback)
DimensionStrategy = _dim_strategy.DimensionStrategy
DimensionMode = _dim_strategy.DimensionMode

__all__ = [
    # Step 1-4: Detection
    'ImageFetcher',
    'YoloDetector',
    'YoloWorldDetector',  # 하위 호환성 별칭
    'MovabilityChecker',
    'MovabilityResult',
    'LabelMappingResult',  # V2 별칭
    # Step 7: Dimension (SAM-3D는 Worker Pool 통해 직접 사용)
    'DimensionCalculator',
    'VolumeCalculator',  # 하위 호환성 별칭
    # Step 8: Absolute Volume (절대 부피 계산)
    'AbsoluteVolumeCalculator',
    'AbsoluteVolumeResult',
    'get_calculator',
    # PLY Preprocessing (GCS 업로드 전 전처리)
    'PLYPreprocessor',
    'PreprocessResult',
    'preprocess_ply',
    # Step 9: Boxer Dimension Estimation (V3)
    'BoxerDimensionEstimator',
    'BoxerResult',
    'get_boxer_estimator',
    'is_boxer_available',
    # Dimension Strategy (Boxer + Fallback)
    'DimensionStrategy',
    'DimensionMode',
]
