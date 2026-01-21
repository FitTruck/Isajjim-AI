"""
AI Pipeline Processors

AI Logic 단계별 모듈 (V2 Pipeline):
1. Firebase Storage에서 이미지 가져오기
2. YOLOE-seg로 객체 탐지 (마스크 포함)
3. (CLIP 제거됨)
4. DB 대조하여 한국어 라벨 매핑
5. (SAM2 제거됨 - YOLOE-seg 마스크 직접 사용)
6. SAM-3D로 3D 변환
7. 상대 부피/치수 계산 (절대 부피는 백엔드)
"""

import importlib

# 숫자/하이픈이 포함된 파일명을 위한 동적 import
_stage1 = importlib.import_module('.1_firebase_images_fetch', package='ai.processors')
_stage2 = importlib.import_module('.2_YOLO_detect', package='ai.processors')
# _stage3 = CLIP 제거됨
_stage4 = importlib.import_module('.4_DB_movability_check', package='ai.processors')
# _stage5 = SAM2 제거됨 (V2 파이프라인에서 YOLOE-seg 마스크 직접 사용)
_stage6 = importlib.import_module('.6_SAM3D_convert', package='ai.processors')
_stage7 = importlib.import_module('.7_volume_calculate', package='ai.processors')

# 클래스 노출
ImageFetcher = _stage1.ImageFetcher
YoloDetector = _stage2.YoloDetector
YoloWorldDetector = _stage2.YoloWorldDetector  # 하위 호환성 별칭
MovabilityChecker = _stage4.MovabilityChecker
MovabilityResult = _stage4.MovabilityResult
LabelMappingResult = _stage4.LabelMappingResult  # V2 별칭
SAM3DConverter = _stage6.SAM3DConverter
SAM3DResult = _stage6.SAM3DResult
VolumeCalculator = _stage7.VolumeCalculator

__all__ = [
    # Step 1-4: Detection
    'ImageFetcher',
    'YoloDetector',
    'YoloWorldDetector',  # 하위 호환성 별칭
    'MovabilityChecker',
    'MovabilityResult',
    'LabelMappingResult',  # V2 별칭
    # Step 6-7: SAM-3D (SAM2 removed in V2)
    'SAM3DConverter',
    'SAM3DResult',
    'VolumeCalculator'
]
