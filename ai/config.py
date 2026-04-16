import torch
import os
from typing import List, Optional, Dict

class Config:
    # 그래픽 사용 유무
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Multi-GPU Configuration ---
    # 사용할 GPU ID 목록 (None이면 자동 감지)
    GPU_IDS: Optional[List[int]] = None

    # 단일 GPU 작업 시 기본 GPU ID
    DEFAULT_GPU_ID: int = 0

    # Multi-GPU 처리 활성화 여부
    ENABLE_MULTI_GPU: bool = True

    # GPU당 최대 동시 이미지 처리 수
    MAX_IMAGES_PER_GPU: int = 1

    @staticmethod
    def get_device(gpu_id: Optional[int] = None) -> str:
        """
        지정된 GPU ID에 대한 디바이스 문자열을 반환합니다.

        Args:
            gpu_id: GPU ID (None이면 DEFAULT_GPU_ID 사용)

        Returns:
            디바이스 문자열 (예: "cuda:0", "cuda:1", "cpu")
        """
        if not torch.cuda.is_available():
            return "cpu"

        if gpu_id is None:
            gpu_id = Config.DEFAULT_GPU_ID

        return f"cuda:{gpu_id}"

    @staticmethod
    def get_available_gpus() -> List[int]:
        """
        사용 가능한 GPU ID 목록을 반환합니다.

        Returns:
            GPU ID 리스트
        """
        if Config.GPU_IDS is not None:
            return Config.GPU_IDS

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))

        return []

    @staticmethod
    def get_spconv_env_vars(gpu_id: int) -> Dict[str, str]:
        """
        특정 GPU에서 spconv 실행을 위한 환경 변수를 반환합니다.

        CUDA_VISIBLE_DEVICES로 GPU를 제한하면 내부적으로 항상 device 0이 됩니다.
        따라서 SPCONV_TUNE_DEVICE는 항상 0으로 설정합니다.

        Args:
            gpu_id: 실제 GPU ID

        Returns:
            환경 변수 딕셔너리
        """
        return {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "SPCONV_TUNE_DEVICE": "0",  # CUDA_VISIBLE_DEVICES로 remap되므로 항상 0
            "SPCONV_ALGO_TIME_LIMIT": "100",
        }

    # --- Models ---
    # YOLOE-seg 모델 사용 (Open-Vocabulary Detection, 세그멘테이션 지원)
    # YOLOv8 26L backbone 기반 고정밀 모델 사용
    YOLO_MODEL_PATH = 'yoloe-26x-seg.pt'

    # CLIP 제거됨 - YOLO 클래스로 직접 DB 매칭
    
    # --- Visualization ---
    # 한글 폰트 경로 (시스템에 맞게 수정 필요)
    FONT_PATH = "fonts/NanumGothic-Regular.ttf" 
    FONT_SIZE_LARGE = 12
    FONT_SIZE_SMALL = 8
    
    # --- Detection Thresholds ---
    # 저조도 환경 등을 고려하여 기본 임계치를 약간 낮게 잡고, 후처리로 필터링
    CONF_THRESHOLD_MAIN = 0.10 
    CONF_THRESHOLD_SMALL = 0.05 
    
    # --- PLY Preprocessing ---
    # GCS 업로드 전 PLY 전처리 설정
    PLY_ENABLE_PREPROCESSING: bool = True  # 전처리 활성화
    PLY_MAX_POINTS: int = 72000            # 다운샘플링 최대 포인트 수 (40% 퀄리티 향상)
    PLY_CONVERT_TO_YUP: bool = True        # Y-up 좌표계 변환 (Three.js 호환)
    PLY_ENABLE_ALIGNMENT: bool = True      # OBB 기반 축 정렬 활성화
    PLY_ENABLE_SCALING: bool = True        # 절대 치수 스케일링 활성화
    PLY_ENABLE_DOWNSAMPLING: bool = True   # Stride 다운샘플링 활성화

    # --- Boxer Configuration (V3 Pipeline) ---
    ENABLE_BOXER: bool = True                    # Boxer 절대 치수 추정 활성화
    BOXER_CONFIDENCE_THRESHOLD: float = 0.3      # 최소 신뢰도 (이하이면 fallback)
    BOXER_FOCAL_LENGTH_FACTOR: float = 0.8       # intrinsics 추정 FOV 계수
    BOXER_MIN_DIMENSION_MM: float = 50.0         # 물리적 타당성 최소 치수 (mm)
    BOXER_MAX_DIMENSION_MM: float = 4000.0       # 물리적 타당성 최대 치수 (mm)

    # --- Advanced Features ---
    # CLAHE가 시간복잡도가 O(N)이긴 한데 저조도 환경의 명암 대비 극대화를 위해 사용
    # 필요 없을 시 관련 함수 처리나 False로 변경하면 됨.
    USE_CLAHE_ENHANCEMENT = True
    
    @staticmethod
    def check_dependencies():
        if not os.path.exists(Config.FONT_PATH):
            print(f"[Warn] Font file not found at {Config.FONT_PATH}. Text will not be rendered correctly.")