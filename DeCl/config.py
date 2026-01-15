import torch
import os

class Config:
    # 그래픽 사용 유무
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Models ---
    # 성능과 속도 균형을 위해 L 버전을 유지하되, 필요시 M이나 X로 변경 가능
    YOLO_MODEL_PATH = 'yolov8l-world.pt' 
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    
    # --- Visualization ---
    # 한글 폰트 경로 (시스템에 맞게 수정 필요)
    FONT_PATH = "fonts/NanumGothic-Regular.ttf" 
    FONT_SIZE_LARGE = 12
    FONT_SIZE_SMALL = 8
    
    # --- Detection Thresholds ---
    # 저조도 환경 등을 고려하여 기본 임계치를 약간 낮게 잡고, 후처리로 필터링
    CONF_THRESHOLD_MAIN = 0.10 
    CONF_THRESHOLD_SMALL = 0.05 
    
    # --- Advanced Features ---
    # CLAHE가 시간복잡도가 O(N)이긴 한데 저조도 환경의 명암 대비 극대화를 위해 사용
    # 필요 없을 시 관련 함수 처리나 False로 변경하면 됨.
    USE_CLAHE_ENHANCEMENT = True
    
    @staticmethod
    def check_dependencies():
        if not os.path.exists(Config.FONT_PATH):
            print(f"[Warn] Font file not found at {Config.FONT_PATH}. Text will not be rendered correctly.")