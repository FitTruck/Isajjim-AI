from ultralytics import YOLO
from config import Config
from utils.image_ops import ImageUtils
import numpy as np

class YoloDetector:
    def __init__(self):
        print(f"[Init] Loading YOLO-World: {Config.YOLO_MODEL_PATH}...")
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        if Config.DEVICE == "cuda":
            self.model.to("cuda")
    
    def set_classes(self, classes):
        self.model.set_classes(classes)
    
    def detect_smart(self, pil_image):
        """
        원본 이미지와 CLAHE(대비 강화) 이미지를 모두 사용하여 탐지
        (Ensemble detection on single image variants)
        """
        results = []
        
        # 1. 기본 탐지
        results.append(self.model.predict(pil_image, conf=Config.CONF_THRESHOLD_MAIN, verbose=False, device=Config.DEVICE)[0])
        
        # 2. 명암 대비 이미지 탐지 (색상 대비가 뚜렷하지 않은 오브젝트들의 경우)
        if Config.USE_CLAHE_ENHANCEMENT:
            cv_img = ImageUtils.pil_to_cv2(pil_image)
            enhanced_cv = ImageUtils.apply_clahe(cv_img)
            enhanced_pil = ImageUtils.cv2_to_pil(enhanced_cv)
            results.append(self.model.predict(enhanced_pil, conf=Config.CONF_THRESHOLD_MAIN, verbose=False, device=Config.DEVICE)[0])
        
        return self._merge_results(results)

    def _merge_results(self, results_list):
        """
        원본과 강화된 이미지에서 나온 박스들을 NMS(Non-Maximum Suppression)로 병합
        """
        import torch
        from torchvision.ops import nms

        all_boxes = []
        all_scores = []
        all_classes = []

        for res in results_list:
            if len(res.boxes) == 0: continue
            all_boxes.append(res.boxes.xyxy)
            all_scores.append(res.boxes.conf)
            all_classes.append(res.boxes.cls)
            
        if not all_boxes:
            return None

        # 텐서 연결
        boxes_cat = torch.cat(all_boxes)
        scores_cat = torch.cat(all_scores)
        classes_cat = torch.cat(all_classes)
        
        # 탐지되어 라벨링 될때 겹치는 박스 제거
        keep_indices = nms(boxes_cat, scores_cat, iou_threshold=0.5)
        
        return {
            "boxes": boxes_cat[keep_indices].cpu().numpy(),
            "scores": scores_cat[keep_indices].cpu().numpy(),
            "classes": classes_cat[keep_indices].cpu().numpy()
        }