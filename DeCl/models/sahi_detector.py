"""
SAHI-Enhanced Object Detector

SAHI(Slicing Aided Hyper Inference)를 사용하여 작은 객체 탐지 성능을 향상시킵니다.
선반 안의 접시, 책장 안의 책 등 작은 객체들을 더 정확하게 검출합니다.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import sys
import os

# DeCl 경로를 sys.path에 추가
decl_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if decl_path not in sys.path:
    sys.path.insert(0, decl_path)

from config import Config
from utils.image_ops import ImageUtils

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False
    print("[Warning] SAHI not installed. Install with: pip install sahi")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[Warning] ultralytics not installed")


class SAHIDetector:
    """
    SAHI를 사용한 슬라이싱 기반 객체 탐지

    작은 객체(선반 안 접시, 책장 안 책 등)를 더 정확하게 검출합니다.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        confidence_threshold: float = 0.10,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.2
    ):
        """
        Args:
            model_path: YOLO 모델 경로
            device: 사용할 디바이스 ('cuda' or 'cpu')
            confidence_threshold: 탐지 신뢰도 임계값
            slice_height: 슬라이스 높이
            slice_width: 슬라이스 너비
            overlap_ratio: 슬라이스 간 겹침 비율
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.device = device or Config.DEVICE
        self.confidence_threshold = confidence_threshold
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        self.model = None
        self.sahi_model = None
        self.classes = []

        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not HAS_YOLO:
            print("[SAHIDetector] YOLO not available")
            return

        print(f"[SAHIDetector] Loading YOLO-World: {self.model_path}")
        self.model = YOLO(self.model_path)

        if self.device == "cuda":
            self.model.to("cuda")

        # SAHI 모델 래퍼 초기화
        if HAS_SAHI:
            try:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device
                )
                print("[SAHIDetector] SAHI model wrapper initialized")
            except Exception as e:
                print(f"[SAHIDetector] Warning: SAHI wrapper failed: {e}")
                self.sahi_model = None

    def set_classes(self, classes: List[str]):
        """탐지할 클래스 설정"""
        self.classes = classes
        if self.model:
            self.model.set_classes(classes)

    def detect(
        self,
        image: Image.Image,
        use_slicing: bool = True,
        return_crops: bool = False
    ) -> Dict:
        """
        이미지에서 객체를 탐지합니다.

        Args:
            image: PIL 이미지
            use_slicing: SAHI 슬라이싱 사용 여부
            return_crops: 크롭 이미지 반환 여부

        Returns:
            {
                "boxes": [[x1, y1, x2, y2], ...],
                "scores": [float, ...],
                "classes": [int, ...],
                "labels": [str, ...],
                "crops": [PIL.Image, ...] (if return_crops=True)
            }
        """
        if self.model is None:
            return {"boxes": [], "scores": [], "classes": [], "labels": []}

        img_array = np.array(image)

        if use_slicing and HAS_SAHI and self.sahi_model:
            return self._detect_with_sahi(image, img_array, return_crops)
        else:
            return self._detect_standard(image, return_crops)

    def _detect_standard(self, image: Image.Image, return_crops: bool = False) -> Dict:
        """표준 YOLO 탐지"""
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )[0]

        boxes = []
        scores = []
        classes = []
        labels = []
        crops = []

        if len(results.boxes) == 0:
            return {"boxes": boxes, "scores": scores, "classes": classes, "labels": labels}

        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())
            label = self.model.names[cls_idx]

            boxes.append(xyxy.tolist())
            scores.append(conf)
            classes.append(cls_idx)
            labels.append(label)

            if return_crops:
                x1, y1, x2, y2 = map(int, xyxy)
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)

        result = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "labels": labels
        }

        if return_crops:
            result["crops"] = crops

        return result

    def _detect_with_sahi(
        self,
        image: Image.Image,
        img_array: np.ndarray,
        return_crops: bool = False
    ) -> Dict:
        """SAHI 슬라이싱을 사용한 탐지"""
        try:
            # SAHI 슬라이싱 예측
            result = get_sliced_prediction(
                image=img_array,
                detection_model=self.sahi_model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5,
                verbose=0
            )

            boxes = []
            scores = []
            classes = []
            labels = []
            crops = []

            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xyxy()
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                scores.append(obj.score.value)
                classes.append(obj.category.id)
                labels.append(obj.category.name)

                if return_crops:
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = image.crop((x1, y1, x2, y2))
                    crops.append(crop)

            result_dict = {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "labels": labels
            }

            if return_crops:
                result_dict["crops"] = crops

            return result_dict

        except Exception as e:
            print(f"[SAHIDetector] SAHI detection failed, falling back to standard: {e}")
            return self._detect_standard(image, return_crops)

    def detect_smart(
        self,
        pil_image: Image.Image,
        use_clahe: bool = True,
        use_slicing: bool = True
    ) -> Optional[Dict]:
        """
        원본 + CLAHE 이미지 앙상블 탐지 (기존 YoloDetector.detect_smart와 호환)

        Args:
            pil_image: PIL 이미지
            use_clahe: CLAHE 대비 강화 사용
            use_slicing: SAHI 슬라이싱 사용

        Returns:
            {
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "classes": np.ndarray
            }
        """
        import torch
        from torchvision.ops import nms

        all_boxes = []
        all_scores = []
        all_classes = []

        # 1. 기본 탐지
        result1 = self.detect(pil_image, use_slicing=use_slicing)
        if result1["boxes"]:
            all_boxes.extend(result1["boxes"])
            all_scores.extend(result1["scores"])
            all_classes.extend(result1["classes"])

        # 2. CLAHE 강화 이미지 탐지
        if use_clahe and Config.USE_CLAHE_ENHANCEMENT:
            cv_img = ImageUtils.pil_to_cv2(pil_image)
            enhanced_cv = ImageUtils.apply_clahe(cv_img)
            enhanced_pil = ImageUtils.cv2_to_pil(enhanced_cv)

            result2 = self.detect(enhanced_pil, use_slicing=use_slicing)
            if result2["boxes"]:
                all_boxes.extend(result2["boxes"])
                all_scores.extend(result2["scores"])
                all_classes.extend(result2["classes"])

        if not all_boxes:
            return None

        # NMS로 중복 제거
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.int64)

        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

        return {
            "boxes": boxes_tensor[keep_indices].numpy(),
            "scores": scores_tensor[keep_indices].numpy(),
            "classes": classes_tensor[keep_indices].numpy()
        }

    def detect_contents(
        self,
        container_crop: Image.Image,
        content_labels: List[str]
    ) -> List[Dict]:
        """
        컨테이너(캐비닛, 책장 등) 내부의 내용물을 탐지합니다.

        Args:
            container_crop: 컨테이너 크롭 이미지
            content_labels: 탐지할 내용물 라벨 리스트

        Returns:
            [{
                "label": str,
                "bbox": [x1, y1, x2, y2],
                "score": float
            }, ...]
        """
        # 내용물 탐지를 위해 클래스 설정
        original_classes = self.classes.copy()
        self.set_classes(content_labels)

        # 슬라이싱 없이 탐지 (이미 크롭된 이미지이므로)
        result = self.detect(container_crop, use_slicing=False)

        # 원래 클래스로 복원
        self.set_classes(original_classes)

        contents = []
        for i in range(len(result["boxes"])):
            contents.append({
                "label": result["labels"][i],
                "bbox": result["boxes"][i],
                "score": result["scores"][i]
            })

        return contents


class EnhancedDetector:
    """
    SAHI + CLAHE를 결합한 향상된 탐지기

    기존 YoloDetector를 대체할 수 있는 드롭인 교체 클래스
    """

    def __init__(
        self,
        model_path: str = None,
        use_sahi: bool = True,
        slice_size: int = 512,
        overlap_ratio: float = 0.2
    ):
        self.use_sahi = use_sahi and HAS_SAHI

        if self.use_sahi:
            self.detector = SAHIDetector(
                model_path=model_path,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_ratio=overlap_ratio
            )
        else:
            # Fallback to standard YOLO
            from models.detector import YoloDetector
            self.detector = YoloDetector()

    def set_classes(self, classes: List[str]):
        """탐지 클래스 설정"""
        if hasattr(self.detector, 'set_classes'):
            self.detector.set_classes(classes)

    def detect_smart(self, pil_image: Image.Image) -> Optional[Dict]:
        """앙상블 탐지"""
        if self.use_sahi:
            return self.detector.detect_smart(pil_image, use_clahe=True, use_slicing=True)
        else:
            return self.detector.detect_smart(pil_image)

    @property
    def model(self):
        """모델 객체 접근"""
        if hasattr(self.detector, 'model'):
            return self.detector.model
        return None
