"""
Stage 2: YOLO-World 객체 탐지

YOLO-World + SAHI(Slicing Aided Hyper Inference)를 사용하여
이미지 내 객체(가구 및 내용물)의 위치(바운딩 박스)와 1차 클래스를 탐지합니다.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

# AI module imports
from ai.config import Config
from ai.utils.image_ops import ImageUtils

# Type hint
from typing import TYPE_CHECKING

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[YoloWorldDetector] ultralytics not installed")

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False
    print("[YoloWorldDetector] SAHI not installed (optional)")

try:
    import torch
    from torchvision.ops import nms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class YoloWorldDetector:
    """
    YOLO-World 기반 객체 탐지기

    AI Logic Step 2: 객체 위치 및 1차 클래스 탐지

    Features:
    - YOLO-World 오픈 보캐블러리 탐지
    - SAHI 슬라이싱으로 작은 객체 탐지 향상
    - CLAHE 대비 강화 앙상블 탐지
    - NMS(Non-Maximum Suppression)로 중복 제거
    """

    def __init__(
        self,
        model_path: str = None,
        use_sahi: bool = True,
        slice_size: int = 512,
        overlap_ratio: float = 0.2,
        confidence_threshold: float = 0.10,
        device_id: Optional[int] = None
    ):
        """
        Args:
            model_path: YOLO 모델 경로 (None이면 Config에서 가져옴)
            use_sahi: SAHI 슬라이싱 사용 여부
            slice_size: SAHI 슬라이스 크기
            overlap_ratio: 슬라이스 간 겹침 비율
            confidence_threshold: 탐지 신뢰도 임계값
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.use_sahi = use_sahi and HAS_SAHI
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold

        # Multi-GPU 지원: 디바이스 설정
        self.device_id = device_id
        self._device = Config.get_device(device_id)

        self.model = None
        self.sahi_model = None
        self.classes = []

        self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not HAS_YOLO:
            print("[YoloWorldDetector] YOLO not available")
            return

        print(f"[YoloWorldDetector] Loading YOLO-World on {self._device}: {self.model_path}")
        self.model = YOLO(self.model_path)

        if "cuda" in self._device:
            self.model.to(self._device)

        # SAHI 모델 래퍼 초기화
        if self.use_sahi:
            try:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self._device
                )
                print(f"[YoloWorldDetector] SAHI enabled on {self._device}")
            except Exception as e:
                print(f"[YoloWorldDetector] SAHI init failed: {e}")
                self.sahi_model = None
                self.use_sahi = False

    def set_classes(self, classes: List[str]):
        """
        탐지할 클래스 설정 (YOLO-World 오픈 보캐블러리)

        Args:
            classes: 탐지할 클래스명 리스트
        """
        self.classes = classes
        if self.model:
            self.model.set_classes(classes)

    def detect(self, image: Image.Image) -> Optional[Dict]:
        """
        이미지에서 객체를 탐지합니다.

        Args:
            image: PIL 이미지

        Returns:
            {
                "boxes": np.ndarray [[x1,y1,x2,y2], ...],
                "scores": np.ndarray [float, ...],
                "classes": np.ndarray [int, ...],
                "labels": [str, ...]
            }
        """
        if self.model is None:
            return None

        if self.use_sahi and self.sahi_model:
            return self._detect_with_sahi(image)
        else:
            return self._detect_standard(image)

    def detect_smart(self, image: Image.Image) -> Optional[Dict]:
        """
        원본 + CLAHE 강화 이미지 앙상블 탐지

        두 이미지에서 탐지된 결과를 NMS로 병합하여
        색상 대비가 낮은 객체도 검출합니다.

        Args:
            image: PIL 이미지

        Returns:
            {
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "classes": np.ndarray
            }
        """
        if not HAS_TORCH:
            return self.detect(image)

        all_boxes = []
        all_scores = []
        all_classes = []

        # 1. 원본 이미지 탐지
        result1 = self.detect(image)
        if result1 and len(result1["boxes"]) > 0:
            all_boxes.extend(result1["boxes"].tolist() if hasattr(result1["boxes"], 'tolist') else result1["boxes"])
            all_scores.extend(result1["scores"].tolist() if hasattr(result1["scores"], 'tolist') else result1["scores"])
            all_classes.extend(result1["classes"].tolist() if hasattr(result1["classes"], 'tolist') else result1["classes"])

        # 2. CLAHE 강화 이미지 탐지
        if Config.USE_CLAHE_ENHANCEMENT:
            cv_img = ImageUtils.pil_to_cv2(image)
            enhanced_cv = ImageUtils.apply_clahe(cv_img)
            enhanced_pil = ImageUtils.cv2_to_pil(enhanced_cv)

            result2 = self.detect(enhanced_pil)
            if result2 and len(result2["boxes"]) > 0:
                all_boxes.extend(result2["boxes"].tolist() if hasattr(result2["boxes"], 'tolist') else result2["boxes"])
                all_scores.extend(result2["scores"].tolist() if hasattr(result2["scores"], 'tolist') else result2["scores"])
                all_classes.extend(result2["classes"].tolist() if hasattr(result2["classes"], 'tolist') else result2["classes"])

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

    def _detect_standard(self, image: Image.Image) -> Optional[Dict]:
        """표준 YOLO 탐지"""
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self._device
        )[0]

        if len(results.boxes) == 0:
            return {"boxes": np.array([]), "scores": np.array([]), "classes": np.array([]), "labels": []}

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        labels = [self.model.names[int(c)] for c in classes]

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "labels": labels
        }

    def _detect_with_sahi(self, image: Image.Image) -> Optional[Dict]:
        """SAHI 슬라이싱 탐지"""
        try:
            img_array = np.array(image)

            result = get_sliced_prediction(
                image=img_array,
                detection_model=self.sahi_model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
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

            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xyxy()
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                scores.append(obj.score.value)
                classes.append(obj.category.id)
                labels.append(obj.category.name)

            return {
                "boxes": np.array(boxes) if boxes else np.array([]),
                "scores": np.array(scores) if scores else np.array([]),
                "classes": np.array(classes) if classes else np.array([]),
                "labels": labels
            }

        except Exception as e:
            print(f"[YoloWorldDetector] SAHI failed, fallback to standard: {e}")
            return self._detect_standard(image)

    def get_label_for_class(self, class_idx: int) -> Optional[str]:
        """클래스 인덱스에 대한 라벨 반환"""
        if self.model and class_idx in self.model.names:
            return self.model.names[class_idx]
        return None
