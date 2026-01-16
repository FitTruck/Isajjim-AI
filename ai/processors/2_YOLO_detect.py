"""
Stage 2: YOLOE 객체 탐지

YOLOE-seg (Open-Vocabulary Detection) 모델을 사용하여 이미지 내 객체의
위치(바운딩 박스), 클래스, 세그멘테이션 마스크를 탐지합니다.

YOLOE는 open-vocabulary 모델로, set_classes()를 통해
탐지할 클래스를 동적으로 설정할 수 있습니다.
가구/가정용품 클래스만 탐지하도록 설정됩니다.

SAHI 제거 - 단일 모델 추론으로 단순화
CLIP 제거 - YOLO 클래스로 직접 DB 매칭
"""

import numpy as np
from typing import List, Dict, Optional
from PIL import Image

# AI module imports
from ai.config import Config
from ai.utils.image_ops import ImageUtils

try:
    from ultralytics import YOLOE
    HAS_YOLO = True
except ImportError:
    try:
        from ultralytics import YOLO as YOLOE
        HAS_YOLO = True
        print("[YoloDetector] YOLOE not available, falling back to YOLO")
    except ImportError:
        HAS_YOLO = False
        print("[YoloDetector] ultralytics not installed")

try:
    import torch
    from torchvision.ops import nms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Objects365 기반 가구/가정용품 클래스 목록 (탐지 대상)
# YOLOE-seg는 Objects365 클래스 인덱스를 사용
FURNITURE_CLASSES = {
    # 가구 (고우선순위)
    "Bed": {"is_movable": True, "base_name": "침대"},
    "Sofa": {"is_movable": True, "base_name": "소파"},
    "Chair": {"is_movable": True, "base_name": "의자"},
    "Desk": {"is_movable": True, "base_name": "책상"},
    "Dining Table": {"is_movable": True, "base_name": "식탁"},
    "Coffee Table": {"is_movable": True, "base_name": "커피테이블"},
    "Nightstand": {"is_movable": True, "base_name": "협탁"},
    "Cabinet/shelf": {"is_movable": True, "base_name": "캐비닛/선반"},
    "Refrigerator": {"is_movable": True, "base_name": "냉장고"},
    "Washing Machine": {"is_movable": True, "base_name": "세탁기"},
    "Microwave": {"is_movable": True, "base_name": "전자레인지"},
    "Oven": {"is_movable": True, "base_name": "오븐"},
    "Air Conditioner": {"is_movable": True, "base_name": "에어컨"},
    "Monitor/TV": {"is_movable": True, "base_name": "TV/모니터"},
    "Mirror": {"is_movable": True, "base_name": "거울"},
    "Storage box": {"is_movable": True, "base_name": "박스/수납함"},

    # 가구 (중우선순위)
    "Stool": {"is_movable": True, "base_name": "스툴"},
    "Bench": {"is_movable": True, "base_name": "벤치"},
    "Toilet": {"is_movable": False, "base_name": "변기"},
    "Sink": {"is_movable": False, "base_name": "싱크대"},
    "Bathtub": {"is_movable": False, "base_name": "욕조"},
    "Luggage": {"is_movable": True, "base_name": "캐리어"},
    "Bicycle": {"is_movable": True, "base_name": "자전거"},
    "Ladder": {"is_movable": True, "base_name": "사다리"},

    # 추가 가정용품
    "Bookshelf": {"is_movable": True, "base_name": "책장"},
    "Wardrobe": {"is_movable": True, "base_name": "옷장"},
    "Drawer": {"is_movable": True, "base_name": "서랍장"},
    "Television": {"is_movable": True, "base_name": "TV"},
    "Computer": {"is_movable": True, "base_name": "컴퓨터"},
    "Laptop": {"is_movable": True, "base_name": "노트북"},
    "Printer": {"is_movable": True, "base_name": "프린터"},
    "Fan": {"is_movable": True, "base_name": "선풍기"},
    "Heater": {"is_movable": True, "base_name": "히터"},
    "Lamp": {"is_movable": True, "base_name": "램프"},
    "Clock": {"is_movable": True, "base_name": "시계"},
    "Vase": {"is_movable": True, "base_name": "꽃병"},
    "Pot/Pan": {"is_movable": True, "base_name": "냄비/팬"},
    "Kettle": {"is_movable": True, "base_name": "주전자"},
    "Toaster": {"is_movable": True, "base_name": "토스터"},
    "Blender": {"is_movable": True, "base_name": "믹서기"},
    "Rice Cooker": {"is_movable": True, "base_name": "밥솥"},
    "Trash Can": {"is_movable": True, "base_name": "쓰레기통"},
    "Plant": {"is_movable": True, "base_name": "화분"},
    "Picture Frame": {"is_movable": True, "base_name": "액자"},
    "Curtain": {"is_movable": True, "base_name": "커튼"},
    "Rug": {"is_movable": True, "base_name": "러그"},
    "Pillow": {"is_movable": True, "base_name": "베개"},
    "Blanket": {"is_movable": True, "base_name": "담요"},
    "Towel": {"is_movable": True, "base_name": "수건"},
    "Basket": {"is_movable": True, "base_name": "바구니"},
    "Suitcase": {"is_movable": True, "base_name": "여행가방"},
    "Backpack": {"is_movable": True, "base_name": "배낭"},
    "Box": {"is_movable": True, "base_name": "박스"},
    "Bag": {"is_movable": True, "base_name": "가방"},
}


class YoloDetector:
    """
    YOLOE 기반 객체 탐지기 (Open-Vocabulary Detection)

    AI Logic Step 2: 객체 위치, 클래스, 세그멘테이션 마스크 탐지

    Features:
    - YOLOE-seg 모델 사용 (Open-Vocabulary, 가구 클래스 설정)
    - 세그멘테이션 마스크 출력 지원
    - CLAHE 대비 강화 앙상블 탐지
    - NMS(Non-Maximum Suppression)로 중복 제거
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.25,
        device_id: Optional[int] = None
    ):
        """
        Args:
            model_path: YOLO 모델 경로 (None이면 Config에서 가져옴)
            confidence_threshold: 탐지 신뢰도 임계값
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.confidence_threshold = confidence_threshold

        # Multi-GPU 지원: 디바이스 설정
        self.device_id = device_id
        self._device = Config.get_device(device_id)

        self.model = None
        self._load_model()

    def _load_model(self):
        """모델 로드 및 가구 클래스 설정"""
        if not HAS_YOLO:
            print("[YoloDetector] YOLO not available")
            return

        print(f"[YoloDetector] Loading YOLOE-seg on {self._device}: {self.model_path}")
        self.model = YOLOE(self.model_path)

        # YOLOE는 open-vocabulary 모델이므로 탐지할 클래스를 설정해야 함
        furniture_class_names = list(FURNITURE_CLASSES.keys())
        print(f"[YoloDetector] Setting {len(furniture_class_names)} furniture classes...")
        self.model.set_classes(furniture_class_names)

        if "cuda" in self._device:
            self.model.to(self._device)

        print(f"[YoloDetector] Model loaded with {len(self.model.names)} classes")

    def detect(self, image: Image.Image, return_masks: bool = False) -> Optional[Dict]:
        """
        이미지에서 객체를 탐지합니다.

        Args:
            image: PIL 이미지
            return_masks: 세그멘테이션 마스크 반환 여부

        Returns:
            {
                "boxes": np.ndarray [[x1,y1,x2,y2], ...],
                "scores": np.ndarray [float, ...],
                "classes": np.ndarray [int, ...],
                "labels": [str, ...],
                "masks": [np.ndarray, ...] (return_masks=True일 때만)
            }
        """
        if self.model is None:
            return None

        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self._device
        )[0]

        if len(results.boxes) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "labels": [],
                "masks": [] if return_masks else None
            }

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        labels = [self.model.names[int(c)] for c in classes]

        result = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "labels": labels
        }

        # 세그멘테이션 마스크 추출
        if return_masks and results.masks is not None:
            masks = []
            for mask in results.masks.data:
                mask_np = mask.cpu().numpy()
                # 이미지 크기에 맞게 리사이즈
                if mask_np.shape != (image.height, image.width):
                    import cv2
                    mask_np = cv2.resize(
                        mask_np,
                        (image.width, image.height),
                        interpolation=cv2.INTER_NEAREST
                    )
                masks.append((mask_np > 0.5).astype(np.uint8) * 255)
            result["masks"] = masks
        else:
            result["masks"] = None

        return result

    def detect_smart(self, image: Image.Image, return_masks: bool = False) -> Optional[Dict]:
        """
        원본 + CLAHE 강화 이미지 앙상블 탐지

        두 이미지에서 탐지된 결과를 NMS로 병합하여
        색상 대비가 낮은 객체도 검출합니다.

        Args:
            image: PIL 이미지
            return_masks: 세그멘테이션 마스크 반환 여부

        Returns:
            {
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "classes": np.ndarray,
                "labels": [str, ...],
                "masks": [np.ndarray, ...] (return_masks=True일 때만)
            }
        """
        if not HAS_TORCH:
            return self.detect(image, return_masks)

        all_boxes = []
        all_scores = []
        all_classes = []
        all_labels = []
        all_masks = []

        # 1. 원본 이미지 탐지
        result1 = self.detect(image, return_masks)
        if result1 and len(result1["boxes"]) > 0:
            all_boxes.extend(result1["boxes"].tolist() if hasattr(result1["boxes"], 'tolist') else result1["boxes"])
            all_scores.extend(result1["scores"].tolist() if hasattr(result1["scores"], 'tolist') else result1["scores"])
            all_classes.extend(result1["classes"].tolist() if hasattr(result1["classes"], 'tolist') else result1["classes"])
            all_labels.extend(result1["labels"])
            if return_masks and result1["masks"]:
                all_masks.extend(result1["masks"])

        # 2. CLAHE 강화 이미지 탐지
        if Config.USE_CLAHE_ENHANCEMENT:
            cv_img = ImageUtils.pil_to_cv2(image)
            enhanced_cv = ImageUtils.apply_clahe(cv_img)
            enhanced_pil = ImageUtils.cv2_to_pil(enhanced_cv)

            result2 = self.detect(enhanced_pil, return_masks)
            if result2 and len(result2["boxes"]) > 0:
                all_boxes.extend(result2["boxes"].tolist() if hasattr(result2["boxes"], 'tolist') else result2["boxes"])
                all_scores.extend(result2["scores"].tolist() if hasattr(result2["scores"], 'tolist') else result2["scores"])
                all_classes.extend(result2["classes"].tolist() if hasattr(result2["classes"], 'tolist') else result2["classes"])
                all_labels.extend(result2["labels"])
                if return_masks and result2["masks"]:
                    all_masks.extend(result2["masks"])

        if not all_boxes:
            return None

        # NMS로 중복 제거
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.int64)

        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        keep_indices_list = keep_indices.tolist()

        result = {
            "boxes": boxes_tensor[keep_indices].numpy(),
            "scores": scores_tensor[keep_indices].numpy(),
            "classes": classes_tensor[keep_indices].numpy(),
            "labels": [all_labels[i] for i in keep_indices_list]
        }

        if return_masks and all_masks:
            result["masks"] = [all_masks[i] for i in keep_indices_list if i < len(all_masks)]
        else:
            result["masks"] = None

        return result

    def filter_furniture_classes(self, detection_result: Dict) -> Dict:
        """
        탐지 결과에서 가구/가정용품 클래스만 필터링합니다.

        Args:
            detection_result: detect() 또는 detect_smart()의 결과

        Returns:
            가구 클래스만 포함된 탐지 결과
        """
        if detection_result is None or len(detection_result["boxes"]) == 0:
            return detection_result

        # 가구 클래스명 목록 (소문자로 정규화)
        furniture_names_lower = {name.lower() for name in FURNITURE_CLASSES.keys()}

        keep_indices = []
        for i, label in enumerate(detection_result["labels"]):
            if label.lower() in furniture_names_lower:
                keep_indices.append(i)

        if not keep_indices:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "labels": [],
                "masks": []
            }

        return {
            "boxes": detection_result["boxes"][keep_indices],
            "scores": detection_result["scores"][keep_indices],
            "classes": detection_result["classes"][keep_indices],
            "labels": [detection_result["labels"][i] for i in keep_indices],
            "masks": [detection_result["masks"][i] for i in keep_indices] if detection_result.get("masks") else None
        }

    def get_label_for_class(self, class_idx: int) -> Optional[str]:
        """클래스 인덱스에 대한 라벨 반환"""
        if self.model and class_idx in self.model.names:
            return self.model.names[class_idx]
        return None

    def get_furniture_info(self, label: str) -> Optional[Dict]:
        """
        라벨에 대한 가구 정보 반환

        Args:
            label: YOLO 탐지 라벨

        Returns:
            {"is_movable": bool, "base_name": str} 또는 None
        """
        # 대소문자 무시 매칭
        for class_name, info in FURNITURE_CLASSES.items():
            if class_name.lower() == label.lower():
                return info
        return None


# 하위 호환성을 위한 별칭
YoloWorldDetector = YoloDetector
