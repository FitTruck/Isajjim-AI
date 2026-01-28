"""
Stage 2: YOLOE 객체 탐지

YOLOE-seg (Open-Vocabulary Detection) 모델을 사용하여 이미지 내 객체의
위치(바운딩 박스), 클래스, 세그멘테이션 마스크를 탐지합니다.

YOLOE는 open-vocabulary 모델로, set_classes()를 통해
탐지할 클래스를 동적으로 설정할 수 있습니다.
"""

import numpy as np
from typing import Dict, List, Optional, Set
from PIL import Image

# AI module imports
from ai.config import Config
from ai.utils.image_ops import ImageUtils
from ai.data.knowledge_base import (
    get_subtypes,
    get_db_key_from_label,
    get_excluded_base_names,
    get_yolo_class_mapping,
    get_all_yolo_classes,
)

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


# knowledge_base.py에서 YOLO 클래스 매핑을 동적으로 가져옴
# 이를 통해 knowledge_base.py가 단일 진실 소스(Single Source of Truth)가 됨
FURNITURE_CLASSES = get_yolo_class_mapping()

# YOLOE set_classes()에 전달할 클래스 목록
YOLO_CLASS_NAMES = get_all_yolo_classes()

# 탐지는 수행하지만 최종 출력에서 제외할 클래스 목록
# knowledge_base.py의 exclude_from_output: True 항목에서 동적으로 로드
EXCLUDED_FROM_OUTPUT = get_excluded_base_names()


def _empty_result() -> Dict:
    """빈 탐지 결과 반환"""
    return {
        "boxes": np.array([]),
        "scores": np.array([]),
        "classes": np.array([]),
        "labels": [],
        "masks": []
    }


def _filter_by_labels(
    detection_result: Dict,
    label_set: Set[str],
    include: bool = True
) -> Dict:
    """
    라벨 기준으로 탐지 결과를 필터링합니다.

    Args:
        detection_result: 탐지 결과
        label_set: 필터링할 라벨 세트 (소문자)
        include: True면 포함, False면 제외

    Returns:
        필터링된 탐지 결과
    """
    if detection_result is None or len(detection_result["boxes"]) == 0:
        return detection_result

    keep_indices = []
    for i, label in enumerate(detection_result["labels"]):
        in_set = label.lower() in label_set
        if (include and in_set) or (not include and not in_set):
            keep_indices.append(i)

    if not keep_indices:
        return _empty_result()

    masks = detection_result.get("masks")
    return {
        "boxes": detection_result["boxes"][keep_indices],
        "scores": detection_result["scores"][keep_indices],
        "classes": detection_result["classes"][keep_indices],
        "labels": [detection_result["labels"][i] for i in keep_indices],
        "masks": [masks[i] for i in keep_indices] if masks else None
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

        print(f"[YoloDetector] Setting {len(YOLO_CLASS_NAMES)} furniture classes from knowledge_base...")
        self.model.set_classes(YOLO_CLASS_NAMES)

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
            {"boxes", "scores", "classes", "labels", "masks"}
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
            "labels": labels,
            "masks": None
        }

        # 세그멘테이션 마스크 추출
        if return_masks and results.masks is not None:
            masks = []
            for mask in results.masks.data:
                mask_np = mask.cpu().numpy()
                if mask_np.shape != (image.height, image.width):
                    import cv2
                    mask_np = cv2.resize(
                        mask_np,
                        (image.width, image.height),
                        interpolation=cv2.INTER_NEAREST
                    )
                masks.append((mask_np > 0.5).astype(np.uint8) * 255)
            result["masks"] = masks

        return result

    def _merge_results(
        self,
        all_boxes: List,
        all_scores: List,
        all_classes: List,
        all_labels: List,
        all_masks: List,
        result: Dict,
        return_masks: bool
    ):
        """탐지 결과를 리스트에 병합"""
        if result is None or len(result["boxes"]) == 0:
            return

        boxes = result["boxes"]
        all_boxes.extend(boxes.tolist() if hasattr(boxes, 'tolist') else boxes)

        scores = result["scores"]
        all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else scores)

        classes = result["classes"]
        all_classes.extend(classes.tolist() if hasattr(classes, 'tolist') else classes)

        all_labels.extend(result["labels"])

        if return_masks and result.get("masks"):
            all_masks.extend(result["masks"])

    def detect_smart(self, image: Image.Image, return_masks: bool = False) -> Optional[Dict]:
        """
        원본 + CLAHE 강화 이미지 앙상블 탐지

        두 이미지에서 탐지된 결과를 NMS로 병합하여
        색상 대비가 낮은 객체도 검출합니다.

        Args:
            image: PIL 이미지
            return_masks: 세그멘테이션 마스크 반환 여부

        Returns:
            {"boxes", "scores", "classes", "labels", "masks"}
        """
        if not HAS_TORCH:
            return self.detect(image, return_masks)

        all_boxes, all_scores, all_classes, all_labels, all_masks = [], [], [], [], []

        # 1. 원본 이미지 탐지
        self._merge_results(
            all_boxes, all_scores, all_classes, all_labels, all_masks,
            self.detect(image, return_masks), return_masks
        )

        # 2. CLAHE 강화 이미지 탐지
        if Config.USE_CLAHE_ENHANCEMENT:
            cv_img = ImageUtils.pil_to_cv2(image)
            enhanced_pil = ImageUtils.cv2_to_pil(ImageUtils.apply_clahe(cv_img))
            self._merge_results(
                all_boxes, all_scores, all_classes, all_labels, all_masks,
                self.detect(enhanced_pil, return_masks), return_masks
            )

        if not all_boxes:
            return None

        # NMS로 중복 제거
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.int64)

        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5).tolist()

        result = {
            "boxes": boxes_tensor[keep_indices].numpy(),
            "scores": scores_tensor[keep_indices].numpy(),
            "classes": classes_tensor[keep_indices].numpy(),
            "labels": [all_labels[i] for i in keep_indices],
            "masks": [all_masks[i] for i in keep_indices if i < len(all_masks)] if return_masks and all_masks else None
        }

        return result

    def filter_furniture_classes(self, detection_result: Dict) -> Dict:
        """
        탐지 결과에서 가구/가정용품 클래스만 필터링합니다.

        Note: YOLOE가 이미 YOLO_CLASS_NAMES만 탐지하도록 설정되어 있어
              이 함수는 보통 불필요하지만, 하위 호환성을 위해 유지됩니다.
        """
        furniture_names = {name.lower() for name in YOLO_CLASS_NAMES}
        return _filter_by_labels(detection_result, furniture_names, include=True)

    def filter_excluded_classes(self, detection_result: Dict) -> Dict:
        """
        탐지 결과에서 EXCLUDED_FROM_OUTPUT에 해당하는 클래스를 제외합니다.
        (예: 아일랜드, 바닥 등)
        """
        excluded_names = {name.lower() for name in EXCLUDED_FROM_OUTPUT}
        return _filter_by_labels(detection_result, excluded_names, include=False)

    def get_label_for_class(self, class_idx: int) -> Optional[str]:
        """클래스 인덱스에 대한 라벨 반환"""
        if self.model and class_idx in self.model.names:
            return self.model.names[class_idx]
        return None

    def get_furniture_info(self, label: str) -> Optional[Dict]:
        """라벨에 대한 가구 정보 반환 (대소문자 무시)"""
        label_lower = label.lower()
        for class_name, info in FURNITURE_CLASSES.items():
            if class_name.lower() == label_lower:
                return info
        return None

    def detect_subtype(
        self,
        crop_image: Image.Image,
        base_label: str,
        confidence_threshold: float = 0.2
    ) -> Optional[str]:
        """
        객체 crop 이미지에서 subtype을 탐지합니다.

        YOLOE의 open-vocabulary 기능을 사용하여
        knowledge_base.py의 subtype prompt로 세부 분류를 수행합니다.

        Args:
            crop_image: 객체가 crop된 PIL 이미지
            base_label: 1차 탐지된 base 라벨 (예: "Sofa", "Bed")
            confidence_threshold: subtype 탐지 신뢰도 임계값

        Returns:
            subtype name (예: "SINGLE_SOFA") 또는 None
        """
        if self.model is None or not HAS_YOLO:
            return None

        # DB 키 찾기
        db_key = get_db_key_from_label(base_label)
        if db_key is None:
            return None

        # subtypes 가져오기
        subtypes = get_subtypes(db_key)
        if not subtypes:
            return None

        # subtype prompt 추출
        subtype_prompts = []
        subtype_map = {}  # prompt -> subtype name 매핑
        for st in subtypes:
            prompt = st.get("prompt")
            name = st.get("name")
            if prompt and name:
                subtype_prompts.append(prompt)
                subtype_map[prompt] = name

        if not subtype_prompts:
            return None

        # YOLOE에 subtype prompt 설정
        try:
            self.model.set_classes(subtype_prompts)

            # subtype 탐지
            results = self.model.predict(
                crop_image,
                conf=confidence_threshold,
                verbose=False,
                device=self._device
            )[0]

            print(f"[YoloDetector] Setting {len(YOLO_CLASS_NAMES)} furniture classes from knowledge_base...")
            # 원래 클래스로 복원
            self.model.set_classes(YOLO_CLASS_NAMES)

            if len(results.boxes) == 0:
                # 원래 클래스로 복원 후 반환
                furniture_class_names = list(FURNITURE_CLASSES.keys())
                self.model.set_classes(furniture_class_names)
                return None

            # 가장 높은 confidence의 결과 선택
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            best_idx = scores.argmax()
            best_class = classes[best_idx]

            # subtype_prompts 리스트에서 직접 인덱스로 접근
            # (model.names를 사용하면 클래스 복원 후 인덱스가 엉망이 됨)
            best_prompt = subtype_prompts[best_class]

            # 원래 클래스로 복원
            furniture_class_names = list(FURNITURE_CLASSES.keys())
            self.model.set_classes(furniture_class_names)

            # prompt -> subtype name 매핑
            return subtype_map.get(best_prompt)

        except Exception as e:
            print(f"[YoloDetector] Subtype detection failed: {e}")
            # 에러 시 원래 클래스로 복원
            try:
                self.model.set_classes(YOLO_CLASS_NAMES)
            except Exception:
                pass
            return None


# 하위 호환성을 위한 별칭
YoloWorldDetector = YoloDetector
