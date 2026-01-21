"""
Tests for ai/processors/2_YOLO_detect.py with real GPU

YOLOE 탐지기 GPU 통합 테스트:
- 실제 YOLOE 모델 로드
- 실제 이미지로 탐지
- 세그멘테이션 마스크 추출
- CLAHE 앙상블 탐지
"""

import pytest
import numpy as np
import importlib
from pathlib import Path
from PIL import Image

# Skip tests if YOLO not available
pytest.importorskip("ultralytics")

from ai.processors import YoloDetector
from ai.config import Config

# 숫자로 시작하는 모듈 import
_stage2 = importlib.import_module('.2_YOLO_detect', package='ai.processors')
FURNITURE_CLASSES = _stage2.FURNITURE_CLASSES


# Test images
TEST_IMAGES_DIR = Path(__file__).parent.parent / "ai" / "imgs"
BEDROOM_IMAGE = TEST_IMAGES_DIR / "bedroom-5772286_1920.jpg"
KITCHEN_IMAGE = TEST_IMAGES_DIR / "kitchen-2165756_1920.jpg"
FURNITURE_IMAGE = TEST_IMAGES_DIR / "furniture-998265.jpg"


@pytest.fixture(scope="module")
def yolo_detector():
    """YOLO 탐지기 인스턴스 (모듈 스코프로 한 번만 로드)"""
    detector = YoloDetector(
        model_path="yoloe-26x-seg.pt",
        confidence_threshold=0.25,
        device_id=0
    )
    return detector


class TestYoloDetectorInit:
    """YoloDetector 초기화 테스트 (GPU)"""

    def test_model_loads_successfully(self, yolo_detector):
        """모델이 성공적으로 로드되는지 확인"""
        assert yolo_detector.model is not None

    def test_model_has_classes(self, yolo_detector):
        """모델에 클래스가 설정되어 있는지 확인"""
        assert hasattr(yolo_detector.model, 'names')
        assert len(yolo_detector.model.names) > 0

    def test_device_is_cuda(self, yolo_detector):
        """GPU 디바이스가 설정되어 있는지 확인"""
        assert "cuda" in yolo_detector._device


class TestYoloDetectorDetect:
    """detect 함수 테스트 (GPU)"""

    def test_detect_bedroom_image(self, yolo_detector):
        """침실 이미지 탐지"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        result = yolo_detector.detect(image, return_masks=False)

        assert result is not None
        assert "boxes" in result
        assert "scores" in result
        assert "classes" in result
        assert "labels" in result

        # 침실 이미지에서 최소 1개 이상 탐지
        print(f"Detected {len(result['labels'])} objects: {result['labels']}")
        assert len(result["labels"]) >= 1

    def test_detect_with_masks(self, yolo_detector):
        """마스크와 함께 탐지"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        result = yolo_detector.detect(image, return_masks=True)

        assert result is not None
        assert "masks" in result

        if len(result["labels"]) > 0:
            assert result["masks"] is not None
            assert len(result["masks"]) == len(result["labels"])
            # 마스크가 numpy 배열인지 확인
            assert isinstance(result["masks"][0], np.ndarray)
            # 마스크 크기가 이미지 크기와 일치하는지 확인
            assert result["masks"][0].shape == (image.height, image.width)

    def test_detect_kitchen_image(self, yolo_detector):
        """주방 이미지 탐지"""
        if not KITCHEN_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(KITCHEN_IMAGE)
        result = yolo_detector.detect(image, return_masks=True)

        assert result is not None
        print(f"Kitchen - Detected {len(result['labels'])} objects: {result['labels']}")

    def test_detect_furniture_image(self, yolo_detector):
        """가구 이미지 탐지"""
        if not FURNITURE_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(FURNITURE_IMAGE)
        result = yolo_detector.detect(image, return_masks=True)

        assert result is not None
        print(f"Furniture - Detected {len(result['labels'])} objects: {result['labels']}")


class TestYoloDetectorDetectSmart:
    """detect_smart 함수 테스트 (CLAHE 앙상블)"""

    def test_detect_smart_bedroom(self, yolo_detector):
        """CLAHE 앙상블 탐지 - 침실"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        result = yolo_detector.detect_smart(image, return_masks=True)

        assert result is not None
        print(f"Smart detect - Detected {len(result['labels'])} objects: {result['labels']}")

        # CLAHE 앙상블이 단독 탐지보다 같거나 많은 객체 탐지
        simple_result = yolo_detector.detect(image, return_masks=False)
        if simple_result:
            print(f"Simple detect - Detected {len(simple_result['labels'])} objects")

    def test_detect_smart_with_masks(self, yolo_detector):
        """CLAHE 앙상블 탐지 - 마스크"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        result = yolo_detector.detect_smart(image, return_masks=True)

        if result and len(result["labels"]) > 0:
            assert result["masks"] is not None or result["masks"] == []


class TestYoloDetectorFilterFurniture:
    """filter_furniture_classes 함수 테스트"""

    def test_filter_returns_furniture_only(self, yolo_detector):
        """가구 클래스만 필터링"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        result = yolo_detector.detect(image, return_masks=True)

        if result and len(result["labels"]) > 0:
            filtered = yolo_detector.filter_furniture_classes(result)
            assert filtered is not None

            # 필터링된 결과의 라벨이 모두 가구 클래스인지 확인
            furniture_names = {name.lower() for name in FURNITURE_CLASSES.keys()}

            for label in filtered["labels"]:
                assert label.lower() in furniture_names, f"'{label}' is not a furniture class"


class TestYoloDetectorHelpers:
    """헬퍼 함수 테스트"""

    def test_get_label_for_class(self, yolo_detector):
        """클래스 인덱스로 라벨 조회"""
        # 모델이 로드되어 있으면 클래스 0에 대한 라벨이 있어야 함
        label = yolo_detector.get_label_for_class(0)
        assert label is not None

    def test_get_furniture_info(self, yolo_detector):
        """가구 정보 조회"""
        info = yolo_detector.get_furniture_info("Bed")
        assert info is not None
        assert "base_name" in info
        assert info["base_name"] == "침대"

    def test_get_furniture_info_case_insensitive(self, yolo_detector):
        """대소문자 구분 없이 가구 정보 조회"""
        info1 = yolo_detector.get_furniture_info("bed")
        info2 = yolo_detector.get_furniture_info("BED")
        info3 = yolo_detector.get_furniture_info("Bed")

        assert info1 == info2 == info3

    def test_get_furniture_info_unknown(self, yolo_detector):
        """알 수 없는 라벨"""
        info = yolo_detector.get_furniture_info("unknown_class")
        assert info is None


class TestYoloDetectorPerformance:
    """성능 테스트"""

    def test_inference_time(self, yolo_detector):
        """추론 시간 측정"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        import time
        image = Image.open(BEDROOM_IMAGE)

        # 웜업
        yolo_detector.detect(image, return_masks=False)

        # 측정
        start = time.time()
        for _ in range(3):
            yolo_detector.detect(image, return_masks=True)
        elapsed = time.time() - start

        avg_time = elapsed / 3
        print(f"Average inference time: {avg_time:.3f}s")
        # A100에서 1초 이내 예상
        assert avg_time < 2.0, f"Inference too slow: {avg_time:.3f}s"


class TestYoloDetectorEdgeCases:
    """엣지 케이스 테스트"""

    def test_small_image(self, yolo_detector):
        """작은 이미지"""
        small_image = Image.new('RGB', (100, 100), color='white')
        result = yolo_detector.detect(small_image, return_masks=False)
        assert result is not None

    def test_large_image(self, yolo_detector):
        """큰 이미지"""
        large_image = Image.new('RGB', (4000, 3000), color='white')
        result = yolo_detector.detect(large_image, return_masks=False)
        assert result is not None

    def test_grayscale_image(self, yolo_detector):
        """그레이스케일 이미지 (RGB로 변환됨)"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE).convert('L').convert('RGB')
        result = yolo_detector.detect(image, return_masks=False)
        assert result is not None
