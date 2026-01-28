"""
Tests for ai/pipeline/furniture_pipeline.py with real GPU

FurniturePipeline GPU 통합 테스트:
- 실제 YOLOE 모델로 객체 탐지
- YOLOE-seg 마스크 생성
- 파이프라인 전체 흐름 테스트
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import base64
import io

# Skip tests if YOLO not available
pytest.importorskip("ultralytics")

from ai.pipeline.furniture_pipeline import (
    FurniturePipeline,
    DetectedObject,
    PipelineResult
)


# Test images
TEST_IMAGES_DIR = Path(__file__).parent.parent / "ai" / "imgs"
BEDROOM_IMAGE = TEST_IMAGES_DIR / "bedroom-5772286_1920.jpg"
KITCHEN_IMAGE = TEST_IMAGES_DIR / "kitchen-2165756_1920.jpg"
FURNITURE_IMAGE = TEST_IMAGES_DIR / "furniture-998265.jpg"


@pytest.fixture(scope="module")
def pipeline():
    """FurniturePipeline 인스턴스 (모듈 스코프로 한 번만 생성)"""
    pipe = FurniturePipeline(
        api_url="http://localhost:8000",
        enable_3d_generation=False,  # 3D 생성은 비활성화 (테스트 속도)
        device_id=0
    )
    return pipe


class TestFurniturePipelineInit:
    """FurniturePipeline 초기화 테스트"""

    def test_pipeline_creates_successfully(self, pipeline):
        """파이프라인이 성공적으로 생성되는지 확인"""
        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.movability_checker is not None
        assert pipeline.dimension_calculator is not None

    def test_class_map_loaded(self, pipeline):
        """클래스 매핑이 로드되었는지 확인"""
        assert pipeline.class_map is not None
        assert len(pipeline.class_map) > 0

    def test_device_set_correctly(self, pipeline):
        """GPU 디바이스가 설정되었는지 확인"""
        assert "cuda" in pipeline._device


class TestFurniturePipelineDetectObjects:
    """detect_objects 함수 테스트 (GPU)"""

    def test_detect_bedroom_objects(self, pipeline):
        """침실 이미지 객체 탐지"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        objects = pipeline.detect_objects(image)

        assert isinstance(objects, list)
        print(f"Detected {len(objects)} objects")

        for obj in objects:
            assert isinstance(obj, DetectedObject)
            assert obj.label is not None
            assert obj.db_key is not None
            assert len(obj.bbox) == 4
            assert len(obj.center_point) == 2
            print(f"  - {obj.label} (conf: {obj.confidence:.2f})")

    def test_detected_objects_have_masks(self, pipeline):
        """탐지된 객체에 마스크가 있는지 확인"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        objects = pipeline.detect_objects(image)

        for obj in objects:
            if obj.yolo_mask is not None:
                assert isinstance(obj.yolo_mask, np.ndarray)
                print(f"  - {obj.label}: mask shape {obj.yolo_mask.shape}")

    def test_detect_kitchen_objects(self, pipeline):
        """주방 이미지 객체 탐지"""
        if not KITCHEN_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(KITCHEN_IMAGE)
        objects = pipeline.detect_objects(image)

        print(f"Kitchen - Detected {len(objects)} objects")
        for obj in objects:
            print(f"  - {obj.label} (conf: {obj.confidence:.2f})")

    def test_detect_empty_image(self, pipeline):
        """빈 이미지 탐지"""
        empty_image = Image.new('RGB', (640, 480), color='white')
        objects = pipeline.detect_objects(empty_image)

        assert isinstance(objects, list)
        # 빈 이미지에서는 객체가 적게 탐지됨
        print(f"Empty image - Detected {len(objects)} objects")


class TestFurniturePipelineMaskConversion:
    """마스크 변환 테스트"""

    def test_yolo_mask_to_base64(self, pipeline):
        """YOLO 마스크 → Base64 변환"""
        # 테스트 마스크 생성
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:300, 200:400] = 255

        b64 = pipeline._yolo_mask_to_base64(mask)

        assert isinstance(b64, str)
        assert len(b64) > 0

        # 디코딩 확인
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (640, 480)

    def test_detected_mask_to_base64(self, pipeline):
        """탐지된 마스크를 Base64로 변환"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        objects = pipeline.detect_objects(image)

        for obj in objects:
            if obj.yolo_mask is not None:
                b64 = pipeline._yolo_mask_to_base64(obj.yolo_mask)
                assert isinstance(b64, str)
                assert len(b64) > 0
                print(f"  - {obj.label}: mask base64 length {len(b64)}")
                break


class TestFurniturePipelineProcessLocalImage:
    """로컬 이미지 처리 테스트"""

    def test_process_local_image(self, pipeline):
        """로컬 이미지로 파이프라인 테스트"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        image = Image.open(BEDROOM_IMAGE)
        objects = pipeline.detect_objects(image)

        # 각 객체에 대해 마스크 변환
        for obj in objects:
            if obj.yolo_mask is not None:
                obj.mask_base64 = pipeline._yolo_mask_to_base64(obj.yolo_mask)

        # 결과 확인
        objects_with_masks = [o for o in objects if o.mask_base64]
        print(f"Objects with masks: {len(objects_with_masks)}/{len(objects)}")

        for obj in objects_with_masks[:3]:  # 처음 3개만 출력
            print(f"  - {obj.label}: bbox={obj.bbox}, mask_b64_len={len(obj.mask_base64)}")


class TestDetectedObjectDataclass:
    """DetectedObject 데이터클래스 테스트"""

    def test_create_detected_object(self):
        """DetectedObject 생성"""
        obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            bbox=[100, 100, 500, 400],
            center_point=[300, 250],
            confidence=0.95
        )

        assert obj.id == 0
        assert obj.label == "침대"
        assert obj.db_key == "bed"
        assert obj.bbox == [100, 100, 500, 400]
        assert obj.confidence == 0.95

    def test_detected_object_optional_fields(self):
        """DetectedObject 선택적 필드"""
        obj = DetectedObject(
            id=1,
            label="소파",
            db_key="sofa"
        )

        assert obj.subtype_name is None
        assert obj.crop_image is None
        assert obj.mask_base64 is None
        assert obj.yolo_mask is None
        assert obj.ply_url is None


class TestPipelineResultDataclass:
    """PipelineResult 데이터클래스 테스트"""

    def test_create_pipeline_result(self):
        """PipelineResult 생성"""
        result = PipelineResult(
            image_id="test-123",
            image_url="http://example.com/image.jpg"
        )

        assert result.image_id == "test-123"
        assert result.image_url == "http://example.com/image.jpg"
        assert result.objects == []
        assert result.status == "pending"
        assert result.error is None

    def test_pipeline_result_with_objects(self):
        """PipelineResult에 객체 추가"""
        obj = DetectedObject(id=0, label="침대", db_key="bed")

        result = PipelineResult(
            image_id="test-456",
            image_url="http://example.com/image.jpg",
            objects=[obj],
            status="completed"
        )

        assert len(result.objects) == 1
        assert result.objects[0].label == "침대"


class TestFurniturePipelineToJsonResponse:
    """to_json_response 함수 테스트 (TDD 문서 포맷)"""

    def test_json_response_format(self, pipeline):
        """TDD 문서 포맷 JSON 응답 형식 검증 (단일 이미지)"""
        # TDD 문서: relative_dimensions 사용, 상대적 값 반환 (절대 부피는 백엔드 계산)
        obj = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            relative_dimensions={
                "bounding_box": {
                    "width": 1.0,
                    "depth": 0.75,
                    "height": 0.45
                },
                "centroid": [0.5, 0.375, 0.225],
                "surface_area": 2.8
            }
        )

        result = PipelineResult(
            image_id="test",
            image_url="http://test.com",
            objects=[obj],
            status="completed"
        )

        json_resp = pipeline.to_json_response([result])

        # TDD 문서 포맷: {"objects": [...]}
        assert "objects" in json_resp
        assert len(json_resp["objects"]) == 1

        obj_data = json_resp["objects"][0]
        # TDD 문서: 4개 필드만 (label, width, depth, height) - volume 제거됨
        assert obj_data["label"] == "침대"
        assert obj_data["width"] == 1.0
        assert obj_data["depth"] == 0.75
        assert obj_data["height"] == 0.45
        assert "volume" not in obj_data  # volume 필드 제거됨
        assert "ratio" not in obj_data

    def test_json_response_without_dimensions(self, pipeline):
        """치수 없는 객체는 응답에 포함되지 않음 (TDD 문서)"""
        # TDD 문서에 따르면 객체는 dimensions가 있어야 응답에 포함됨
        obj = DetectedObject(id=0, label="의자", db_key="chair")

        result = PipelineResult(
            image_id="test",
            image_url="http://test.com",
            objects=[obj],
            status="completed"
        )

        json_resp = pipeline.to_json_response([result])

        # TDD 포맷: relative_dimensions가 없으면 객체가 포함되지 않음
        assert "objects" in json_resp
        assert len(json_resp["objects"]) == 0


class TestFurniturePipelineToJsonResponseV2:
    """to_json_response_v2 함수 테스트 (TDD 문서 Section 4.1 포맷)"""

    def test_json_response_v2_format(self, pipeline):
        """TDD 문서 Section 4.1 포맷 JSON 응답 형식 검증 (다중 이미지)"""
        obj1 = DetectedObject(
            id=0,
            label="소파",
            db_key="sofa",
            relative_dimensions={
                "bounding_box": {"width": 1.0, "depth": 0.45, "height": 0.425},
                "centroid": [0.5, 0.225, 0.2125],
                "surface_area": 2.0
            }
        )
        obj2 = DetectedObject(
            id=1,
            label="의자",
            db_key="chair",
            relative_dimensions={
                "bounding_box": {"width": 0.5, "depth": 0.56, "height": 1.0},
                "centroid": [0.25, 0.28, 0.5],
                "surface_area": 1.8
            }
        )

        result1 = PipelineResult(
            image_id="uuid-1",
            image_url="http://test.com/1.jpg",
            objects=[obj1],
            status="completed",
            user_image_id=101
        )
        result2 = PipelineResult(
            image_id="uuid-2",
            image_url="http://test.com/2.jpg",
            objects=[obj2],
            status="completed",
            user_image_id=102
        )

        json_resp = pipeline.to_json_response_v2([result1, result2])

        # TDD 문서 Section 4.1 포맷: {"results": [{image_id, objects}, ...]}
        assert "results" in json_resp
        assert len(json_resp["results"]) == 2

        # 첫 번째 이미지 결과 검증
        first_result = json_resp["results"][0]
        assert first_result["image_id"] == 101
        assert len(first_result["objects"]) == 1
        assert first_result["objects"][0]["label"] == "소파"
        assert first_result["objects"][0]["width"] == 1.0
        assert "volume" not in first_result["objects"][0]  # volume 제거됨

        # 두 번째 이미지 결과 검증
        second_result = json_resp["results"][1]
        assert second_result["image_id"] == 102
        assert len(second_result["objects"]) == 1
        assert second_result["objects"][0]["label"] == "의자"

    def test_json_response_v2_empty_objects(self, pipeline):
        """탐지된 객체가 없는 이미지도 결과에 포함"""
        result = PipelineResult(
            image_id="uuid-1",
            image_url="http://test.com/1.jpg",
            objects=[],
            status="completed",
            user_image_id=101
        )

        json_resp = pipeline.to_json_response_v2([result])

        assert "results" in json_resp
        assert len(json_resp["results"]) == 1
        assert json_resp["results"][0]["image_id"] == 101
        assert json_resp["results"][0]["objects"] == []

    def test_json_response_v2_filters_no_dimensions(self, pipeline):
        """치수 없는 객체는 V2 응답에서도 필터링됨"""
        obj_with_dims = DetectedObject(
            id=0,
            label="침대",
            db_key="bed",
            relative_dimensions={
                "bounding_box": {"width": 2000.0, "depth": 1500.0, "height": 450.0},
                "centroid": [1000, 750, 225],
                "surface_area": 5000
            }
        )
        obj_without_dims = DetectedObject(
            id=1,
            label="의자",
            db_key="chair"
            # relative_dimensions 없음
        )

        result = PipelineResult(
            image_id="uuid-1",
            image_url="http://test.com/1.jpg",
            objects=[obj_with_dims, obj_without_dims],
            status="completed",
            user_image_id=101
        )

        json_resp = pipeline.to_json_response_v2([result])

        assert len(json_resp["results"]) == 1
        assert len(json_resp["results"][0]["objects"]) == 1
        assert json_resp["results"][0]["objects"][0]["label"] == "침대"


class TestFurniturePipelinePerformance:
    """성능 테스트"""

    def test_detection_time(self, pipeline):
        """탐지 시간 측정"""
        if not BEDROOM_IMAGE.exists():
            pytest.skip("Test image not found")

        import time
        image = Image.open(BEDROOM_IMAGE)

        # 웜업
        pipeline.detect_objects(image)

        # 측정
        start = time.time()
        for _ in range(3):
            pipeline.detect_objects(image)
        elapsed = time.time() - start

        avg_time = elapsed / 3
        print(f"Average detection time: {avg_time:.3f}s")
        assert avg_time < 3.0, f"Detection too slow: {avg_time:.3f}s"


class TestFurniturePipelineMultipleImages:
    """여러 이미지 처리 테스트"""

    def test_process_multiple_local_images(self, pipeline):
        """여러 로컬 이미지 처리"""
        test_images = [BEDROOM_IMAGE, KITCHEN_IMAGE, FURNITURE_IMAGE]
        existing_images = [img for img in test_images if img.exists()]

        if len(existing_images) < 2:
            pytest.skip("Not enough test images")

        all_objects = []
        for img_path in existing_images:
            image = Image.open(img_path)
            objects = pipeline.detect_objects(image)
            all_objects.extend(objects)
            print(f"{img_path.name}: {len(objects)} objects detected")

        print(f"Total objects across all images: {len(all_objects)}")
        assert len(all_objects) > 0
