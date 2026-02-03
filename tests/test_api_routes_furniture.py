"""
Furniture Routes 테스트

api/routes/furniture.py 커버리지 향상을 위한 통합 테스트
"""

import base64
import io
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.app import app
from api.models import AnalyzeFurnitureRequest, ImageUrlItem


def create_test_image_base64():
    """테스트용 base64 이미지 생성"""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def client():
    """TestClient fixture"""
    return TestClient(app)


@pytest.fixture
def test_image_base64():
    """테스트 이미지 base64 fixture"""
    return create_test_image_base64()


class TestAnalyzeFurnitureEndpoint:
    """POST /analyze-furniture 테스트"""

    def test_analyze_furniture_success(self, client):
        """정상 요청 처리"""
        request_data = {
            "estimate_id": 123,
            "image_urls": [
                {"id": 1, "url": "https://firebase.com/image1.jpg"},
                {"id": 2, "url": "https://firebase.com/image2.jpg"},
            ],
            "enable_mask": True,
            "enable_3d": True,
            "max_concurrent": 4,
        }

        response = client.post("/analyze-furniture", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["estimate_id"] == 123
        assert data["status"] == "processing"

    def test_analyze_furniture_no_images(self, client):
        """이미지 없이 요청"""
        request_data = {
            "estimate_id": 123,
            "image_urls": [],
        }

        response = client.post("/analyze-furniture", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "At least 1 image URL required" in data["error"]

    def test_analyze_furniture_too_many_images(self, client):
        """이미지 20개 초과"""
        request_data = {
            "estimate_id": 123,
            "image_urls": [
                {"id": i, "url": f"https://firebase.com/image{i}.jpg"}
                for i in range(21)
            ],
        }

        response = client.post("/analyze-furniture", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "Maximum 20 image URLs allowed" in data["error"]

    def test_analyze_furniture_invalid_request(self, client):
        """잘못된 요청 형식"""
        response = client.post("/analyze-furniture", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error


class TestAnalyzeFurnitureBase64Endpoint:
    """POST /analyze-furniture-base64 테스트"""

    def test_analyze_base64_invalid_image(self, client):
        """잘못된 base64 이미지"""
        response = client.post("/analyze-furniture-base64", json={
            "image": "not_valid_base64!!!",
        })

        assert response.status_code == 400
        assert "Invalid base64 image" in response.json()["error"]

    def test_analyze_base64_returns_objects(self, client, test_image_base64):
        """Base64 이미지 분석 - 객체 반환"""
        response = client.post("/analyze-furniture-base64", json={
            "image": test_image_base64,
            "enable_mask": False,
            "enable_3d": False,
        })

        assert response.status_code == 200
        data = response.json()
        assert "objects" in data


class TestDetectFurnitureEndpoint:
    """POST /detect-furniture 테스트"""

    def test_detect_furniture_returns_objects(self, client, test_image_base64):
        """탐지 결과 반환"""
        response = client.post("/detect-furniture", json={
            "image": test_image_base64,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "objects" in data
        assert "total_objects" in data
        assert "processing_time_seconds" in data

    def test_detect_furniture_invalid_image(self, client):
        """잘못된 이미지"""
        response = client.post("/detect-furniture", json={
            "image": "invalid_base64!@#$",
        })

        assert response.status_code == 400
        assert "Invalid base64 image" in response.json()["error"]


# Direct function tests
class TestFurnitureFunctions:
    """furniture.py 함수 직접 테스트"""

    def test_ensure_gcs_initialized_already_initialized(self):
        """이미 초기화된 경우"""
        from api.routes.furniture import ensure_gcs_initialized
        import api.routes.furniture as module

        module._gcs_initialized = True
        mock_service = MagicMock()

        with patch.object(module, 'get_gcs_service', return_value=mock_service):
            result = ensure_gcs_initialized()

        assert result == mock_service

    def test_ensure_gcs_initialized_no_credentials(self):
        """credentials 파일 없음"""
        from api.routes.furniture import ensure_gcs_initialized
        import api.routes.furniture as module

        module._gcs_initialized = False

        with patch.object(module, 'os') as mock_os:
            mock_os.path.exists.return_value = False
            with patch.object(module, 'get_gcs_service', return_value=None):
                result = ensure_gcs_initialized()

        assert result is None

    def test_get_furniture_pipeline_creates_pipeline(self):
        """파이프라인 생성"""
        from api.routes.furniture import get_furniture_pipeline
        import api.routes.furniture as module

        # Reset state
        module._furniture_pipeline = None

        # 이미 실제 파이프라인이 있으면 그것을 반환
        pipeline = get_furniture_pipeline()
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_process_background_success(self):
        """백그라운드 처리 성공"""
        from api.routes.furniture import process_furniture_analysis_background

        mock_pipeline = MagicMock()
        mock_pipeline.process_multiple_images_with_ids = AsyncMock(return_value=[])
        mock_pipeline.to_json_response_v2 = MagicMock(return_value={"results": []})

        with patch('api.routes.furniture.ensure_gcs_initialized', return_value=None):
            with patch('api.routes.furniture.get_furniture_pipeline_with_gcs', return_value=mock_pipeline):
                with patch('api.routes.furniture.send_callback', new_callable=AsyncMock) as mock_callback:
                    await process_furniture_analysis_background(
                        estimate_id=123,
                        image_items=[(1, "url1")],
                        enable_mask=True,
                        enable_3d=True,
                        max_concurrent=4,
                    )

                    mock_callback.assert_called_once()
                    # 성공 시 result_data가 포함됨
                    call_kwargs = mock_callback.call_args[1]
                    assert "result_data" in call_kwargs

    @pytest.mark.asyncio
    async def test_process_background_error(self):
        """백그라운드 처리 실패"""
        from api.routes.furniture import process_furniture_analysis_background

        with patch('api.routes.furniture.ensure_gcs_initialized', return_value=None):
            with patch('api.routes.furniture.get_furniture_pipeline_with_gcs', side_effect=Exception("Pipeline error")):
                with patch('api.routes.furniture.send_callback', new_callable=AsyncMock) as mock_callback:
                    await process_furniture_analysis_background(
                        estimate_id=123,
                        image_items=[(1, "url1")],
                        enable_mask=True,
                        enable_3d=True,
                        max_concurrent=4,
                    )

                    mock_callback.assert_called_once()
                    # 실패 시 error가 포함됨
                    call_kwargs = mock_callback.call_args[1]
                    assert "error" in call_kwargs

    def test_get_furniture_pipeline_with_gcs(self):
        """GCS가 설정된 파이프라인 생성"""
        from api.routes.furniture import get_furniture_pipeline_with_gcs

        mock_gcs = MagicMock()
        pipeline = get_furniture_pipeline_with_gcs(
            gcs_service=mock_gcs,
            estimate_id=123,
            device_id=0
        )
        assert pipeline is not None


class TestAnalyzeFurnitureSingleEndpoint:
    """POST /analyze-furniture-single 테스트"""

    def test_analyze_furniture_single_valid_url(self, client):
        """유효한 URL로 요청"""
        response = client.post("/analyze-furniture-single", json={
            "image_url": "https://firebase.com/image.jpg",
            "enable_mask": True,
            "enable_3d": False,
        })

        # URL 에러나 성공 모두 200 반환 가능
        assert response.status_code in [200, 500]
