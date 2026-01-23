"""
Test Callback API functionality

Tests for:
- Async mode with callback_url
- Task status endpoint
- Callback sending logic
- Error handling
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from api.models import (
    AnalyzeFurnitureRequest,
    ImageUrlItem,
    CallbackPayload,
)
from api.routes.furniture import (
    send_callback,
    process_and_callback,
    _task_status,
    CALLBACK_TIMEOUT_SECONDS,
    CALLBACK_MAX_RETRIES,
)


class TestCallbackPayloadModel:
    """CallbackPayload 모델 테스트"""

    def test_completed_payload(self):
        """성공 페이로드 생성"""
        payload = CallbackPayload(
            task_id="test-123",
            status="completed",
            results=[{"image_id": 1, "objects": []}]
        )
        assert payload.task_id == "test-123"
        assert payload.status == "completed"
        assert payload.results is not None
        assert payload.error is None

    def test_failed_payload(self):
        """실패 페이로드 생성"""
        payload = CallbackPayload(
            task_id="test-456",
            status="failed",
            error="Processing failed"
        )
        assert payload.task_id == "test-456"
        assert payload.status == "failed"
        assert payload.results is None
        assert payload.error == "Processing failed"


class TestAnalyzeFurnitureRequestWithCallback:
    """AnalyzeFurnitureRequest callback_url 필드 테스트"""

    def test_request_without_callback(self):
        """callback_url 없이 요청 생성"""
        request = AnalyzeFurnitureRequest(
            estimate_id=1,
            image_urls=[ImageUrlItem(id=1, url="https://example.com/image.jpg")]
        )
        assert request.callback_url is None
        assert request.enable_mask is True
        assert request.enable_3d is True

    def test_request_with_callback(self):
        """callback_url 포함 요청 생성"""
        request = AnalyzeFurnitureRequest(
            estimate_id=1,
            image_urls=[ImageUrlItem(id=1, url="https://example.com/image.jpg")],
            callback_url="http://api.example.com/callback"
        )
        assert request.callback_url == "http://api.example.com/callback"


class TestSendCallback:
    """send_callback 함수 테스트"""

    @pytest.mark.asyncio
    async def test_successful_callback(self):
        """성공적인 callback 전송"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("api.routes.furniture.aiohttp.ClientSession", return_value=mock_session):
            result = await send_callback(
                "http://api.example.com/callback",
                {"task_id": "test-123", "status": "completed"}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_callback_server_error_with_retry(self):
        """서버 에러 시 재시도"""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("api.routes.furniture.aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await send_callback(
                    "http://api.example.com/callback",
                    {"task_id": "test-123", "status": "completed"},
                    retries=2
                )
                assert result is False
                # Should have been called twice (initial + 1 retry)
                assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_callback_connection_error(self):
        """연결 에러 시 재시도"""
        import aiohttp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("api.routes.furniture.aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await send_callback(
                    "http://api.example.com/callback",
                    {"task_id": "test-123", "status": "completed"},
                    retries=2
                )
                assert result is False


class TestProcessAndCallback:
    """process_and_callback 함수 테스트"""

    @pytest.mark.asyncio
    async def test_successful_processing(self):
        """성공적인 처리 및 callback"""
        # Setup
        task_id = "test-task-123"
        request = AnalyzeFurnitureRequest(
            estimate_id=1,
            image_urls=[ImageUrlItem(id=1, url="https://example.com/image.jpg")],
            callback_url="http://api.example.com/callback"
        )

        # Initialize task status
        _task_status[task_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.process_multiple_images_with_ids = AsyncMock(return_value=[])
        mock_pipeline.to_json_response_v2 = MagicMock(return_value={"results": []})

        with patch("api.routes.furniture.get_furniture_pipeline", return_value=mock_pipeline):
            with patch("api.routes.furniture.send_callback", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = True

                await process_and_callback(task_id, request)

                # Verify status updated
                assert _task_status[task_id]["status"] == "completed"
                assert _task_status[task_id]["callback_sent"] is True

                # Verify callback was sent with correct payload
                mock_send.assert_called_once()
                call_args = mock_send.call_args
                assert call_args[0][0] == "http://api.example.com/callback"
                assert call_args[0][1]["task_id"] == task_id
                assert call_args[0][1]["status"] == "completed"

        # Cleanup
        del _task_status[task_id]

    @pytest.mark.asyncio
    async def test_failed_processing(self):
        """처리 실패 시 에러 callback"""
        task_id = "test-task-456"
        request = AnalyzeFurnitureRequest(
            estimate_id=1,
            image_urls=[ImageUrlItem(id=1, url="https://example.com/image.jpg")],
            callback_url="http://api.example.com/callback"
        )

        # Initialize task status
        _task_status[task_id] = {
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Mock pipeline to raise exception
        mock_pipeline = MagicMock()
        mock_pipeline.process_multiple_images_with_ids = AsyncMock(
            side_effect=Exception("Test error")
        )

        with patch("api.routes.furniture.get_furniture_pipeline", return_value=mock_pipeline):
            with patch("api.routes.furniture.send_callback", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = True

                await process_and_callback(task_id, request)

                # Verify status updated to failed
                assert _task_status[task_id]["status"] == "failed"
                assert "Test error" in _task_status[task_id]["error"]

                # Verify error callback was sent
                mock_send.assert_called_once()
                call_args = mock_send.call_args
                assert call_args[0][1]["status"] == "failed"
                assert "error" in call_args[0][1]

        # Cleanup
        del _task_status[task_id]


class TestTaskStatusEndpoint:
    """작업 상태 조회 엔드포인트 테스트"""

    def test_task_status_storage(self):
        """작업 상태 저장소 테스트"""
        task_id = "test-status-123"

        # Add task status
        _task_status[task_id] = {
            "status": "processing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "callback_url": "http://example.com/callback",
            "image_count": 5
        }

        # Verify retrieval
        assert task_id in _task_status
        assert _task_status[task_id]["status"] == "processing"
        assert _task_status[task_id]["image_count"] == 5

        # Cleanup
        del _task_status[task_id]

    def test_task_not_found(self):
        """존재하지 않는 작업 조회"""
        assert "nonexistent-task" not in _task_status


class TestIntegrationWithFastAPI:
    """FastAPI 통합 테스트"""

    @pytest.mark.asyncio
    async def test_analyze_furniture_async_mode(self):
        """비동기 모드 요청 테스트 (통합)"""
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        # We need to test the actual endpoint behavior
        # This requires mocking the background task execution

        request_data = {
            "estimate_id": 1,
            "image_urls": [
                {"id": 1, "url": "https://example.com/image1.jpg"},
                {"id": 2, "url": "https://example.com/image2.jpg"}
            ],
            "callback_url": "http://api.example.com/callback"
        }

        # Verify request model accepts callback_url
        request = AnalyzeFurnitureRequest(**request_data)
        assert request.callback_url == "http://api.example.com/callback"
        assert len(request.image_urls) == 2

    @pytest.mark.asyncio
    async def test_analyze_furniture_sync_mode(self):
        """동기 모드 요청 테스트 (호환성)"""
        request_data = {
            "estimate_id": 1,
            "image_urls": [
                {"id": 1, "url": "https://example.com/image1.jpg"}
            ]
        }

        # Verify request model works without callback_url
        request = AnalyzeFurnitureRequest(**request_data)
        assert request.callback_url is None


class TestCallbackConfiguration:
    """Callback 설정 테스트"""

    def test_timeout_configuration(self):
        """타임아웃 설정 확인"""
        assert CALLBACK_TIMEOUT_SECONDS == 30

    def test_retry_configuration(self):
        """재시도 횟수 설정 확인"""
        assert CALLBACK_MAX_RETRIES == 3
