"""
Callback Service 테스트

api/services/callback.py 커버리지 향상을 위한 단위 테스트
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp

from api.services.callback import send_callback


class TestSendCallback:
    """send_callback 함수 테스트"""

    @pytest.mark.asyncio
    async def test_send_callback_success_with_result_data(self):
        """결과 데이터로 콜백 성공"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            result = await send_callback(
                estimate_id=123,
                result_data={"results": [{"label": "SOFA"}]}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_callback_success_with_error(self):
        """에러 메시지로 콜백 성공"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            result = await send_callback(
                estimate_id=123,
                error="Processing failed"
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_callback_no_data_default_error(self):
        """데이터 없이 콜백 (기본 에러 메시지)"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            result = await send_callback(estimate_id=123)

        assert result is True
        # Verify default error payload was sent
        call_args = mock_session.post.call_args
        assert call_args[1]["json"] == {"error": "Unknown error"}

    @pytest.mark.asyncio
    async def test_send_callback_http_error_status(self):
        """HTTP 에러 상태 코드"""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            with patch('api.services.callback.CALLBACK_RETRY_COUNT', 0):
                result = await send_callback(
                    estimate_id=123,
                    result_data={"results": []}
                )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_callback_network_error_with_retry(self):
        """네트워크 오류로 재시도"""
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            with patch('api.services.callback.CALLBACK_RETRY_COUNT', 2):
                result = await send_callback(
                    estimate_id=123,
                    result_data={"results": []}
                )

        assert result is False
        # 초기 시도 + 2번 재시도 = 3번 호출
        assert mock_session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_send_callback_unexpected_exception(self):
        """예상치 못한 예외"""
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=RuntimeError("Unexpected error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            with patch('api.services.callback.CALLBACK_RETRY_COUNT', 0):
                result = await send_callback(
                    estimate_id=123,
                    result_data={"results": []}
                )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_callback_url_format(self):
        """콜백 URL 형식 확인"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            with patch('api.services.callback.CALLBACK_URL_TEMPLATE', 'https://api.example.com/callback/{estimateId}'):
                await send_callback(estimate_id=456, result_data={"results": []})

        call_args = mock_session.post.call_args
        called_url = call_args[0][0]
        assert "456" in called_url

    @pytest.mark.asyncio
    async def test_send_callback_content_type_header(self):
        """Content-Type 헤더 확인"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
            await send_callback(estimate_id=123, result_data={"results": []})

        call_args = mock_session.post.call_args
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_callback_status_codes_success_range(self):
        """2xx 상태 코드 범위 성공 처리"""
        for status_code in [200, 201, 202, 204]:
            mock_response = MagicMock()
            mock_response.status = status_code
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
                result = await send_callback(
                    estimate_id=123,
                    result_data={"results": []}
                )

            assert result is True, f"Status {status_code} should be success"

    @pytest.mark.asyncio
    async def test_send_callback_status_codes_failure_range(self):
        """4xx/5xx 상태 코드 범위 실패 처리"""
        for status_code in [400, 401, 403, 404, 500, 502, 503]:
            mock_response = MagicMock()
            mock_response.status = status_code
            mock_response.text = AsyncMock(return_value="Error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            with patch('api.services.callback.aiohttp.ClientSession', return_value=mock_session):
                with patch('api.services.callback.CALLBACK_RETRY_COUNT', 0):
                    result = await send_callback(
                        estimate_id=123,
                        result_data={"results": []}
                    )

            assert result is False, f"Status {status_code} should be failure"
