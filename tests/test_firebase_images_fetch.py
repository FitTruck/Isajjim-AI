"""
Tests for ai/processors/1_firebase_images_fetch.py

ImageFetcher 클래스의 단위 테스트:
- 동기 이미지 가져오기
- 비동기 이미지 가져오기
- 다중 이미지 가져오기
- 오류 처리
"""

import pytest
import io
import importlib
from unittest.mock import patch, MagicMock
from PIL import Image

from ai.processors import ImageFetcher

# 숫자로 시작하는 모듈은 직접 import
_stage1 = importlib.import_module('.1_firebase_images_fetch', package='ai.processors')


class TestImageFetcherInit:
    """ImageFetcher 초기화 테스트"""

    def test_default_timeout(self):
        """기본 타임아웃"""
        fetcher = ImageFetcher()
        assert fetcher.timeout == 30

    def test_custom_timeout(self):
        """사용자 정의 타임아웃"""
        fetcher = ImageFetcher(timeout=60)
        assert fetcher.timeout == 60


class TestFetchSync:
    """fetch_sync 함수 테스트"""

    def test_successful_fetch(self):
        """성공적인 이미지 가져오기"""
        fetcher = ImageFetcher()

        # 테스트용 이미지 생성
        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        with patch.object(_stage1, 'requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = img_bytes.getvalue()
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch_sync("http://example.com/image.png")

            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.size == (100, 100)

    def test_http_error(self):
        """HTTP 오류"""
        fetcher = ImageFetcher()

        with patch.object(_stage1, 'requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_requests.get.return_value = mock_response

            result = fetcher.fetch_sync("http://example.com/notfound.png")

            assert result is None

    def test_timeout_error(self):
        """타임아웃 오류"""
        fetcher = ImageFetcher()

        with patch.object(_stage1, 'requests') as mock_requests:
            import requests
            mock_requests.get.side_effect = requests.Timeout()
            mock_requests.Timeout = requests.Timeout

            result = fetcher.fetch_sync("http://example.com/slow.png")

            assert result is None

    def test_general_exception(self):
        """일반 예외"""
        fetcher = ImageFetcher()
        import requests as real_requests

        with patch.object(_stage1, 'requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection error")
            # requests.Timeout을 실제 클래스로 설정
            mock_requests.Timeout = real_requests.Timeout

            result = fetcher.fetch_sync("http://example.com/error.png")

            assert result is None

    def test_no_requests_library(self):
        """requests 라이브러리가 없는 경우"""
        fetcher = ImageFetcher()
        original_value = _stage1.HAS_REQUESTS
        try:
            _stage1.HAS_REQUESTS = False
            result = fetcher.fetch_sync("http://example.com/image.png")
            assert result is None
        finally:
            _stage1.HAS_REQUESTS = original_value


class TestFetchAsync:
    """fetch_async 함수 테스트"""

    @pytest.mark.asyncio
    async def test_fallback_to_sync_when_no_aiohttp(self):
        """aiohttp가 없으면 동기 방식으로 폴백"""
        fetcher = ImageFetcher()
        original_value = _stage1.HAS_AIOHTTP

        # 테스트용 이미지
        test_image = Image.new('RGB', (50, 50), color='green')

        try:
            _stage1.HAS_AIOHTTP = False
            with patch.object(fetcher, 'fetch_sync') as mock_sync:
                mock_sync.return_value = test_image
                result = await fetcher.fetch_async("http://example.com/image.png")

                mock_sync.assert_called_once_with("http://example.com/image.png")
                assert result is test_image
        finally:
            _stage1.HAS_AIOHTTP = original_value

    @pytest.mark.asyncio
    async def test_general_exception(self):
        """일반 예외 발생 시 None 반환"""
        fetcher = ImageFetcher()

        with patch.object(fetcher, 'fetch_async', wraps=fetcher.fetch_async):
            # aiohttp 세션 오류 시뮬레이션
            with patch.object(_stage1, 'aiohttp') as mock_aiohttp:
                mock_aiohttp.ClientSession.side_effect = Exception("Connection error")
                mock_aiohttp.ClientTimeout.return_value = MagicMock()

                result = await fetcher.fetch_async("http://example.com/error.png")

                assert result is None


class TestFetchMultipleSync:
    """fetch_multiple_sync 함수 테스트"""

    def test_multiple_urls(self):
        """여러 URL 순차 가져오기"""
        fetcher = ImageFetcher()

        urls = [
            "http://example.com/image1.png",
            "http://example.com/image2.png",
            "http://example.com/image3.png"
        ]

        # 테스트용 이미지
        test_image = Image.new('RGB', (50, 50), color='red')

        with patch.object(fetcher, 'fetch_sync') as mock_fetch:
            mock_fetch.return_value = test_image
            results = fetcher.fetch_multiple_sync(urls)

            assert len(results) == 3
            assert all(url in [r[0] for r in results] for url in urls)
            assert mock_fetch.call_count == 3

    def test_partial_failure(self):
        """일부 URL 실패"""
        fetcher = ImageFetcher()

        urls = ["http://example.com/ok.png", "http://example.com/fail.png"]
        test_image = Image.new('RGB', (50, 50), color='red')

        def mock_fetch(url):
            if "fail" in url:
                return None
            return test_image

        with patch.object(fetcher, 'fetch_sync', side_effect=mock_fetch):
            results = fetcher.fetch_multiple_sync(urls)

            assert len(results) == 2
            assert results[0][1] is not None
            assert results[1][1] is None


class TestFetchMultipleAsync:
    """fetch_multiple_async 함수 테스트"""

    @pytest.mark.asyncio
    async def test_multiple_urls(self):
        """여러 URL 동시 가져오기"""
        fetcher = ImageFetcher()

        urls = [
            "http://example.com/image1.png",
            "http://example.com/image2.png"
        ]

        test_image = Image.new('RGB', (50, 50), color='blue')

        async def mock_fetch_async(url):
            return test_image

        with patch.object(fetcher, 'fetch_async', side_effect=mock_fetch_async):
            results = await fetcher.fetch_multiple_async(urls)

            assert len(results) == 2
            assert all(r[1] is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """동시 요청 수 제한"""
        fetcher = ImageFetcher()

        urls = [f"http://example.com/image{i}.png" for i in range(10)]
        test_image = Image.new('RGB', (50, 50), color='red')

        async def mock_fetch_async(url):
            return test_image

        with patch.object(fetcher, 'fetch_async', side_effect=mock_fetch_async):
            results = await fetcher.fetch_multiple_async(urls, max_concurrent=3)

            assert len(results) == 10

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """예외 처리"""
        fetcher = ImageFetcher()

        urls = ["http://example.com/ok.png", "http://example.com/error.png"]

        async def mock_fetch_async(url):
            if "error" in url:
                raise Exception("Fetch error")
            return Image.new('RGB', (50, 50), color='red')

        with patch.object(fetcher, 'fetch_async', side_effect=mock_fetch_async):
            results = await fetcher.fetch_multiple_async(urls)

            assert len(results) == 2
            # 예외가 발생한 URL은 None으로 처리됨
            assert results[1][1] is None
