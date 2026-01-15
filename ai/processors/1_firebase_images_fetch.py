"""
Stage 1: Firebase Storage 이미지 가져오기

Firebase Storage URL에서 이미지를 다운로드합니다.
비동기(async) 및 동기(sync) 버전 모두 제공합니다.
"""

import io
import asyncio
from typing import Optional, List, Tuple
from PIL import Image

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ImageFetcher:
    """
    Firebase Storage 또는 일반 URL에서 이미지를 가져오는 클래스

    AI Logic Step 1: 이미지 다운로드
    """

    def __init__(self, timeout: int = 30):
        """
        Args:
            timeout: 요청 타임아웃 (초)
        """
        self.timeout = timeout

    async def fetch_async(self, url: str) -> Optional[Image.Image]:
        """
        비동기로 URL에서 이미지를 가져옵니다.

        Args:
            url: 이미지 URL (Firebase Storage 또는 일반 URL)

        Returns:
            PIL Image 객체 또는 None (실패 시)
        """
        if not HAS_AIOHTTP:
            print("[ImageFetcher] aiohttp not installed, falling back to sync")
            return self.fetch_sync(url)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        print(f"[ImageFetcher] HTTP {response.status} for {url}")
                        return None

                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    return image

        except asyncio.TimeoutError:
            print(f"[ImageFetcher] Timeout fetching {url}")
            return None
        except Exception as e:
            print(f"[ImageFetcher] Error fetching {url}: {e}")
            return None

    def fetch_sync(self, url: str) -> Optional[Image.Image]:
        """
        동기로 URL에서 이미지를 가져옵니다.

        Args:
            url: 이미지 URL

        Returns:
            PIL Image 객체 또는 None (실패 시)
        """
        if not HAS_REQUESTS:
            print("[ImageFetcher] requests not installed")
            return None

        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                print(f"[ImageFetcher] HTTP {response.status_code} for {url}")
                return None

            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image

        except requests.Timeout:
            print(f"[ImageFetcher] Timeout fetching {url}")
            return None
        except Exception as e:
            print(f"[ImageFetcher] Error fetching {url}: {e}")
            return None

    async def fetch_multiple_async(
        self,
        urls: List[str],
        max_concurrent: int = 5
    ) -> List[Tuple[str, Optional[Image.Image]]]:
        """
        여러 URL에서 이미지를 동시에 가져옵니다.

        Args:
            urls: 이미지 URL 리스트
            max_concurrent: 최대 동시 요청 수

        Returns:
            [(url, image), ...] 리스트
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str):
            async with semaphore:
                image = await self.fetch_async(url)
                return (url, image)

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ImageFetcher] Exception for {urls[i]}: {result}")
                processed.append((urls[i], None))
            else:
                processed.append(result)

        return processed

    def fetch_multiple_sync(self, urls: List[str]) -> List[Tuple[str, Optional[Image.Image]]]:
        """
        여러 URL에서 이미지를 순차적으로 가져옵니다.

        Args:
            urls: 이미지 URL 리스트

        Returns:
            [(url, image), ...] 리스트
        """
        results = []
        for url in urls:
            image = self.fetch_sync(url)
            results.append((url, image))
        return results
