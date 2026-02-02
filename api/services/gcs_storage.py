"""
GCS Storage Service

PLY 파일을 Google Cloud Storage에 업로드하고 Public URL을 반환합니다.
"""

import asyncio
import base64
import logging
import re
import uuid
from datetime import datetime
from typing import Optional, Union

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# 허용되는 파일명 패턴 (ply/ 프리픽스, 알파벳/숫자/언더스코어/하이픈/점)
VALID_FILENAME_PATTERN = re.compile(r'^ply/[a-zA-Z0-9_\-\.]+\.ply$')


class GCSStorageService:
    """
    Google Cloud Storage PLY 업로드 서비스.

    Usage:
        gcs = GCSStorageService(bucket_name, credentials_path)
        url = await gcs.upload_ply_base64(ply_b64, "ply/sofa_123.ply")
    """

    def __init__(self, bucket_name: str, credentials_path: str):
        """
        GCS 클라이언트 초기화.

        Args:
            bucket_name: GCS 버킷 이름
            credentials_path: 서비스 계정 JSON 파일 경로
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self._client = storage.Client(credentials=credentials)
            self._bucket = self._client.bucket(bucket_name)
            self._bucket_name = bucket_name
            logger.info(f"[GCS] Initialized with bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"[GCS] Failed to initialize: {e}")
            raise

    def _validate_filename(self, filename: str) -> str:
        """
        파일명 검증 및 정제.

        Args:
            filename: GCS 파일 경로 (예: "ply/sofa_123.ply")

        Returns:
            검증된 파일명

        Raises:
            ValueError: 유효하지 않은 파일명
        """
        # ply/ 프리픽스 확인
        if not filename.startswith("ply/"):
            raise ValueError("Filename must start with 'ply/'")

        # 경로 순회 공격 방지
        if ".." in filename or "//" in filename:
            raise ValueError("Invalid filename: path traversal detected")

        # 허용된 문자만 사용
        if not VALID_FILENAME_PATTERN.match(filename):
            raise ValueError(f"Invalid filename format: {filename}")

        return filename

    def generate_unique_filename(
        self,
        label: str,
        estimate_id: Optional[int] = None,
        image_id: Optional[int] = None,
        object_idx: Optional[int] = None,
    ) -> str:
        """
        고유한 PLY 파일명 생성.

        Format: ply/est{estimate_id}_img{image_id}_{label}_{timestamp}_{uuid}.ply

        Args:
            label: 객체 라벨 (예: "SOFA")
            estimate_id: 견적 ID (옵션)
            image_id: 이미지 ID (옵션)
            object_idx: 객체 인덱스 (옵션)

        Returns:
            고유한 파일 경로 (예: "ply/est123_img101_SOFA_20260202_abc12345.ply")
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        # 라벨 정리 (공백 제거, 소문자)
        safe_label = label.replace(" ", "_").upper()

        parts = ["ply"]
        filename_parts = []

        if estimate_id is not None:
            filename_parts.append(f"est{estimate_id}")
        if image_id is not None:
            filename_parts.append(f"img{image_id}")
        if object_idx is not None:
            filename_parts.append(f"obj{object_idx}")

        filename_parts.append(safe_label)
        filename_parts.append(timestamp)
        filename_parts.append(unique_id)

        filename = "_".join(filename_parts) + ".ply"
        return f"ply/{filename}"

    async def upload_ply_base64(self, ply_b64: str, filename: str) -> str:
        """
        Base64 인코딩된 PLY 데이터를 GCS에 업로드.

        Args:
            ply_b64: Base64 인코딩된 PLY 데이터
            filename: GCS 내 파일 경로 (예: "ply/sofa_123.ply")

        Returns:
            Public URL (예: "https://storage.googleapis.com/bucket/ply/sofa_123.ply")
        """
        try:
            ply_bytes = base64.b64decode(ply_b64)
            return await self.upload_ply_bytes(ply_bytes, filename)
        except Exception as e:
            logger.error(f"[GCS] Failed to decode base64: {e}")
            raise

    async def upload_ply_bytes(self, ply_data: bytes, filename: str) -> str:
        """
        PLY 바이트 데이터를 GCS에 업로드.

        Args:
            ply_data: PLY 파일 바이트 데이터
            filename: GCS 내 파일 경로

        Returns:
            Public URL

        Raises:
            ValueError: 유효하지 않은 파일명
        """
        # 파일명 검증
        filename = self._validate_filename(filename)

        def _upload() -> str:
            blob = self._bucket.blob(filename)
            blob.upload_from_string(
                ply_data, content_type="application/octet-stream"
            )
            # Public URL 반환
            url = f"https://storage.googleapis.com/{self._bucket_name}/{filename}"
            logger.info(f"[GCS] Uploaded: {filename} ({len(ply_data)} bytes)")
            return url

        try:
            return await asyncio.to_thread(_upload)
        except Exception as e:
            logger.error(f"[GCS] Upload failed for {filename}: {e}")
            raise

    async def upload_multiple_ply(
        self,
        items: list[tuple[str, str]],
    ) -> list[Union[str, Exception]]:
        """
        여러 PLY 파일을 병렬로 업로드.

        Args:
            items: [(ply_b64, filename), ...] 튜플 리스트

        Returns:
            업로드된 URL 리스트. 실패한 경우 Exception 객체 포함.
            호출자는 isinstance(result, Exception)으로 확인해야 함.
        """
        tasks = [
            self.upload_ply_base64(ply_b64, filename)
            for ply_b64, filename in items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# 전역 GCS 서비스 인스턴스 (lazy initialization)
_gcs_service: Optional[GCSStorageService] = None


def initialize_gcs_service(
    bucket_name: str, credentials_path: str
) -> GCSStorageService:
    """
    전역 GCS 서비스 초기화.

    Args:
        bucket_name: GCS 버킷 이름
        credentials_path: 서비스 계정 JSON 파일 경로

    Returns:
        GCSStorageService 인스턴스
    """
    global _gcs_service
    _gcs_service = GCSStorageService(bucket_name, credentials_path)
    return _gcs_service


def get_gcs_service() -> Optional[GCSStorageService]:
    """전역 GCS 서비스 인스턴스 반환."""
    return _gcs_service
