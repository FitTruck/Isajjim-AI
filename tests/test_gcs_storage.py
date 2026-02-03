"""
Tests for GCS Storage Service.

테스트 실행:
    pytest tests/test_gcs_storage.py -v
"""

import base64
import pytest
from unittest.mock import Mock, patch

# Import the service
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.gcs_storage import (
    GCSStorageService,
    initialize_gcs_service,
    get_gcs_service,
)


class TestGCSStorageService:
    """GCSStorageService 단위 테스트"""

    def test_generate_unique_filename_basic(self):
        """기본 파일명 생성 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()
                mock_storage.Client.return_value = Mock()

                service = GCSStorageService("test-bucket", "/fake/path.json")
                filename = service.generate_unique_filename("SOFA")

                assert filename.startswith("ply/")
                assert "SOFA" in filename
                assert filename.endswith(".ply")

    def test_generate_unique_filename_with_estimate_id(self):
        """estimate_id 포함 파일명 생성 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()
                mock_storage.Client.return_value = Mock()

                service = GCSStorageService("test-bucket", "/fake/path.json")
                filename = service.generate_unique_filename(
                    "BED",
                    estimate_id=123,
                    image_id=456,
                    object_idx=0
                )

                assert filename.startswith("ply/")
                assert "est123" in filename
                assert "img456" in filename
                assert "obj0" in filename
                assert "BED" in filename
                assert filename.endswith(".ply")

    def test_generate_unique_filename_sanitizes_label(self):
        """라벨에 공백이 있으면 언더스코어로 변환"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()
                mock_storage.Client.return_value = Mock()

                service = GCSStorageService("test-bucket", "/fake/path.json")
                filename = service.generate_unique_filename("dining table")

                assert "DINING_TABLE" in filename
                assert " " not in filename

    @pytest.mark.asyncio
    async def test_upload_ply_bytes(self):
        """PLY 바이트 업로드 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()

                mock_bucket = Mock()
                mock_blob = Mock()
                mock_bucket.blob.return_value = mock_blob
                mock_client = Mock()
                mock_client.bucket.return_value = mock_bucket
                mock_storage.Client.return_value = mock_client

                service = GCSStorageService("test-bucket", "/fake/path.json")

                ply_data = b"ply\nformat ascii 1.0\nelement vertex 3\n"
                url = await service.upload_ply_bytes(ply_data, "ply/test.ply")

                assert url == "https://storage.googleapis.com/test-bucket/ply/test.ply"
                mock_blob.upload_from_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_ply_base64(self):
        """Base64 PLY 업로드 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()

                mock_bucket = Mock()
                mock_blob = Mock()
                mock_bucket.blob.return_value = mock_blob
                mock_client = Mock()
                mock_client.bucket.return_value = mock_bucket
                mock_storage.Client.return_value = mock_client

                service = GCSStorageService("test-bucket", "/fake/path.json")

                ply_data = b"ply\nformat ascii 1.0\n"
                ply_b64 = base64.b64encode(ply_data).decode('utf-8')

                url = await service.upload_ply_base64(ply_b64, "ply/test.ply")

                assert url == "https://storage.googleapis.com/test-bucket/ply/test.ply"

    @pytest.mark.asyncio
    async def test_upload_multiple_ply(self):
        """여러 PLY 병렬 업로드 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()

                mock_bucket = Mock()
                mock_blob = Mock()
                mock_bucket.blob.return_value = mock_blob
                mock_client = Mock()
                mock_client.bucket.return_value = mock_bucket
                mock_storage.Client.return_value = mock_client

                service = GCSStorageService("test-bucket", "/fake/path.json")

                ply_data = b"ply\n"
                ply_b64 = base64.b64encode(ply_data).decode('utf-8')

                items = [
                    (ply_b64, "ply/test1.ply"),
                    (ply_b64, "ply/test2.ply"),
                    (ply_b64, "ply/test3.ply"),
                ]

                urls = await service.upload_multiple_ply(items)

                assert len(urls) == 3
                assert all("https://storage.googleapis.com/test-bucket/ply/" in str(u) for u in urls)


class TestGCSServiceGlobal:
    """전역 GCS 서비스 관리 테스트"""

    def test_initialize_and_get_gcs_service(self):
        """전역 서비스 초기화 및 조회 테스트"""
        with patch('api.services.gcs_storage.service_account') as mock_sa:
            with patch('api.services.gcs_storage.storage') as mock_storage:
                mock_sa.Credentials.from_service_account_file.return_value = Mock()
                mock_storage.Client.return_value = Mock()

                # Reset global state
                import api.services.gcs_storage as gcs_module
                gcs_module._gcs_service = None

                # Initialize
                service = initialize_gcs_service("test-bucket", "/fake/path.json")
                assert service is not None

                # Get
                retrieved = get_gcs_service()
                assert retrieved is service


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
