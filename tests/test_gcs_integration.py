"""
GCS Integration Tests (requires actual GCS credentials).

이 테스트는 실제 GCS에 업로드하므로 CI에서는 스킵됩니다.

테스트 실행:
    pytest tests/test_gcs_integration.py -v -m "not ci_skip"
"""

import asyncio
import base64
import os
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import GCS_BUCKET_NAME, GCS_CREDENTIALS_PATH
from api.services.gcs_storage import GCSStorageService


# 실제 credentials가 있을 때만 테스트 실행
CREDENTIALS_EXISTS = os.path.exists(GCS_CREDENTIALS_PATH)


@pytest.mark.skipif(not CREDENTIALS_EXISTS, reason="GCS credentials not found")
class TestGCSIntegration:
    """실제 GCS 업로드 통합 테스트"""

    @pytest.fixture
    def gcs_service(self):
        """GCS 서비스 인스턴스 생성"""
        return GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS_PATH)

    @pytest.mark.asyncio
    async def test_real_upload_ply(self, gcs_service):
        """실제 PLY 파일 업로드 테스트"""
        # 간단한 PLY 데이터 생성
        ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 0 0
0 1 0
"""
        ply_bytes = ply_content.encode('utf-8')
        ply_b64 = base64.b64encode(ply_bytes).decode('utf-8')

        # 파일명 생성
        filename = gcs_service.generate_unique_filename(
            label="TEST_SOFA",
            estimate_id=999,
            image_id=1,
            object_idx=0
        )

        # 업로드
        url = await gcs_service.upload_ply_base64(ply_b64, filename)

        # 검증
        assert url.startswith("https://storage.googleapis.com/")
        assert GCS_BUCKET_NAME in url
        assert "TEST_SOFA" in url
        assert filename in url

        print(f"\n[Test] Uploaded to: {url}")

    @pytest.mark.asyncio
    async def test_real_upload_multiple(self, gcs_service):
        """여러 PLY 파일 병렬 업로드 테스트"""
        ply_content = "ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n"
        ply_b64 = base64.b64encode(ply_content.encode()).decode('utf-8')

        items = []
        for i in range(3):
            filename = gcs_service.generate_unique_filename(
                label=f"TEST_OBJ_{i}",
                estimate_id=998,
                image_id=1,
                object_idx=i
            )
            items.append((ply_b64, filename))

        # 병렬 업로드
        urls = await gcs_service.upload_multiple_ply(items)

        # 검증
        assert len(urls) == 3
        for url in urls:
            if not isinstance(url, Exception):
                assert "https://storage.googleapis.com/" in url
                print(f"\n[Test] Uploaded: {url}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
