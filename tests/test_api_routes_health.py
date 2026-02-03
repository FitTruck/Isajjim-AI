"""
Health Routes 테스트

api/routes/health.py 커버리지 향상을 위한 통합 테스트
"""

import os
import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    """TestClient fixture"""
    return TestClient(app)


class TestHealthEndpoint:
    """GET /health 테스트"""

    def test_health_check_returns_healthy(self, client):
        """헬스 체크 성공"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data


class TestGpuStatusEndpoint:
    """GET /gpu-status 테스트"""

    def test_gpu_status_endpoint_returns_json(self, client):
        """GPU 상태 엔드포인트가 JSON 응답 반환"""
        response = client.get("/gpu-status")
        assert response.status_code == 200
        data = response.json()
        # 풀이 있거나 에러가 있거나 둘 중 하나
        assert "total_gpus" in data or "error" in data


class TestAssetsListEndpoint:
    """GET /assets-list 테스트"""

    def test_assets_list_returns_json(self, client):
        """에셋 목록 엔드포인트가 JSON 응답 반환"""
        response = client.get("/assets-list")
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total_files" in data
        assert "total_size_bytes" in data


class TestAssetsEndpoint:
    """GET /assets/{filename} 테스트"""

    def test_get_asset_not_found(self, client):
        """존재하지 않는 파일 요청"""
        response = client.get("/assets/nonexistent_file_12345.ply")
        assert response.status_code == 404


# Direct function tests for better coverage
class TestHealthFunctions:
    """health.py 함수 직접 테스트"""

    @pytest.mark.asyncio
    async def test_health_check_direct(self):
        """health_check 함수 직접 테스트"""
        from api.routes.health import health_check
        result = await health_check()
        assert result["status"] == "healthy"
        assert "device" in result

    @pytest.mark.asyncio
    async def test_gpu_status_direct_with_mock_pool(self):
        """gpu_status 함수 직접 테스트 (mock pool)"""
        from api.routes.health import gpu_status

        mock_pool = MagicMock()
        mock_pool.get_status.return_value = {
            "total_gpus": 4,
            "available_gpus": 3,
            "gpus": {"0": {"available": True}}
        }
        mock_pool.get_pipelines_status.return_value = {
            "initialized_pipelines": 4,
            "gpus": {0: {"has_pipeline": True}}
        }

        with patch('ai.gpu.get_gpu_pool', return_value=mock_pool):
            result = await gpu_status()

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_gpu_status_direct_with_exception(self):
        """gpu_status 함수 직접 테스트 (예외)"""
        from api.routes.health import gpu_status

        with patch('ai.gpu.get_gpu_pool', side_effect=RuntimeError("Pool error")):
            result = await gpu_status()

        assert result.status_code == 200
        data = result.body.decode()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_list_assets_direct(self):
        """list_assets 함수 직접 테스트"""
        from api.routes.health import list_assets

        result = await list_assets()
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_assets_no_directory(self):
        """list_assets - 디렉토리 없을 때"""
        from api.routes.health import list_assets

        with patch('api.routes.health.os.path.exists', return_value=False):
            result = await list_assets()

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["files"] == []

    @pytest.mark.asyncio
    async def test_list_assets_with_error(self):
        """list_assets - 디렉토리 읽기 오류"""
        from api.routes.health import list_assets

        with patch('api.routes.health.os.path.exists', return_value=True):
            with patch('api.routes.health.os.listdir', side_effect=PermissionError("Access denied")):
                result = await list_assets()

        assert result.status_code == 500
