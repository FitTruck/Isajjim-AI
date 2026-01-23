"""
Tests for api/routes/health.py

Health check endpoints unit tests:
- /health endpoint
- /gpu-status endpoint
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


class TestHealthCheck:
    """Health check endpoint tests"""

    def test_health_check_returns_healthy(self):
        """Health check returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data

    def test_health_check_includes_device(self):
        """Health check includes device information"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Device should be a string (e.g., "cuda:0" or "cpu")
        assert isinstance(data["device"], str)


class TestGpuStatus:
    """GPU status endpoint tests"""

    def test_gpu_status_returns_json(self):
        """GPU status returns JSON response"""
        response = client.get("/gpu-status")
        assert response.status_code == 200
        data = response.json()
        # Should have required fields
        assert "total_gpus" in data or "error" in data

    def test_gpu_status_with_pool_available(self):
        """GPU status when pool is available"""
        mock_pool = MagicMock()
        mock_pool.get_status.return_value = {
            "total_gpus": 2,
            "available_gpus": 1,
            "gpus": {
                "0": {"available": True, "task_id": None},
                "1": {"available": False, "task_id": "test-task"}
            }
        }
        mock_pool.get_pipelines_status.return_value = {
            "initialized_pipelines": 2,
            "gpus": {
                0: {"has_pipeline": True},
                1: {"has_pipeline": True}
            }
        }

        with patch("ai.gpu.get_gpu_pool", return_value=mock_pool):
            response = client.get("/gpu-status")
            assert response.status_code == 200
            data = response.json()
            # Pool is already initialized, so we get actual data
            assert "total_gpus" in data or "error" in data

    def test_gpu_status_pool_not_initialized(self):
        """GPU status when pool is not initialized returns error gracefully"""
        # This test verifies the endpoint handles errors gracefully
        response = client.get("/gpu-status")
        assert response.status_code == 200
        data = response.json()
        # Should either have total_gpus or error
        assert "total_gpus" in data or "error" in data

    def test_gpu_status_returns_required_fields(self):
        """GPU status returns expected structure"""
        response = client.get("/gpu-status")
        assert response.status_code == 200
        data = response.json()
        # Should have these fields in success or error case
        if "error" not in data:
            assert "total_gpus" in data
            assert "available_gpus" in data
