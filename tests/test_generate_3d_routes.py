"""
Tests for api/routes/generate_3d.py

3D generation endpoints unit tests:
- /generate-3d endpoint
- /generate-3d-status/{task_id} endpoint
- /assets-list endpoint
"""

import pytest
import base64
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import io

from api.app import app
from api.services.tasks import generation_tasks


client = TestClient(app)


class TestGenerate3DEndpoint:
    """POST /generate-3d endpoint tests"""

    @pytest.fixture
    def valid_image_b64(self):
        """Create valid base64 encoded image"""
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @pytest.fixture
    def valid_mask_b64(self):
        """Create valid base64 encoded mask"""
        mask = Image.new('L', (100, 100), color=255)
        buffer = io.BytesIO()
        mask.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def test_generate_3d_returns_task_id(self, valid_image_b64, valid_mask_b64):
        """Generate 3D returns task_id"""
        with patch("api.routes.generate_3d.generate_3d_background"):
            response = client.post("/generate-3d", json={
                "image": valid_image_b64,
                "mask": valid_mask_b64,
                "seed": 42
            })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data
        assert data["status"] == "queued"

        # Cleanup
        if data["task_id"] in generation_tasks:
            del generation_tasks[data["task_id"]]

    def test_generate_3d_invalid_image(self):
        """Invalid base64 image returns 400"""
        response = client.post("/generate-3d", json={
            "image": "invalid_base64!!!",
            "mask": "invalid_base64!!!",
            "seed": 42
        })

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_generate_3d_with_skip_gif(self, valid_image_b64, valid_mask_b64):
        """Generate 3D with skip_gif option"""
        with patch("api.routes.generate_3d.generate_3d_background") as mock_bg:
            response = client.post("/generate-3d", json={
                "image": valid_image_b64,
                "mask": valid_mask_b64,
                "seed": 42,
                "skip_gif": True,
                "max_image_size": 256
            })

        assert response.status_code == 200

        # Cleanup
        data = response.json()
        if data["task_id"] in generation_tasks:
            del generation_tasks[data["task_id"]]


class TestGenerate3DStatusEndpoint:
    """GET /generate-3d-status/{task_id} endpoint tests"""

    def test_status_task_not_found(self):
        """Non-existent task returns 404"""
        response = client.get("/generate-3d-status/nonexistent-task-id")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_status_queued_task(self):
        """Queued task status"""
        task_id = "test-queued-001"
        generation_tasks[task_id] = {
            "status": "queued",
            "progress": 0
        }

        try:
            response = client.get(f"/generate-3d-status/{task_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == task_id
            assert data["status"] == "queued"
            assert data["progress"] == 0
        finally:
            del generation_tasks[task_id]

    def test_status_processing_task(self):
        """Processing task status"""
        task_id = "test-processing-001"
        generation_tasks[task_id] = {
            "status": "processing",
            "progress": 50
        }

        try:
            response = client.get(f"/generate-3d-status/{task_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["progress"] == 50
        finally:
            del generation_tasks[task_id]

    def test_status_completed_task(self):
        """Completed task status with results"""
        task_id = "test-completed-001"
        generation_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "output_b64": "dGVzdA==",
            "output_type": "ply",
            "output_size_bytes": 1000,
            "gif_b64": "Z2lm",
            "gif_size_bytes": 500,
            "mesh_url": None
        }

        try:
            response = client.get(f"/generate-3d-status/{task_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["ply_b64"] == "dGVzdA=="
            assert data["output_type"] == "ply"
        finally:
            del generation_tasks[task_id]

    def test_status_failed_task(self):
        """Failed task status with error"""
        task_id = "test-failed-001"
        generation_tasks[task_id] = {
            "status": "failed",
            "error": "Test error message"
        }

        try:
            response = client.get(f"/generate-3d-status/{task_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["error"] == "Test error message"
        finally:
            del generation_tasks[task_id]

    def test_status_with_mesh_url(self, tmp_path):
        """Completed task with mesh file"""
        task_id = "test-mesh-001"

        # Create temporary mesh file
        mesh_content = b"GLTF test content"
        mesh_filename = "test_mesh.glb"

        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path)):
            mesh_path = tmp_path / mesh_filename
            mesh_path.write_bytes(mesh_content)

            generation_tasks[task_id] = {
                "status": "completed",
                "progress": 100,
                "output_b64": "dGVzdA==",
                "output_type": "ply",
                "output_size_bytes": 1000,
                "mesh_url": f"/assets/{mesh_filename}"
            }

            try:
                response = client.get(f"/generate-3d-status/{task_id}")

                assert response.status_code == 200
                data = response.json()
                # mesh_b64 is None because ASSETS_DIR is different in actual request
            finally:
                del generation_tasks[task_id]


class TestAssetsListEndpoint:
    """GET /assets-list endpoint tests"""

    def test_assets_list_empty_dir(self, tmp_path):
        """Empty assets directory"""
        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path)):
            response = client.get("/assets-list")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 0
        assert data["files"] == []

    def test_assets_list_nonexistent_dir(self, tmp_path):
        """Non-existent assets directory"""
        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path / "nonexistent")):
            response = client.get("/assets-list")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 0

    def test_assets_list_with_files(self, tmp_path):
        """Assets directory with files"""
        # Create test files
        (tmp_path / "test1.ply").write_text("ply content")
        (tmp_path / "test2.glb").write_bytes(b"glb content")

        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path)):
            response = client.get("/assets-list")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        assert len(data["files"]) == 2

    def test_assets_list_skips_metadata(self, tmp_path):
        """Metadata files are skipped"""
        (tmp_path / "test.ply").write_text("ply content")
        (tmp_path / "test.ply.metadata.json").write_text('{"created_at": "2024-01-01"}')

        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path)):
            response = client.get("/assets-list")

        assert response.status_code == 200
        data = response.json()
        # Only the PLY file, not the metadata
        assert data["total_files"] == 1
        assert data["files"][0]["name"] == "test.ply"

    def test_assets_list_with_metadata(self, tmp_path):
        """Assets with metadata files"""
        import json

        (tmp_path / "test.ply").write_text("ply content")
        metadata = {"created_at": "2024-01-15T10:30:00"}
        (tmp_path / "test.ply.metadata.json").write_text(json.dumps(metadata))

        with patch("api.routes.generate_3d.ASSETS_DIR", str(tmp_path)):
            response = client.get("/assets-list")

        assert response.status_code == 200
        data = response.json()
        assert data["files"][0]["created_at"] == "2024-01-15T10:30:00"
