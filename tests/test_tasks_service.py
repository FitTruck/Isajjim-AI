"""
Tests for api/services/tasks.py

Background task service unit tests:
- generate_3d_background function
- Task status storage
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

from api.services.tasks import (
    generate_3d_background,
    generation_tasks
)


class TestGenerationTasksStorage:
    """generation_tasks dictionary tests"""

    def test_generation_tasks_is_dict(self):
        """generation_tasks is a dictionary"""
        assert isinstance(generation_tasks, dict)

    def test_can_add_task(self):
        """Can add task to generation_tasks"""
        task_id = "test-task-001"
        generation_tasks[task_id] = {"status": "queued"}
        assert task_id in generation_tasks
        assert generation_tasks[task_id]["status"] == "queued"
        # Cleanup
        del generation_tasks[task_id]

    def test_can_update_task(self):
        """Can update task status"""
        task_id = "test-task-002"
        generation_tasks[task_id] = {"status": "queued"}
        generation_tasks[task_id]["status"] = "processing"
        assert generation_tasks[task_id]["status"] == "processing"
        # Cleanup
        del generation_tasks[task_id]


class TestGenerate3DBackground:
    """generate_3d_background function tests"""

    @pytest.fixture
    def temp_image_files(self, tmp_path):
        """Create temporary image and mask files"""
        # Create test image
        img = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / "test_image.png"
        img.save(str(image_path))

        # Create test mask
        mask = Image.new('L', (100, 100), color=255)
        mask_path = tmp_path / "test_mask.png"
        mask.save(str(mask_path))

        return str(image_path), str(mask_path)

    def test_subprocess_failure_sets_failed_status(self, temp_image_files):
        """Subprocess failure sets status to failed"""
        image_path, mask_path = temp_image_files
        task_id = "test-fail-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        # Mock subprocess to fail
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Subprocess error"

        with patch("api.services.tasks.subprocess.run", return_value=mock_result):
            generate_3d_background(task_id, image_path, mask_path, seed=42)

        assert generation_tasks[task_id]["status"] == "failed"
        assert "error" in generation_tasks[task_id]

        # Cleanup
        del generation_tasks[task_id]

    def test_timeout_sets_failed_status(self, temp_image_files):
        """Subprocess timeout sets status to failed"""
        import subprocess
        image_path, mask_path = temp_image_files
        task_id = "test-timeout-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        with patch("api.services.tasks.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=600)):
            generate_3d_background(task_id, image_path, mask_path, seed=42)

        assert generation_tasks[task_id]["status"] == "failed"
        assert "timed out" in generation_tasks[task_id]["error"]

        # Cleanup
        del generation_tasks[task_id]

    def test_exception_sets_failed_status(self, temp_image_files):
        """General exception sets status to failed"""
        image_path, mask_path = temp_image_files
        task_id = "test-exception-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        with patch("api.services.tasks.subprocess.run", side_effect=Exception("Unexpected error")):
            generate_3d_background(task_id, image_path, mask_path, seed=42)

        assert generation_tasks[task_id]["status"] == "failed"
        assert "Unexpected error" in generation_tasks[task_id]["error"]

        # Cleanup
        del generation_tasks[task_id]

    def test_successful_completion_with_ply(self, tmp_path):
        """Successful completion with PLY output"""
        # Create test files that won't be cleaned up during test
        img = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / "test_image_success.png"
        img.save(str(image_path))

        mask = Image.new('L', (100, 100), color=255)
        mask_path = tmp_path / "test_mask_success.png"
        mask.save(str(mask_path))

        task_id = "test-success-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        # Create a temporary PLY file that will be "generated"
        ply_content = b"""ply
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

        # Mock subprocess to succeed and create PLY
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Processing complete"
        mock_result.stderr = ""

        def mock_subprocess_run(cmd, **kwargs):
            # Create the PLY file that would be output
            # cmd format: [python, script, image, mask, seed, ply_path, assets_dir, ...]
            if len(cmd) > 5:
                ply_path = cmd[5]  # PLY path is 6th argument (index 5)
                with open(ply_path, 'wb') as f:
                    f.write(ply_content)
            return mock_result

        with patch("api.services.tasks.subprocess.run", side_effect=mock_subprocess_run):
            generate_3d_background(task_id, str(image_path), str(mask_path), seed=42)

        assert generation_tasks[task_id]["status"] == "completed"
        assert generation_tasks[task_id]["ply_b64"] is not None
        assert generation_tasks[task_id]["output_type"] == "ply"

        # Cleanup
        del generation_tasks[task_id]

    def test_no_output_sets_failed(self, temp_image_files):
        """No output files sets status to failed"""
        image_path, mask_path = temp_image_files
        task_id = "test-no-output-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        # Mock subprocess to succeed but not create any files
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Processing complete"
        mock_result.stderr = ""

        with patch("api.services.tasks.subprocess.run", return_value=mock_result):
            generate_3d_background(task_id, image_path, mask_path, seed=42)

        assert generation_tasks[task_id]["status"] == "failed"
        assert "Neither GIF nor PLY" in generation_tasks[task_id]["error"]

        # Cleanup
        del generation_tasks[task_id]

    def test_extracts_gif_data(self, temp_image_files):
        """Extracts GIF data from subprocess output"""
        image_path, mask_path = temp_image_files
        task_id = "test-gif-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        # Mock subprocess output with GIF data
        gif_b64 = "dGVzdF9naWZfZGF0YQ=="  # base64 for "test_gif_data"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"GIF_DATA_START{gif_b64}GIF_DATA_END"
        mock_result.stderr = ""

        with patch("api.services.tasks.subprocess.run", return_value=mock_result):
            generate_3d_background(task_id, image_path, mask_path, seed=42)

        assert generation_tasks[task_id]["status"] == "completed"
        assert generation_tasks[task_id]["gif_b64"] == gif_b64
        assert generation_tasks[task_id]["output_type"] == "gif"

        # Cleanup
        del generation_tasks[task_id]

    def test_extracts_mesh_url(self, tmp_path):
        """Extracts mesh URL from subprocess output"""
        # Create test files
        img = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / "test_image_mesh.png"
        img.save(str(image_path))

        mask = Image.new('L', (100, 100), color=255)
        mask_path = tmp_path / "test_mask_mesh.png"
        mask.save(str(mask_path))

        task_id = "test-mesh-url-001"

        # Initialize task
        generation_tasks[task_id] = {"status": "queued"}

        # Create PLY for successful completion
        ply_content = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"

        # Mock subprocess output with mesh URL
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "MESH_URL_START/assets/mesh.glbMESH_URL_END\nPLY_URL_START/assets/test.plyPLY_URL_END"
        mock_result.stderr = ""

        def mock_subprocess_run(cmd, **kwargs):
            # cmd format: [python, script, image, mask, seed, ply_path, assets_dir, ...]
            if len(cmd) > 5:
                ply_path = cmd[5]  # PLY path is 6th argument (index 5)
                with open(ply_path, 'wb') as f:
                    f.write(ply_content)
            return mock_result

        with patch("api.services.tasks.subprocess.run", side_effect=mock_subprocess_run):
            generate_3d_background(task_id, str(image_path), str(mask_path), seed=42)

        assert generation_tasks[task_id]["status"] == "completed"
        assert generation_tasks[task_id]["mesh_url"] == "/assets/mesh.glb"
        assert generation_tasks[task_id]["ply_url"] == "/assets/test.ply"

        # Cleanup
        del generation_tasks[task_id]

    def test_image_resize_large_image(self, tmp_path):
        """Large images are resized - verify via logging/status"""
        # Create large test image
        large_img = Image.new('RGB', (2000, 2000), color='red')
        image_path = tmp_path / "large_image.png"
        large_img.save(str(image_path))

        # Create matching mask
        mask = Image.new('L', (2000, 2000), color=255)
        mask_path = tmp_path / "large_mask.png"
        mask.save(str(mask_path))

        task_id = "test-resize-001"
        generation_tasks[task_id] = {"status": "queued"}

        # Track if resize happened via captured print output
        resize_happened = []

        # Mock subprocess to fail (we're testing resize, not full workflow)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "test"

        original_print = print

        def capture_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            if "Resized image" in msg:
                resize_happened.append(True)
            original_print(*args, **kwargs)

        with patch("api.services.tasks.subprocess.run", return_value=mock_result), \
             patch("builtins.print", side_effect=capture_print):
            generate_3d_background(
                task_id, str(image_path), str(mask_path),
                seed=42, max_image_size=512
            )

        # Verify resize was attempted (file may be cleaned up, but resize log should exist)
        assert len(resize_happened) > 0, "Image should have been resized"

        # Cleanup
        del generation_tasks[task_id]

    def test_skip_gif_flag_passed(self, temp_image_files):
        """skip_gif flag is passed to subprocess"""
        image_path, mask_path = temp_image_files
        task_id = "test-skip-gif-001"

        generation_tasks[task_id] = {"status": "queued"}

        captured_cmd = []

        def capture_cmd(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "test"
            return mock_result

        with patch("api.services.tasks.subprocess.run", side_effect=capture_cmd):
            generate_3d_background(task_id, image_path, mask_path, seed=42, skip_gif=True)

        assert "--skip-gif" in captured_cmd

        # Cleanup
        del generation_tasks[task_id]
