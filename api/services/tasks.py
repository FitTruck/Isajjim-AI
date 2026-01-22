"""
Background Task Service

3D generation 등 비동기 작업 관리
"""

import os
import sys
import base64
import subprocess
from typing import Dict

from api.config import ASSETS_DIR

# Task storage for async 3D generation
generation_tasks: Dict[str, Dict] = {}


def generate_3d_background(
    task_id: str,
    image_temp_path: str,
    mask_temp_path: str,
    seed: int,
    skip_gif: bool = True,
    max_image_size: int = 512
):
    """
    Background task for 3D generation.
    Updates generation_tasks dict with status and results.
    """
    ply_temp_path = None
    gif_temp_path = None

    try:
        generation_tasks[task_id]["status"] = "processing"

        # Create temp file for output PLY
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_temp_path = tmp.name
            gif_temp_path = ply_temp_path.replace(".ply", ".gif")

        # Get the subprocess script path (project root is 3 levels up from this file)
        # This file: api/services/tasks.py -> api/services -> api -> project_root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess_script = os.path.join(project_root, "ai", "subprocess", "generate_3d_worker.py")

        print(f"[Task {task_id}] Running 3D generation in subprocess...")
        print(f"[Task {task_id}] Options: skip_gif={skip_gif}, max_image_size={max_image_size}")

        # Resize image and mask if needed for faster processing
        if max_image_size > 0:
            try:
                from PIL import Image

                # Resize image
                img = Image.open(image_temp_path)
                w, h = img.size
                if max(w, h) > max_image_size:
                    scale = max_image_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    img.save(image_temp_path)
                    print(f"[Task {task_id}] Resized image: {w}x{h} -> {new_w}x{new_h}")

                # Resize mask to match
                mask = Image.open(mask_temp_path)
                if mask.size != img.size:
                    mask = mask.resize(img.size, Image.NEAREST)
                    mask.save(mask_temp_path)
                    print(f"[Task {task_id}] Resized mask to match: {img.size}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not resize image: {e}")

        # Run subprocess with sam3d-objects conda environment
        sam3d_python = os.path.expanduser("~/miniconda3/envs/sam3d-objects/bin/python")
        python_executable = sam3d_python if os.path.exists(sam3d_python) else sys.executable

        # Build command with optional skip-gif flag
        cmd = [
            python_executable,
            subprocess_script,
            image_temp_path,
            mask_temp_path,
            str(seed),
            ply_temp_path,
            ASSETS_DIR,
        ]
        if skip_gif:
            cmd.append("--skip-gif")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Print subprocess output for debugging
        if result.stdout:
            print(f"[Task {task_id}][Subprocess stdout]:\n{result.stdout}")
        if result.stderr:
            print(f"[Task {task_id}][Subprocess stderr]:\n{result.stderr}")

        # Check if subprocess succeeded
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"[Task {task_id}] Subprocess failed with return code {result.returncode}")
            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id]["error"] = error_msg
            return

        # Extract GIF data from subprocess output
        gif_b64 = None
        if "GIF_DATA_START" in result.stdout and "GIF_DATA_END" in result.stdout:
            try:
                start_idx = result.stdout.find("GIF_DATA_START") + len("GIF_DATA_START")
                end_idx = result.stdout.find("GIF_DATA_END")
                gif_b64 = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] Extracted GIF: {len(gif_b64)} chars (base64)")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract GIF data: {e}")

        # Extract mesh URL from subprocess output
        mesh_url = None
        if "MESH_URL_START" in result.stdout and "MESH_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("MESH_URL_START") + len("MESH_URL_START")
                end_idx = result.stdout.find("MESH_URL_END")
                mesh_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] Extracted mesh URL: {mesh_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract mesh URL: {e}")

        # Extract PLY URL from subprocess output
        ply_url = None
        if "PLY_URL_START" in result.stdout and "PLY_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("PLY_URL_START") + len("PLY_URL_START")
                end_idx = result.stdout.find("PLY_URL_END")
                ply_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] Extracted PLY URL: {ply_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract PLY URL: {e}")

        # Read PLY as primary output
        ply_b64 = None
        ply_size_bytes = None

        if os.path.exists(ply_temp_path):
            print(f"[Task {task_id}] Reading PLY from {ply_temp_path}")
            with open(ply_temp_path, "rb") as f:
                ply_bytes = f.read()

            # Validate PLY header
            try:
                header_text = ply_bytes[:min(50000, len(ply_bytes))].decode("utf-8", errors="ignore")
                if "end_header" not in header_text:
                    print(f"[Task {task_id}] WARNING: PLY missing 'end_header' in first 50KB")
                else:
                    print(f"[Task {task_id}] PLY header valid (ASCII format)")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not validate PLY header: {e}")

            ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
            ply_size_bytes = len(ply_bytes)
            print(f"[Task {task_id}] PLY loaded: {ply_size_bytes} bytes")

        # Determine primary output
        gif_size_bytes = len(gif_b64) if gif_b64 else None
        output_b64 = ply_b64 if ply_b64 else gif_b64
        output_type = "ply" if ply_b64 else "gif"
        output_size_bytes = ply_size_bytes if ply_b64 else gif_size_bytes

        if output_b64:
            print(f"[Task {task_id}] 3D generation successful ({output_type}): {output_size_bytes} bytes")
        else:
            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id]["error"] = "Neither GIF nor PLY file was generated"
            return

        generation_tasks[task_id]["status"] = "completed"
        generation_tasks[task_id]["output_b64"] = output_b64
        generation_tasks[task_id]["output_type"] = output_type
        generation_tasks[task_id]["output_size_bytes"] = output_size_bytes
        generation_tasks[task_id]["ply_b64"] = ply_b64
        generation_tasks[task_id]["ply_size_bytes"] = ply_size_bytes
        generation_tasks[task_id]["ply_url"] = ply_url
        generation_tasks[task_id]["gif_b64"] = gif_b64
        generation_tasks[task_id]["gif_size_bytes"] = gif_size_bytes
        generation_tasks[task_id]["mesh_url"] = mesh_url
        generation_tasks[task_id]["progress"] = 100

    except subprocess.TimeoutExpired:
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id]["error"] = "3D generation timed out (exceeded 10 minutes)"
    except Exception as e:
        print(f"[Task {task_id}] Error in 3D generation: {e}")
        import traceback
        traceback.print_exc()
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary files
        for path in [image_temp_path, mask_temp_path, ply_temp_path, gif_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"[Task {task_id}] Cleaned up temp file: {path}")
                except Exception:
                    pass
