#!/usr/bin/env python3
"""
SAM2 & SAM-3D API Test Script

API 서버가 실행 중인 상태에서 SAM2 마스크 생성과 SAM-3D 3D 변환을 테스트합니다.

Usage:
    python test_sam_api.py
"""

import sys
import os
import time
import base64
import json
import requests
from pathlib import Path
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE = "DeCl/imgs/test.jpg"
PROJECT_ROOT = Path(__file__).parent


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title: str):
    print(f"\n--- {title} ---")


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data))


def test_health():
    """Test health endpoint"""
    print_header("1. Testing Health Endpoint")

    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        data = resp.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Model: {data.get('model')}")
        print(f"  Device: {data.get('device')}")
        print(f"  Model Loaded: {data.get('model_loaded')}")

        if data.get('status') == 'healthy' and data.get('model_loaded'):
            print("\n[PASS] Health check passed")
            return True
        else:
            print("\n[FAIL] Server not ready")
            return False
    except Exception as e:
        print(f"\n[FAIL] Health check failed: {e}")
        return False


def test_sam2_segment(image_path: str):
    """Test SAM2 segmentation endpoint"""
    print_header("2. Testing SAM2 Segmentation")

    # Load test image
    print_section(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    print(f"  Image size: {img.size}")

    # Convert to base64
    img_b64 = image_to_base64(image_path)

    # Use center point of the sofa (from previous YOLO detection)
    # sofa bbox was [83, 481, 702, 816]
    center_x = (83 + 702) // 2  # ~392
    center_y = (481 + 816) // 2  # ~648

    print_section(f"Requesting segmentation at point ({center_x}, {center_y})")

    # Call /segment endpoint (uses x, y parameters)
    payload = {
        "image": img_b64,
        "x": float(center_x),
        "y": float(center_y)
    }

    start_time = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/segment",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            masks = data.get('masks', [])
            print(f"  Response time: {elapsed:.2f}s")
            print(f"  Masks returned: {len(masks)}")

            if masks:
                # Get best mask (highest score)
                best_mask = max(masks, key=lambda m: m.get('score', 0))
                print(f"  Best mask score: {best_mask.get('score', 0):.3f}")

                # Save mask image for verification
                mask_b64 = best_mask.get('mask', '')
                if mask_b64:
                    mask_img = base64_to_image(mask_b64)
                    mask_path = PROJECT_ROOT / "test_mask_output.png"
                    mask_img.save(mask_path)
                    print(f"  Mask saved to: {mask_path}")

                print("\n[PASS] SAM2 segmentation successful")
                return best_mask.get('mask', '')
            else:
                print("\n[WARN] No masks returned")
                return None
        else:
            print(f"  Error: {resp.status_code} - {resp.text[:200]}")
            print("\n[FAIL] SAM2 segmentation failed")
            return None

    except Exception as e:
        print(f"\n[FAIL] SAM2 request error: {e}")
        return None


def test_sam2_segment_binary(image_path: str):
    """Test SAM2 binary mask endpoint"""
    print_header("3. Testing SAM2 Binary Mask")

    img_b64 = image_to_base64(image_path)

    # Use center point of the sofa
    center_x = (83 + 702) // 2
    center_y = (481 + 816) // 2

    print_section(f"Requesting binary mask at point ({center_x}, {center_y})")

    # /segment-binary expects points as [{"x": float, "y": float}, ...]
    payload = {
        "image": img_b64,
        "points": [{"x": float(center_x), "y": float(center_y)}]
    }

    start_time = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/segment-binary",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            # Response uses 'mask' key, not 'masked_image'
            mask_b64 = data.get('mask', '')
            score = data.get('score', 0)

            print(f"  Response time: {elapsed:.2f}s")
            print(f"  Score: {score:.3f}")

            if mask_b64:
                # Save masked image
                masked_img = base64_to_image(mask_b64)
                masked_path = PROJECT_ROOT / "test_masked_output.png"
                masked_img.save(masked_path)
                print(f"  Masked image size: {masked_img.size}")
                print(f"  Masked image saved to: {masked_path}")

                print("\n[PASS] SAM2 binary mask successful")
                return mask_b64
            else:
                print("\n[WARN] No mask returned")
                return None
        else:
            print(f"  Error: {resp.status_code} - {resp.text[:200]}")
            print("\n[FAIL] SAM2 binary mask failed")
            return None

    except Exception as e:
        print(f"\n[FAIL] SAM2 binary mask error: {e}")
        return None


def test_sam3d_generation(image_path: str, mask_b64: str = None):
    """Test SAM-3D generation endpoint"""
    print_header("4. Testing SAM-3D Generation")

    if not mask_b64:
        print("  [INFO] No mask provided, generating one first...")
        mask_b64 = test_sam2_segment(image_path)
        if not mask_b64:
            print("\n[FAIL] Could not generate mask for SAM-3D test")
            return None

    img_b64 = image_to_base64(image_path)

    print_section("Submitting 3D generation request")

    payload = {
        "image": img_b64,
        "mask": mask_b64,
        "seed": 42
    }

    start_time = time.time()
    try:
        # Submit generation request
        resp = requests.post(
            f"{API_BASE_URL}/generate-3d",
            json=payload,
            timeout=30
        )

        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text[:200]}")
            print("\n[FAIL] SAM-3D submission failed")
            return None

        data = resp.json()
        task_id = data.get('task_id')
        print(f"  Task ID: {task_id}")

        if not task_id:
            print("\n[FAIL] No task_id returned")
            return None

        # Poll for completion
        print_section("Polling for completion (this may take 1-3 minutes)")
        max_polls = 60  # 5 minutes max
        poll_interval = 5

        for i in range(max_polls):
            time.sleep(poll_interval)
            elapsed = time.time() - start_time

            status_resp = requests.get(
                f"{API_BASE_URL}/generate-3d-status/{task_id}",
                timeout=10
            )

            if status_resp.status_code != 200:
                print(f"  Poll {i+1}: Error {status_resp.status_code}")
                continue

            status_data = status_resp.json()
            status = status_data.get('status', 'unknown')

            print(f"  [{elapsed:.0f}s] Status: {status}")

            if status == 'completed':
                print_section("Generation completed!")

                # Check outputs
                gif_url = status_data.get('gif_url')
                ply_url = status_data.get('ply_url')
                mesh_url = status_data.get('mesh_url')

                print(f"  GIF URL: {gif_url or 'N/A'}")
                print(f"  PLY URL: {ply_url or 'N/A'}")
                print(f"  Mesh URL: {mesh_url or 'N/A'}")

                # Download and save GIF if available
                if gif_url:
                    gif_data = status_data.get('gif_data')
                    if gif_data:
                        gif_bytes = base64.b64decode(gif_data)
                        gif_path = PROJECT_ROOT / "test_3d_output.gif"
                        with open(gif_path, 'wb') as f:
                            f.write(gif_bytes)
                        print(f"  GIF saved to: {gif_path}")

                print(f"\n  Total time: {elapsed:.1f}s")
                print("\n[PASS] SAM-3D generation successful")
                return status_data

            elif status == 'failed':
                error = status_data.get('error', 'Unknown error')
                print(f"  Error: {error}")
                print("\n[FAIL] SAM-3D generation failed")
                return None

        print("\n[FAIL] Timeout waiting for SAM-3D generation")
        return None

    except Exception as e:
        print(f"\n[FAIL] SAM-3D error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "=" * 60)
    print(" SAM2 & SAM-3D API Test")
    print("=" * 60)

    test_image = str(PROJECT_ROOT / TEST_IMAGE)

    if not Path(test_image).exists():
        print(f"[ERROR] Test image not found: {test_image}")
        return 1

    all_passed = True

    # Test 1: Health check
    if not test_health():
        print("\n[FATAL] Server not healthy. Cannot continue.")
        return 1

    # Test 2: SAM2 Segmentation
    mask_b64 = None
    try:
        mask_b64 = test_sam2_segment(test_image)
        if not mask_b64:
            all_passed = False
    except Exception as e:
        print(f"\n[FAIL] SAM2 segment error: {e}")
        all_passed = False

    # Test 3: SAM2 Binary Mask
    try:
        masked_image = test_sam2_segment_binary(test_image)
        if not masked_image:
            all_passed = False
    except Exception as e:
        print(f"\n[FAIL] SAM2 binary mask error: {e}")
        all_passed = False

    # Test 4: SAM-3D Generation (uses mask from test 2)
    try:
        result = test_sam3d_generation(test_image, mask_b64)
        if not result:
            all_passed = False
    except Exception as e:
        print(f"\n[FAIL] SAM-3D error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    print_header("API Test Summary")
    if all_passed:
        print("[SUCCESS] All API tests passed!")
        print("\nGenerated files:")
        print("  - test_mask_output.png (SAM2 mask)")
        print("  - test_masked_output.png (masked image)")
        print("  - test_3d_output.gif (3D rotation GIF)")
        return 0
    else:
        print("[PARTIAL] Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
