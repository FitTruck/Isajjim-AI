"""
Step 6: SAM-3D 3D Conversion

SAM-3D (Sam-3d-objects)를 사용하여 2D 이미지 + 마스크에서 3D 모델을 생성합니다.
- Gaussian Splat, PLY, GIF, GLB 파일 생성
- subprocess를 통한 GPU 격리 (spconv 충돌 방지)
"""

import os
import sys
import base64
import tempfile
import subprocess
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Config for GPU settings
from ai.config import Config


@dataclass
class SAM3DResult:
    """SAM-3D 변환 결과"""
    success: bool
    ply_b64: Optional[str] = None
    ply_size_bytes: Optional[int] = None
    ply_url: Optional[str] = None
    gif_b64: Optional[str] = None
    gif_size_bytes: Optional[int] = None
    mesh_url: Optional[str] = None
    mesh_b64: Optional[str] = None
    mesh_format: Optional[str] = None
    error: Optional[str] = None


class SAM3DConverter:
    """
    SAM-3D를 사용한 3D 변환기

    Usage:
        converter = SAM3DConverter(assets_dir="./assets", device_id=0)
        result = converter.convert(image_path, mask_path, seed=42)
    """

    def __init__(self, assets_dir: str = "./assets", device_id: Optional[int] = None):
        """
        Args:
            assets_dir: 생성된 에셋을 저장할 디렉토리
            device_id: GPU 디바이스 ID (None이면 기본값 사용)
        """
        self.assets_dir = assets_dir
        os.makedirs(assets_dir, exist_ok=True)

        # Multi-GPU 지원: 디바이스 설정
        self.device_id = device_id if device_id is not None else Config.DEFAULT_GPU_ID

        # subprocess 스크립트 경로
        self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.subprocess_script = os.path.join(
            self.script_dir, "ai", "subprocess", "generate_3d_worker.py"
        )

    def convert(
        self,
        image_path: str,
        mask_path: str,
        seed: int = 42,
        timeout: int = 600
    ) -> SAM3DResult:
        """
        3D 변환 실행

        Args:
            image_path: 입력 이미지 경로
            mask_path: 마스크 이미지 경로
            seed: 랜덤 시드
            timeout: 타임아웃 (초)

        Returns:
            SAM3DResult
        """
        ply_temp_path = None

        try:
            # 임시 PLY 파일 경로
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                ply_temp_path = tmp.name

            print(f"[SAM3DConverter] Running 3D generation on GPU {self.device_id}...")
            print(f"[SAM3DConverter] Script: {self.subprocess_script}")

            # GPU 격리를 위한 환경 변수 설정
            # CUDA_VISIBLE_DEVICES로 GPU를 제한하면 내부적으로 항상 device 0이 됨
            env = os.environ.copy()
            spconv_env = Config.get_spconv_env_vars(self.device_id)
            env.update(spconv_env)

            print(f"[SAM3DConverter] Environment: CUDA_VISIBLE_DEVICES={spconv_env['CUDA_VISIBLE_DEVICES']}")

            # subprocess 실행
            result = subprocess.run(
                [
                    sys.executable,
                    self.subprocess_script,
                    image_path,
                    mask_path,
                    str(seed),
                    ply_temp_path,
                    self.assets_dir
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            # 로그 출력
            if result.stdout:
                print(f"[SAM3DConverter][stdout]:\n{result.stdout}")
            if result.stderr:
                print(f"[SAM3DConverter][stderr]:\n{result.stderr}")

            # 실패 체크
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                return SAM3DResult(
                    success=False,
                    error=f"Subprocess failed with return code {result.returncode}: {error_msg}"
                )

            # 결과 파싱
            return self._parse_result(result.stdout, ply_temp_path)

        except subprocess.TimeoutExpired:
            return SAM3DResult(
                success=False,
                error=f"3D generation timed out (exceeded {timeout} seconds)"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SAM3DResult(success=False, error=str(e))
        finally:
            # 임시 파일 정리
            if ply_temp_path and os.path.exists(ply_temp_path):
                try:
                    os.unlink(ply_temp_path)
                except:
                    pass

    def convert_from_base64(
        self,
        image_b64: str,
        mask_b64: str,
        seed: int = 42,
        timeout: int = 600
    ) -> SAM3DResult:
        """
        base64 이미지에서 3D 변환

        Args:
            image_b64: base64 인코딩된 이미지
            mask_b64: base64 인코딩된 마스크
            seed: 랜덤 시드
            timeout: 타임아웃 (초)

        Returns:
            SAM3DResult
        """
        image_temp_path = None
        mask_temp_path = None

        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(base64.b64decode(image_b64))
                image_temp_path = tmp.name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(base64.b64decode(mask_b64))
                mask_temp_path = tmp.name

            return self.convert(image_temp_path, mask_temp_path, seed, timeout)

        finally:
            # 임시 파일 정리
            for path in [image_temp_path, mask_temp_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass

    def _parse_result(self, stdout: str, ply_temp_path: str) -> SAM3DResult:
        """subprocess 출력에서 결과 파싱"""
        gif_b64 = None
        mesh_url = None
        ply_url = None

        # GIF 데이터 추출
        if "GIF_DATA_START" in stdout and "GIF_DATA_END" in stdout:
            try:
                start_idx = stdout.find("GIF_DATA_START") + len("GIF_DATA_START")
                end_idx = stdout.find("GIF_DATA_END")
                gif_b64 = stdout[start_idx:end_idx].strip()
                print(f"[SAM3DConverter] Extracted GIF: {len(gif_b64)} chars (base64)")
            except Exception as e:
                print(f"[SAM3DConverter] Warning: Could not extract GIF data: {e}")

        # Mesh URL 추출
        if "MESH_URL_START" in stdout and "MESH_URL_END" in stdout:
            try:
                start_idx = stdout.find("MESH_URL_START") + len("MESH_URL_START")
                end_idx = stdout.find("MESH_URL_END")
                mesh_url = stdout[start_idx:end_idx].strip()
                print(f"[SAM3DConverter] Extracted mesh URL: {mesh_url}")
            except Exception as e:
                print(f"[SAM3DConverter] Warning: Could not extract mesh URL: {e}")

        # PLY URL 추출
        if "PLY_URL_START" in stdout and "PLY_URL_END" in stdout:
            try:
                start_idx = stdout.find("PLY_URL_START") + len("PLY_URL_START")
                end_idx = stdout.find("PLY_URL_END")
                ply_url = stdout[start_idx:end_idx].strip()
                print(f"[SAM3DConverter] Extracted PLY URL: {ply_url}")
            except Exception as e:
                print(f"[SAM3DConverter] Warning: Could not extract PLY URL: {e}")

        # PLY 파일 읽기
        ply_b64 = None
        ply_size_bytes = None

        if os.path.exists(ply_temp_path):
            print(f"[SAM3DConverter] Reading PLY from {ply_temp_path}")
            with open(ply_temp_path, "rb") as f:
                ply_bytes = f.read()

            # PLY 헤더 검증
            self._validate_ply(ply_bytes)

            ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
            ply_size_bytes = len(ply_bytes)
            print(f"[SAM3DConverter] PLY loaded: {ply_size_bytes} bytes")

        # 결과 없음 체크
        if not ply_b64 and not gif_b64:
            return SAM3DResult(
                success=False,
                error="Neither GIF nor PLY file was generated"
            )

        # Mesh base64 인코딩
        mesh_b64 = None
        mesh_format = None
        if mesh_url:
            mesh_filename = mesh_url.split("/")[-1]
            mesh_path = os.path.join(self.assets_dir, mesh_filename)

            if mesh_filename.endswith(".glb"):
                mesh_format = "glb"
            elif mesh_filename.endswith(".ply"):
                mesh_format = "ply"

            if os.path.exists(mesh_path):
                try:
                    with open(mesh_path, "rb") as f:
                        mesh_b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    print(f"[SAM3DConverter] Warning: Could not encode mesh: {e}")

        return SAM3DResult(
            success=True,
            ply_b64=ply_b64,
            ply_size_bytes=ply_size_bytes,
            ply_url=ply_url,
            gif_b64=gif_b64,
            gif_size_bytes=len(gif_b64) if gif_b64 else None,
            mesh_url=mesh_url,
            mesh_b64=mesh_b64,
            mesh_format=mesh_format
        )

    def _validate_ply(self, ply_bytes: bytes) -> bool:
        """PLY 헤더 검증"""
        try:
            header_text = ply_bytes[:min(50000, len(ply_bytes))].decode("utf-8", errors="ignore")
            if "end_header" not in header_text:
                print("[SAM3DConverter] WARNING: PLY missing 'end_header' in first 50KB")
                full_text = ply_bytes.decode("utf-8", errors="ignore")
                if "end_header" not in full_text:
                    print("[SAM3DConverter] ERROR: PLY file corrupted or not ASCII format")
                    return False
                else:
                    print("[SAM3DConverter] Found end_header after 50KB - file is valid")
            else:
                print("[SAM3DConverter] PLY header valid (ASCII format)")
            return True
        except Exception as e:
            print(f"[SAM3DConverter] Warning: Could not validate PLY header: {e}")
            return False
