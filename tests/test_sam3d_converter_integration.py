"""
Tests for SAM3DConverter with test_outputs files

test_outputs 디렉토리의 실제 원본 이미지와 마스크로 SAM3DConverter 테스트:
- SAM3DResult 데이터클래스
- 3D 변환 (subprocess 기반)
"""

import pytest
import base64
from pathlib import Path
from PIL import Image

from ai.processors import SAM3DConverter, SAM3DResult


# Test output directories
TEST_OUTPUTS_DIR = Path(__file__).parent.parent / "test_outputs"
BEDROOM_DIR = TEST_OUTPUTS_DIR / "bedroom-5772286_1920"
QA_BEDROOM_DIR = TEST_OUTPUTS_DIR / "qa" / "bedroom-5772286_1920"


class TestSAM3DResultDataclass:
    """SAM3DResult 데이터클래스 테스트"""

    def test_create_success_result(self):
        """성공 결과 생성"""
        result = SAM3DResult(
            success=True,
            ply_b64="test_ply_base64",
            ply_size_bytes=1000,
            ply_url="/assets/test.ply",
            gif_b64="test_gif_base64",
            gif_size_bytes=500,
            mesh_url="/assets/test.glb",
            mesh_b64="test_mesh_base64",
            mesh_format="glb"
        )

        assert result.success is True
        assert result.ply_b64 == "test_ply_base64"
        assert result.ply_size_bytes == 1000
        assert result.mesh_format == "glb"
        assert result.error is None

    def test_create_failure_result(self):
        """실패 결과 생성"""
        result = SAM3DResult(
            success=False,
            error="Test error message"
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.ply_b64 is None

    def test_default_values(self):
        """기본값 확인"""
        result = SAM3DResult(success=True)

        assert result.success is True
        assert result.ply_b64 is None
        assert result.ply_size_bytes is None
        assert result.ply_url is None
        assert result.gif_b64 is None
        assert result.mesh_url is None
        assert result.error is None


class TestSAM3DConverterInit:
    """SAM3DConverter 초기화 테스트"""

    def test_init_default(self):
        """기본 초기화"""
        converter = SAM3DConverter()

        assert converter.assets_dir == "./assets"
        assert converter.device_id == 0

    def test_init_custom_assets_dir(self, tmp_path):
        """사용자 정의 assets 디렉토리"""
        custom_dir = tmp_path / "custom_assets"
        converter = SAM3DConverter(assets_dir=str(custom_dir))

        assert converter.assets_dir == str(custom_dir)
        assert custom_dir.exists()

    def test_init_custom_device(self):
        """사용자 정의 GPU 디바이스"""
        converter = SAM3DConverter(device_id=1)
        assert converter.device_id == 1

    def test_subprocess_script_exists(self):
        """subprocess 스크립트 존재 확인"""
        converter = SAM3DConverter()
        assert Path(converter.subprocess_script).exists(), \
            f"Subprocess script not found: {converter.subprocess_script}"


class TestSAM3DConverterTestOutputs:
    """test_outputs 파일을 사용한 테스트"""

    def test_original_image_exists(self):
        """원본 이미지 존재 확인"""
        original = BEDROOM_DIR / "00_original.jpg"
        if not BEDROOM_DIR.exists():
            pytest.skip(f"Test data directory not found: {BEDROOM_DIR}")
        assert original.exists(), f"Original image not found: {original}"

    def test_mask_files_exist(self):
        """마스크 파일 존재 확인"""
        if not BEDROOM_DIR.exists():
            pytest.skip(f"Test data directory not found: {BEDROOM_DIR}")
        mask_files = list(BEDROOM_DIR.glob("mask_*.png"))
        print(f"Found {len(mask_files)} mask files")
        if len(mask_files) == 0:
            pytest.skip("No mask files found in test directory")

        for mask_file in mask_files[:5]:
            print(f"  - {mask_file.name}")

    def test_load_original_image(self):
        """원본 이미지 로드"""
        original = BEDROOM_DIR / "00_original.jpg"
        if not original.exists():
            pytest.skip("Original image not found")

        img = Image.open(original)
        print(f"Original image size: {img.size}")
        assert img.size[0] > 0 and img.size[1] > 0

    def test_load_mask_image(self):
        """마스크 이미지 로드"""
        mask_files = list(BEDROOM_DIR.glob("mask_*.png"))
        if not mask_files:
            pytest.skip("No mask files found")

        mask = Image.open(mask_files[0])
        print(f"Mask image size: {mask.size}")
        print(f"Mask mode: {mask.mode}")

    def test_image_to_base64(self):
        """이미지 → base64 변환"""
        original = BEDROOM_DIR / "00_original.jpg"
        if not original.exists():
            pytest.skip("Original image not found")

        with open(original, "rb") as f:
            image_bytes = f.read()

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert len(b64) > 0
        print(f"Image base64 length: {len(b64)}")

    def test_mask_to_base64(self):
        """마스크 → base64 변환"""
        mask_files = list(BEDROOM_DIR.glob("mask_*.png"))
        if not mask_files:
            pytest.skip("No mask files found")

        with open(mask_files[0], "rb") as f:
            mask_bytes = f.read()

        b64 = base64.b64encode(mask_bytes).decode("utf-8")
        assert len(b64) > 0
        print(f"Mask base64 length: {len(b64)}")


@pytest.mark.slow
class TestSAM3DConverterConvert:
    """SAM-3D 변환 테스트 (느림 - 실제 3D 생성)"""

    @pytest.fixture
    def converter(self, tmp_path):
        """SAM3DConverter 인스턴스"""
        return SAM3DConverter(assets_dir=str(tmp_path / "assets"), device_id=0)

    def test_convert_with_bed_mask(self, converter):
        """침대 마스크로 3D 변환"""
        original = BEDROOM_DIR / "00_original.jpg"
        bed_mask = BEDROOM_DIR / "mask_05_Bed.png"

        if not original.exists() or not bed_mask.exists():
            pytest.skip("Test files not found")

        print(f"Converting: {original.name} + {bed_mask.name}")

        result = converter.convert(
            image_path=str(original),
            mask_path=str(bed_mask),
            seed=42,
            timeout=300
        )

        print(f"Result: success={result.success}")
        if not result.success:
            print(f"Error: {result.error}")

        # 결과 확인 (성공 여부는 sam-3d-objects 설치 상태에 따라 다름)
        if result.success:
            print(f"PLY size: {result.ply_size_bytes} bytes")
            print(f"Mesh URL: {result.mesh_url}")
            assert result.ply_b64 is not None or result.gif_b64 is not None

    def test_convert_from_base64(self, converter):
        """base64에서 3D 변환"""
        original = BEDROOM_DIR / "00_original.jpg"
        bed_mask = BEDROOM_DIR / "mask_05_Bed.png"

        if not original.exists() or not bed_mask.exists():
            pytest.skip("Test files not found")

        # 파일을 base64로 인코딩
        with open(original, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        with open(bed_mask, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")

        result = converter.convert_from_base64(
            image_b64=image_b64,
            mask_b64=mask_b64,
            seed=42,
            timeout=300
        )

        print(f"Result: success={result.success}")
        if not result.success:
            print(f"Error: {result.error}")


class TestSAM3DConverterParseResult:
    """_parse_result 테스트"""

    def test_parse_result_with_gif(self, tmp_path):
        """GIF 데이터가 있는 출력 파싱"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # PLY 파일 생성
        ply_path = tmp_path / "test.ply"
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
        ply_path.write_text(ply_content)

        stdout = """
Processing image...
GIF_DATA_STARTdGVzdF9naWZfZGF0YQ==GIF_DATA_END
Done!
"""

        result = converter._parse_result(stdout, str(ply_path))

        assert result.success is True
        assert result.gif_b64 == "dGVzdF9naWZfZGF0YQ=="
        assert result.ply_b64 is not None

    def test_parse_result_with_mesh_url(self, tmp_path):
        """Mesh URL이 있는 출력 파싱"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 메시 파일 생성
        mesh_path = tmp_path / "mesh_test.glb"
        mesh_path.write_bytes(b"test mesh content")

        # PLY 파일 생성
        ply_path = tmp_path / "test.ply"
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
        ply_path.write_text(ply_content)

        stdout = f"""
Processing image...
MESH_URL_START/assets/mesh_test.glbMESH_URL_END
PLY_URL_START/assets/test.plyPLY_URL_END
Done!
"""

        result = converter._parse_result(stdout, str(ply_path))

        assert result.success is True
        assert result.mesh_url == "/assets/mesh_test.glb"
        assert result.mesh_format == "glb"
        assert result.mesh_b64 is not None

    def test_parse_result_no_output(self, tmp_path):
        """출력이 없는 경우"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # PLY 파일 없음
        ply_path = tmp_path / "nonexistent.ply"

        stdout = "Processing failed"

        result = converter._parse_result(stdout, str(ply_path))

        assert result.success is False
        assert "Neither GIF nor PLY" in result.error


class TestSAM3DConverterValidatePly:
    """_validate_ply 테스트"""

    def test_validate_valid_ply(self):
        """유효한 PLY 검증"""
        converter = SAM3DConverter()

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
        result = converter._validate_ply(ply_content)
        assert result is True

    def test_validate_invalid_ply(self):
        """유효하지 않은 PLY 검증"""
        converter = SAM3DConverter()

        ply_content = b"not a ply file"

        result = converter._validate_ply(ply_content)
        assert result is False
