"""
Additional tests for ai/processors/6_SAM3D_convert.py

SAM3DConverter 추가 테스트:
- convert 메서드 (subprocess 실행)
- convert_from_base64 메서드
- 타임아웃 및 에러 처리
"""

import os
import base64
from unittest.mock import patch, MagicMock
from PIL import Image

from ai.processors import SAM3DConverter, SAM3DResult


class TestSAM3DConverterConvert:
    """convert 메서드 테스트"""

    def test_convert_subprocess_success(self, tmp_path):
        """subprocess 성공 시나리오"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 테스트 이미지/마스크 생성
        image_path = tmp_path / "test_image.png"
        mask_path = tmp_path / "test_mask.png"
        Image.new('RGB', (100, 100), 'red').save(image_path)
        Image.new('L', (100, 100), 255).save(mask_path)

        # subprocess 모킹
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Processing complete..."
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result):
            # PLY 파일 생성 모킹
            with patch.object(converter, '_parse_result') as mock_parse:
                mock_parse.return_value = SAM3DResult(success=True, ply_b64="test")
                converter.convert(str(image_path), str(mask_path))

                assert mock_parse.called

    def test_convert_subprocess_failure(self, tmp_path):
        """subprocess 실패 시나리오"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 테스트 이미지/마스크 생성
        image_path = tmp_path / "test_image.png"
        mask_path = tmp_path / "test_mask.png"
        Image.new('RGB', (100, 100), 'red').save(image_path)
        Image.new('L', (100, 100), 255).save(mask_path)

        # subprocess 실패 모킹
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "CUDA error"

        with patch('subprocess.run', return_value=mock_result):
            result = converter.convert(str(image_path), str(mask_path))

            assert result.success is False
            assert "CUDA error" in result.error

    def test_convert_timeout(self, tmp_path):
        """subprocess 타임아웃"""
        import subprocess as sp

        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 테스트 이미지/마스크 생성
        image_path = tmp_path / "test_image.png"
        mask_path = tmp_path / "test_mask.png"
        Image.new('RGB', (100, 100), 'red').save(image_path)
        Image.new('L', (100, 100), 255).save(mask_path)

        # 타임아웃 모킹
        with patch('subprocess.run', side_effect=sp.TimeoutExpired(cmd="test", timeout=1)):
            result = converter.convert(str(image_path), str(mask_path), timeout=1)

            assert result.success is False
            assert "timed out" in result.error

    def test_convert_exception(self, tmp_path):
        """subprocess 예외 발생"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 존재하지 않는 파일 사용 - subprocess 실행 전 예외 발생 가능
        with patch('subprocess.run', side_effect=Exception("Unexpected error")):
            result = converter.convert("nonexistent.png", "nonexistent_mask.png")

            assert result.success is False
            assert "Unexpected error" in result.error


class TestSAM3DConverterConvertFromBase64:
    """convert_from_base64 메서드 테스트"""

    def test_convert_from_base64_success(self, tmp_path):
        """base64에서 변환 성공"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 테스트 이미지/마스크 base64 생성
        img = Image.new('RGB', (100, 100), 'blue')
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        mask = Image.new('L', (100, 100), 255)
        buffer = io.BytesIO()
        mask.save(buffer, format='PNG')
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # convert 메서드 모킹
        with patch.object(converter, 'convert') as mock_convert:
            mock_convert.return_value = SAM3DResult(success=True, ply_b64="test")
            converter.convert_from_base64(image_b64, mask_b64)

            assert mock_convert.called
            # convert가 임시 파일 경로로 호출되었는지 확인
            call_args = mock_convert.call_args[0]
            assert call_args[0].endswith('.png')
            assert call_args[1].endswith('.png')

    def test_convert_from_base64_cleanup(self, tmp_path):
        """base64 변환 후 임시 파일 정리"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # 테스트 이미지/마스크 base64 생성
        img = Image.new('RGB', (50, 50), 'green')
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        created_files = []

        def mock_convert(image_path, mask_path, seed, timeout):
            # 생성된 임시 파일 경로 저장
            created_files.append(image_path)
            created_files.append(mask_path)
            return SAM3DResult(success=True)

        with patch.object(converter, 'convert', side_effect=mock_convert):
            converter.convert_from_base64(image_b64, mask_b64)

        # 임시 파일이 정리되었는지 확인
        for path in created_files:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"


class TestSAM3DConverterParseResultExtended:
    """_parse_result 메서드 확장 테스트"""

    def test_parse_result_with_all_data(self, tmp_path):
        """모든 데이터가 있는 출력 파싱"""
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

        # GLB 파일 생성
        mesh_path = tmp_path / "mesh_test.glb"
        mesh_path.write_bytes(b"GLTF binary content")

        stdout = """
Processing image...
GIF_DATA_STARTdGVzdF9naWZfZGF0YQ==GIF_DATA_END
MESH_URL_START/assets/mesh_test.glbMESH_URL_END
PLY_URL_START/assets/test.plyPLY_URL_END
Done!
"""

        result = converter._parse_result(stdout, str(ply_path))

        assert result.success is True
        assert result.gif_b64 == "dGVzdF9naWZfZGF0YQ=="
        assert result.mesh_url == "/assets/mesh_test.glb"
        assert result.mesh_format == "glb"
        assert result.ply_url == "/assets/test.ply"
        assert result.ply_b64 is not None
        assert result.mesh_b64 is not None  # GLB 파일이 존재하므로 인코딩됨

    def test_parse_result_ply_mesh(self, tmp_path):
        """PLY 형식 mesh 파싱"""
        converter = SAM3DConverter(assets_dir=str(tmp_path))

        # PLY 파일 생성
        ply_path = tmp_path / "test.ply"
        ply_content = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
"""
        ply_path.write_text(ply_content)

        # 또 다른 PLY 메시 파일
        mesh_path = tmp_path / "mesh_test.ply"
        mesh_path.write_text(ply_content)

        stdout = """
MESH_URL_START/assets/mesh_test.plyMESH_URL_END
"""

        result = converter._parse_result(stdout, str(ply_path))

        assert result.success is True
        assert result.mesh_format == "ply"


class TestSAM3DConverterValidatePly:
    """_validate_ply 메서드 테스트"""

    def test_validate_ply_valid(self):
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

    def test_validate_ply_invalid(self):
        """유효하지 않은 PLY (end_header 없음)"""
        converter = SAM3DConverter()

        ply_content = b"not a ply file at all"

        result = converter._validate_ply(ply_content)
        assert result is False

    def test_validate_ply_large_header(self):
        """큰 헤더를 가진 PLY (50KB 이후에 end_header)"""
        converter = SAM3DConverter()

        # 50KB 이상의 헤더
        padding = b" " * 60000
        ply_content = b"ply\nformat ascii 1.0\n" + padding + b"\nend_header\n0 0 0\n"

        result = converter._validate_ply(ply_content)
        assert result is True

    def test_validate_ply_exception(self):
        """PLY 검증 중 예외"""
        converter = SAM3DConverter()

        # 디코딩 불가능한 바이트
        ply_content = bytes([0xff, 0xfe, 0x00, 0x01])

        # 예외가 발생해도 False 반환
        result = converter._validate_ply(ply_content)
        assert result is False


class TestSAM3DConverterEdgeCases:
    """엣지 케이스 테스트"""

    def test_assets_dir_creation(self, tmp_path):
        """assets 디렉토리 자동 생성"""
        new_assets_dir = tmp_path / "new_assets"
        assert not new_assets_dir.exists()

        SAM3DConverter(assets_dir=str(new_assets_dir))

        assert new_assets_dir.exists()

    def test_device_id_custom(self):
        """커스텀 device_id 설정"""
        converter = SAM3DConverter(device_id=2)
        assert converter.device_id == 2

    def test_device_id_default(self):
        """기본 device_id"""
        from ai.config import Config
        converter = SAM3DConverter()
        assert converter.device_id == Config.DEFAULT_GPU_ID
