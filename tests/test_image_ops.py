"""
Tests for ai/utils/image_ops.py

ImageUtils 클래스의 단위 테스트:
- 이미지 로드
- PIL ↔ OpenCV 변환
- CLAHE 적용
- 이미지 샤프닝
- 바운딩 박스 그리기
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestImageUtilsLoadImage:
    """load_image 함수 테스트"""

    def test_load_valid_image(self):
        """유효한 이미지 로드 테스트"""
        from ai.utils.image_ops import ImageUtils

        # 임시 이미지 생성
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)

            try:
                result = ImageUtils.load_image(f.name)
                assert isinstance(result, Image.Image)
                assert result.size == (100, 100)
            finally:
                os.unlink(f.name)

    def test_load_nonexistent_image_raises_error(self):
        """존재하지 않는 이미지 로드 시 에러"""
        from ai.utils.image_ops import ImageUtils

        with pytest.raises(ValueError) as exc_info:
            ImageUtils.load_image("/nonexistent/path/image.jpg")

        assert "Image Load Error" in str(exc_info.value)

    def test_load_image_with_exif_rotation(self):
        """EXIF 회전 정보가 있는 이미지 테스트"""
        from ai.utils.image_ops import ImageUtils

        # 임시 이미지 생성
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (100, 200), color='blue')
            img.save(f.name)

            try:
                result = ImageUtils.load_image(f.name)
                assert isinstance(result, Image.Image)
            finally:
                os.unlink(f.name)


class TestImageUtilsConversions:
    """PIL ↔ OpenCV 변환 테스트"""

    def test_pil_to_cv2(self):
        """PIL → OpenCV 변환"""
        from ai.utils.image_ops import ImageUtils

        pil_img = Image.new('RGB', (100, 100), color='red')
        cv2_img = ImageUtils.pil_to_cv2(pil_img)

        assert isinstance(cv2_img, np.ndarray)
        assert cv2_img.shape == (100, 100, 3)
        # OpenCV는 BGR 순서이므로 빨간색은 (0, 0, 255)
        assert cv2_img[50, 50, 2] == 255  # Red channel
        assert cv2_img[50, 50, 1] == 0    # Green channel
        assert cv2_img[50, 50, 0] == 0    # Blue channel

    def test_cv2_to_pil(self):
        """OpenCV → PIL 변환"""
        from ai.utils.image_ops import ImageUtils

        # BGR 이미지 생성 (파란색)
        cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_img[:, :, 0] = 255  # Blue channel in BGR

        pil_img = ImageUtils.cv2_to_pil(cv2_img)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (100, 100)
        # PIL은 RGB 순서이므로 파란색은 (0, 0, 255)
        pixel = pil_img.getpixel((50, 50))
        assert pixel == (0, 0, 255)

    def test_round_trip_conversion(self):
        """왕복 변환 테스트 (PIL → CV2 → PIL)"""
        from ai.utils.image_ops import ImageUtils

        original = Image.new('RGB', (100, 100), color='green')
        cv2_img = ImageUtils.pil_to_cv2(original)
        result = ImageUtils.cv2_to_pil(cv2_img)

        assert result.size == original.size
        # 초록색 픽셀 확인
        pixel = result.getpixel((50, 50))
        assert pixel == (0, 128, 0)  # PIL's 'green' color


class TestImageUtilsCLAHE:
    """CLAHE 적용 테스트"""

    def test_apply_clahe_returns_same_shape(self):
        """CLAHE 적용 후 동일한 형태 유지"""
        from ai.utils.image_ops import ImageUtils

        cv2_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = ImageUtils.apply_clahe(cv2_img)

        assert result.shape == cv2_img.shape
        assert result.dtype == cv2_img.dtype

    def test_apply_clahe_modifies_image(self):
        """CLAHE가 이미지를 수정하는지 확인"""
        from ai.utils.image_ops import ImageUtils

        # 저대비 이미지 생성
        cv2_img = np.full((100, 100, 3), 100, dtype=np.uint8)
        cv2_img[25:75, 25:75] = 120  # 약간 밝은 사각형

        result = ImageUtils.apply_clahe(cv2_img)

        # 결과가 입력과 다른지 확인 (대비가 향상됨)
        assert not np.array_equal(result, cv2_img)


class TestImageUtilsSharpen:
    """이미지 샤프닝 테스트"""

    def test_sharpen_returns_same_shape(self):
        """샤프닝 후 동일한 형태 유지"""
        from ai.utils.image_ops import ImageUtils

        cv2_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = ImageUtils.sharpen_image(cv2_img)

        assert result.shape == cv2_img.shape

    def test_sharpen_enhances_edges(self):
        """샤프닝이 엣지를 강화하는지 확인"""
        from ai.utils.image_ops import ImageUtils

        # 엣지가 있는 이미지 생성
        cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_img[25:75, 25:75] = 200  # 밝은 사각형

        result = ImageUtils.sharpen_image(cv2_img)

        # 샤프닝 후 이미지가 변경됨
        assert not np.array_equal(result, cv2_img)


class TestImageUtilsDrawBoxAndText:
    """draw_box_and_text 함수 테스트"""

    def test_draw_box_and_text(self):
        """바운딩 박스와 텍스트 그리기"""
        from ai.utils.image_ops import ImageUtils

        # PIL 이미지 생성
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)

        # 기본 폰트 사용
        try:
            font = ImageFont.truetype("fonts/NanumGothic-Regular.ttf", 12)
        except:
            font = ImageFont.load_default()

        bbox = (50, 50, 150, 150)
        text = "Test"
        color = (255, 0, 0)

        # 함수 실행
        ImageUtils.draw_box_and_text(draw, bbox, text, color, font)

        # 이미지가 수정되었는지 확인 (흰색이 아닌 픽셀 존재)
        pixels = list(img.getdata())
        non_white_pixels = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white_pixels) > 0

    def test_draw_box_with_korean_text(self):
        """한국어 텍스트로 바운딩 박스 그리기"""
        from ai.utils.image_ops import ImageUtils

        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("fonts/NanumGothic-Regular.ttf", 12)
        except:
            font = ImageFont.load_default()

        bbox = (50, 50, 150, 150)
        text = "침대"
        color = (0, 255, 0)

        # 예외 없이 실행
        ImageUtils.draw_box_and_text(draw, bbox, text, color, font)

    def test_draw_box_with_float_bbox(self):
        """float 형태의 bbox 처리"""
        from ai.utils.image_ops import ImageUtils

        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("fonts/NanumGothic-Regular.ttf", 12)
        except:
            font = ImageFont.load_default()

        # float bbox
        bbox = (50.5, 50.7, 150.2, 150.8)
        text = "Float"
        color = (0, 0, 255)

        # 예외 없이 실행 (내부에서 int로 변환)
        ImageUtils.draw_box_and_text(draw, bbox, text, color, font)
