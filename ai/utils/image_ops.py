import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

class ImageUtils:
    @staticmethod
    def load_image(image_path):
        try:
            original_pil = Image.open(image_path)
            original_pil = ImageOps.exif_transpose(original_pil)
            return original_pil
        except Exception as e:
            raise ValueError(f"Image Load Error: {e}")

    @staticmethod
    def pil_to_cv2(pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    @staticmethod
    def cv2_to_pil(cv2_image):
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def apply_clahe(cv2_image):
        """
        저조도/저대비(흰 벽+흰 가구) 환경 개선을 위한 CLAHE 적용
        """
        # Lab 색공간으로 변환 (L: Lightness 채널만 조절하기 위함)
        lab = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 객체 생성 (Clip Limit: 대비 제한 임계값)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # 채널 병합 및 변환
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    @staticmethod
    def sharpen_image(cv2_image):
        """이미지 윤곽선 강화"""
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        return cv2.filter2D(cv2_image, -1, kernel)

    @staticmethod
    def draw_box_and_text(draw_obj, bbox, text, color, font):
        x1, y1, x2, y2 = map(int, bbox)
        draw_obj.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Text background
        left, top, right, bottom = draw_obj.textbbox((x1, y1), text, font=font)
        text_height = bottom - top
        draw_obj.rectangle([x1, y1 - text_height - 5, x1 + (right-left) + 10, y1], fill=color)
        draw_obj.text((x1 + 5, y1 - text_height - 5), text, fill=(255, 255, 255), font=font)