import os
import json
import cv2
import numpy as np
from PIL import Image

class OutputManager:
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # 크롭 이미지 저장을 위한 하위 폴더 생성
        self.crop_dir = os.path.join(base_dir, "crops")
        os.makedirs(self.crop_dir, exist_ok=True)

    def save_results(self, image_path, detections, drawn_image):
        """
        1. 전체 결과 이미지 저장 (박스 그려진 버전)
        2. 객체별 크롭 이미지 저장 (박스 없는 깨끗한 버전)
        3. SAM 3D 연동용 JSON 데이터 저장
        """
        # 1. 전체 결과 이미지 저장
        res_cv = cv2.cvtColor(np.array(drawn_image), cv2.COLOR_RGB2BGR)
        full_result_path = os.path.join(self.base_dir, "final_result.jpg")
        cv2.imwrite(full_result_path, res_cv)
        print(f"[Output] 전체 결과 저장됨: {full_result_path}")

        # 2. 객체별 크롭 이미지 저장을 위해 원본 이미지 다시 로드 (박스 없는 버전)
        try:
            clean_image = Image.open(image_path)
            self._save_crops(clean_image, detections)
        except Exception as e:
            print(f"[Error] 크롭 저장 실패: {e}")

        # 3. SAM 3D 프롬프트 데이터 저장
        self._save_sam_json(detections)

    def _save_crops(self, clean_image, detections):
        """탐지된 객체를 각각 잘라내어 이미지로 저장"""
        print(f"[Output] 객체 크롭 이미지 저장 중... ({self.crop_dir})")
        
        for i, item in enumerate(detections):
            bbox = item['bbox'] # [x1, y1, x2, y2]
            label = item['label'].replace(" ", "_").replace("/", "_")
            
            # Crop (PIL 사용)
            # 좌표가 이미지 범위를 벗어나지 않게 클리핑
            w, h = clean_image.size
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            
            if x2 - x1 < 5 or y2 - y1 < 5: continue # 너무 작은 이미지는 패스
            # 만약 작은 이미지 사용시 이 부분 수정 필요

            crop = clean_image.crop((x1, y1, x2, y2))
            
            filename = f"{i:02d}_{label}.jpg"
            save_path = os.path.join(self.crop_dir, filename)
            crop.save(save_path)

    # SAM 입력 프롬프트값 수정이 필요하면 여기서 수정
    def _save_sam_json(self, detections):
        """SAM 3D 입력용 프롬프트 데이터 생성 (BBox, Center Point)"""
        sam_data = []
        for i, item in enumerate(detections):
            bbox = item['bbox']
            # 중심점 계산 (Point Prompt용)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            sam_entry = {
                "id": i,
                "label": item['label'],
                "is_movable": item['is_movable'],
                "prompt_type": "box_and_point",
                "bbox_2d": [int(x) for x in bbox], # [x1, y1, x2, y2]
                "point_prompt": [int(center_x), int(center_y)],
                "confidence_score": item.get('score', 0.0) # 만약 score가 있다면
            }
            sam_data.append(sam_entry)
            
        json_path = os.path.join(self.base_dir, "sam_3d_prompts.json")
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(sam_data, f, indent=4, ensure_ascii=False)
            
        print(f"[Output] SAM 3D 프롬프트 데이터 저장됨: {json_path}")