from PIL import Image, ImageDraw, ImageFont
from config import Config
from data.knowledge_base import FURNITURE_DB # Local KB 로드
from utils.image_ops import ImageUtils
from models.detector import YoloDetector
from models.classifier import ClipClassifier
from models.refiners.ac_refiner import ACRefiner

# 가구분석
class FurnitureAnalyzer:
    def __init__(self):
        Config.check_dependencies()
        
        self.detector = YoloDetector()
        self.classifier = ClipClassifier()
        self.ac_refiner = ACRefiner(self.classifier)
        
        # 폰트 로드
        try:
            self.font = ImageFont.truetype(Config.FONT_PATH, Config.FONT_SIZE_LARGE)
        except:
            self.font = ImageFont.load_default()

        # [DB Integration Note]
        # 현재는 knowledge_base.py의 딕셔너리를 메모리에 로드하여 사용합니다.
        # 추후 실제 DB 연결 시 아래 코드를 활성화하세요.
        # self.db_connection = connect_to_db(host=..., port=...) 
        
        self.search_classes = []
        self.class_map = {} 
        
        # Local DB에서 검색 키워드 추출
        for key, info in FURNITURE_DB.items():
            for syn in info['synonyms']:
                self.search_classes.append(syn)
                self.class_map[syn] = key

    def analyze(self, image_path):
        # 1. 이미지 로드
        pil_image = ImageUtils.load_image(image_path)
        img_w, img_h = pil_image.size
        draw = ImageDraw.Draw(pil_image)

        # 2. YOLO 탐지 설정
        self.detector.set_classes(self.search_classes)
        
        # 3. 객체 탐지 (CLAHE 적용)
        results = self.detector.detect_smart(pil_image)
        
        final_data = []
        if results is None:
            return [], pil_image

        boxes = results["boxes"]
        classes = results["classes"]
        
        # 4. 결과 분석 Loop
        for box, cls_idx in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            if (x2-x1) < 20 or (y2-y1) < 20: continue
            
            # YOLO Label -> DB Key 매핑
            detected_label = self.detector.model.names[int(cls_idx)]
            db_key = self.class_map.get(detected_label)
            if not db_key: continue
            
            # [DB Integration Note]
            # 실제 DB 사용 시: db_info = self.db_connection.query("SELECT * FROM furniture WHERE key=?", db_key)
            db_info = FURNITURE_DB[db_key] # 현재는 Local Dict 사용
            
            # 객체 Crop
            crop = pil_image.crop((x1, y1, x2, y2))
            
            # 5. 상세 분류 (Refinement)
            if 'subtypes' in db_info and db_info['subtypes']:
                if db_info.get('check_strategy') == "AC_STRATEGY":
                    final_info = self.ac_refiner.refine([x1, y1, x2, y2], (img_w, img_h), crop, db_info['subtypes'])
                else:
                    final_info = self.classifier.classify(crop, db_info['subtypes'])
            else:
                # Subtypes가 없으면 DB의 기본 정보 사용 (Fallback)
                final_info = {
                    "name": db_info.get('base_name', detected_label),
                    "score": 1.0,
                    "is_movable": db_info.get('is_movable', True)
                }
                  
            # 6. 시각화 및 결과 저장
            status_text = f"{final_info['name']} ({int(final_info['score']*100)}%)"
            color = (0, 255, 0) if final_info['is_movable'] else (255, 50, 50)
            
            ImageUtils.draw_box_and_text(draw, (x1,y1,x2,y2), status_text, color, self.font)
            
            final_data.append({
                "label": final_info['name'],
                "bbox": [x1, y1, x2, y2],
                "is_movable": final_info['is_movable'],
                "status": status_text
            })

        return final_data, pil_image