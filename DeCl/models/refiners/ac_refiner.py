# Issue 1 & 3의 핵심 로직이 담긴 Refiner
import numpy as np

class ACRefiner:
    def __init__(self, clip_classifier):
        self.clip = clip_classifier

    def refine(self, bbox, image_wh, crop_pil, original_subtypes):
        """
        에어컨 전용 정밀 분석 로직
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_wh
        w, h = x2 - x1, y2 - y1
        ratio = w / h if h > 0 else 0

        # --- Logic 1: Geometry & Position Check (천장형 vs 벽걸이) ---
        is_touching_top = (y1 < img_h * 0.05) # 이미지 상단 5% 이내 시작
        
        if is_touching_top and ratio > 1.5:
            # 상단에 붙어있고 가로로 길면 -> 천장형일 확률 매우 높음
            return self._find_subtype(original_subtypes, "천장형")
        
        if ratio > 2.5: 
            # 위치와 상관없이 극도로 가로로 길면 -> 벽걸이일 확률 높음 (천장형 1way도 포함될 수 있으나 CLIP으로 검증)
            # 단, 상단에 붙어있지 않다면 벽걸이가 유력
             return self._find_subtype(original_subtypes, "벽걸이")

        # --- Logic 2: Tower Type vs Cabinet Check (스탠드 에어컨) ---
        if ratio < 0.6: # 세로로 긴 형태 (타워형)
            # 머리 부분(Head)을 잘라서 CLIP 확인 ↓
            # "vents", "fan" 등의 특징이 상단에 있는지 확인
            head_result = self.clip.classify_head_check(crop_pil, original_subtypes)
            
            # CLIP이 스탠드 에어컨으로 강하게 확신하거나, 규칙상 스탠드 형태면
            if "스탠드" in head_result['name']:
                return head_result
            else:
                # CLIP이 에어컨이 아니라고 하면(가구와 혼동 시), 보수적으로 판단하거나 사용자 확인 요청
                # 여기서는 일단 스탠드로 분류하되 Score를 낮춤
                res = self._find_subtype(original_subtypes, "스탠드")
                res['score'] = 0.6 # Low confidence
                return res

        # Fallback: 일반 CLIP 분류
        return self.clip.classify(crop_pil, original_subtypes)

    def _find_subtype(self, subtypes, keyword):
        for s in subtypes:
            if keyword in s['name']:
                res = s.copy()
                res['score'] = 0.95 # 점수 차이로 구별
                return res
        return subtypes[0] # Default