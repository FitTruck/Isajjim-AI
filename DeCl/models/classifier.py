import torch
from transformers import CLIPProcessor, CLIPModel
from config import Config
from PIL import Image

class ClipClassifier:
    def __init__(self):
        print(f"[Init] Loading CLIP: {Config.CLIP_MODEL_ID}...")
        # Use safetensors to avoid torch.load security vulnerability with torch < 2.6
        self.model = CLIPModel.from_pretrained(Config.CLIP_MODEL_ID, use_safetensors=True).to(Config.DEVICE)
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_ID)

    def classify(self, pil_crop, candidates):
        """
        기본적인 CLIP Classification
        """
        if not candidates: return None
        
        prompts = [c['prompt'] for c in candidates]
        inputs = self.processor(text=prompts, images=pil_crop, return_tensors="pt", padding=True).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
        best_idx = probs.argmax().item()
        score = probs[0][best_idx].item()
        
        res = candidates[best_idx].copy()
        res['score'] = score
        return res

    def classify_head_check(self, pil_crop, candidates):
        """
        타워형 에어컨 vs 옷장 구분을 위해
        이미지의 상단 25%만 잘라서(Head Crop) 특징을 분석
        -> 오류가 생길 확률이 높아서 차후 수정 필수
        """
        w, h = pil_crop.size
        if h < 50: # 너무 작으면 전체 사용
            return self.classify(pil_crop, candidates)
            
        # 상단 25% Crop
        head_crop = pil_crop.crop((0, 0, w, int(h * 0.25)))
        
        # '에어컨 송풍구' vs '매끈한 나무' 류의 구체적 프롬프트로 비교 필요
        # 여기서는 단순화를 위해 기존 candidates 사용하되, 입력 이미지만 Head로 변경
        return self.classify(head_crop, candidates)