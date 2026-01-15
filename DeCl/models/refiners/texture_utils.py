import cv2
import numpy as np

def check_vent_pattern(image_crop):
    """
    Issue 3 해결: 타워형 에어컨의 송풍구 패턴(높은 엣지 밀도 + 주기적 수평선) 감지.
    """
    if image_crop.size == 0: return False
    
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 1. 전체 엣지 밀도 계산
    total_pixels = image_crop.shape * image_crop.shape
    edge_pixels = np.count_nonzero(edges)
    density = edge_pixels / total_pixels
    
    # 2. 수평 투영 (Horizontal Projection)을 통한 반복 패턴 확인
    horizontal_sum = np.sum(edges, axis=1)
    peaks = 0
    threshold = np.max(horizontal_sum) * 0.3
    
    # 피크 개수 카운트 (송풍구 날개 개수 추정)
    for val in horizontal_sum:
        if val > threshold:
            peaks += 1
            
    # 에어컨 판단 조건: 엣지 밀도가 적당히 높고(0.05 이상), 수평 패턴이 존재
    is_ac_pattern = density > 0.05 and peaks > 5
    return is_ac_pattern