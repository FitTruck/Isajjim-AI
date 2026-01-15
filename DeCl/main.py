import os
import cv2
import numpy as np
from core.analyzer import FurnitureAnalyzer
from utils.output_manager import OutputManager

def main():
    img_dir = "imgs"
    os.makedirs(img_dir, exist_ok=True) # 입력 디렉토리 설정
    
    image_path = os.path.join(img_dir, "test5.jpg") # 테스트 이미지 입력
    
    # 만약 이미지가 없으면 더미데이터 (0ㅇ로 구성된) 생성
    '''
    if not os.path.exists(image_path):
        print(f"[Info] '{image_path}'가 없어 테스트용 이미지를 생성합니다.")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy)
    '''

    # print("=== SAM 3D 기반 AI 이사 견적 서비스 (Local DB Mode) ===")
    
    try:
        # 1. 초기화
        analyzer = FurnitureAnalyzer()
        output_manager = OutputManager(base_dir="outputs") # 출력 폴더 설정
        
        # 2. 분석 실행
        # print(f"\n[Processing] Analyzing: {image_path}")
        data, res_image = analyzer.analyze(image_path)
        
        # 3. 콘솔 출력
        print(f"\n[Result] 탐지된 가구 수: {len(data)}")
        # 어쩌다보니 최종적으로 아이콘 형태로 출력되었는데 O, X로 변경해도 됩니다.
        for i, item in enumerate(data):
            icon = "✅" if item['is_movable'] else "🚫"
            print(f"{i+1}. {icon} {item['label']} : {item['status']}")

        # 4. 결과 파일 저장 (박스 이미지, 크롭, SAM 데이터)
        # analyze 함수에서 그린 박스 이미지(res_image)와 원본 경로(image_path)를 모두 전달
        output_manager.save_results(image_path, data, res_image)
        
    except Exception as e:
        print(f"[Critical Error] 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()