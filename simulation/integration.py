"""
시뮬레이션 모듈을 메인 API와 통합하는 예시

사용법:
1. api/app.py에서 라우터 등록
2. analyze-furniture 결과와 연동
"""

from typing import Optional
from .models import FurnitureItem, SimulationData, TruckSpec, get_furniture_color


def create_simulation_data_from_analysis(
    estimate_id: int,
    analysis_results: list[dict],
    truck_type: str = "2.5ton",
    base_url: str = "http://localhost:8000",
    scale_factor: float = 1.0,
) -> SimulationData:
    """
    analyze-furniture API 결과를 SimulationData로 변환

    Args:
        estimate_id: 견적 ID
        analysis_results: /analyze-furniture 응답의 results 배열
        truck_type: 트럭 유형 (1ton, 2.5ton, 5ton)
        base_url: PLY 파일 서빙 URL
        scale_factor: 치수 스케일 (API가 cm 반환 시 0.01)

    Example:
        # API 결과
        results = {
            "results": [
                {
                    "image_id": 101,
                    "objects": [
                        {
                            "label": "sofa",
                            "type": "THREE_SEATER_SOFA",
                            "width": 200.0,
                            "depth": 90.0,
                            "height": 85.0,
                            "ply_url": "/assets/sofa_001.ply"  # 옵션
                        }
                    ]
                }
            ]
        }

        # 변환
        sim_data = create_simulation_data_from_analysis(
            estimate_id=123,
            analysis_results=results["results"],
            scale_factor=0.01  # cm → m
        )
    """
    from .models import TRUCK_PRESETS

    # 트럭 선택
    truck = TRUCK_PRESETS.get(truck_type, TRUCK_PRESETS["2.5ton"])

    # 가구 아이템 변환
    furniture_items = []
    item_counter = {}

    for img_result in analysis_results:
        image_id = img_result.get("image_id", 0)

        for obj in img_result.get("objects", []):
            label = obj.get("label", "unknown")

            # 고유 ID 생성
            item_counter[label] = item_counter.get(label, 0) + 1
            item_id = f"{estimate_id}_{label}_{item_counter[label]:03d}"

            # PLY URL 처리
            ply_path = obj.get("ply_url")
            ply_url = None
            if ply_path:
                if ply_path.startswith("http"):
                    ply_url = ply_path
                else:
                    ply_url = f"{base_url}{ply_path}"

            item = FurnitureItem(
                id=item_id,
                label=label,
                label_ko=_get_korean_label(label),
                type=obj.get("type"),
                width=obj.get("width", 1.0) * scale_factor,
                depth=obj.get("depth", 1.0) * scale_factor,
                height=obj.get("height", 1.0) * scale_factor,
                weight=_estimate_weight(label),
                ply_url=ply_url,
                color=get_furniture_color(label),
            )
            furniture_items.append(item)

    return SimulationData(
        estimate_id=estimate_id,
        truck=truck,
        furniture=furniture_items,
    )


def _get_korean_label(label: str) -> str:
    """영문 라벨을 한글로 변환"""
    translations = {
        "sofa": "소파",
        "chair": "의자",
        "table": "테이블",
        "dining table": "식탁",
        "desk": "책상",
        "bed": "침대",
        "wardrobe": "옷장",
        "cabinet": "수납장",
        "shelf": "선반",
        "tv stand": "TV 스탠드",
        "coffee table": "커피테이블",
        "armchair": "안락의자",
        "bookshelf": "책장",
        "dresser": "화장대",
        "nightstand": "협탁",
    }
    return translations.get(label.lower(), label)


def _estimate_weight(label: str) -> Optional[float]:
    """가구 유형별 예상 무게 (kg)"""
    weights = {
        "sofa": 45,
        "chair": 5,
        "table": 25,
        "dining table": 35,
        "desk": 25,
        "bed": 60,
        "wardrobe": 80,
        "cabinet": 40,
        "shelf": 20,
        "tv stand": 30,
        "coffee table": 15,
        "armchair": 25,
        "bookshelf": 35,
        "dresser": 50,
        "nightstand": 10,
    }
    return weights.get(label.lower())


# ==================== API 통합 예시 ====================

INTEGRATION_EXAMPLE = """
# api/app.py에 추가

from simulation import simulation_router

# 라우터 등록
app.include_router(simulation_router)

# 또는 특정 prefix로
# app.include_router(simulation_router, prefix="/api/v1")


# ==================== 분석 결과와 연동 ====================

# api/routes/furniture.py에서

from simulation.integration import create_simulation_data_from_analysis

@router.post("/analyze-furniture")
async def analyze_furniture(request: FurnitureAnalysisRequest):
    # ... 기존 분석 로직 ...

    # 분석 완료 후 시뮬레이션 데이터 생성
    sim_data = create_simulation_data_from_analysis(
        estimate_id=request.estimate_id,
        analysis_results=results,
        scale_factor=0.01,  # cm → m
        base_url="https://your-api.com"
    )

    # 시뮬레이션 데이터 저장 (캐시 또는 DB)
    await save_simulation_data(request.estimate_id, sim_data)

    return {
        "success": True,
        "results": results,
        "simulation_url": f"/simulation/?estimate_id={request.estimate_id}"
    }


# ==================== React Native 연동 ====================

// TruckSimulationScreen.tsx

import { WebView } from 'react-native-webview';

export function TruckSimulationScreen({ route }) {
  const { estimateId, furnitureData } = route.params;
  const webViewRef = useRef(null);

  const onWebViewLoad = () => {
    // 시뮬레이터에 데이터 전송
    webViewRef.current?.injectJavaScript(`
      window.initSimulation(${JSON.stringify(furnitureData)});
      true;
    `);
  };

  return (
    <WebView
      ref={webViewRef}
      source={{ uri: `https://your-api.com/simulation/?estimate_id=${estimateId}` }}
      onLoad={onWebViewLoad}
      onMessage={(event) => {
        const data = JSON.parse(event.nativeEvent.data);
        console.log('시뮬레이션 상태:', data);
      }}
    />
  );
}
"""

if __name__ == "__main__":
    print(INTEGRATION_EXAMPLE)
