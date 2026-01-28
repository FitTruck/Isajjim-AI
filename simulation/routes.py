"""시뮬레이션 API 라우트"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from .models import (
    SimulationData,
    SimulationState,
    FurnitureItem,
    FurniturePosition,
    TruckSpec,
    TRUCK_PRESETS,
    get_furniture_color,
)
from .optimizer import optimize_placement, PY3DBP_AVAILABLE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])

# 시뮬레이션 상태 저장 (실제 운영에서는 DB 사용)
_simulation_states: dict[int, SimulationState] = {}

# Static 파일 경로
STATIC_DIR = Path(__file__).parent / "static"


@router.get("/")
async def get_simulator_page() -> HTMLResponse:
    """시뮬레이터 HTML 페이지 반환"""
    html_path = STATIC_DIR / "simulator.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Simulator not found")

    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/trucks")
async def get_truck_presets() -> dict[str, TruckSpec]:
    """사용 가능한 트럭 프리셋 목록"""
    return TRUCK_PRESETS


@router.get("/data/{estimate_id}")
async def get_simulation_data(
    estimate_id: int,
    truck_type: str = Query(default="2.5ton", description="트럭 유형")
) -> SimulationData:
    """
    시뮬레이션에 필요한 데이터 반환

    실제 구현에서는 estimate_id로 DB에서 분석 결과를 조회합니다.
    """
    # 트럭 프리셋 선택
    truck = TRUCK_PRESETS.get(truck_type, TRUCK_PRESETS["2.5ton"])

    # TODO: estimate_id로 실제 분석 결과 조회
    # 현재는 샘플 데이터 반환
    furniture = _get_sample_furniture(estimate_id)

    return SimulationData(
        estimate_id=estimate_id,
        truck=truck,
        furniture=furniture,
    )


@router.post("/state/{estimate_id}")
async def save_simulation_state(
    estimate_id: int,
    state: SimulationState
) -> dict:
    """시뮬레이션 상태 저장"""
    _simulation_states[estimate_id] = state
    logger.info(f"Saved simulation state for estimate {estimate_id}")
    return {"success": True, "message": "State saved"}


@router.get("/state/{estimate_id}")
async def load_simulation_state(estimate_id: int) -> Optional[SimulationState]:
    """저장된 시뮬레이션 상태 불러오기"""
    state = _simulation_states.get(estimate_id)
    if not state:
        raise HTTPException(status_code=404, detail="No saved state found")
    return state


@router.get("/static/{filename}")
async def get_static_file(filename: str) -> FileResponse:
    """정적 파일 제공"""
    file_path = STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ==================== 최적화 API ====================

from pydantic import BaseModel

class OptimizeRequest(BaseModel):
    """최적화 요청"""
    truck_type: str = "2.5ton"
    items: list[dict]  # [{"id", "width", "depth", "height"}, ...]
    algorithm: str = "auto"  # "auto" | "py3dbp" | "blf"


class PlacementResponse(BaseModel):
    """배치 결과"""
    id: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    rotated: bool


class OptimizeResponse(BaseModel):
    """최적화 응답"""
    success: bool
    placements: list[PlacementResponse]
    unplaced_ids: list[str]
    load_percent: float
    algorithm: str
    message: str


@router.post("/optimize")
async def optimize_truck_loading(request: OptimizeRequest) -> OptimizeResponse:
    """
    3D Bin Packing 최적화 API

    트럭 적재 최적화를 수행합니다.

    - **algorithm**: "auto" (py3dbp 가능시 사용, 아니면 BLF), "py3dbp", "blf"
    - **truck_type**: "1ton", "2.5ton", "5ton", "11ton"

    Returns:
        각 가구의 최적 배치 위치 (Three.js 좌표계)
    """
    # 트럭 규격 가져오기
    truck = TRUCK_PRESETS.get(request.truck_type, TRUCK_PRESETS["2.5ton"])

    # 알고리즘 선택
    if request.algorithm == "auto":
        algo = "py3dbp" if PY3DBP_AVAILABLE else "blf"
    else:
        algo = request.algorithm

    logger.info(f"Optimization request: {len(request.items)} items, truck={request.truck_type}, algo={algo}")

    # 최적화 실행
    result = optimize_placement(
        truck_width=truck.width,
        truck_depth=truck.depth,
        truck_height=truck.height,
        items=request.items,
        algorithm=algo
    )

    logger.info(f"Optimization result: {result.load_percent}% load, {len(result.placements)} placed")

    return OptimizeResponse(
        success=result.success,
        placements=[
            PlacementResponse(
                id=p.id, x=p.x, y=p.y, z=p.z,
                width=p.width, depth=p.depth, height=p.height,
                rotated=p.rotated
            )
            for p in result.placements
        ],
        unplaced_ids=result.unplaced_ids,
        load_percent=result.load_percent,
        algorithm=result.algorithm,
        message=result.message
    )


@router.get("/optimizer-status")
async def get_optimizer_status() -> dict:
    """최적화 엔진 상태 확인"""
    return {
        "py3dbp_available": PY3DBP_AVAILABLE,
        "recommended_algorithm": "py3dbp" if PY3DBP_AVAILABLE else "blf",
        "install_command": "pip install py3dbp" if not PY3DBP_AVAILABLE else None
    }


def _get_sample_furniture(estimate_id: int) -> list[FurnitureItem]:
    """
    샘플 가구 데이터 (테스트용) - 실제 PLY 파일 사용

    실제 구현에서는 analyze-furniture 결과를 DB에서 조회
    """
    # 테스트 서버 URL (8080 포트)
    base_url = "http://localhost:8080"

    # 실제 존재하는 PLY 파일 사용
    # 2.png 이미지에서 생성된 PLY 파일들
    sample_items = [
        FurnitureItem(
            id=f"{estimate_id}_bed_001",
            label="bed",
            label_ko="침대",
            type="BED",
            width=0.97, depth=1.0, height=0.48,
            weight=60,
            ply_url=f"{base_url}/test-ply/2_BED_1.ply",  # 15MB
            color=get_furniture_color("bed"),
        ),
        FurnitureItem(
            id=f"{estimate_id}_nightstand_001",
            label="nightstand",
            label_ko="협탁",
            type="NIGHTSTAND",
            width=0.9, depth=0.77, height=1.01,
            weight=15,
            ply_url=f"{base_url}/test-ply/2_NIGHTSTAND_0.ply",  # 12MB
            color=get_furniture_color("cabinet"),
        ),
        FurnitureItem(
            id=f"{estimate_id}_tv_001",
            label="television",
            label_ko="TV",
            type="MONITOR_TV",
            width=1.03, depth=0.11, height=0.57,
            weight=10,
            ply_url=f"{base_url}/test-ply/2_MONITOR_TV_3.ply",  # 2.2MB
            color="#333333",
        ),
        FurnitureItem(
            id=f"{estimate_id}_plant_001",
            label="potted plant",
            label_ko="화분",
            type="POTTED_PLANT",
            width=0.43, depth=0.43, height=1.01,
            weight=5,
            ply_url=f"{base_url}/test-ply/2_POTTED_PLANT_2.ply",  # 4.4MB
            color="#228B22",
        ),
    ]

    return sample_items


def integrate_with_analysis_results(
    estimate_id: int,
    analysis_results: list[dict],
    base_url: str = "http://localhost:8000"
) -> list[FurnitureItem]:
    """
    analyze-furniture 결과를 FurnitureItem 리스트로 변환

    Usage:
        results = await analyze_furniture(...)
        furniture = integrate_with_analysis_results(
            estimate_id=123,
            analysis_results=results["results"],
            base_url="https://your-api.com"
        )
    """
    furniture_items = []

    for img_result in analysis_results:
        image_id = img_result.get("image_id")

        for i, obj in enumerate(img_result.get("objects", [])):
            item = FurnitureItem(
                id=f"{estimate_id}_{image_id}_{obj['label']}_{i:03d}",
                label=obj["label"],
                type=obj.get("type"),
                width=obj["width"] / 100,  # cm → m 변환 (필요시)
                depth=obj["depth"] / 100,
                height=obj["height"] / 100,
                ply_url=obj.get("ply_url"),  # 분석 결과에 포함된 경우
                color=get_furniture_color(obj["label"]),
            )
            furniture_items.append(item)

    return furniture_items
