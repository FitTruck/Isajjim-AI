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
    TRUCK_PRESETS_CM,
    get_furniture_color,
)
from .optimizer import optimize_placement, PY3DBP_AVAILABLE
from .obb_packer import optimize_obb, TRUCK_PRESETS_CM as OBB_TRUCK_PRESETS
from .ply_alignment import PLYAlignmentService, AlignmentResult, OPEN3D_AVAILABLE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])

# 시뮬레이션 상태 저장 (실제 운영에서는 DB 사용)
_simulation_states: dict[int, SimulationState] = {}

# Static 파일 경로
STATIC_DIR = Path(__file__).parent / "static"
ASSETS_DIR = Path(__file__).parent / "assets"


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


@router.get("/assets/{filename:path}")
async def get_asset_file(filename: str) -> FileResponse:
    """PLY 에셋 파일 제공 (aligned/ 서브디렉토리 포함)"""
    file_path = ASSETS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {filename}")
    return FileResponse(file_path, media_type="application/octet-stream")


# ==================== 최적화 API ====================

from pydantic import BaseModel

class OptimizeRequest(BaseModel):
    """최적화 요청"""
    truck_type: str = "2.5ton"
    items: list[dict]  # [{"id", "width", "depth", "height"}, ...]
    algorithm: str = "blf"  # "blf" | "py3dbp"


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


# ==================== OBB 최적화 API ====================

class OBBOptimizeRequest(BaseModel):
    """OBB 최적화 요청"""
    items: list[dict]  # [{"id", "width", "depth", "height"}, ...]
    truck_type: Optional[str] = None  # None = 자동 선택
    unit: str = "m"  # "m" | "cm"
    support_ratio: float = 0.7  # 지지 비율 (기본 70%)


class OBBPlacementResponse(BaseModel):
    """OBB 배치 결과"""
    id: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    orientation: int


class OBBOptimizeResponse(BaseModel):
    """OBB 최적화 응답"""
    success: bool
    truck_type: str
    placements: list[OBBPlacementResponse]
    unplaced_ids: list[str]
    volume_utilization: float
    message: str


@router.post("/optimize-obb")
async def optimize_obb_loading(request: OBBOptimizeRequest) -> OBBOptimizeResponse:
    """
    OBB 기반 3D Bin Packing 최적화 API

    Extreme Points 알고리즘을 사용하여 트럭 적재 최적화를 수행합니다.

    특징:
    - 6방향 회전 지원
    - 70% 지지 규칙 (안정성 보장)
    - 벽면 채우기 우선순위 (Y → Z → X)
    - 자동 트럭 선택 (truck_type=None)

    Args:
        items: [{"id", "width", "depth", "height"}, ...]
        truck_type: "1ton" | "2.5ton" | "5ton" | None (자동)
        unit: "m" | "cm" (기본 "m")
        support_ratio: 지지 비율 (기본 0.7)

    Returns:
        각 가구의 최적 배치 위치 (Three.js 좌표계)
    """
    logger.info(
        f"OBB Optimization: {len(request.items)} items, "
        f"truck={request.truck_type or 'auto'}, unit={request.unit}"
    )

    result = optimize_obb(
        items=request.items,
        truck_type=request.truck_type,
        unit=request.unit,
        support_ratio=request.support_ratio
    )

    logger.info(
        f"OBB Result: {result.volume_utilization}% utilization, "
        f"{len(result.placed_items)} placed, {len(result.unplaced_items)} unplaced"
    )

    return OBBOptimizeResponse(
        success=result.success,
        truck_type=result.truck_type,
        placements=[
            OBBPlacementResponse(
                id=p.item_id,
                x=p.x,
                y=p.y,
                z=p.z,
                width=p.width,
                depth=p.depth,
                height=p.height,
                orientation=p.orientation
            )
            for p in result.placed_items
        ],
        unplaced_ids=result.unplaced_items,
        volume_utilization=result.volume_utilization,
        message=result.message
    )


# ==================== PLY 정렬 API ====================

class AlignPLYRequest(BaseModel):
    """PLY 정렬 요청"""
    ply_base64: str  # Base64 인코딩된 PLY 데이터
    convert_to_yup: bool = True  # Y-up 좌표계로 변환 여부


class AlignPLYResponse(BaseModel):
    """PLY 정렬 응답"""
    success: bool
    aligned_ply_base64: Optional[str] = None
    width: float
    depth: float
    height: float
    point_count: int
    message: str


@router.post("/align-ply")
async def align_ply(request: AlignPLYRequest) -> AlignPLYResponse:
    """
    PLY 객체 정렬 API

    OBB(Oriented Bounding Box) 기반으로 PLY 객체를 정렬합니다.

    기능:
    - OBB.R.T 역회전으로 주축 정렬
    - 바닥에 배치 (Z-min 또는 Y-min = 0)
    - Z-up → Y-up 좌표계 변환 (선택)

    Args:
        ply_base64: Base64 인코딩된 PLY 데이터
        convert_to_yup: True면 Three.js 호환 Y-up 좌표계로 변환

    Returns:
        정렬된 PLY (Base64) + AABB 치수
    """
    if not OPEN3D_AVAILABLE:
        return AlignPLYResponse(
            success=False,
            width=0, depth=0, height=0,
            point_count=0,
            message="Open3D not installed. Run: pip install open3d"
        )

    try:
        service = PLYAlignmentService(convert_to_yup=request.convert_to_yup)
        aligned_base64, result = service.align_from_base64(request.ply_base64)

        return AlignPLYResponse(
            success=result.success,
            aligned_ply_base64=aligned_base64 if result.success else None,
            width=result.width,
            depth=result.depth,
            height=result.height,
            point_count=result.point_count,
            message=result.message
        )
    except Exception as e:
        logger.error(f"PLY alignment failed: {e}")
        return AlignPLYResponse(
            success=False,
            width=0, depth=0, height=0,
            point_count=0,
            message=str(e)
        )


@router.get("/alignment-status")
async def get_alignment_status() -> dict:
    """PLY 정렬 서비스 상태 확인"""
    return {
        "open3d_available": OPEN3D_AVAILABLE,
        "install_command": "pip install open3d" if not OPEN3D_AVAILABLE else None,
        "features": [
            "OBB-based axis alignment",
            "Z-up to Y-up coordinate conversion",
            "Floor placement (min=0)"
        ]
    }


def _get_sample_furniture(estimate_id: int) -> list[FurnitureItem]:
    """
    샘플 가구 데이터 (테스트용) - PLY 파일 있으면 3D 렌더링

    실제 구현에서는 analyze-furniture 결과를 DB에서 조회

    PLY 모델 비율 기준으로 실제 가구 크기 설정:
    - 침대: 싱글(1.0x2.0), 더블(1.4x2.0), 퀸(1.5x2.0)
    - 소파: 2인(1.4x0.85), 3인(2.0x0.9)
    - 테이블: 거실(1.0x0.6), 소형(0.6x0.4)
    - 협탁: 0.4~0.5 정사각형
    - 캐비닛: 0.6x0.4 x 높이 1.0~1.5
    """
    # PLY 파일이 있는 가구만 3D 렌더링 (크기 1.3배 적용)
    sample_items = [
        # === 침대 (BED) ===
        # 2_BED_1.ply: 싱글침대 (1.3x2.28x0.59)
        FurnitureItem(
            id=f"{estimate_id}_bed_001",
            label="bed",
            label_ko="싱글침대",
            type="BED",
            width=1.3, depth=2.28, height=0.59,
            weight=65,
            ply_url="/simulation/assets/aligned/2_BED_1.ply",
            color=get_furniture_color("bed"),
        ),
        # 1_BED_6.ply: 더블침대 (1.82x2.6x0.65)
        FurnitureItem(
            id=f"{estimate_id}_bed_002",
            label="bed",
            label_ko="더블침대",
            type="BED",
            width=1.82, depth=2.6, height=0.65,
            weight=91,
            ply_url="/simulation/assets/aligned/1_BED_6.ply",
            color=get_furniture_color("bed"),
        ),
        # 9_BED_0.ply: 퀸침대 (1.95x2.6x0.72)
        FurnitureItem(
            id=f"{estimate_id}_bed_003",
            label="bed",
            label_ko="퀸침대",
            type="BED",
            width=1.95, depth=2.6, height=0.72,
            weight=104,
            ply_url="/simulation/assets/aligned/9_BED_0.ply",
            color=get_furniture_color("bed"),
        ),
        # 5_BED_0.ply: 싱글침대2 (1.17x2.47x0.59)
        FurnitureItem(
            id=f"{estimate_id}_bed_004",
            label="bed",
            label_ko="싱글침대2",
            type="BED",
            width=1.17, depth=2.47, height=0.59,
            weight=59,
            ply_url="/simulation/assets/aligned/5_BED_0.ply",
            color=get_furniture_color("bed"),
        ),
        # === 소파 (SOFA) ===
        # 9_SOFA_1.ply: 1인소파 (1.04x1.11x0.98)
        FurnitureItem(
            id=f"{estimate_id}_sofa_001",
            label="sofa",
            label_ko="1인소파",
            type="SOFA",
            width=1.04, depth=1.11, height=0.98,
            weight=33,
            ply_url="/simulation/assets/aligned/9_SOFA_1.ply",
            color=get_furniture_color("sofa"),
        ),
        # 12_SOFA_0.ply: 2인소파 (2.34x1.17x1.04)
        FurnitureItem(
            id=f"{estimate_id}_sofa_002",
            label="sofa",
            label_ko="2인소파",
            type="SOFA",
            width=2.34, depth=1.17, height=1.04,
            weight=52,
            ply_url="/simulation/assets/aligned/12_SOFA_0.ply",
            color=get_furniture_color("sofa"),
        ),
        # === 테이블 (TABLE) ===
        # 4_COFFEE_TABLE_1.ply: 거실테이블 (1.04x0.65x0.59)
        FurnitureItem(
            id=f"{estimate_id}_table_001",
            label="coffee table",
            label_ko="거실테이블",
            type="COFFEE_TABLE",
            width=1.04, depth=0.65, height=0.59,
            weight=26,
            ply_url="/simulation/assets/aligned/4_COFFEE_TABLE_1.ply",
            color=get_furniture_color("table"),
        ),
        # 11_COFFEE_TABLE_1.ply: 소형테이블 (0.65x0.52x0.52)
        FurnitureItem(
            id=f"{estimate_id}_table_002",
            label="coffee table",
            label_ko="소형테이블",
            type="COFFEE_TABLE",
            width=0.65, depth=0.52, height=0.52,
            weight=13,
            ply_url="/simulation/assets/aligned/11_COFFEE_TABLE_1.ply",
            color=get_furniture_color("table"),
        ),
        # === 의자/스툴 (CHAIR) ===
        # 4_CHAIR_STOOL_3.ply: 스툴 (0.52x0.52x0.59)
        FurnitureItem(
            id=f"{estimate_id}_chair_001",
            label="chair",
            label_ko="스툴",
            type="CHAIR_STOOL",
            width=0.52, depth=0.52, height=0.59,
            weight=5,
            ply_url="/simulation/assets/aligned/4_CHAIR_STOOL_3.ply",
            color=get_furniture_color("chair"),
        ),
        # === 수납장 (CABINET) ===
        # 8_CABINET_1.ply: 수납장 (0.78x0.52x1.56)
        FurnitureItem(
            id=f"{estimate_id}_cabinet_001",
            label="cabinet",
            label_ko="수납장",
            type="CABINET",
            width=0.78, depth=0.52, height=1.56,
            weight=39,
            ply_url="/simulation/assets/aligned/8_CABINET_1.ply",
            color=get_furniture_color("cabinet"),
        ),
        # === 협탁 (NIGHTSTAND) ===
        # 2_NIGHTSTAND_0.ply: 협탁 (0.52x0.52x0.72)
        FurnitureItem(
            id=f"{estimate_id}_nightstand_001",
            label="nightstand",
            label_ko="협탁",
            type="NIGHTSTAND",
            width=0.52, depth=0.52, height=0.72,
            weight=16,
            ply_url="/simulation/assets/aligned/2_NIGHTSTAND_0.ply",
            color=get_furniture_color("cabinet"),
        ),
        # 10_NIGHTSTAND_0.ply: 협탁2 (0.46x0.59x0.78)
        FurnitureItem(
            id=f"{estimate_id}_nightstand_002",
            label="nightstand",
            label_ko="협탁2",
            type="NIGHTSTAND",
            width=0.46, depth=0.59, height=0.78,
            weight=13,
            ply_url="/simulation/assets/aligned/10_NIGHTSTAND_0.ply",
            color=get_furniture_color("cabinet"),
        ),
        # === TV (MONITOR_TV) ===
        # 2_MONITOR_TV_3.ply: 55인치TV (1.59x0.09x0.91)
        FurnitureItem(
            id=f"{estimate_id}_tv_001",
            label="television",
            label_ko="55인치TV",
            type="MONITOR_TV",
            width=1.59, depth=0.09, height=0.91,
            weight=20,
            ply_url="/simulation/assets/aligned/2_MONITOR_TV_3.ply",
            color="#333333",
        ),
        # === 화분 (POTTED_PLANT) ===
        # 2_POTTED_PLANT_4.ply: 대형화분 (0.46x0.46x1.17)
        FurnitureItem(
            id=f"{estimate_id}_plant_001",
            label="potted plant",
            label_ko="대형화분",
            type="POTTED_PLANT",
            width=0.46, depth=0.46, height=1.17,
            weight=13,
            ply_url="/simulation/assets/aligned/2_POTTED_PLANT_4.ply",
            color="#228B22",
        ),
        # 2_POTTED_PLANT_2.ply: 중형화분 (0.39x0.39x0.72)
        FurnitureItem(
            id=f"{estimate_id}_plant_002",
            label="potted plant",
            label_ko="중형화분",
            type="POTTED_PLANT",
            width=0.39, depth=0.39, height=0.72,
            weight=7,
            ply_url="/simulation/assets/aligned/2_POTTED_PLANT_2.ply",
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
