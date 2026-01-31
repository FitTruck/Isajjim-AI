"""시뮬레이션 데이터 모델"""

from pydantic import BaseModel, Field
from typing import Optional


class TruckSpec(BaseModel):
    """트럭 규격 (미터 단위)"""
    name: str = "1톤 트럭"
    width: float = Field(default=1.7, description="너비 (m)")
    depth: float = Field(default=2.8, description="길이 (m)")
    height: float = Field(default=1.7, description="높이 (m)")
    max_weight: float = Field(default=1000, description="최대 적재 중량 (kg)")

    @property
    def volume(self) -> float:
        """적재 용량 (m³)"""
        return self.width * self.depth * self.height


class FurnitureItem(BaseModel):
    """가구 아이템"""
    id: str = Field(..., description="고유 ID")
    label: str = Field(..., description="가구 이름 (영문)")
    label_ko: Optional[str] = Field(None, description="가구 이름 (한글)")
    type: Optional[str] = Field(None, description="세부 유형")
    width: float = Field(..., description="너비 (m)")
    depth: float = Field(..., description="깊이 (m)")
    height: float = Field(..., description="높이 (m)")
    weight: Optional[float] = Field(None, description="무게 (kg)")
    ply_url: Optional[str] = Field(None, description="PLY 파일 URL")
    color: Optional[str] = Field(None, description="표시 색상 (hex)")

    @property
    def volume(self) -> float:
        """부피 (m³)"""
        return self.width * self.depth * self.height


class FurniturePosition(BaseModel):
    """배치된 가구 위치"""
    id: str
    x: float
    y: float
    z: float
    rotation_y: float = 0.0


class SimulationData(BaseModel):
    """시뮬레이션 초기 데이터"""
    estimate_id: int
    truck: TruckSpec = Field(default_factory=TruckSpec)
    furniture: list[FurnitureItem] = Field(default_factory=list)


class SimulationState(BaseModel):
    """시뮬레이션 상태 (저장/불러오기용)"""
    estimate_id: int
    truck: TruckSpec
    placed_items: list[FurniturePosition] = Field(default_factory=list)
    load_percent: float = 0.0


# 트럭 프리셋 (미터 단위)
TRUCK_PRESETS = {
    "1ton": TruckSpec(
        name="1톤 트럭",
        width=1.7, depth=2.8, height=1.7, max_weight=1000
    ),
    "2.5ton": TruckSpec(
        name="2.5톤 트럭",
        width=2.0, depth=4.3, height=1.9, max_weight=2500
    ),
    "5ton": TruckSpec(
        name="5톤 트럭",
        width=2.3, depth=6.2, height=2.4, max_weight=5000
    ),
    "11ton": TruckSpec(
        name="11톤 트럭",
        width=2.4, depth=9.0, height=2.6, max_weight=11000
    ),
}

# 트럭 프리셋 (cm 단위) - OBB 패커용 (TRUCK_PRESETS와 일치)
TRUCK_PRESETS_CM = {
    "1ton": {"width": 170, "depth": 280, "height": 170},
    "2.5ton": {"width": 200, "depth": 430, "height": 190},
    "5ton": {"width": 230, "depth": 620, "height": 240},
}

# 가구별 기본 색상
FURNITURE_COLORS = {
    "sofa": "#4a90d9",
    "chair": "#2e8b57",
    "table": "#8b4513",
    "desk": "#a0522d",
    "bed": "#9370db",
    "wardrobe": "#708090",
    "cabinet": "#556b2f",
    "shelf": "#cd853f",
    "default": "#888888",
}


def get_furniture_color(label: str) -> str:
    """가구 라벨에 맞는 색상 반환"""
    label_lower = label.lower()
    for key, color in FURNITURE_COLORS.items():
        if key in label_lower:
            return color
    return FURNITURE_COLORS["default"]
