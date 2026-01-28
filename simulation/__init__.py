# Truck Loading Simulation Module
"""
트럭 적재 시뮬레이션 모듈

사용자가 3D 가구 객체를 트럭에 직접 배치해볼 수 있는
인터랙티브 시뮬레이션을 제공합니다.

Components:
- routes.py: FastAPI 라우트 (/simulation/*)
- models.py: Pydantic 데이터 모델
- static/simulator.html: Three.js 기반 3D 시뮬레이터
"""

from .routes import router as simulation_router
from .models import SimulationData, FurnitureItem, TruckSpec

__all__ = [
    "simulation_router",
    "SimulationData",
    "FurnitureItem",
    "TruckSpec",
]
