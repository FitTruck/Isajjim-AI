"""
독립적인 시뮬레이션 테스트 서버

메인 API에 영향 없이 시뮬레이션만 테스트합니다.

Usage:
    # 프로젝트 루트에서 실행
    python -m simulation.test_server

    # 브라우저에서 접속
    http://localhost:8080/
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from simulation.routes import router as simulation_router

# 독립 FastAPI 앱
app = FastAPI(
    title="Truck Loading Simulation (Test)",
    description="트럭 적재 시뮬레이션 테스트 서버",
    version="0.1.0",
)

# CORS 설정 (React Native WebView 등)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(simulation_router)

# 정적 파일 (PLY 테스트용)
ASSETS_DIR = PROJECT_ROOT / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# 테스트 PLY 파일 (2.png 등에서 생성된 파일들)
TEST_PLY_DIR = PROJECT_ROOT / "tests" / "new_images_output"
if TEST_PLY_DIR.exists():
    app.mount("/test-ply", StaticFiles(directory=str(TEST_PLY_DIR)), name="test-ply")


@app.get("/")
async def root():
    """루트 - 시뮬레이터로 리다이렉트"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/simulation/")


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "ok", "service": "simulation-test"}


if __name__ == "__main__":
    print("=" * 50)
    print("트럭 적재 시뮬레이션 테스트 서버")
    print("=" * 50)
    print()
    print("브라우저에서 접속: http://localhost:8080")
    print()
    print("API 문서: http://localhost:8080/docs")
    print()
    print("종료: Ctrl+C")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
