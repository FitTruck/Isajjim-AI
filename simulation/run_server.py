"""시뮬레이션 서버 실행 스크립트"""
import sys
from pathlib import Path

# 부모 디렉토리를 path에 추가 (simulation 패키지 임포트용)
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from simulation.routes import router

app = FastAPI(title="Truck Loading Simulator")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)

# 정적 파일 서빙
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/simulation/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Truck Loading Simulator API", "docs": "/docs"}


if __name__ == "__main__":
    print("=" * 50)
    print("트럭 적재 시뮬레이션 테스트 서버")
    print("=" * 50)
    print()
    print("브라우저에서 접속: http://localhost:8080/simulation/simulator")
    print()
    print("API 문서: http://localhost:8080/docs")
    print()
    print("종료: Ctrl+C")
    print("=" * 50)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8080)
