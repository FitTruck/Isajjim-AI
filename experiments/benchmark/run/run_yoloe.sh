#!/bin/bash
# YOLOE Inference Benchmark 실행 스크립트
# SAM3D와 동일한 500개 Pix3D 샘플 사용
#
# Usage: bash experiments/benchmark/run/run_yoloe.sh

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(cd ../.. && pwd)"

echo "=========================================="
echo "YOLOE Inference Benchmark (500 samples)"
echo "=========================================="

# 1. ultralytics 설치 확인
echo "[$(date)] Checking ultralytics..."
conda run -n sam3d-objects python -c "from ultralytics import YOLOE" 2>/dev/null \
    || { echo "Installing ultralytics..."; conda run -n sam3d-objects pip install ultralytics; }

# 2. YOLOE 모델 다운로드 확인
MODEL_PATH="${PROJECT_ROOT}/yoloe-26x-seg.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[$(date)] Downloading yoloe-26x-seg.pt..."
    conda run -n sam3d-objects python -c "
from ultralytics import YOLOE
import os
os.chdir('${PROJECT_ROOT}')
YOLOE('yoloe-26x-seg.pt')
print('Download complete!')
"
fi
echo "[$(date)] Model ready: $MODEL_PATH"

# 3. 벤치마크 실행
echo "[$(date)] Starting YOLOE benchmark"
conda run -n sam3d-objects python scripts/workers/worker_yoloe.py \
    --gpu 0 \
    --samples data/benchmark_samples.json \
    --output results/yoloe.csv
echo "[$(date)] YOLOE benchmark complete!"

echo "=========================================="
echo "[$(date)] Done! Results: results/yoloe.csv"
echo "=========================================="
