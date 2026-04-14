#!/bin/bash
# CD 계산용 PLY 생성 — 3종 순차 실행
# Usage: conda run -n sam3d-objects bash experiments/benchmark/run/run_cd_ply.sh
#
# 예상 시간:
#   Ours (o5):     ~5.5s × 500 = ~46min
#   Original (o1): ~22s × 500  = ~3h
#   Fast-SAM3D:    ~7s × 500   = ~1h
#   총합: ~5h

set -e
# benchmark/ 루트로 이동
cd "$(dirname "$0")/.."

SAMPLES=data/benchmark_samples.json
PLY_BASE=results/ply_for_cd

echo "=== [1/3] Ours (o5) — ~46min ==="
python scripts/workers/worker_ablation.py \
    --gpu 0 --config o5 \
    --samples "$SAMPLES" \
    --output results/cd_ours.csv \
    --save-ply-dir "$PLY_BASE/ours"

echo "=== [2/3] Original (o1 = gaussian-only SS=25/SLaT=25) — ~3h ==="
python scripts/workers/worker_ablation.py \
    --gpu 0 --config o1 \
    --samples "$SAMPLES" \
    --output results/cd_original.csv \
    --save-ply-dir "$PLY_BASE/original"

echo "=== [3/3] Fast-SAM3D — ~1h ==="
python scripts/workers/worker_fastsam3d.py \
    --gpu 0 \
    --samples "$SAMPLES" \
    --output results/cd_fastsam3d.csv \
    --save-ply-dir "$PLY_BASE/fastsam3d"

echo "=== Done! PLYs saved to $PLY_BASE ==="
echo "Run: python scripts/evaluate/compute_cd_500.py"
