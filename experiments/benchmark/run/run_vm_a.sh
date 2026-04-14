#!/bin/bash
# VM-A 실행 스크립트: R1 (Original/Baseline) → R6 (Ours/O5)
# Usage: bash experiments/benchmark/run/run_vm_a.sh

set -e
# benchmark/ 루트로 이동 (data/, scripts/, results/ 경로 기준)
cd "$(dirname "$0")/.."

echo "=========================================="
echo "VM-A: R1 (Original/Baseline) → R6 (Ours/O5)"
echo "=========================================="

# R1: Original default = Ablation Baseline
echo "[$(date)] Starting R1: Original/Baseline (~10.4h)"
conda run -n sam3d-objects python scripts/workers/worker_ablation.py \
    --gpu 0 --config baseline \
    --samples data/benchmark_samples.json --output results/original.csv
echo "[$(date)] R1 complete!"

# R6: Ours = Ablation O5
echo "[$(date)] Starting R6: Ours/O5 (~1.8h)"
conda run -n sam3d-objects python scripts/workers/worker_ablation.py \
    --gpu 0 --config o5 \
    --samples data/benchmark_samples.json --output results/ours.csv
echo "[$(date)] R6 complete!"

echo "=========================================="
echo "[$(date)] VM-A all done!"
echo "=========================================="
