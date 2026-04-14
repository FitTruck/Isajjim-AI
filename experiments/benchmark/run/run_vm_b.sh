#!/bin/bash
# VM-B 실행 스크립트: R3 (O1) → R4 (O2) → R5 (O4)
# Usage: bash experiments/benchmark/run/run_vm_b.sh

set -e
# benchmark/ 루트로 이동 (data/, scripts/, results/ 경로 기준)
cd "$(dirname "$0")/.."

echo "=========================================="
echo "VM-B: R3 (O1) → R4 (O2) → R5 (O4)"
echo "=========================================="

# R3: +O1 Gaussian-Only (~3.2h)
echo "[$(date)] Starting R3: +O1 Gaussian-Only"
conda run -n sam3d-objects python scripts/workers/worker_ablation.py \
    --gpu 0 --config o1 \
    --samples data/benchmark_samples.json --output results/ablation_o1.csv
echo "[$(date)] R3 complete!"

# R4: +O2 VRAM Unload (~3.2h)
echo "[$(date)] Starting R4: +O2 VRAM Unload"
conda run -n sam3d-objects python scripts/workers/worker_ablation.py \
    --gpu 0 --config o2 \
    --samples data/benchmark_samples.json --output results/ablation_o2.csv
echo "[$(date)] R4 complete!"

# R5: +O4 Steps Reduction (~2.1h)
echo "[$(date)] Starting R5: +O4 Steps 14/4"
conda run -n sam3d-objects python scripts/workers/worker_ablation.py \
    --gpu 0 --config o4 \
    --samples data/benchmark_samples.json --output results/ablation_o4.csv
echo "[$(date)] R5 complete!"

echo "=========================================="
echo "[$(date)] VM-B all done!"
echo "=========================================="
