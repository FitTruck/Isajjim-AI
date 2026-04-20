#!/bin/bash
# ABO 정확도 실험 end-to-end 실행 스크립트
#
# 단계:
#   1) 메타데이터 다운로드
#   2) KB 매핑 생성
#   3) 500 샘플 층화 추출
#   4) 이미지 + 메시 다운로드
#   5) YOLOE-seg 마스크 전처리
#   6) Worker 샘플 포맷 변환
#   7) SAM-3D 2 설정 실행 (Proposed, Baseline-B)
#   8) 정확도 집계 (표 2, 표 3)
#
# Usage:
#   bash experiments/benchmark/run/run_abo_accuracy.sh [--skip-download] [--gpu 0]

set -euo pipefail

GPU=0
SKIP_DOWNLOAD=0
SKIP_PREP=0
RUN_ONLY=""  # proposed | baseline_b | (empty=both)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-prep) SKIP_PREP=1; shift ;;
        --run-only) RUN_ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "=== ABO Accuracy Pipeline ==="
echo "ROOT=$ROOT GPU=$GPU SKIP_DOWNLOAD=$SKIP_DOWNLOAD SKIP_PREP=$SKIP_PREP RUN_ONLY=$RUN_ONLY"

# conda env activation
if ! conda info --envs >/dev/null 2>&1; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate sam3d-objects

RESULTS_DIR="$ROOT/experiments/benchmark/results"
mkdir -p "$RESULTS_DIR"

# ─── Phase 1: 데이터 준비 ───
if [ "$SKIP_PREP" -eq 0 ]; then
    echo -e "\n[1/6] ABO 메타데이터 다운로드..."
    python experiments/benchmark/scripts/abo/1_fetch_metadata.py

    echo -e "\n[2/6] KB 매핑..."
    python experiments/benchmark/scripts/abo/2_build_kb_mapping.py

    echo -e "\n[3/6] 500 샘플 층화 추출..."
    python experiments/benchmark/scripts/abo/3_select_500.py

    if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
        echo -e "\n[4/6] 이미지 + 메시 다운로드..."
        python experiments/benchmark/scripts/abo/4_download_assets.py
    fi

    echo -e "\n[5/6] YOLOE-seg 마스크 전처리..."
    CUDA_VISIBLE_DEVICES=$GPU python experiments/benchmark/scripts/abo/5_yoloe_preprocess.py

    echo -e "\n[6/6] Worker 샘플 변환..."
    python experiments/benchmark/scripts/abo/6_prepare_worker_samples.py
fi

WORKER_JSON="$ROOT/experiments/benchmark/data/abo/abo_worker_samples.json"
if [ ! -s "$WORKER_JSON" ]; then
    echo "ERROR: $WORKER_JSON 이 비어있거나 존재하지 않음. --skip-prep 제거 후 재실행." >&2
    exit 1
fi

N_SAMPLES=$(python -c "import json; print(len(json.load(open('$WORKER_JSON'))))")
echo "Worker 샘플 수: $N_SAMPLES"

# ─── Phase 2: SAM-3D 실행 ───
run_config() {
    local CFG=$1
    local OUT="$RESULTS_DIR/${CFG}.csv"
    echo -e "\n=== SAM-3D 실행: $CFG → $OUT ==="
    python experiments/benchmark/scripts/workers/worker_ablation.py \
        --gpu "$GPU" \
        --config "$CFG" \
        --samples "$WORKER_JSON" \
        --output "$OUT"
}

if [ -z "$RUN_ONLY" ] || [ "$RUN_ONLY" = "proposed" ]; then
    run_config abo_proposed
fi
if [ -z "$RUN_ONLY" ] || [ "$RUN_ONLY" = "baseline_b" ]; then
    run_config abo_baseline_b
fi

# ─── Phase 3: 집계 ───
echo -e "\n=== 정확도 집계 (표 2, 표 3) ==="
python experiments/benchmark/scripts/evaluate/compute_abo_accuracy.py \
    --proposed-csv "$RESULTS_DIR/abo_proposed.csv" \
    --baseline-csv "$RESULTS_DIR/abo_baseline_b.csv" \
    --output-dir "$RESULTS_DIR"

echo -e "\n=== 완료 ==="
echo "결과:"
echo "  - $RESULTS_DIR/abo_accuracy_summary.csv  (표 2)"
echo "  - $RESULTS_DIR/abo_accuracy_by_category.csv  (표 3)"
echo "  - $RESULTS_DIR/abo_tables.md  (마크다운 붙여넣기용)"
