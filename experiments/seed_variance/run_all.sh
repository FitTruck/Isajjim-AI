#!/bin/bash
# ============================================================================
# Seed Variance Experiment - Full Pipeline
# ============================================================================
#
# Usage:
#   bash run_all.sh              # 전체 실행 (다운로드 → 샘플링 → Phase 1 → Phase 2 → 분석)
#   bash run_all.sh --phase 1    # Phase 1만 실행 (Original SAM-3D, 250 runs)
#   bash run_all.sh --phase 2    # Phase 2만 실행 (Optimized SAM-3D, 50 runs)
#   bash run_all.sh --analyze    # 분석만 실행
#
# 예상 시간 (2× L4 GPU):
#   Phase 1: ~5시간 (250회 × ~150초 / 2 GPUs)
#   Phase 2: ~6분 (50회 × ~13초 / 2 GPUs)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
PHASE="all"
ANALYZE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --analyze)
            ANALYZE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Seed Variance Experiment"
echo "============================================"
echo "Phase: $PHASE"
echo "Script dir: $SCRIPT_DIR"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "sam3d"; then
        conda activate sam3d
        echo "Activated conda env: sam3d"
    fi
fi

# ============================================================================
# Step 0: Data preparation
# ============================================================================
if [ "$ANALYZE_ONLY" = false ]; then
    echo ""
    echo "Step 0: Data preparation (Pix3D download + sampling)"
    echo "--------------------------------------------"

    if [ ! -f "data/selected_samples.json" ]; then
        # Check if zip is downloaded
        if [ ! -f "data/pix3d.zip" ] && [ ! -d "data/pix3d" ]; then
            echo "Downloading Pix3D dataset..."
            mkdir -p data
            wget -q --show-progress -O data/pix3d.zip 'http://pix3d.csail.mit.edu/data/pix3d.zip'
        fi

        # Extract if needed
        if [ -f "data/pix3d.zip" ] && [ ! -d "data/pix3d" ]; then
            echo "Extracting Pix3D (img + mask only, skipping 3D models)..."
            python download_pix3d.py
        elif [ -d "data/pix3d" ] && [ ! -f "data/selected_samples.json" ]; then
            echo "Sampling from existing Pix3D data..."
            python download_pix3d.py
        fi
    else
        echo "Samples already selected: data/selected_samples.json"
    fi
fi

# ============================================================================
# Step 1 & 2: Run experiment
# ============================================================================
if [ "$ANALYZE_ONLY" = false ]; then
    echo ""
    echo "Step 1-2: Running experiment (Phase $PHASE)"
    echo "--------------------------------------------"

    GPU_IDS="0,1"

    if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
        echo ""
        echo ">>> Phase 1: Original SAM-3D (250 runs)"
        echo ">>> Estimated time: ~5 hours on 2× L4 GPU"
        echo ">>> Started at: $(date)"
        python run_experiment.py --phase 1 --gpu-ids "$GPU_IDS"
        echo ">>> Phase 1 completed at: $(date)"
    fi

    if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
        echo ""
        echo ">>> Phase 2: Optimized SAM-3D (50 runs)"
        echo ">>> Estimated time: ~6 minutes on 2× L4 GPU"
        echo ">>> Started at: $(date)"
        python run_experiment.py --phase 2 --gpu-ids "$GPU_IDS"
        echo ">>> Phase 2 completed at: $(date)"
    fi
fi

# ============================================================================
# Step 3: Analysis
# ============================================================================
echo ""
echo "Step 3: Analysis"
echo "--------------------------------------------"
python analyze_results.py

echo ""
echo "============================================"
echo "Experiment complete!"
echo "Results: results/analysis_report.md"
echo "Summary: results/summary.json"
echo "============================================"
