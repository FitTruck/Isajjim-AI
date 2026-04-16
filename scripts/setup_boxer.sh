#!/bin/bash
# Boxer Setup Script
# Meta의 Boxer 모델 클론 및 체크포인트 다운로드
#
# Usage: bash scripts/setup_boxer.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOXER_DIR="${REPO_ROOT}/boxer"

echo "=== Boxer Setup ==="

# 1. Clone Boxer repository
if [ -d "$BOXER_DIR" ]; then
    echo "[INFO] Boxer directory already exists at ${BOXER_DIR}"
    echo "[INFO] Pulling latest changes..."
    cd "$BOXER_DIR" && git pull && cd "$REPO_ROOT"
else
    echo "[INFO] Cloning Boxer repository..."
    git clone https://github.com/facebookresearch/boxer.git "$BOXER_DIR"
fi

# 2. Install Boxer as editable package
echo "[INFO] Installing Boxer package..."
pip install -e "$BOXER_DIR"

# 3. Download checkpoints
echo "[INFO] Downloading Boxer checkpoints..."
cd "$BOXER_DIR" && bash scripts/download_ckpts.sh && cd "$REPO_ROOT"

echo "=== Boxer Setup Complete ==="
echo "[INFO] Boxer installed at: ${BOXER_DIR}"
echo "[INFO] Checkpoints at: ${BOXER_DIR}/checkpoints/"
