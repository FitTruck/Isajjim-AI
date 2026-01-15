#!/bin/bash
# =============================================================================
# Data Directory Setup Script for SAM3D API
# Sets up persistent data directory with models and checkpoints
#
# This script:
# 1. Creates /data/sam3d directory structure
# 2. Clones sam-3d-objects repository
# 3. Downloads HuggingFace checkpoints (~12GB)
# 4. Downloads YOLO model (~92MB)
#
# Usage: sudo bash scripts/vm-setup-data.sh
# =============================================================================

set -e

DATA_DIR="/data/sam3d"
CURRENT_USER=${SUDO_USER:-$USER}

echo "=== Setting up SAM3D data directory ==="
echo "Data directory: $DATA_DIR"
echo "Owner: $CURRENT_USER"

# Create directory structure
echo "Creating directories..."
sudo mkdir -p $DATA_DIR/{sam-3d-objects,models,huggingface,torch,assets}

# Set ownership
sudo chown -R $CURRENT_USER:$CURRENT_USER $DATA_DIR

# Clone sam-3d-objects repository
if [ ! -d "$DATA_DIR/sam-3d-objects/.git" ]; then
    echo ""
    echo "=== Cloning sam-3d-objects repository ==="
    git clone https://github.com/facebookresearch/sam-3d-objects.git $DATA_DIR/sam-3d-objects
else
    echo "sam-3d-objects already cloned, pulling latest..."
    cd $DATA_DIR/sam-3d-objects && git pull
fi

# Download HuggingFace checkpoints
if [ ! -d "$DATA_DIR/sam-3d-objects/checkpoints/hf" ] || [ -z "$(ls -A $DATA_DIR/sam-3d-objects/checkpoints/hf 2>/dev/null)" ]; then
    echo ""
    echo "=== Downloading HuggingFace checkpoints (~12GB) ==="
    echo "This may take 10-30 minutes depending on network speed..."

    # Install huggingface-cli if not available
    if ! command -v huggingface-cli &> /dev/null; then
        pip install 'huggingface-hub[cli]<1.0'
    fi

    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        echo "Note: HF_TOKEN not set. If download fails, run: huggingface-cli login"
    fi

    cd $DATA_DIR/sam-3d-objects
    mkdir -p checkpoints

    # Download from HuggingFace
    huggingface-cli download \
        --repo-type model \
        --local-dir checkpoints/hf-download \
        --max-workers 4 \
        facebook/sam-3d-objects

    # Move to correct location
    if [ -d "checkpoints/hf-download/checkpoints" ]; then
        mv checkpoints/hf-download/checkpoints checkpoints/hf
        rm -rf checkpoints/hf-download
    else
        mv checkpoints/hf-download checkpoints/hf
    fi

    echo "Checkpoints downloaded successfully!"
else
    echo "Checkpoints already exist at $DATA_DIR/sam-3d-objects/checkpoints/hf"
fi

# Download YOLO model
if [ ! -f "$DATA_DIR/models/yolov8l-world.pt" ]; then
    echo ""
    echo "=== Downloading YOLO model (~92MB) ==="

    cd $DATA_DIR/models
    wget -q --show-progress \
        https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt

    echo "YOLO model downloaded successfully!"
else
    echo "YOLO model already exists at $DATA_DIR/models/yolov8l-world.pt"
fi

# Summary
echo ""
echo "=== Data Setup Complete ==="
echo ""
echo "Directory structure:"
du -sh $DATA_DIR/*
echo ""
echo "Checkpoints:"
ls -la $DATA_DIR/sam-3d-objects/checkpoints/hf/ 2>/dev/null | head -10
echo ""
echo "Models:"
ls -la $DATA_DIR/models/
echo ""
echo "Total size:"
du -sh $DATA_DIR

echo ""
echo "=== Ready for Docker deployment ==="
