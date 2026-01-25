#!/bin/bash
# =============================================================================
# SAM3D API Docker Entrypoint
#
# This script:
# 1. Verifies volume mounts exist
# 2. Creates symlinks for hardcoded paths in the application
# 3. Starts the API server
# =============================================================================

set -e

echo "=== SAM3D API Container Startup ==="
echo "Time: $(date)"

# =============================================================================
# Verify Data Volumes
# =============================================================================

echo "Checking data volumes..."

# Check sam-3d-objects checkpoints
if [ ! -d "/data/sam3d/sam-3d-objects/checkpoints/hf" ]; then
    echo "ERROR: sam-3d-objects checkpoints not found!"
    echo "Expected path: /data/sam3d/sam-3d-objects/checkpoints/hf"
    echo ""
    echo "Please run on host:"
    echo "  sudo bash scripts/vm-setup-data.sh"
    echo ""
    echo "Or ensure volume is mounted correctly in docker-compose.yml"
    exit 1
fi

# Check YOLO model (YOLOE-seg)
if [ ! -f "/data/sam3d/models/yoloe-26x-seg.pt" ]; then
    echo "ERROR: YOLO model not found!"
    echo "Expected path: /data/sam3d/models/yoloe-26x-seg.pt"
    exit 1
fi

# Check sam-3d-objects notebook (for inference imports)
if [ ! -d "/data/sam3d/sam-3d-objects/notebook" ]; then
    echo "ERROR: sam-3d-objects notebook directory not found!"
    echo "Expected path: /data/sam3d/sam-3d-objects/notebook"
    exit 1
fi

echo "All required data volumes verified."

# =============================================================================
# Create Symlinks for Hardcoded Paths
# =============================================================================

echo "Creating symlinks..."
cd /app

# Remove existing symlinks/directories if they exist
rm -rf ./sam-3d-objects 2>/dev/null || true
rm -f ./yoloe-26x-seg.pt 2>/dev/null || true

# Create symlinks
# These paths are hardcoded in api/app.py and persistent_3d_worker.py
ln -sf /data/sam3d/sam-3d-objects ./sam-3d-objects
ln -sf /data/sam3d/models/yoloe-26x-seg.pt ./yoloe-26x-seg.pt

# Create assets directory if not exists (for generated outputs)
mkdir -p ./assets

# Verify symlinks
echo "Symlinks created:"
ls -la ./sam-3d-objects ./yoloe-26x-seg.pt

# Verify checkpoint access
echo ""
echo "Verifying checkpoint access..."
ls ./sam-3d-objects/checkpoints/hf/ | head -5

# =============================================================================
# GPU Information
# =============================================================================

echo ""
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not available (GPU may not be accessible)"
fi

# =============================================================================
# Start Application
# =============================================================================

echo ""
echo "=== Starting SAM3D API ==="
echo "Command: $@"

# Virtual environment is already in PATH via Dockerfile ENV
# Execute the passed command (default: uvicorn)
exec "$@"
