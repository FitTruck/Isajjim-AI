#!/bin/bash
# =============================================================================
# NVIDIA Container Toolkit Installation Script
# Required for GPU passthrough in Docker containers
#
# Prerequisites: Docker must be installed first (run vm-setup-docker.sh)
#
# Usage: sudo bash scripts/vm-setup-nvidia-toolkit.sh
# =============================================================================

set -e

echo "=== Installing NVIDIA Container Toolkit ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please run vm-setup-docker.sh first."
    exit 1
fi

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA driver is not installed."
    exit 1
fi

echo "NVIDIA Driver detected:"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv

# Add NVIDIA repository GPG key
echo "Adding NVIDIA repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
echo "Restarting Docker..."
sudo systemctl restart docker

# Verify installation
echo ""
echo "=== Verification ==="
echo "Testing GPU access in Docker container..."

docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== NVIDIA Container Toolkit Installation Complete ==="
echo ""
echo "You can now run GPU-enabled containers with: docker run --gpus all ..."
