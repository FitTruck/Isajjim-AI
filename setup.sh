#!/bin/bash

# 1. Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "#######################################################################"
    echo "WARNING: You are running this script as a subshell (./setup.sh)."
    echo "To remain inside the conda environment after the script finishes,"
    echo "please run: source setup.sh"
    echo "#######################################################################"
    sleep 2
fi

# Exit on error
set -e

echo "--- 1. Handling Repository ---"
REPO_DIR="sam-3d-objects"
if [ -d "$REPO_DIR" ]; then
    echo "Directory '$REPO_DIR' exists. Updating..."
    cd "$REPO_DIR"
    git pull
else
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    cd "$REPO_DIR"
fi

echo "--- 2. Checking Miniconda ---"
CONDA_ROOT="$HOME/miniconda3"
if [ -d "$CONDA_ROOT" ]; then
    echo "Miniconda already installed at $CONDA_ROOT."
else
    echo "Installing Miniconda..."
    curl -fsSL -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$CONDA_ROOT"
    rm Miniconda3.sh
fi

# Initialize conda for the current shell session
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda init bash

echo "--- 3. Setting up Conda Environment ---"
ENV_NAME="sam3d-objects"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists. Updating..."
    conda env update -f environments/default.yml --prune
else
    conda env create -f environments/default.yml
fi

conda activate "$ENV_NAME"

echo "--- 4. Installing PyTorch & Dependencies ---"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

echo "--- 5. Installing Inference & Patching ---"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

if [ -f "./patching/hydra" ]; then
    chmod +x ./patching/hydra
    ./patching/hydra
fi

echo "--- 6. Downloading Model Checkpoints ---"
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
if [ ! -d "checkpoints/${TAG}" ]; then
    mkdir -p checkpoints
    huggingface-cli download \
      --repo-type model \
      --local-dir checkpoints/${TAG}-download \
      --max-workers 1 \
      facebook/sam-3d-objects

    mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
    rm -rf checkpoints/${TAG}-download
else
    echo "Checkpoints already present. Skipping download."
fi

echo "--- 7. Final Requirements (Root) ---"
cd ..
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt from $(pwd)..."

    pip install -r requirements.txt
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
else
    echo "No requirements.txt found in $(pwd)."
fi

echo "--- Setup Complete! ---"
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "SUCCESS: You are now active in the '$ENV_NAME' environment."
else
    echo "To activate the environment now, run: conda activate $ENV_NAME"
fi
