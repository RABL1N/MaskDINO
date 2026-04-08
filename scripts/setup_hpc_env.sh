#!/bin/bash
# Run this once on the HPC login node to set up the Python environment.
# Usage: bash setup_hpc_env.sh

set -e

PROJECT_DIR=/dtu/3d-imaging-center/projects/2026_QIM_MoldColonies_DTI/analysis/MaskDINO
cd $PROJECT_DIR

# Create virtual environment (no GPU needed for this)
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Detectron2 (install from source — no-build-isolation so torch is visible during build)
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# MaskDINO dependencies
pip install -r requirements.txt

# NOTE: The CUDA kernel (deformable attention) will be compiled automatically
# on first run of the batch job, which runs on a GPU node with CUDA available.

echo "Python environment setup complete. Submit the job with: bsub < submit_fungi.sh"
