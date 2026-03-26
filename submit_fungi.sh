#!/bin/bash
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -J fungi_finetune
#BSUB -o logs/fungi_%J.out
#BSUB -e logs/fungi_%J.err
#BSUB -u s234806@dtu.dk
#BSUB -B                      # email when job starts
#BSUB -N                      # email when job ends (success or failure)

set -e

PROJECT_DIR=/dtu/3d-imaging-center/projects/2026_QIM_MoldColonies_DTI/analysis/MaskDINO
cd $PROJECT_DIR

# Load CUDA module
module load cuda/12.4

# Activate virtual environment
source .venv/bin/activate

# Compile CUDA kernel (checks for any compiled .so regardless of Python version)
if [ -z "$(find maskdino/modeling/pixel_decoder/ops/build -name 'MultiScaleDeformableAttention*.so' 2>/dev/null)" ]; then
    echo "Compiling CUDA ops..."
    export CUDA_HOME=/appl/cuda/12.4.0
    cd maskdino/modeling/pixel_decoder/ops
    sh make.sh
    cd $PROJECT_DIR
else
    echo "CUDA ops already compiled, skipping."
fi

mkdir -p logs

python train_net.py --num-gpus 1 \
    --config-file configs/fungi/maskdino_R50_finetune_fungi.yaml \
    MODEL.WEIGHTS checkpoints/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth \
    OUTPUT_DIR output/fungi_finetune
