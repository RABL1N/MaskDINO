#!/bin/bash
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -J fungi_finetune_r50
#BSUB -o logs/fungi_r50_%J.out
#BSUB -e logs/fungi_r50_%J.err
#BSUB -u s234806@dtu.dk
#BSUB -B
#BSUB -N

set -e

PROJECT_DIR=/dtu/3d-imaging-center/projects/2026_QIM_MoldColonies_DTI/analysis/MaskDINO
cd $PROJECT_DIR

module load cuda/12.4
source .venv/bin/activate

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
    OUTPUT_DIR output/fungi_r50_48img_fulltune
