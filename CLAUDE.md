# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python environment**: Use the `.venv` virtualenv in the project root.
```bash
source .venv/bin/activate
```

**CUDA kernel** (must be compiled before first use):
```bash
export CUDA_HOME=/usr/local/cuda
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd -
```

**Key dependencies**: PyTorch 2.x, Detectron2 (installed from source), timm.

## Common Commands

**Inference (demo)**:
```bash
python demo/demo.py \
  --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
  --input path/to/image.jpg \
  --output output/ \
  --opts MODEL.WEIGHTS checkpoints/<model>.pth \
         MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD 0.5
```

**Training**:
```bash
python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
  MODEL.WEIGHTS /path/to/pretrained.pth
```

**Evaluation only**:
```bash
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
  MODEL.WEIGHTS checkpoints/<model>.pth
```

## Architecture Overview

MaskDINO is a unified Transformer model for instance, panoptic, and semantic segmentation. The main model class is `maskdino/maskdino.py`.

**Forward pass flow**:
1. **Backbone** (`maskdino/modeling/backbone/`) — ResNet or Swin extracts multi-scale features (res2–res5)
2. **Pixel Decoder** (`maskdino/modeling/pixel_decoder/maskdino_encoder.py`) — Multi-scale deformable attention encoder. Outputs `mask_features` (1/4 resolution, used for mask generation) and `multi_scale_features` (inputs to decoder)
3. **Transformer Decoder** (`maskdino/modeling/transformer_decoder/`) — DINO-style decoder with 300 object queries. Two-stage: encoder first generates box proposals, decoder refines them. Uses denoising (DN) training during training only
4. **Inference heads** in `maskdino.py`: `instance_inference()`, `panoptic_inference()`, `semantic_inference()` — selected by config flags `MODEL.MaskDINO.TEST.INSTANCE_ON/PANOPTIC_ON/SEMANTIC_ON`

**Loss** (`maskdino/modeling/criterion.py`): Hungarian matching + focal classification loss + BCE/Dice mask loss + GIoU box loss, applied at each decoder layer (deep supervision).

## Config System

Configs are layered YAML files in `configs/`. Task-specific configs inherit from base configs (e.g. `Base-COCO-InstanceSegmentation.yaml`). Runtime overrides go after `--opts KEY VALUE`.

Key config knobs:
- `MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD` — confidence threshold for filtering predictions (0.25 default; `--confidence-threshold` in demo.py is a no-op bug)
- `MODEL.MaskDINO.TEST.INSTANCE_ON/PANOPTIC_ON/SEMANTIC_ON` — which inference head to use
- `MODEL.MaskDINO.DN` — denoising mode: `"no"`, `"standard"` (boxes), `"seg"` (boxes+masks+labels)
- `MODEL.MaskDINO.TWO_STAGE` — enables encoder-based query initialization
- `MODEL.MaskDINO.INITIALIZE_BOX_TYPE` — `"bitmask"` = maskenhanced box init (used by maskenhanced checkpoints)

## Checkpoints

Pre-trained checkpoints are stored in `checkpoints/`. Model `.pth` files are gitignored. The naming convention encodes architecture: `maskdino_<backbone>_<epochs>ep_<queries>q_hid<encoder_ffn_dim>_<scales>sd<downsample>_<task>_<variant>_<metric>.pth`.

## Known Issues / Local Fixes

- **`ms_deform_attn_cuda.cu`**: `value.type()` changed to `value.scalar_type()` on lines 69 and 139 to fix compilation with PyTorch 2.x
- **`demo/predictor.py`**: Added score filtering for instance segmentation (upstream bug — `--confidence-threshold` flag was parsed but never applied). Now reads threshold from `MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD`
