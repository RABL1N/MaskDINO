"""
Register the fungi instance segmentation dataset with Detectron2.

Import this module before calling train_net.py, or add it to maskdino/__init__.py.
The dataset must first be exported via:
    python datasets/prepare_fungi.py --db ... --images ... --out datasets/fungi
"""

import os
from detectron2.data.datasets import register_coco_instances

_FUNGI_SPLITS = {
    "fungi_train": ("fungi/train", "fungi/annotations/instances_train.json"),
    "fungi_val":   ("fungi/val",   "fungi/annotations/instances_val.json"),
}


def register_fungi(root="datasets"):
    for name, (image_dir, json_file) in _FUNGI_SPLITS.items():
        register_coco_instances(
            name,
            {},
            os.path.join(root, json_file),
            os.path.join(root, image_dir),
        )


# Auto-register when this module is imported
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_fungi(_root)
