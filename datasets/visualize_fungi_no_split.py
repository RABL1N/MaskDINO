"""
Visualize COCO instance segmentation annotations on top of images for a splitless dataset.

Draws each instance mask as a semi-transparent colour overlay with a contour outline.

Usage:
    python datasets/visualize_fungi_no_split.py --dataset datasets/fungi_no_split
    python datasets/visualize_fungi_no_split.py --dataset datasets/fungi_no_split --alpha 0.45
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

MASK_ALPHA = 0.40       # overlay opacity for real instances
CONTOUR_THICKNESS = 2

# Visually distinct palette (BGR)
_PALETTE = [
    (220,  60,  60), ( 60, 180,  60), ( 60,  60, 220), (220, 200,  60),
    ( 60, 200, 200), (200,  60, 200), (255, 140,   0), ( 50, 205,  50),
    (255,  20, 147), (  0, 191, 255), (255, 165,   0), (127,   0, 255),
    (  0, 255, 127), (255,  99,  71), ( 70, 130, 180), (210, 105,  30),
]


def _palette_color(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


def decode_rle(segmentation) -> np.ndarray:
    """Return a uint8 H×W binary mask from a COCO compressed-RLE dict."""
    rle = segmentation
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    return mask_utils.decode(rle).astype(np.uint8)


def draw_annotations(img: np.ndarray, anns: list, alpha: float) -> np.ndarray:
    """Overlay all annotations onto img (float copy), return uint8 result."""
    out = img.astype(np.float32)
    overlay = img.copy().astype(np.float32)

    # Sort annotations by area descending so smaller ones are drawn last/on top
    sorted_anns = sorted(anns, key=lambda a: a["area"], reverse=True)

    for idx, ann in enumerate(sorted_anns):
        mask = decode_rle(ann["segmentation"])
        color = _palette_color(idx)
        for c in range(3):
            overlay[:, :, c] = np.where(mask, color[c], overlay[:, :, c])

    # Blend overlay only where any mask is active
    all_masks = np.zeros(img.shape[:2], dtype=np.uint8)
    for ann in anns:
        all_masks = np.maximum(all_masks, decode_rle(ann["segmentation"]))

    blend_alpha = np.where(all_masks[:, :, None], alpha, 0.0).astype(np.float32)
    out = overlay * blend_alpha + out * (1.0 - blend_alpha)
    result = np.clip(out, 0, 255).astype(np.uint8)

    # Draw contours
    for idx, ann in enumerate(sorted_anns):
        mask = decode_rle(ann["segmentation"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, _palette_color(idx), CONTOUR_THICKNESS)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/fungi_no_split", help="Path to splitless dataset directory")
    parser.add_argument("--alpha", type=float, default=MASK_ALPHA, help="Mask overlay opacity (0–1)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        dataset_dir = Path(__file__).resolve().parent.parent.parent / args.dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found at {args.dataset}")

    ann_path = dataset_dir / "annotations" / "instances.json"
    img_dir  = dataset_dir / "images"
    out_dir  = dataset_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {ann_path}")

    with open(ann_path) as f:
        coco = json.load(f)

    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    print(f"Visualizing images from {img_dir} using {ann_path}")
    
    count = 0
    for img_info in sorted(coco["images"], key=lambda x: x["file_name"]):
        img_path = img_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        anns = ann_by_image.get(img_info["id"], [])
        result = draw_annotations(img, anns, args.alpha)

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {img_info['file_name']}  →  {out_path.relative_to(dataset_dir.parent)}  "
              f"({len(anns)} inst)")
        count += 1

    print(f"\nDone. Visualized {count} images and saved to {out_dir}")


if __name__ == "__main__":
    main()
