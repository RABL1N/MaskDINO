"""
Visualize COCO instance segmentation annotations on top of images.

Draws each instance mask as a semi-transparent colour overlay with a contour outline.
Ignore-region annotations (iscrowd=1) are drawn in red with a dashed outline.

Usage:
    python datasets/visualize_fungi.py --dataset datasets/fungi
    python datasets/visualize_fungi.py --dataset datasets/fungi --splits val
    python datasets/visualize_fungi.py --dataset datasets/fungi --alpha 0.45
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

MASK_ALPHA = 0.40       # overlay opacity for real instances
IGNORE_ALPHA = 0.30     # overlay opacity for ignore regions
CONTOUR_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Visually distinct palette (BGR) — cycles if more instances than colours
_PALETTE = [
    (220,  60,  60), ( 60, 180,  60), ( 60,  60, 220), (220, 200,  60),
    ( 60, 200, 200), (200,  60, 200), (255, 140,   0), ( 50, 205,  50),
    (255,  20, 147), (  0, 191, 255), (255, 165,   0), (127,   0, 255),
    (  0, 255, 127), (255,  99,  71), ( 70, 130, 180), (210, 105,  30),
]
_IGNORE_COLOR = (0, 0, 220)   # red (BGR) for iscrowd=1


def _palette_color(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


def decode_rle(segmentation) -> np.ndarray:
    """Return a uint8 H×W binary mask from a COCO compressed-RLE dict."""
    rle = segmentation
    if isinstance(rle["counts"], list):
        # uncompressed RLE — shouldn't happen after prepare_fungi, but handle anyway
        rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    return mask_utils.decode(rle).astype(np.uint8)


def draw_annotations(img: np.ndarray, anns: list, alpha: float) -> np.ndarray:
    """Overlay all annotations onto img (float copy), return uint8 result."""
    out = img.astype(np.float32)
    overlay = img.copy().astype(np.float32)

    real = sorted([a for a in anns if a["iscrowd"] == 0], key=lambda a: a["area"], reverse=True)
    ignore = [a for a in anns if a["iscrowd"] == 1]

    for idx, ann in enumerate(real):
        mask = decode_rle(ann["segmentation"])
        color = _palette_color(idx)
        for c in range(3):
            overlay[:, :, c] = np.where(mask, color[c], overlay[:, :, c])

    for ann in ignore:
        mask = decode_rle(ann["segmentation"])
        for c in range(3):
            overlay[:, :, c] = np.where(mask, _IGNORE_COLOR[c], overlay[:, :, c])

    # Blend overlay only where any mask is active
    all_masks = np.zeros(img.shape[:2], dtype=np.uint8)
    for ann in anns:
        all_masks = np.maximum(all_masks, decode_rle(ann["segmentation"]))

    blend_alpha = np.where(all_masks[:, :, None], alpha, 0.0).astype(np.float32)
    out = overlay * blend_alpha + out * (1.0 - blend_alpha)
    result = np.clip(out, 0, 255).astype(np.uint8)

    # Contours for real instances
    for idx, ann in enumerate(real):
        mask = decode_rle(ann["segmentation"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, _palette_color(idx), CONTOUR_THICKNESS)

    # Dashed-style contours for ignore regions (drawn as thin solid lines, different colour)
    for ann in ignore:
        mask = decode_rle(ann["segmentation"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, _IGNORE_COLOR, CONTOUR_THICKNESS)

    # Annotation count label (top-left corner)
    label = f"{len(real)} inst" + (f"  {len(ignore)} ign" if ignore else "")
    cv2.putText(result, label, (12, 32), FONT, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(result, label, (12, 32), FONT, 0.9, (20, 20, 20),    1, cv2.LINE_AA)

    return result


def process_split(dataset_dir: Path, split: str, alpha: float) -> None:
    ann_path = dataset_dir / "annotations" / f"instances_{split}.json"
    img_dir  = dataset_dir / split
    out_dir  = dataset_dir / "labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        coco = json.load(f)

    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_info in sorted(coco["images"], key=lambda x: x["file_name"]):
        img_path = img_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        anns = ann_by_image.get(img_info["id"], [])
        result = draw_annotations(img, anns, alpha)

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {split}/{img_info['file_name']}  →  {out_path.relative_to(dataset_dir.parent)}  "
              f"({len([a for a in anns if not a['iscrowd']])} inst)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/fungi", help="Path to fungi dataset directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--alpha", type=float, default=MASK_ALPHA, help="Mask overlay opacity (0–1)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    for split in args.splits:
        print(f"\n[{split}]")
        process_split(dataset_dir, split, args.alpha)

    print("\nDone.")


if __name__ == "__main__":
    main()
