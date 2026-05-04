"""
Plot input | prediction | label side-by-side for every val image.

Usage:
    python datasets/plot_val_comparison.py \
        --dataset datasets/fungi_01_05_26 \
        --preds pred/fungi_r50_48img_lr1e6/model_0008499/val \
        --output val_comparison.jpg
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/fungi_01_05_26")
    parser.add_argument("--preds", default="pred/fungi_r50_48img_lr1e6/model_0008499/val")
    parser.add_argument("--output", default="val_comparison.jpg")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    pred_dir    = Path(args.preds)
    val_dir     = dataset_dir / "val"
    label_dir   = dataset_dir / "labels" / "val"

    names = sorted(p.name for p in val_dir.glob("*.jpg"))

    rows_per_grid = 3
    chunks = [names[i:i + rows_per_grid] for i in range(0, len(names), rows_per_grid)]

    out_stem = Path(args.output).stem
    out_suffix = Path(args.output).suffix
    out_dir = Path(args.output).parent
    col_titles = ["Input", "Prediction", "Label"]

    for grid_idx, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 3, figsize=(18, 6 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=16, fontweight="bold", pad=8)

        for row, name in enumerate(chunk):
            stem = Path(name).stem
            input_img = load_rgb(val_dir / name)
            pred_img  = load_rgb(pred_dir / name)
            label_img = load_rgb(label_dir / name)

            for col, img in enumerate([input_img, pred_img, label_img]):
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

            axes[row, 0].set_ylabel(stem, fontsize=9, rotation=0, labelpad=160, va="center")

        plt.tight_layout()
        out_path = out_dir / f"{out_stem}_{grid_idx}{out_suffix}"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
