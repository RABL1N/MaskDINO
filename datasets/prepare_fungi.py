"""
Export SAM3 annotations to COCO instance segmentation format for MaskDINO fine-tuning.

All segmented instances are treated as a single class ("fungus") — no species
classification. Species labels are irrelevant; any instance SAM3 segmented is
a real fungal colony and counts as a positive training target.

Usage:
    python datasets/prepare_fungi.py \
        --db /path/to/sam3.db \
        --images /path/to/session_images \
        --out datasets/fungi \
        --val-fraction 0.2 \
        --exclude session_20260220_094350

Output:
    datasets/fungi/
        train/          # training images
        val/            # validation images
        annotations/
            instances_train.json
            instances_val.json

Category handling:
  - Archived sessions or --exclude list   → excluded entirely
  - Any segmented instance (any category) → iscrowd=0, category_id=1 ("fungus")
  - Ignore (99)                           → dropped entirely
"""

import argparse
import json
import random
import shutil
import sqlite3
from pathlib import Path

import numpy as np

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("WARNING: pycocotools not found. Install with: pip install pycocotools")


IGNORE_CATEGORY_ID = 99

SINGLE_CLASS = [{"id": 1, "name": "fungus", "supercategory": "fungus"}]


def uncomp_rle_to_coco_rle(rle_dict):
    """Convert SAM3 uncompressed RLE to COCO compressed RLE."""
    h, w = rle_dict["size"]
    counts = rle_dict["counts"]

    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:  # odd indices = foreground pixels
            flat[pos : pos + c] = 1
        pos += c

    binary_mask = flat.reshape((h, w), order="F")  # COCO uses Fortran (column-major) order
    encoded = mask_utils.encode(np.asfortranarray(binary_mask))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def load_sessions_and_instances(db_path, exclude_ids):
    """Load non-archived sessions (minus --exclude list) from the SQLite DB.

    All instances are included as targets except those with category Ignore (99).
    Returns only sessions that have at least one instance.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        "SELECT id, image_filename, width, height FROM sessions WHERE is_archived = 0"
    )
    sessions = {
        row[0]: {"id": row[0], "filename": row[1], "width": row[2], "height": row[3], "instances": []}
        for row in cur.fetchall()
        if row[0] not in exclude_ids
    }

    cur.execute(
        "SELECT session_id, instance_index, mask_rle, score FROM instances "
        "WHERE category_id != ? OR category_id IS NULL",
        (IGNORE_CATEGORY_ID,),
    )
    for session_id, idx, mask_rle_str, score in cur.fetchall():
        if session_id not in sessions:
            continue
        sessions[session_id]["instances"].append({
            "instance_index": idx,
            "mask_rle": json.loads(mask_rle_str),
            "score": score,
        })

    con.close()
    return {sid: s for sid, s in sessions.items() if s["instances"]}


def build_coco_json(sessions):
    """Build a COCO-format dict. All instances become category_id=1, iscrowd=0."""
    images = []
    annotations = []
    ann_id = 1

    for img_id, (session_id, session) in enumerate(sessions.items(), start=1):
        images.append({
            "id": img_id,
            "file_name": session_id + ".jpg",
            "width": session["width"],
            "height": session["height"],
        })

        for inst in session["instances"]:
            coco_rle = uncomp_rle_to_coco_rle(inst["mask_rle"])
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": coco_rle,
                "area": int(mask_utils.area(coco_rle)),
                "bbox": mask_utils.toBbox(coco_rle).tolist(),
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to sam3.db")
    parser.add_argument("--images", required=True, help="Path to session_images directory")
    parser.add_argument("--out", default="datasets/fungi", help="Output dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=[], metavar="SESSION_ID",
                        help="Session IDs to exclude (e.g. if archived in Docker but not in local DB)")
    args = parser.parse_args()

    if not HAS_PYCOCOTOOLS:
        raise SystemExit("pycocotools is required. Install with: pip install pycocotools")

    random.seed(args.seed)
    exclude_ids = set(args.exclude)

    sessions = load_sessions_and_instances(args.db, exclude_ids)
    n_instances = sum(len(s["instances"]) for s in sessions.values())
    print(f"Sessions: {len(sessions)}")
    print(f"Instances (targets): {n_instances}")
    if exclude_ids:
        print(f"Excluded: {sorted(exclude_ids)}")

    # Train/val split by session
    session_ids = list(sessions.keys())
    random.shuffle(session_ids)
    n_val = max(1, round(len(session_ids) * args.val_fraction))
    val_ids = set(session_ids[:n_val])
    train_ids = set(session_ids[n_val:])
    print(f"Train: {len(train_ids)} sessions  Val: {len(val_ids)} sessions")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    images_base = Path(args.images)

    def copy_images(split_ids, split_name):
        missing = []
        for sid in split_ids:
            src = images_base / sid / "image.jpg"
            dst = out / split_name / f"{sid}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing.append(sid)
        if missing:
            print(f"  WARNING: images not found for {split_name}: {missing}")

    print("\nCopying images...")
    copy_images(train_ids, "train")
    copy_images(val_ids, "val")

    print("\nBuilding COCO JSON...")
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        split_sessions = {sid: sessions[sid] for sid in split_ids}
        coco = build_coco_json(split_sessions)
        coco["categories"] = SINGLE_CLASS
        out_path = out / "annotations" / f"instances_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"  {split_name}: {len(split_sessions)} images, {len(coco['annotations'])} instances → {out_path}")

    print("\nDone. NUM_CLASSES = 1  (single class: fungus)")


if __name__ == "__main__":
    main()
