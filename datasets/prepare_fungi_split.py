"""
Export SAM3 annotations to a split COCO instance segmentation dataset (train/val).

All segmented instances are treated as a single class ("mold") without any 
species, ignore, or overgrowth distinctions (all iscrowd=0, category_id=1).

Usage:
    python datasets/prepare_fungi_split.py \
        --db ../SAM3/app/data/sam3.db \
        --images ../SAM3/dataset \
        --out datasets/fungi_split \
        --val-fraction 0.2

Output:
    datasets/fungi_split/
        train/          # training images
        val/            # validation images
        annotations/
            instances_train.json
            instances_val.json
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


SINGLE_CLASS = [{"id": 1, "name": "mold", "supercategory": "mold"}]


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

    binary_mask = flat.reshape((h, w))  # SAM3 RLE is row-major; asfortranarray handles COCO encoding
    encoded = mask_utils.encode(np.asfortranarray(binary_mask))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def load_sessions_and_instances(db_path, image_filenames, exclude_ids):
    """Load active, favourited sessions (minus --exclude list) matching dataset images.

    All database instances are stored under 'instances' and treated identically.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        "SELECT id, image_filename, width, height FROM sessions "
        "WHERE is_archived = 0 AND is_favourited = 1"
    )
    
    sessions = {}
    for row in cur.fetchall():
        session_id, image_filename, width, height = row
        if session_id in exclude_ids:
            continue
        if image_filename in image_filenames:
            sessions[session_id] = {
                "id": session_id,
                "filename": image_filename,
                "width": width,
                "height": height,
                "instances": [],
            }

    cur.execute(
        "SELECT session_id, instance_index, mask_rle, score FROM instances"
    )
    for session_id, idx, mask_rle_str, score in cur.fetchall():
        if session_id not in sessions:
            continue
        inst = {
            "instance_index": idx,
            "mask_rle": json.loads(mask_rle_str),
            "score": score,
        }
        sessions[session_id]["instances"].append(inst)

    con.close()
    return sessions


def build_coco_json(sessions):
    """Build a COCO-format dict.

    All instances → category_id=1, iscrowd=0
    """
    images = []
    annotations = []
    ann_id = 1

    for img_id, (session_id, session) in enumerate(sessions.items(), start=1):
        images.append({
            "id": img_id,
            "file_name": session["filename"],
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
    parser = argparse.ArgumentParser(description="Prepare SAM3 dataset with train/val splits (no ignore distinctions).")
    parser.add_argument("--db", default="SAM3/app/data/sam3.db", help="Path to sam3.db")
    parser.add_argument("--images", default="SAM3/dataset", help="Path to images directory")
    parser.add_argument("--out", default="MaskDINO/datasets/fungi_split", help="Output dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of sessions to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="SESSION_ID",
                        help="Session IDs to exclude")
    args = parser.parse_args()

    if not HAS_PYCOCOTOOLS:
        raise SystemExit("pycocotools is required. Install with: pip install pycocotools")

    db_path = Path(args.db)
    if not db_path.exists():
        db_path = Path(__file__).resolve().parent.parent.parent / args.db
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found at {args.db}")

    images_dir = Path(args.images)
    if not images_dir.exists():
        images_dir = Path(__file__).resolve().parent.parent.parent / args.images
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {args.images}")

    exclude_ids = set(args.exclude)

    # Get image filenames from the directory
    image_filenames = {
        f.name for f in images_dir.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    }
    print(f"Found {len(image_filenames)} images in {images_dir}")

    sessions = load_sessions_and_instances(db_path, image_filenames, exclude_ids)
    print(f"Matched sessions in DB: {len(sessions)}")

    # Split sessions
    session_ids = list(sessions.keys())
    random.seed(args.seed)
    random.shuffle(session_ids)
    n_val = max(1, round(len(session_ids) * args.val_fraction))
    val_ids = set(session_ids[:n_val])
    train_ids = set(session_ids[n_val:])
    print(f"Split details — Train: {len(train_ids)} sessions  Val: {len(val_ids)} sessions")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent.parent.parent / args.out

    # Create directory structures
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Copy images and build json
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        print(f"\nProcessing {split_name} split...")
        split_sessions = {sid: sessions[sid] for sid in split_ids}

        # Copy images
        for sid, session in split_sessions.items():
            src = images_dir / session["filename"]
            dest = out / split_name / session["filename"]
            shutil.copy2(src, dest)
        print(f"  Copied {len(split_sessions)} images to {out}/{split_name}")

        # Build COCO JSON
        coco = build_coco_json(split_sessions)
        coco["categories"] = SINGLE_CLASS
        out_path = out / "annotations" / f"instances_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        
        n_instances = sum(len(s["instances"]) for s in split_sessions.values())
        print(f"  Saved {split_name} COCO annotations to {out_path}")
        print(f"  Total images: {len(coco['images'])}, annotations: {n_instances}")

    print("\nDone. NUM_CLASSES = 1 (single class: mold, all masks treated equally)")


if __name__ == "__main__":
    main()
