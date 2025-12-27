#!/usr/bin/env python
"""
Prepare Tiny-ImageNet-200 (if downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- Reorganizes val images into class subfolders under val/.

Usage:
  python scripts/prepare_tiny_imagenet.py --root data/datasets/tiny-imagenet-200
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prepare(root: Path) -> None:
    val_dir = root / "val"
    images_dir = val_dir / "images"
    anno = val_dir / "val_annotations.txt"
    if not images_dir.exists() or not anno.exists():
        raise FileNotFoundError("Expected val/images and val_annotations.txt under Tiny-ImageNet val/")

    with anno.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fname, cls = parts[0], parts[1]
            src = images_dir / fname
            dst_dir = val_dir / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / fname
            if src.exists():
                shutil.move(str(src), str(dst))
    # Remove empty images dir if all moved
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
    print("Prepared Tiny-ImageNet val/ into class subfolders.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tiny-ImageNet-200 val set (class subfolders).")
    parser.add_argument("--root", type=str, required=True, help="Path to tiny-imagenet-200 directory")
    args = parser.parse_args()
    prepare(Path(args.root))


if __name__ == "__main__":
    main()


