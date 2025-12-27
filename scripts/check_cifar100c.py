#!/usr/bin/env python
"""
Quick check that CIFAR-100-C is present and readable.
Reports corruption types and array shapes using the npy files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CIFAR-100-C extraction.")
    parser.add_argument("--root", type=str, default="data/datasets/cifar-100-c")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    labels = root / "CIFAR-100-C" / "labels.npy"
    if not labels.exists():
        raise FileNotFoundError(f"labels.npy not found under {labels.parent}")

    y = np.load(labels)
    print(f"labels.npy shape: {y.shape}")
    samples_per_corruption = {}
    for npy in sorted((root / "CIFAR-100-C").glob("*.npy")):
        if npy.name == "labels.npy":
            continue
        arr = np.load(npy, mmap_mode="r")
        samples_per_corruption[npy.stem] = arr.shape[0]
        print(f"{npy.name}: shape={arr.shape}")
    print("Corruptions:", ", ".join(samples_per_corruption.keys()))


if __name__ == "__main__":
    main()


