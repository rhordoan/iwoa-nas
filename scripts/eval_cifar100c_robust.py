#!/usr/bin/env python
"""
Quick CIFAR-100-C robustness check using a torchvision MobileNetV3-small baseline.
This is a lightweight baseline entrypoint; replace the model load with your best NAS-found model when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def load_cifar100c(root: Path, batch_size: int, num_workers: int):
    tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    ds = torchvision.datasets.ImageFolder(str(root), transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness on CIFAR-100-C with a baseline MobileNetV3.")
    parser.add_argument("--data_root", type=str, default="data/datasets/cifar-100-c/CIFAR-100-C")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = load_cifar100c(Path(args.data_root), args.batch_size, args.num_workers)

    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 100)
    model.to(device)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / max(1, total)
    print(f"CIFAR-100-C robustness (baseline MobileNetV3-small): top-1={acc:.4f}")


if __name__ == "__main__":
    main()


