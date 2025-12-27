from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from src.evaluator import Candidate, EvalResult
from src.eval.cache import EvalCache
from src.eval.ofa_loader import get_net_for_config
from src.nas.codec import Codec


def _round_key(cfg: Dict[str, Any], ndigits: int = 3) -> str:
    def r(x):
        return round(float(x), ndigits)

    return json_key(
        {
            "depth": r(cfg.get("depth_mult", cfg.get("depth", 0))),
            "width": r(cfg.get("width_mult", 0)),
            "res_mult": r(cfg.get("res_mult", 0)),
            "kernel_size": r(cfg.get("kernel_size", 0)),
            "expand_ratio": r(cfg.get("expand_ratio", 0)),
            "image_size": int(cfg.get("image_size", 224)),
        }
    )


def json_key(d: Dict[str, Any]) -> str:
    return "|".join(f"{k}:{v}" for k, v in sorted(d.items()))


def build_cifar100_val_loader(data_root: Path, batch_size: int = 128, num_workers: int = 4):
    tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    ds = torchvision.datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_cifar100_train_loader(data_root: Path, batch_size: int = 128, num_workers: int = 4):
    tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    ds = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def build_imagenet_val_loader(data_root: Path, batch_size: int = 128, num_workers: int = 4, subset: Optional[int] = None):
    # Accept either data_root/val/* or data_root/* as class folders
    root = data_root / "val" if (data_root / "val").exists() else data_root
    tfm = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = torchvision.datasets.ImageFolder(str(root), transform=tfm)
    if subset is not None and subset > 0 and subset < len(ds):
        ds = Subset(ds, list(range(subset)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def build_imagenet_train_loader(data_root: Path, batch_size: int = 128, num_workers: int = 4, subset: Optional[int] = None):
    tfm = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_root = data_root / "train"
    root = train_root if train_root.exists() else (data_root / "val" if (data_root / "val").exists() else data_root)
    ds = torchvision.datasets.ImageFolder(str(root), transform=tfm)
    if subset is not None and subset > 0 and subset < len(ds):
        ds = Subset(ds, list(range(subset)))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def load_imagenet_wnid_to_idx(cache_path: Path = Path("data/imagenet_class_index.json")) -> Dict[str, int]:
    """
    Load mapping from WordNet ID to ImageNet-1k class index.
    Downloads the standard imagenet_class_index.json once if missing.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with urlopen(url) as resp, cache_path.open("wb") as f:
            f.write(resp.read())
    with cache_path.open("r") as f:
        data = json.load(f)
    return {v[0]: int(k) for k, v in data.items()}


@dataclass
class OFARealEvaluator:
    codec: Codec
    checkpoint_path: Path
    net_id: str
    cache: EvalCache
    data_root: Path
    device: torch.device
    batch_size: int = 128
    num_workers: int = 4
    imagenet_root: Optional[Path] = None
    imagenet_subset: Optional[int] = None
    imagenet_remap_wnids: bool = False
    head_ft_steps: int = 0
    head_ft_lr: float = 5e-4
    head_ft_weight_decay: float = 0.0

    def evaluate(self, cand: Candidate) -> EvalResult:
        cfg = self.codec.to_ofa_config(cand.to_vector())
        key = _round_key(cfg, ndigits=3)
        cached = self.cache.get(key)
        if cached:
            return EvalResult(
                val_acc=float(cached["val_acc"]),
                flops_g=float(cached["flops_g"]),
                latency_ms=float(cached["latency_ms"]),
                params_m=float(cached["params_m"]),
                epochs=int(cached.get("epochs", 0)),
            )

        subnet, image_size = get_net_for_config(cfg, checkpoint_path=self.checkpoint_path, net_id=self.net_id, device=self.device)
        # Select dataset / classes
        if self.imagenet_root:
            loader = build_imagenet_val_loader(
                self.imagenet_root,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                subset=self.imagenet_subset,
            )
            num_classes = len(getattr(loader.dataset, "classes", [])) or 1000
            wnid_to_idx = load_imagenet_wnid_to_idx() if self.imagenet_remap_wnids else None
            class_to_wnid = {i: c for i, c in enumerate(getattr(loader.dataset, "classes", []))}
            train_loader = (
                build_imagenet_train_loader(
                    self.imagenet_root,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    subset=self.imagenet_subset,
                )
                if self.head_ft_steps > 0
                else None
            )
        else:
            loader = build_cifar100_val_loader(self.data_root, batch_size=self.batch_size, num_workers=self.num_workers)
            num_classes = 100
            wnid_to_idx = None
            class_to_wnid = None
            train_loader = (
                build_cifar100_train_loader(self.data_root, batch_size=self.batch_size, num_workers=self.num_workers)
                if self.head_ft_steps > 0
                else None
            )

        # Ensure head matches dataset classes
        if hasattr(subnet, "classifier") and isinstance(subnet.classifier, torch.nn.Sequential):
            last = subnet.classifier[-1]
            if getattr(last, "out_features", None) != num_classes:
                subnet.classifier[-1] = torch.nn.Linear(last.in_features, num_classes, bias=True).to(self.device)
        elif hasattr(subnet, "fc"):
            if getattr(subnet.fc, "out_features", None) != num_classes:
                subnet.fc = torch.nn.Linear(subnet.fc.in_features, num_classes, bias=True).to(self.device)

        # Optional head-only fine-tune
        if self.head_ft_steps > 0 and train_loader is not None:
            # freeze body
            for p in subnet.parameters():
                p.requires_grad = False
            # unfreeze head
            head_params = []
            if hasattr(subnet, "classifier") and isinstance(subnet.classifier, torch.nn.Sequential):
                for p in subnet.classifier.parameters():
                    p.requires_grad = True
                head_params = list(subnet.classifier.parameters())
            elif hasattr(subnet, "fc"):
                for p in subnet.fc.parameters():
                    p.requires_grad = True
                head_params = list(subnet.fc.parameters())
            if head_params:
                opt = optim.Adam(
                    head_params, lr=self.head_ft_lr, weight_decay=self.head_ft_weight_decay
                )
                subnet.train()
                steps = 0
                for images, labels in train_loader:
                    # Optional remap for ImageNet subsets with WNID folders
                    if wnid_to_idx and class_to_wnid:
                        mapped = []
                        keep_idx = []
                        for i, t in enumerate(labels.tolist()):
                            wnid = class_to_wnid.get(t)
                            if wnid in wnid_to_idx:
                                mapped.append(wnid_to_idx[wnid])
                                keep_idx.append(i)
                        if not keep_idx:
                            continue
                        labels = torch.tensor(mapped, device=self.device)
                        images = images[keep_idx].to(self.device, non_blocking=True)
                    else:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    logits = subnet(images)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    opt.step()
                    steps += 1
                    if steps >= self.head_ft_steps:
                        break
            subnet.eval()

        correct = total = 0
        params_m = sum(p.numel() for p in subnet.parameters()) / 1e6
        # rough flops proxy (same as before)
        flops_g = 2 * params_m * (image_size * image_size) / 1e3

        subnet.eval()
        with torch.no_grad():
            for images, labels in loader:
                # Optional remap for ImageNet subsets with WNID folders (e.g., ImageNet-A)
                if wnid_to_idx and class_to_wnid:
                    mapped = []
                    keep_idx = []
                    for i, t in enumerate(labels.tolist()):
                        wnid = class_to_wnid.get(t)
                        if wnid in wnid_to_idx:
                            mapped.append(wnid_to_idx[wnid])
                            keep_idx.append(i)
                    if not keep_idx:
                        continue
                    labels = torch.tensor(mapped, device=self.device)
                    images = images[keep_idx].to(self.device, non_blocking=True)
                else:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                logits = subnet(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(1, total)
        # measure a single forward latency median over few runs
        dummy = torch.randn(1, 3, image_size, image_size, device=self.device)
        timings = []
        for _ in range(3):
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            _ = subnet(dummy)
            t1.record()
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            timings.append(t0.elapsed_time(t1))  # ms
        latency_ms = float(sum(timings) / max(1, len(timings))) if timings else 0.0

        result = EvalResult(
            val_acc=float(acc),
            flops_g=float(flops_g),
            latency_ms=float(latency_ms),
            params_m=float(params_m),
            epochs=0,
        )
        self.cache.set(key, result.__dict__)
        return result

