from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast


class ObjectiveFn(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...


@dataclass
class CEC_Evaluator:
    """Lightweight evaluator wrapper with an evaluation budget.

    `iwoa.py` expects:
    - `calls`: number of objective evaluations performed
    - `max_fes`: max function evaluations (budget)
    - `stop_flag`: set when budget exhausted (or external stop)
    - `__call__(x) -> float`: evaluates objective and increments `calls`
    """

    objective: Callable[[np.ndarray], float]
    max_fes: int
    calls: int = 0
    stop_flag: bool = False

    def __call__(self, x: np.ndarray) -> float:
        if self.stop_flag:
            return float("inf")
        if self.calls >= self.max_fes:
            self.stop_flag = True
            return float("inf")
        self.calls += 1
        return float(self.objective(x))


@dataclass
class Candidate:
    depth_mult: float
    width_mult: float
    res_mult: float

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Candidate":
        return cls(float(vec[0]), float(vec[1]), float(vec[2]))

    def to_vector(self) -> np.ndarray:
        return np.array([self.depth_mult, self.width_mult, self.res_mult], dtype=np.float32)


@dataclass
class EvalResult:
    val_acc: float
    flops_g: float
    latency_ms: float
    params_m: float
    epochs: int


class CIFAR100Evaluator:
    """Proxy evaluator that trains a scaled MobileNetV3 on CIFAR-100.

    Designed for fast proxy training (few epochs, AMP, persistent workers).
    """

    def __init__(
        self,
        *,
        data_root: str = "./data",
        base_image_size: int = 224,
        batch_size: int = 128,
        epochs: int = 5,
        num_workers: int = 8,
        persistent_workers: bool = True,
        amp: bool = True,
        device: Optional[str] = None,
        max_train_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.base_image_size = base_image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.amp = amp
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_train_batches = max_train_batches
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _build_model(self, cand: Candidate) -> nn.Module:
        width_mult = max(0.35, float(cand.width_mult))
        # Depth multiplier is mapped to dropout + effective epochs in this proxy.
        dropout = float(np.clip(0.2 * cand.depth_mult, 0.0, 0.5))
        model = torchvision.models.mobilenet_v3_small(
            weights=None,
            num_classes=100,
            width_mult=width_mult,
            dropout=dropout,
        )
        return model

    def _get_transforms(self, image_size: int) -> Tuple[T.Compose, T.Compose]:
        train_tfms = T.Compose(
            [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        val_tfms = T.Compose(
            [
                T.Resize(image_size + 16),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        return train_tfms, val_tfms

    def _make_dataloaders(self, image_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_tfms, val_tfms = self._get_transforms(image_size)
        train_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True, download=True, transform=train_tfms
        )
        val_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=val_tfms
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _estimate_flops(self, model: nn.Module, image_size: int) -> float:
        params = sum(p.numel() for p in model.parameters())
        # Very rough proxy: 2 * params * spatial_size (in GFLOPs).
        spatial = image_size * image_size
        gflops = 2 * params * spatial / 1e9
        return float(gflops)

    def _measure_latency(self, model: nn.Module, image_size: int, steps: int = 10) -> float:
        model.eval()
        dummy = torch.randn(1, 3, image_size, image_size, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        with torch.no_grad():
            for _ in range(steps):
                start = time.perf_counter()
                _ = model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                timings.append(end - start)
        return float(np.median(timings) * 1000.0)  # ms

    def evaluate_candidate(self, cand: Candidate) -> EvalResult:
        image_size = max(32, int(self.base_image_size * cand.res_mult))
        model = self._build_model(cand).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-5)
        scaler = GradScaler(enabled=self.amp and self.device.type == "cuda")

        train_loader, val_loader = self._make_dataloaders(image_size)
        effective_epochs = max(1, int(round(self.epochs * cand.depth_mult)))

        best_val_acc = 0.0
        for epoch in range(effective_epochs):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.amp and self.device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if self.max_train_batches and batch_idx + 1 >= self.max_train_batches:
                    break

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    with autocast(enabled=self.amp and self.device.type == "cuda"):
                        logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(1, total)
            best_val_acc = max(best_val_acc, val_acc)

        flops_g = self._estimate_flops(model, image_size)
        latency_ms = self._measure_latency(model, image_size, steps=5 if self.device.type == "cuda" else 2)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        return EvalResult(
            val_acc=float(best_val_acc),
            flops_g=flops_g,
            latency_ms=latency_ms,
            params_m=float(params_m),
            epochs=effective_epochs,
        )

