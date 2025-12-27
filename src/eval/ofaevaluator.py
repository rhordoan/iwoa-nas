from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch

from src.evaluator import Candidate, EvalResult
from src.eval.ofa_loader import load_ofa_net, make_image_size
from src.nas.codec import Codec
from src.latency.lut import LatencyLUT


class OFAEvaluator:
    """
    Zero-cost-ish evaluator for OFA MobileNetV3 using a single forward pass on dummy data.
    Latency is measured directly; val_acc is a proxy (1 / (1 + latency/100)).
    """

    def __init__(self, codec: Codec, latency_lut: Optional[Path] = None, device: str | None = None) -> None:
        self.codec = codec
        self.latency_lut_path = latency_lut
        self.lut = LatencyLUT(latency_lut) if latency_lut else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mode = "proxy"  # or "real"
        self.real_eval = None  # set externally when mode=real

    def _measure_latency(self, net: torch.nn.Module, image_size: int) -> float:
        dummy = torch.randn(1, 3, image_size, image_size, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        with torch.no_grad():
            for _ in range(3):
                t0 = time.perf_counter()
                _ = net(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(timings)) if timings else 0.0

    def _lut_latency(self, net: torch.nn.Module) -> Optional[float]:
        if not self.lut:
            return None
        total = 0.0
        for name, module in net.named_modules():
            if any(p.requires_grad for p in module.parameters()):
                val = self.lut.get_latency(name)
                if val is None:
                    return None
                total += float(val)
        return total

    def evaluate_candidate(self, cand: Candidate) -> EvalResult:
        if self.mode == "real" and self.real_eval:
            return self.real_eval(cand)
        cfg = self.codec.to_ofa_config(cand.to_vector())
        net = load_ofa_net(cfg, device=self.device.type)
        image_size = make_image_size(cfg)

        latency_ms = self._lut_latency(net)
        if latency_ms is None:
            latency_ms = self._measure_latency(net, image_size)

        params_m = sum(p.numel() for p in net.parameters()) / 1e6
        flops_g = 2 * params_m * (image_size * image_size) / 1e3
        val_acc = 1.0 / (1.0 + latency_ms / 100.0)
        return EvalResult(
            val_acc=float(val_acc),
            flops_g=float(flops_g),
            latency_ms=float(latency_ms),
            params_m=float(params_m),
            epochs=0,
        )

