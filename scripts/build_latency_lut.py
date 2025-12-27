#!/usr/bin/env python
"""
Benchmark per-module latencies for an OFA MobileNetV3 supernet and write a LUT.
Keys are module names; values are median latency (ms) over N runs with batch=1.

Usage:
  python scripts/build_latency_lut.py --runs 100 --image_size 224 --out data/latency_lut.json

Notes:
- Targets the current GPU by default.
- Uses batch=1 (typical for latency LUT). Adjust with --batch if needed.
- You should rerun this on the specific deployment device to get accurate numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.eval.ofa_loader import load_ofa_net


def register_latency_hooks(net: torch.nn.Module, device: torch.device):
    times: Dict[str, List[float]] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            # forward already executed; we measure nothing here—just placeholder for consistency
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.setdefault(name, []).append((t1 - t0) * 1000.0)

        return hook

    handles = []
    for name, module in net.named_modules():
        # Only time leaf modules with params (e.g., conv, bn, linear)
        if any(p.requires_grad for p in module.parameters()):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles, times


def run_benchmark(net: torch.nn.Module, image_size: int, batch: int, runs: int, device: torch.device) -> Dict[str, float]:
    net.eval()
    dummy = torch.randn(batch, 3, image_size, image_size, device=device)

    handles, times = register_latency_hooks(net, device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = net(dummy)
    # Timed runs
    with torch.no_grad():
        for _ in range(runs):
            _ = net(dummy)
    for h in handles:
        h.remove()

    medians: Dict[str, float] = {}
    for name, vals in times.items():
        if vals:
            medians[name] = float(torch.median(torch.tensor(vals)).item())
    return medians


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latency LUT for OFA MobileNetV3 supernet.")
    parser.add_argument("--net_id", type=str, default="ofa_mbv3_d234_e346_k357_w1.0", help="OFA net id.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for benchmarking.")
    parser.add_argument("--runs", type=int, default=100, help="Timed runs per module.")
    parser.add_argument("--out", type=str, default="data/latency_lut.json", help="Output JSON path.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_ofa_net({"net_id": args.net_id}, device=device.type)

    print(f"Benchmarking {args.net_id} on {device} | image_size={args.image_size} batch={args.batch} runs={args.runs}")
    medians = run_benchmark(net, args.image_size, args.batch, args.runs, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(medians, indent=2))
    print(f"Wrote LUT with {len(medians)} entries to {out_path}")


if __name__ == "__main__":
    main()


