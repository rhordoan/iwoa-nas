#!/usr/bin/env python
"""
Build a hardware-specific latency LUT keyed by OFA MobileNetV3 block configs.

Each key encodes:
  stage_idx, stride, ks, expand_ratio, in_ch, out_ch, image_size

We enumerate the active subnet across all ks/expand/depth combinations for MBV3,
benchmark each unique block, and write a JSON LUT:
  { "<stage>-s<stride>-k<ks>-e<exp>-c<in>-o<out>-h<res>": latency_ms }

This can be summed for a candidate architecture without re-timing.

Usage:
  PYTHONPATH=. python scripts/build_latency_lut_full.py --runs 100 --image_size 224 --out data/latency_lut_full.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

from src.eval.ofa_loader import load_ofa_supernet


DEFAULT_NET_ID = "ofa_mbv3_d234_e346_k357_w1.0"
DEFAULT_CKPT = Path("data/checkpoints/ofa/ofa_mbv3_d234_e346_k357_w1.0.pth")


def block_key(stage_idx: int, stride: int, ks: int, exp: int, in_ch: int, out_ch: int, res: int) -> str:
    return f"stage{stage_idx}-s{stride}-k{ks}-e{exp}-c{in_ch}-o{out_ch}-h{res}"


def enumerate_blocks(active_net, image_size: int, ks_sel: int, exp_sel: int) -> List[Tuple[str, torch.nn.Module]]:
    """
    Enumerate blocks in the active subnet and build keys using the selected ks/exp.
    """
    keyed = []
    idx = 0
    for name, module in active_net.named_modules():
        if ("InvertedResidual" in module.__class__.__name__) or ("MBConv" in module.__class__.__name__):
            stride = getattr(module, "stride", 1)
            if isinstance(stride, (tuple, list)):
                stride = stride[0]
            in_ch = getattr(module, "in_channels", 0)
            out_ch = getattr(module, "out_channels", 0)
            key = block_key(idx, stride, ks_sel, exp_sel, int(in_ch), int(out_ch), image_size)
            keyed.append((key, module))
            idx += 1
    return keyed


def measure_block_latency(module: torch.nn.Module, image_size: int, in_ch: int, runs: int, device: torch.device) -> float:
    # Create a synthetic input matching the module's expected channels.
    c = in_ch if in_ch > 0 else 16
    x = torch.randn(1, c, image_size, image_size, device=device)
    module.eval().to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = module(x)
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = module(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return float(torch.median(torch.tensor(times)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full OFA MBV3 block-level latency LUT.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--net_id", type=str, default=DEFAULT_NET_ID)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/latency_lut_full.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_ofa_supernet(Path(args.checkpoint), net_id=args.net_id, device=device)

    # Sweep all ks/exp/depth combinations from the net lists (best-effort).
    ks_list = getattr(net, "ks_list", [3, 5, 7])
    exp_list = getattr(net, "expand_ratio_list", [3, 4, 6])
    depth_list = getattr(net, "depth_list", [2, 3, 4])

    seen: Set[str] = set()
    lut: Dict[str, float] = {}

    for ks, exp, depth in itertools.product(ks_list, exp_list, depth_list):
        try:
            net.set_active_subnet(ks=ks, e=exp, d=depth)
            active = net.get_active_subnet(preserve_weight=True)
        except Exception:
            continue
        keyed = enumerate_blocks(active, args.image_size, ks_sel=ks, exp_sel=exp)
        for key, module in keyed:
            if key in seen:
                continue
            seen.add(key)
            # Approximate input channels if available from the key
            parts = key.split("-")
            in_ch = 16
            for p in parts:
                if p.startswith("c"):
                    try:
                        in_ch = int(p[1:])
                    except Exception:
                        pass
            latency = measure_block_latency(module, args.image_size, in_ch, args.runs, device)
            lut[key] = latency
            print(f"{key}: {latency:.4f} ms")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lut, indent=2))
    print(f"Wrote full LUT with {len(lut)} entries to {out_path}")


if __name__ == "__main__":
    main()

