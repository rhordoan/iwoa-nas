#!/usr/bin/env python
"""
Quick asset status checker for RI-NAS.
- Verifies presence/size of required downloads.
- Suggests resume commands (via bootstrap_download_assets.sh) if incomplete.
- Optionally checks local cache for Nanbeige4-3B-Thinking (Hugging Face) using local_files_only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_file(path: Path, expected_bytes: Optional[int] = None) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"missing ({path})"
    size = path.stat().st_size
    if expected_bytes and size < expected_bytes:
        return False, f"incomplete ({fmt_bytes(size)}/{fmt_bytes(expected_bytes)})"
    return True, f"ok ({fmt_bytes(size)})"


def check_cifar100c() -> Tuple[bool, str]:
    tar = DATA / "datasets" / "cifar-100-c" / "CIFAR-100-C.tar"
    extracted = DATA / "datasets" / "cifar-100-c" / "CIFAR-100-C"
    ok_tar, msg_tar = check_file(tar, expected_bytes=2_918_473_216)
    ok_ext = extracted.exists()
    if ok_tar and ok_ext:
        return True, f"tar {msg_tar}, extracted present"
    if not ok_tar:
        return False, f"tar {msg_tar}"
    return False, f"tar ok; extracted missing -> rerun bootstrap_download_assets.sh"


def check_nasbench201() -> Tuple[bool, str]:
    path = DATA / "benchmarks" / "NAS-Bench-201-v1_1-096897.pth"
    return check_file(path, expected_bytes=4_700_000_000)  # ~4.7G


def check_ofa() -> Tuple[bool, str]:
    path = DATA / "checkpoints" / "ofa" / "ofa_mbv3_d234_e346_k357_w1.0.pth"
    return check_file(path, expected_bytes=25_000_000)  # ~30MB


def check_nanbeige_local() -> Tuple[bool, str]:
    """
    Check if Nanbeige4-3B-Thinking is cached locally (no network).
    """
    model_id = "Nanbeige/Nanbeige4-3B-Thinking-2511"
    try:
        snapshot_download(model_id, local_files_only=True, allow_patterns=["*"])
        return True, "ok (found in local HF cache)"
    except Exception as exc:  # noqa: BLE001
        return False, f"not cached locally ({exc})"


def main() -> None:
    checks = {
        "cifar-100-c": check_cifar100c(),
        "nasbench201": check_nasbench201(),
        "ofa_supernet": check_ofa(),
        "nanbeige4-3b-thinking": check_nanbeige_local(),
    }

    print("== Asset status ==")
    for name, (ok, msg) in checks.items():
        status = "OK" if ok else "MISSING/INCOMPLETE"
        print(f"{name:20} {status:18} {msg}")

    print("\nIf any are missing:")
    print("  bash scripts/bootstrap_download_assets.sh   # resume CIFAR-100-C / OFA / NAS-Bench-201")
    print("  python scripts/fetch_nanbeige.py            # download Nanbeige4-3B-Thinking from HF")


if __name__ == "__main__":
    main()

