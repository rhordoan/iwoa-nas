#!/usr/bin/env python
"""
Lightweight sanity check for NAS-Bench-201 availability.
- Verifies the benchmark file is present.
- Optionally loads nas_201_api (if installed) and queries a random architecture.

Usage:
  python scripts/nasbench201_sanity.py --file data/benchmarks/NAS-Bench-201-v1_1-096897.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS-Bench-201 sanity check.")
    parser.add_argument("--file", type=str, default="data/benchmarks/NAS-Bench-201-v1_1-096897.pth")
    args = parser.parse_args()

    bench_path = Path(args.file)
    if not bench_path.exists():
        raise FileNotFoundError(f"NAS-Bench-201 file not found at {bench_path}")

    try:
        from nas_201_api import NASBench201API as API  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"nas_201_api not installed or failed to import: {exc}")
        print("Benchmark file is present; install nas_201_api to query it.")
        return

    api = API(str(bench_path))
    idx = random.randint(0, len(api)-1)
    info = api.query_by_index(idx)
    print(f"Loaded NAS-Bench-201. Random arch idx={idx}")
    print(f"Metrics keys: {list(info.keys())[:5]} ...")


if __name__ == "__main__":
    main()


