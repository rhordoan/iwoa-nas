import argparse
import time
from pathlib import Path

import numpy as np

from src.evaluator import CIFAR100Evaluator
from src.nas.utils import append_rows_csv, load_search_space, sample_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search over compound scaling multipliers.")
    parser.add_argument("--config", type=str, default="configs/search_space.yaml", help="Search space config.")
    parser.add_argument("--samples", type=int, default=100, help="Number of random candidates to evaluate.")
    parser.add_argument("--output", type=str, default="data/nas_data.csv", help="CSV log path.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional cap per epoch (debug).")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--resume", action="store_true", help="Append without truncating existing CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    space = load_search_space(args.config)
    rng = np.random.default_rng(args.seed)
    evaluator = CIFAR100Evaluator(
        data_root=args.data_root,
        base_image_size=int(space.get("base_image_size", 224)),
        batch_size=int(space.get("defaults", {}).get("batch_size", 128)),
        epochs=int(space.get("defaults", {}).get("epochs", 5)),
        num_workers=int(space.get("defaults", {}).get("num_workers", 8)),
        persistent_workers=bool(space.get("defaults", {}).get("persistent_workers", True)),
        amp=bool(space.get("defaults", {}).get("amp", True)),
        device=args.device,
        max_train_batches=args.max_train_batches,
        seed=args.seed,
    )

    fieldnames = [
        "depth_mult",
        "width_mult",
        "res_mult",
        "val_acc",
        "flops_g",
        "latency_ms",
        "params_m",
        "epochs",
        "seed",
        "wall_time_s",
    ]

    csv_path = Path(args.output)
    if not args.resume and csv_path.exists():
        csv_path.unlink()

    start = time.time()
    rows = []
    for idx in range(args.samples):
        cand = sample_candidate(space, rng)
        tick = time.time()
        result = evaluator.evaluate_candidate(cand)
        wall = time.time() - tick
        row = {
            "depth_mult": cand.depth_mult,
            "width_mult": cand.width_mult,
            "res_mult": cand.res_mult,
            "val_acc": result.val_acc,
            "flops_g": result.flops_g,
            "latency_ms": result.latency_ms,
            "params_m": result.params_m,
            "epochs": result.epochs,
            "seed": args.seed,
            "wall_time_s": wall,
        }
        rows.append(row)
        append_rows_csv(csv_path, fieldnames, [row])
        elapsed = time.time() - start
        print(
            f"[{idx+1}/{args.samples}] acc={result.val_acc:.4f} "
            f"flops={result.flops_g:.2f}G lat={result.latency_ms:.1f}ms "
            f"elapsed={elapsed/60:.1f}m"
        )


if __name__ == "__main__":
    main()

