import argparse
import json
from pathlib import Path

from src.evaluator import Candidate, CIFAR100Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single RI-NAS candidate on CIFAR-100.")
    parser.add_argument("--depth_mult", type=float, required=True, help="Depth multiplier (continuous).")
    parser.add_argument("--width_mult", type=float, required=True, help="Width multiplier (continuous).")
    parser.add_argument("--res_mult", type=float, required=True, help="Resolution multiplier (continuous).")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset cache directory.")
    parser.add_argument("--base_image_size", type=int, default=224, help="Base image size before scaling.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for proxy training.")
    parser.add_argument("--epochs", type=int, default=5, help="Proxy training epochs.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--no_persistent_workers", action="store_true", help="Disable persistent workers.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional cap per epoch for fast tests.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional path to save JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cand = Candidate(depth_mult=args.depth_mult, width_mult=args.width_mult, res_mult=args.res_mult)
    evaluator = CIFAR100Evaluator(
        data_root=args.data_root,
        base_image_size=args.base_image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        persistent_workers=not args.no_persistent_workers,
        amp=not args.no_amp,
        device=args.device,
        max_train_batches=args.max_train_batches,
        seed=args.seed,
    )
    result = evaluator.evaluate_candidate(cand)
    payload = {
        "depth_mult": cand.depth_mult,
        "width_mult": cand.width_mult,
        "res_mult": cand.res_mult,
        "val_acc": result.val_acc,
        "flops_g": result.flops_g,
        "latency_ms": result.latency_ms,
        "params_m": result.params_m,
        "epochs": result.epochs,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

