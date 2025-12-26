import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot convergence (accuracy vs wall-clock).")
    parser.add_argument("--csv", type=str, default="data/nas_data.csv", help="Input CSV log.")
    parser.add_argument("--out", type=str, default="results/convergence.png", help="Output plot path.")
    parser.add_argument("--title", type=str, default="RI-NAS Convergence", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df.sort_values(by=["generation", "wall_time_s"])
    df["cum_time_h"] = df["wall_time_s"].cumsum() / 3600.0

    plt.figure(figsize=(8, 4))
    plt.plot(df["cum_time_h"], df["val_acc"], marker="o", linestyle="-", label="RI-NAS")
    plt.xlabel("Wall-clock time (hours)")
    plt.ylabel("Validation accuracy")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

