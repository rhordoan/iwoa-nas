import argparse
from pathlib import Path

from src.surrogates.llm_ranker import train_llm_ranker
from src.surrogates.mlp_proxy import train_mlp_proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP proxy and LLM ranker surrogates.")
    parser.add_argument("--csv", type=str, default="data/nas_data.csv", help="Input dataset CSV.")
    parser.add_argument("--mlp_out", type=str, default="results/surrogates/mlp_proxy.pth", help="MLP checkpoint.")
    parser.add_argument("--llm_out", type=str, default="results/surrogates/llm_ranker", help="LLM output directory.")
    parser.add_argument("--skip_mlp", action="store_true", help="Skip MLP training.")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM training.")
    parser.add_argument("--base_image_size", type=int, default=224, help="Base image size for text prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_mlp:
        mlp_metrics = train_mlp_proxy(args.csv, output_path=args.mlp_out)
        print(f"MLP proxy -> MAE: {mlp_metrics.val_mae:.4f} MSE: {mlp_metrics.val_mse:.4f} Pearson: {mlp_metrics.val_pearson:.3f}")
    if not args.skip_llm:
        llm_metrics = train_llm_ranker(
            args.csv,
            output_dir=args.llm_out,
            base_image_size=args.base_image_size,
        )
        print(f"LLM ranker -> MSE: {llm_metrics.val_mse:.4f} Pearson: {llm_metrics.val_pearson:.3f}")


if __name__ == "__main__":
    main()

