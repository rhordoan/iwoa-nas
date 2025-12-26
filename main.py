import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.evaluator import Candidate, CIFAR100Evaluator
from src.nas.utils import append_rows_csv, bounds_from_space, load_search_space, sample_candidate, vector_to_text
from src.surrogates.llm_ranker import load_llm_ranker
from src.surrogates.mlp_proxy import MLPProxy, load_mlp_proxy, train_mlp_proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid RI-NAS search (IWOA + LLM + gradient guide).")
    parser.add_argument("--config", type=str, default="configs/search_space.yaml", help="Search space config.")
    parser.add_argument("--pop_size", type=int, default=50, help="Population size.")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations.")
    parser.add_argument("--use_llm_filter", action="store_true", help="Enable LLM filter.")
    parser.add_argument("--use_gradient_guide", action="store_true", help="Enable gradient-based refinement.")
    parser.add_argument("--mlp_path", type=str, default="results/surrogates/mlp_proxy.pth", help="MLP checkpoint.")
    parser.add_argument("--llm_path", type=str, default="results/surrogates/llm_ranker", help="LLM checkpoint dir.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset directory.")
    parser.add_argument("--csv_log", type=str, default="data/nas_data.csv", help="Dataset CSV (append).")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional cap per epoch for evaluator.")
    parser.add_argument("--device", type=str, default=None, help="Device override for evaluator and surrogates.")
    parser.add_argument("--retrain_every", type=int, default=5, help="Retrain surrogates every N generations.")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    return parser.parse_args()


def predict_mlp(model: MLPProxy, vecs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(vecs, dtype=torch.float32, device=device)
        preds = model(x).detach().cpu().numpy()
    return preds


def mlp_gradients(model: MLPProxy, vecs: np.ndarray, device: torch.device) -> np.ndarray:
    grads = []
    for vec in vecs:
        x = torch.tensor(vec, dtype=torch.float32, device=device, requires_grad=True)
        pred = model(x)
        pred.backward()
        grads.append(x.grad.detach().cpu().numpy())
    return np.stack(grads, axis=0)


def llm_score(tokenizer, model, cands: List[Candidate], base_image_size: int, device: torch.device) -> np.ndarray:
    texts = [vector_to_text(c, base_image_size) for c in cands]
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits.squeeze(-1).detach().cpu().numpy()
    return logits


def iwoa_move(pop: np.ndarray, bounds: np.ndarray, progress: float, global_best: np.ndarray) -> np.ndarray:
    lb, ub = bounds
    a = 2.0 * (1.0 - progress**1.5)
    spiral_c = 1.0 * (1 - progress)
    new_pop = []
    for i in range(pop.shape[0]):
        x = pop[i]
        r1, r2 = np.random.rand(), np.random.rand()
        A, C = 2 * a * r1 - a, 2 * r2
        p = np.random.rand()
        if p < 0.5:
            if abs(A) < 1:
                D = np.abs(C * global_best - x)
                x_new = global_best - A * D
            else:
                rand_idx = np.random.randint(0, pop.shape[0])
                D = np.abs(C * pop[rand_idx] - x)
                x_new = pop[rand_idx] - A * D
        else:
            D = np.abs(global_best - x)
            l = np.random.uniform(-1, 1)
            x_new = D * np.exp(spiral_c * l) * np.cos(2 * np.pi * l) + global_best
        x_new = np.clip(x_new, lb, ub)
        new_pop.append(x_new)
    return np.stack(new_pop, axis=0)


def ensure_bounds(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(pop, lb, ub)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    space = load_search_space(args.config)
    lb, ub = bounds_from_space(space)
    bounds = np.stack([lb, ub], axis=0)
    base_image_size = int(space.get("base_image_size", 224))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Surrogates
    mlp_model: Optional[MLPProxy] = None
    if Path(args.mlp_path).exists():
        mlp_model = load_mlp_proxy(args.mlp_path, device=device)
    else:
        mlp_model = MLPProxy().to(device)
    llm_tokenizer = llm_model = None
    if args.use_llm_filter and Path(args.llm_path).exists():
        llm_tokenizer, llm_model = load_llm_ranker(args.llm_path)
        llm_model.to(device)

    evaluator = CIFAR100Evaluator(
        data_root=args.data_root,
        base_image_size=base_image_size,
        batch_size=int(space.get("defaults", {}).get("batch_size", 128)),
        epochs=int(space.get("defaults", {}).get("epochs", 5)),
        num_workers=int(space.get("defaults", {}).get("num_workers", 8)),
        persistent_workers=bool(space.get("defaults", {}).get("persistent_workers", True)),
        amp=bool(space.get("defaults", {}).get("amp", True)),
        device=args.device,
        max_train_batches=args.max_train_batches,
        seed=args.seed,
    )

    # Init population
    population = np.stack([sample_candidate(space).to_vector() for _ in range(args.pop_size)], axis=0)

    # Logging setup
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
        "generation",
        "wall_time_s",
    ]
    csv_path = args.csv_log

    for gen in range(args.generations):
        progress = gen / max(1, args.generations - 1)

        # Surrogate scoring (higher is better)
        if mlp_model is None:
            mlp_model = load_mlp_proxy(args.mlp_path, device=device)
        surrogate_scores = predict_mlp(mlp_model, population, device=device)
        best_idx = int(np.argmax(surrogate_scores))
        global_best = population[best_idx]

        offspring = iwoa_move(population, bounds, progress, global_best)
        candidates = [Candidate.from_vector(vec) for vec in offspring]

        # LLM filtering
        elite_indices = list(range(len(candidates)))
        if args.use_llm_filter and llm_tokenizer and llm_model:
            llm_scores = llm_score(llm_tokenizer, llm_model, candidates, base_image_size, device)
            elite_count = max(1, int(0.1 * len(candidates)))
            elite_indices = list(np.argsort(-llm_scores)[:elite_count])
        elites = [candidates[i] for i in elite_indices]
        elite_vecs = np.stack([e.to_vector() for e in elites], axis=0)

        # Gradient guidance
        if args.use_gradient_guide and mlp_model:
            grads = mlp_gradients(mlp_model, elite_vecs, device=device)
            step = 0.05
            elite_vecs = elite_vecs + step * grads
            elite_vecs = ensure_bounds(elite_vecs, lb, ub)
            elites = [Candidate.from_vector(vec) for vec in elite_vecs]

        # Real evaluation
        for cand in elites:
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
                "generation": gen,
                "wall_time_s": wall,
            }
            append_rows_csv(csv_path, fieldnames, [row])
            print(
                f"[Gen {gen}] acc={result.val_acc:.4f} flops={result.flops_g:.2f}G "
                f"lat={result.latency_ms:.1f}ms vec={cand.to_vector()}"
            )

        # Update population with elites + random injects
        new_pop = elite_vecs.tolist()
        while len(new_pop) < args.pop_size:
            new_pop.append(sample_candidate(space).to_vector())
        population = np.stack(new_pop[: args.pop_size], axis=0)

        # Periodic surrogate retrain
        if (gen + 1) % args.retrain_every == 0:
            if Path(csv_path).exists():
                mlp_metrics = train_mlp_proxy(csv_path, output_path=args.mlp_path)
                print(
                    f"[Retrain] MLP proxy -> MAE {mlp_metrics.val_mae:.4f} "
                    f"Pearson {mlp_metrics.val_pearson:.3f}"
                )
                mlp_model = load_mlp_proxy(args.mlp_path, device=device)
            else:
                print(f"[Retrain] Skipping surrogate retrain; CSV not found at {csv_path}")

    print("Search complete.")


if __name__ == "__main__":
    main()

