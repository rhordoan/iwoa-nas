import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluator import Candidate, CIFAR100Evaluator, EvalResult
from src.eval.ofaevaluator import OFAEvaluator
from src.eval.ofa_real_eval import OFARealEvaluator
from src.eval.cache import EvalCache
from src.llm.buffer import ExperienceBuffer
from src.llm.online_lora import train_online_lora
from src.nas.codec import Codec
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
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional cap per validation epoch for evaluator.")
    parser.add_argument("--device", type=str, default=None, help="Device override for evaluator and surrogates.")
    parser.add_argument("--retrain_every", type=int, default=5, help="Retrain surrogates every N generations.")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument("--evaluator", choices=["proxy", "ofa"], default="proxy", help="Evaluator backend.")
    parser.add_argument("--latency_lut", type=str, default=None, help="Optional latency_lut.json for OFA path.")
    parser.add_argument("--use_semantic_bubble", action="store_true", help="Enable semantic bubble-net (LLM mutations).")
    parser.add_argument("--p_semantic", type=float, default=0.2, help="Probability of LLM mutation per offspring.")
    parser.add_argument("--llm_model_name", type=str, default="Nanbeige/Nanbeige4-3B-Thinking-2511", help="Generative LLM for mutations.")
    parser.add_argument("--buffer_batch_size", type=int, default=4, help="Online LoRA trigger size.")
    parser.add_argument("--lora_output_dir", type=str, default="results/llm/nanbeige-lora", help="LoRA adapter output dir.")
    parser.add_argument("--ofa_eval_mode", choices=["proxy", "real"], default="proxy", help="OFA evaluator mode.")
    parser.add_argument("--ofa_cache", type=str, default="data/ofa_eval_cache.json", help="Cache file for OFA real eval results.")
    parser.add_argument("--imagenet_root", type=str, default=None, help="Path to ImageNet dataset (train/val) for OFA real eval.")
    parser.add_argument("--imagenet_subset", type=int, default=1000, help="Optional limit on number of validation samples for real eval.")
    parser.add_argument("--imagenet_remap_wnids", action="store_true", help="If set, map ImageFolder class WNIDs to ImageNet-1k indices for subsets like ImageNet-A/R.")
    parser.add_argument("--ofa_head_ft_steps", type=int, default=0, help="Optional head fine-tune steps before real eval.")
    parser.add_argument("--ofa_head_ft_lr", type=float, default=5e-4, help="Head fine-tune learning rate.")
    parser.add_argument("--ofa_head_ft_weight_decay", type=float, default=0.0, help="Head fine-tune weight decay.")
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


def iwoa_move(pop: np.ndarray, bounds: np.ndarray, progress: float, global_best: np.ndarray, eigvecs: np.ndarray | None = None) -> np.ndarray:
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
            else:
                rand_idx = np.random.randint(0, pop.shape[0])
                D = np.abs(C * pop[rand_idx] - x)
                x_new = pop[rand_idx] - A * D
            if eigvecs is not None:
                D = eigvecs @ D
            x_new = global_best - A * D
        else:
            D = np.abs(global_best - x)
            l = np.random.uniform(-1, 1)
            x_new = D * np.exp(spiral_c * l) * np.cos(2 * np.pi * l) + global_best
        x_new = np.clip(x_new, lb, ub)
        new_pop.append(x_new)
    return np.stack(new_pop, axis=0)


def ensure_bounds(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(pop, lb, ub)


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None


def load_generative_llm(model_name: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
    return tokenizer, model


def llm_mutate(
    vec: np.ndarray,
    tokenizer,
    model,
    codec: Codec,
    device: torch.device,
    base_image_size: int,
    max_new_tokens: int = 128,
) -> Optional[np.ndarray]:
    prompt = vector_to_text(codec.vec_to_candidate(vec), base_image_size)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    payload = parse_json_from_text(text)
    if not payload:
        return None
    return codec.json_to_vec(payload)


def compute_rotation_basis(pop: np.ndarray) -> Optional[np.ndarray]:
    if pop.shape[0] < 2:
        return None
    cov = np.cov(pop.T)
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    _, eigvecs = np.linalg.eigh(cov)
    return eigvecs


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    space = load_search_space(args.config)
    lb, ub = bounds_from_space(space)
    bounds = np.stack([lb, ub], axis=0)
    base_image_size = int(space.get("base_image_size", 224))
    codec = Codec.from_yaml(args.config)

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

    evaluator = None
    if args.evaluator == "proxy":
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
            max_val_batches=args.max_val_batches,
            seed=args.seed,
        )
    else:
        evaluator = OFAEvaluator(codec=codec, latency_lut=Path(args.latency_lut) if args.latency_lut else None, device=args.device)
        if args.ofa_eval_mode == "real":
            cache = EvalCache(Path(args.ofa_cache))
            evaluator.mode = "real"
            evaluator.real_eval = OFARealEvaluator(
                codec=codec,
                checkpoint_path=Path("data/checkpoints/ofa/ofa_mbv3_d234_e346_k357_w1.0.pth"),
                net_id="ofa_mbv3_d234_e346_k357_w1.0",
                cache=cache,
                data_root=Path(args.data_root) / "datasets" / "cifar-100-python",
                device=device,
                imagenet_root=Path(args.imagenet_root) if args.imagenet_root else None,
                imagenet_subset=args.imagenet_subset,
                imagenet_remap_wnids=args.imagenet_remap_wnids,
                head_ft_steps=args.ofa_head_ft_steps,
                head_ft_lr=args.ofa_head_ft_lr,
                head_ft_weight_decay=args.ofa_head_ft_weight_decay,
            ).evaluate

    # Init population
    population = np.stack([sample_candidate(space).to_vector() for _ in range(args.pop_size)], axis=0)

    # LLM generative (semantic bubble-net)
    gen_tokenizer = gen_model = None
    if args.use_semantic_bubble:
        gen_tokenizer, gen_model = load_generative_llm(args.llm_model_name)

    # Experience buffer for online LoRA
    buffer = ExperienceBuffer(max_size=128)
    best_val_acc = -1e9

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
        eigvecs = compute_rotation_basis(population)

        # Surrogate scoring (higher is better)
        if mlp_model is None:
            mlp_model = load_mlp_proxy(args.mlp_path, device=device)
        surrogate_scores = predict_mlp(mlp_model, population, device=device)
        best_idx = int(np.argmax(surrogate_scores))
        global_best = population[best_idx]

        offspring = iwoa_move(population, bounds, progress, global_best, eigvecs=eigvecs)
        candidates = [Candidate.from_vector(vec) for vec in offspring]

        # LLM filtering (ranker)
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

        # Semantic bubble-net mutation
        if args.use_semantic_bubble and gen_tokenizer and gen_model:
            mutated = []
            for vec in elite_vecs:
                if random.random() < args.p_semantic:
                    mut = llm_mutate(vec, gen_tokenizer, gen_model, codec, device, base_image_size)
                    if mut is not None:
                        mutated.append(mut)
                        continue
                mutated.append(vec)
            elite_vecs = ensure_bounds(np.stack(mutated, axis=0), lb, ub)
            elites = [Candidate.from_vector(v) for v in elite_vecs]

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
            # Success buffer: only if improvement over running best
            if result.val_acc > best_val_acc:
                best_val_acc = result.val_acc
                payload = codec.vec_to_json(cand.to_vector())
                buffer.add_if_improved(True, prompt=vector_to_text(cand, base_image_size), response=str(payload), json_payload=payload)

        # Update population with elites + random injects
        new_pop = elite_vecs.tolist()
        while len(new_pop) < args.pop_size:
            new_pop.append(sample_candidate(space).to_vector())
        population = np.stack(new_pop[: args.pop_size], axis=0)

        # Online LoRA on success buffer
        if len(buffer) >= args.buffer_batch_size and args.use_semantic_bubble and gen_tokenizer and gen_model:
            entries = [{"response": e.response, "json_payload": e.json_payload} for e in buffer.pop_all()]
            train_online_lora(entries, model_name=args.llm_model_name, output_dir=args.lora_output_dir)

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

