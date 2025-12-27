# Context Snapshot – Neuro-Symbolic NAS (RI-IWOA + OFA)

Last updated: 2025-12-28

## Current state
- OFA real-eval on ImageNet-A yields ~0.001 accuracy with head-only fine-tuning (50 steps on 500 images). ImageNet-A is adversarial; small adaptation is insufficient, so low accuracy is expected.
- Latency LUT: `data/latency_lut_full.json` is built for OFA MobileNetV3; `OFAEvaluator` uses it when present, otherwise falls back to timing.
- NAS search loop with RI-IWOA, semantic bubble-net, and online LoRA is wired; CSV logs in `data/nas_data*.csv`.
- Long search log: `data/logs/long_search.log` (20 gens on ImageNet-A real eval, still ~0.001 acc).

## Assets present
- OFA supernet weights: `data/ofa/ofa_mbv3_d234_e346_k357_w1.2`.
- NAS-Bench-201: `data/nasbench201/`; sanity script `scripts/nasbench201_sanity.py`.
- CIFAR-100 and CIFAR-100-C under `data/datasets/`; checker `scripts/check_cifar100c.py`.
- ImageNet-A extracted at `data/datasets/imagenet-a` (class folders directly under root).
- Tiny-ImageNet prep script: `scripts/prepare_tiny_imagenet.py`.
- Nanbeige LLM fetcher: `scripts/fetch_nanbeige.py` (model `Nanbeige/Nanbeige4-3B-Thinking-2511`).

## How to run
```bash
cd /home/shadeform/iwoa-nas
source .venv/bin/activate
export PYTHONPATH=.

# Proxy eval (latency-based)
python main.py --evaluator ofa --ofa_eval_mode proxy --csv_log data/nas_data.csv

# Real eval on ImageNet-A with head FT (example)
TOKENIZERS_PARALLELISM=false \
python main.py \
  --evaluator ofa --ofa_eval_mode real \
  --imagenet_root data/datasets/imagenet-a \
  --imagenet_subset 500 \
  --imagenet_remap_wnids \
  --ofa_head_ft_steps 50 \
  --ofa_head_ft_lr 5e-4 \
  --ofa_head_ft_weight_decay 0.0 \
  --ofa_cache data/ofa_eval_cache_imagenet_a.json \
  --generations 1 --pop_size 4 \
  --csv_log data/nas_data.csv
```

## Knobs to improve accuracy
- Use more/cleaner data: If you have ImageNet-1k, set `--imagenet_root /path/to/imagenet` (val/train ImageFolder). Increase `--imagenet_subset` (e.g., 5000) or drop it to use full set.
- Stronger adaptation: try `--ofa_head_ft_steps 500-2000`, `--ofa_head_ft_lr 5e-4`, `--ofa_head_ft_weight_decay 1e-4`. Optionally unfreeze the last block (code change) for a heavier tune.
- Caching: set `--ofa_cache` to reuse real-eval results keyed by rounded configs.

## Notes
- ImageNet-A layout is flat; loader handles it.
- CIFAR-100 auto-downloads; head resized to dataset classes.
- Latency LUT builder: `scripts/build_latency_lut_full.py` benchmarks all unique OFA blocks.
- LLM: Nanbeige model via HF; online LoRA uses QLoRA (NF4).
