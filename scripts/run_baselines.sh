#!/usr/bin/env bash
set -euo pipefail

# Random search baseline
python scripts/random_search.py --samples 100 --output data/random_baseline.csv

# Standard IWOA baseline (no LLM / gradient guide; uses surrogate MLP only if available)
python main.py --pop_size 50 --generations 20 --csv_log data/iwoa_baseline.csv

# Full RI-NAS (LLM + gradient guide)
python main.py --use_llm_filter --use_gradient_guide --pop_size 50 --generations 30 --csv_log data/rinas_run.csv

# Plot convergence for the latest run
python analysis/plot_convergence.py --csv data/rinas_run.csv --out results/convergence.png

