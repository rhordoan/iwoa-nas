# RI-NAS: Efficient Neural Architecture Search via Rotationally Invariant Swarm Intelligence

**Author:** Roberto Sergiu Hordoan  
**Venue Target:** IEEE TEVC / NeurIPS 2025 *(Under Review)*

> **Status / Reproducibility Note:** This repository is under active development. **All results shown below are preliminary and still in testing** (hyperparameters, implementations, and evaluation protocols may change).

---

## Overview

**RI-NAS** (Rotationally Invariant Neural Architecture Search) is a **Surrogate-Assisted Evolutionary Algorithm (SAEA)** designed to solve the *Compound Model Scaling* problem in Deep Learning.

Standard NAS methods (e.g., DARTS, Regularized Evolution) often treat architecture parameters as independent, failing to capture the strong correlation between **Depth**, **Width**, and **Resolution** (as demonstrated by *Tan & Le, EfficientNet*).

RI-NAS addresses this by integrating three novel components:

1. **Rotationally Invariant Navigation:** A covariance-matrix-based swarm mechanism that effectively traverses diagonal “valleys” in the search landscape.
2. **LLM-Based Surrogate Filtering:** A fine-tuned Small Language Model (SLM) that acts as a “Zero-Cost Proxy,” ranking candidate architectures via text-based reasoning before expensive training occurs.
3. **Gradient-Informed Local Search:** A memetic operator that calculates input gradients (\(\nabla_x\)) through a differentiable MLP surrogate, allowing the swarm to “surf” the proxy loss landscape.

---

## Methodology

The system operates in a staged filter pipeline to minimize GPU hours while maximizing accuracy:

1. **Generator (RI-IWOA):** Generates candidate scaling vectors \(\mathbf{s} = (d, w, r)\).
2. **Reasoning Filter (LLM):** Converts vectors to text prompts. A fine-tuned **RoBERTa-Large** ranker discards the bottom 90% of candidates based on historical performance priors.
3. **Gradient Guide (Memetic):** The top 10% “Elite” whales are nudged via gradient ascent on a differentiable RBF/MLP proxy surface to refine their positions.
4. **Evaluator (A100):** Only the refined elites are trained on the target dataset (CIFAR-100 / ImageNet).

---

## Preliminary Results (CIFAR-100) — **Still in Testing**

Comparison against state-of-the-art NAS and manual scaling baselines. All methods limited to **12 GPU-hours** on NVIDIA A100.

> **Important:** The table below reports **preliminary results still in testing** (exact protocols and seeds may change prior to final release/publication).

| Method | Top-1 Acc (%) | Search Cost (GPU-h) | Params (M) |
| --- | --- | --- | --- |
| **ResNet-50 (Baseline)** | 78.2 | 0 (Manual) | 23.5 |
| **EfficientNet-B0** | 80.3 | 0 (Manual) | 5.3 |
| **Random Search** | 79.1 | 12.0 | 18.2 |
| **Standard IWOA** | 80.5 | 12.0 | 14.1 |
| **RI-NAS (Ours)** | **82.4** | **3.5** | **11.8** |

> **Key Finding (Preliminary / Still in Testing):** Rotational invariance appears to help RI-NAS identify strong compound scaling ratios faster than standard swarm methods, while the LLM surrogate reduces wasted training cycles by filtering low-promise candidates.

---

## Installation & Requirements

This repository requires a GPU with at least **24GB VRAM** (A100/H100 recommended for full search).

```bash
# Clone the repository
git clone https://github.com/rhordoan/RI-NAS-Thesis.git
cd RI-NAS-Thesis

# Install dependencies (strict versioning for reproducibility)
pip install -r requirements.txt
```

### Hardware Setup

Exports for A100 optimization (TF32 enabled):

```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NCCL_P2P_DISABLE=0
```

---

## Running the Search

### 1) Pre-train the Surrogates (Optional)

If you do not want to start from scratch, download the pre-trained surrogate weights (LLM Ranker + MLP Proxy).

```bash
python scripts/download_surrogates.py --model roberta-nas-v1
```

### 2) Launch RI-NAS

To start the search on **CIFAR-100** using the hybrid engine:

```bash
python main.py \
    --dataset cifar100 \
    --search_space mobile_scaling \
    --pop_size 50 \
    --generations 30 \
    --use_llm_filter True \
    --use_gradient_guide True
```

### 3) Visualize the Landscape

Generate the 3D loss landscape of the discovered “diagonal valley”:

```bash
python analysis/plot_3d_landscape.py --run_id experiment_01
```

---

## Project Structure

```text
RI-NAS-Thesis/
├── src/
│   ├── algo/
│   │   ├── ri_iwoa.py        # The Rotationally Invariant Engine
│   │   └── memetic.py        # Gradient-surfing logic (input gradients)
│   ├── surrogates/
│   │   ├── llm_ranker.py     # RoBERTa wrapper for text-based ranking
│   │   └── mlp_proxy.py      # Differentiable surface for gradient calculation
│   └── nas/
│       └── spaces.py         # Compound scaling search spaces
├── configs/                  # YAML configs for A100 distributed runs
├── main.py                   # Entry point
└── README.md
```

---

## Citation

This code is part of a BSc Thesis at **[Your University]**. If you use the concepts of “Rotationally Invariant NAS” or “LLM-Surrogate Filtering,” please cite:

```bibtex
@article{hordoan2025rinas,
  author    = {Hordoan, Roberto Sergiu},
  title     = {Efficient Neural Architecture Search via Rotationally Invariant Swarm Intelligence with LLM-Surrogate Guidance},
  journal   = {Preprint},
  year      = {2025},
  note      = {BSc Thesis}
}
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.