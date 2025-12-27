# Implementation Plan: Neuro-Symbolic NAS using RI-IWOA and Online Evolutionary LoRA

## 1. System Architecture Overview

The system consists of three decoupled engines that interact cyclically:

1. **The Evaluation Engine (Oracle):** A zero-cost evaluator using a One-Shot Supernet and Hardware Look-Up Tables (LUT).
2. **The Cognitive Engine (LLM):** A "Thinking" Small Language Model (SLM) equipped with an Online LoRA adapter to learn from search history.
3. **The Optimization Engine (RI-IWOA):** A numerical evolutionary algorithm that manages the population and drives global exploration via Covariance Matrix Adaptation.

---

## 2. Phase 1: The Evaluation Engine (Zero-Cost Proxy)

**Goal:** Decouple the search from the training loop to ensure feasibility on consumer hardware.

### 2.1. The Supernet (One-Shot NAS)

- **Asset:** Use the **Once-for-All (OFA)** pre-trained MobileNetV3 supernet (trained on ImageNet).
- **Mechanism:** Instead of training candidate architectures, the system extracts the specific sub-network weights from the supernet and runs inference on a validation batch.
- **Implementation:**
  - Load the OFA supernet into GPU memory.
  - Implement `get_net_for_config(config_dict)`: Returns a `torch.nn.Module` ready for inference.

### 2.2. Latency Look-Up Table (LUT)

- **Rationale:** Real-time latency measurement is noisy. A pre-computed table ensures consistency.
- **Construction:**
  - Iterate through every distinct operation type in the search space (e.g., *MBConv_k3_e3*, *MBConv_k5_e6*).
  - Benchmark each operation 500 times on the target hardware (e.g., laptop GPU or Jetson).
  - Store mean latency in `latency_lut.json`.
- **Inference:** Total Latency = Sum of layer latencies from the LUT.

### 2.3. The Codec

- **Role:** Translates between the continuous vector space of the optimizer and the discrete JSON space of the LLM.
- **Functions:**
  - `vec_to_json(vector)`: Maps continuous ranges to discrete attributes (e.g., Kernel 3).
  - `json_to_vec(json)`: Maps the LLM's output back to the numerical population.

---

## 3. Phase 2: The Cognitive Engine (Online Evolutionary LoRA)

**Goal:** A "Thinking" mutation operator that improves its suggestions based on environmental feedback.

### 3.1. Model Configuration

- **Base Model:** `Nanbeige4-3B-Thinking` (or equivalent reasoning-tuned SLM).
- **Quantization:** 4-bit Normal Float (NF4) via `bitsandbytes` to fit in <4GB VRAM.
- **Adapter:** Initialize a blank **LoRA (Low-Rank Adaptation)** adapter (Rank=8, Alpha=16) targeting attention modules (`q_proj`, `v_proj`).

### 3.2. The Experience Buffer (Trace-Preserving)

- **Logic:** To maintain the model's reasoning capability, we train on the full "Chain of Thought," not just the final answer.
- **Storage:** A FIFO buffer storing only **successful** mutations.
- **Entry Format:**

```text
User: Optimize Layer 2...
Assistant: <thinking> Layer 2 has high latency... I will reduce kernel size. </thinking>
JSON: { "layer_2": ... }
```

- **Condition:** An entry is added ONLY if `Fitness(Child) > Fitness(Parent)`.

### 3.3. The Online Training Loop

- **Trigger:** Activates every time the buffer collects new samples (e.g., N samples / batch size threshold).
- **Process:**
  1. **Pause Search.**
  2. Format buffer into a causal language modeling dataset.
  3. Run 1-2 epochs of QLoRA fine-tuning on the adapter.
  4. **Clear Buffer** (to prevent overfitting to old data).
  5. **Resume Search.**

---

## 4. Phase 3: The Numerical Engine (RI-IWOA)

**Goal:** Handle global exploration and prevent mode collapse using statistical mathematics.

### 4.1. Covariance Matrix Adaptation (The "RI" Component)

- **State:** Maintain a global Covariance Matrix \(C\) derived from the top 50% of the population.
- **Eigen-Decomposition:** Every \(k\) generations, compute Eigenvectors \(B\) and Eigenvalues \(D\) of \(C\).
- **Rotation:** During the "Search" phase, mutation steps are rotated by \(B\) to align with the fitness landscape's valleys.

### 4.2. The Semantic Bubble-Net (The "Cognitive" Component)

- **Logic:** Replaces the standard geometric spiral (\(A\)) with an LLM call.
- **Probabilistic Switch:**

```python
if random.random() < p_semantic:
    # LLM Reasoning Attack
    child = llm.generate_mutation(parent, context="reduction of parameters")
else:
    # Standard Numerical Spiral
    child = geometric_spiral(parent, best_whale)
```

- **Fallback:** If the LLM generates invalid JSON, discard the result and revert to the geometric spiral for that iteration.

---

## 5. Phase 4: Execution Workflow

This is the main loop of the experiment.

1. **Initialization:**
   - Spawn Population \(P\) randomly.
   - Initialize `Nanbeige-4B` in 4-bit mode.
   - Build/Load Latency LUT.

2. **Generational Loop:**
   - **Sort Population** by Fitness \(F\).
   - **Update Global Best** \(\mathbf{x}^*\).
   - **Update Covariance:** If `gen % 5 == 0`, recalculate Eigen-basis.
   - **Whale Cycle:**
     - For each whale, determine phase: *Encircling*, *Search*, or *Attacking*.
     - If *Attacking*: Trigger **Semantic Bubble-Net** (LLM) or Geometric Spiral.
     - If *Search*: Apply **Rotationally Invariant** mutation.
   - **Evaluation:**
     - Evaluate new candidates via Supernet.
     - **Feedback:** If a Semantic Mutation improved fitness, add the trace to the **Experience Buffer**.
   - **Learning:**
     - Check `if len(buffer) > batch_size`.
     - If true, trigger **Online LoRA Fine-Tuning**.

3. **Termination:**
   - After Max Generations, run a local Nelder-Mead search on \(\mathbf{x}^*\) for final integer tuning.

4. **Final Export:**
   - Output the architecture config and the trained LoRA adapter.

---

## 6. Phase 5: Validation Protocols

### 6.1. Algorithmic Proof (NAS-Bench-201)

- **Objective:** Statistical validation of convergence speed.
- **Setup:** Replace the OFA Supernet with the `nasbench201_api`.
- **Metric:** Plot "Best Accuracy vs. Search Time" over 30 independent runs.
- **Success Criteria:** RI-IWOA must reach the global optimum (or near-optimum) faster than Random Search and Standard WOA.

### 6.2. The "Max Grade" Benchmark (CIFAR-100 & Robustness)

- **Objective:** Demonstrate real-world applicability and model quality.
- **Search:** Perform the full search on CIFAR-100 using the OFA Supernet.
- **Retraining:** Take the single best discovered architecture and train it from scratch (100 epochs, Cosine Annealing).
- **Robustness Test:** Evaluate the trained model on **CIFAR-100-C** (Corrupted) to measure stability against noise/blur.

---

## 7. Implementation Checklist

- [ ] **GPU Environment:** Verify `bitsandbytes` 4-bit loading works on the target machine.
- [ ] **LUT:** Ensure `latency_lut.json` is generated for the specific hardware used for the demo.
- [ ] **Parsing Logic:** Write robust Regex to extract JSON from the LLM's "Thinking" output.
- [ ] **Buffer Logic:** Verify that *only* successful mutations are entering the training buffer.
- [ ] **Rotation Math:** Ensure Eigen-decomposition handles singular matrices (add small \(\epsilon\) to diagonal).

---

## Appendix A: Dependencies & Download Links

This section lists **canonical download/install pages** for every external dependency or asset referenced above.

### A.1. Pretrained Supernet (OFA)

- **Once-for-All (MIT HAN Lab) code + pretrained checkpoints/model zoo**: [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all)
  - Use the repo’s README “model zoo / pretrained models” instructions to download the exact MobileNetV3 OFA supernet checkpoint you need.
- **OFA-Sys implementation (alternate OFA repo)**: [OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)

### A.2. LLM / SLM + Online LoRA Tooling

- **Base model download (Hugging Face model hub)**:
  - Default checkpoint: `Nanbeige/Nanbeige4-3B-Thinking-2511`
  - If you later pick a different reasoning-tuned SLM, put its Hugging Face model card link here.
- **Transformers (inference/training)**: [huggingface/transformers](https://github.com/huggingface/transformers)
- **PEFT (LoRA / QLoRA adapters)**: [huggingface/peft](https://github.com/huggingface/peft)
- **bitsandbytes (4-bit NF4 quantization)**: [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

### A.3. Benchmarks / Datasets (Validation Protocols)

- **NAS-Bench-201 (benchmark + API)**: [D-X-Y/NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
  - This repo provides the API usage and points to the official benchmark file download(s).
- **CIFAR-100 (official download)**: [CIFAR-10 and CIFAR-100 datasets (Toronto)](https://www.cs.toronto.edu/~kriz/cifar.html)
- **CIFAR-100-C (corruptions benchmark)**: [hendrycks/robustness](https://github.com/hendrycks/robustness)
  - This repo documents how to obtain and use CIFAR-C style corruptions for CIFAR-10/100.

### A.4. Core Frameworks (Install Pages)

- **PyTorch**: [PyTorch – Get Started](https://pytorch.org/get-started/locally/)


