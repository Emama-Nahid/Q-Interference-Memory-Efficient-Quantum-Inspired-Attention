# Q-Interference

Q-Interference is a GPT-style language model with **quantum-inspired phase-aware attention** and an **exact memory-bypassed factorization**.

The model keeps the standard GPT backbone and only changes the attention compatibility score:

- **Baseline GPT**: standard scaled dot-product attention
- **Naive interference attention**: direct phase-aware token-pair-feature interaction
- **Q-Interference**: exact trigonometric factorization that avoids explicitly materializing the large `T x T x d_h` intermediate tensor

This repository is designed as a **VS Code-ready starter project** for:

- debugging on a single GPU
- scaling to multi-GPU training with `torchrun`
- running controlled baseline comparisons
- collecting memory, perplexity, throughput, and ablation results for a NeurIPS-style paper

## Repository layout

```text
Q-Interference/
├── .vscode/
├── configs/
├── scripts/
├── src/q_bypass_gpt/
│   ├── data/
│   ├── models/
│   ├── modules/
│   ├── tests/
│   └── utils/
└── outputs/
```

## Main experiment logic

You should run experiments in this order:

1. **Debug**: single GPU, tiny config
2. **Controlled comparison**: baseline GPT vs Q-Interference
3. **Ablation study**
4. **Memory profiling across sequence lengths**
5. **Larger multi-GPU runs**

## Setup

```bash
conda create -n qbypassgpt python=3.10 -y
conda activate qbypassgpt

pip install -r requirements.txt
pip install -e .
```

## Quick start

### 1) Run tests

```bash
pytest -q src/q_bypass_gpt/tests
```

### 2) Single-GPU debug run

```bash
CUDA_VISIBLE_DEVICES=0 python -m q_bypass_gpt.train --config configs/debug_qbypass.yaml
```

### 3) Baseline run

```bash
CUDA_VISIBLE_DEVICES=0 python -m q_bypass_gpt.train --config configs/debug_baseline.yaml
```

### 4) Multi-GPU run with torchrun

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m q_bypass_gpt.train --config configs/main_qbypass.yaml
```

## Evaluation

### Evaluate a checkpoint

```bash
python -m q_bypass_gpt.evaluate \
  --config configs/main_qbypass.yaml \
  --checkpoint outputs/main_qbypass/checkpoints/best.pt \
  --split test
```

### Generate text

```bash
python -m q_bypass_gpt.generate \
  --config configs/main_qbypass.yaml \
  --checkpoint outputs/main_qbypass/checkpoints/best.pt \
  --prompt "Quantum-inspired attention may help"
```

### Profile attention memory

```bash
python -m q_bypass_gpt.profiler --config configs/debug_qbypass.yaml
```

## Suggested paper results to report

### Main table
- validation loss
- test perplexity
- peak GPU memory
- tokens/sec
- parameters

### Key ablations
- baseline GPT
- naive interference attention
- Q-Interference
- no-phase variant
- different amplitude activations
- different phase bounding strategies
- context length scaling

## Recommended first paper figures

1. **Peak memory vs sequence length**
2. **Perplexity vs training steps**
3. **Throughput vs sequence length**
4. **Ablation table**
5. **Naive vs bypass exactness sanity test**

## Notes

- The repository is intentionally modular and readable so you can later share it on GitHub with your professor.
- The implementation is a strong starter codebase, not a final benchmark-optimized framework.
- For the strongest paper, keep the comparison controlled: same optimizer, same data, same parameter scale, same training budget.
