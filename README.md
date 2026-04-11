[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

# TCFDS — Data-Aware Spectral Compression for LLMs

**Compression par Feuilletage Dynamique Spectral**

TCFDS compresses large language models by replacing dense weight matrices with low-rank factored representations, guided by actual input activation statistics. Unlike quantization methods that reduce bit precision, TCFDS performs **structural compression** — it reduces the number of parameters by exploiting the low spectral entropy of weight matrices. The result: 2-5x smaller models that preserve output quality, with no fine-tuning required.

---

## What is TCFDS?

Most LLM compression tools (GPTQ, AWQ) work by **quantizing** weights — reducing the number of bits per parameter (e.g. 16-bit to 4-bit). This is effective but fundamentally limited: you can't go below 1 bit per weight without losing information.

TCFDS takes a different approach: **factorization**. It decomposes each weight matrix `W (m x n)` into three smaller matrices `U (m x k)`, `S (k)`, `V (n x k)` where `k << min(m, n)`. This exploits the fact that weight matrices in trained LLMs have low **spectral entropy** — most of the information is concentrated in a few dominant singular values.

The key innovation is **data-aware decomposition**: instead of minimizing the abstract matrix approximation error `||W - W'||`, TCFDS minimizes the **expected output error** `E[||Wx - W'x||^2]` over the actual input distribution. This is achieved by weighting the SVD with the covariance matrix of input activations collected during calibration.

Because TCFDS operates on the parameter structure rather than precision, it is **composable with quantization** — you can apply GPTQ or AWQ to the factored U, S, V matrices for further savings.

---

## How It Works

### The Core Idea

A standard `nn.Linear` layer stores a dense weight matrix `W` with `m x n` parameters. TCFDS replaces it with a factored representation:

```
Standard nn.Linear:              TCFDS Factored (TCFDSLinear):

x ──> [ W (m x n) ] ──> y       x ──> [V (n x k)] ──> [* S (k)] ──> [U.T (k x m)] ──> y

Parameters: m * n                Parameters: k * (m + n) + k
Example: 2048 x 5632 = 11.5M    Example: k=200 → 200 * 7680 + 200 = 1.5M  (7.5x smaller)
```

The rank `k` is selected per-layer via a spectral energy threshold: keep enough singular values to capture `(1 - eps^2/2)` of the total spectral energy, where `eps` controls the compression-quality tradeoff.

### Data-Aware SVD

Standard SVD decomposes `W` to minimize `||W - W_k||_F` (Frobenius norm). This treats all input directions equally, which is suboptimal — some directions are never activated in practice.

TCFDS computes the **input activation covariance** `C = E[x x^T]` from calibration data, then decomposes in the covariance-weighted space:

```
1. Compute C^{1/2} via eigendecomposition of C
2. Form W_weighted = W @ C^{1/2}
3. SVD on W_weighted → U, S, V_weighted
4. Transform back: V = C^{-1/2} @ V_weighted
```

This ensures the approximation error is small **where it matters** — along the input directions that the model actually uses. The data-aware error is typically 20-40% lower than the Frobenius error for the same rank.

### Adaptive Epsilon

Not all layers are equally sensitive to compression. TCFDS calibrates sensitivity per-layer by injecting small noise and measuring the impact on loss:

```
adaptive_eps(layer) = base_eps * (0.3 + 1.7 * (1.0 - sensitivity))
```

Sensitive layers (attention projections) get tighter tolerances. Robust layers (feed-forward) get compressed more aggressively.

### Compression Pipeline

```
┌─────────────┐     ┌──────────────────┐      ┌───────────────────────┐
│  Load Model │ ──> │  Baseline PPL    │ ──>  │  Calibrate            │
│    (HF)     │     │  + sample outputs│      │  - Collect covariances│
└─────────────┘     └──────────────────┘      │  - Measure sensitivity│
                                              └───────────┬───────────┘
                                                          │
                                                          v
                                              ┌───────────────────────┐
                                              │  Stream Compress      │
                                              │  For each block:      │
                                              │   - Collect act. covs │
                                              │   - Data-aware SVD    │
                                              │   - Replace layers    │
                                              │   - Free memory       │
                                              │  Peak mem: ~230 MB    │
                                              └───────────┬───────────┘
                                                          │
                                                          v
                                              ┌───────────────────────┐
                                              │  Verify & Save        │
                                              │  - Check modules      │
                                              │  - Check memory       │
                                              │  - Check SVD struct   │
                                              │  - Measure post-PPL   │
                                              └───────────────────────┘
```

Block-by-block streaming limits peak memory to ~230 MB (vs ~5 GB if processing all layers at once).

---

## Comparison with Existing Methods

| Feature | **TCFDS** | GPTQ | AWQ | LoRA | QLoRA | SqueezeLLM | AQLM |
|---|---|---|---|---|---|---|---|
| **Approach** | Data-aware SVD factorization | 2nd-order quantization | Activation-aware quantization | Low-rank adaptation | Quantized + LoRA | Non-uniform quantization | Additive quantization |
| **Compression type** | Structural (fewer params) | Precision (fewer bits) | Precision (fewer bits) | Additive (extra params) | Precision + additive | Precision (fewer bits) | Precision (codebooks) |
| **Typical ratio** | 2-5x | 2-4x (4-bit) | 2-4x (4-bit) | N/A (adds params) | 2-4x + adapters | 2-4x | 2-8x |
| **Requires fine-tuning** | No | No | No | Yes | Yes | No | Yes (codebook) |
| **Calibration data** | Yes (64 sentences) | Yes (~128 samples) | Yes (~128 samples) | Training set | Training set | Yes | Yes |
| **Error bound** | Spectral isometry (proven) | 2nd-order Taylor (approx.) | Empirical | Task-specific | Task-specific | Empirical | Empirical |
| **Composable with quant** | Yes | No (already quantized) | No (already quantized) | Yes | Includes quant | No | No |
| **GPU required** | Optional | Yes | Yes | Yes | Yes | Yes | Yes |
| **Inference overhead** | Minimal (2 matmuls) | None | None | Small (adapter) | Small (adapter) | None | Codebook lookup |

### Key differences

- **TCFDS vs GPTQ/AWQ**: TCFDS reduces parameter count; quantization reduces bit-width. They are orthogonal and composable. TCFDS provides proven error bounds; quantization methods rely on empirical validation.
- **TCFDS vs LoRA/QLoRA**: LoRA adds trainable low-rank matrices during fine-tuning but doesn't compress the base model. TCFDS compresses without any training. LoRA is better when you need task-specific adaptation; TCFDS is better for deployment compression.
- **TCFDS vs SqueezeLLM/AQLM**: These are advanced quantization variants. TCFDS operates on a fundamentally different axis (structure vs precision) and can be combined with them.

---

## Quick Start

### Install

```bash
pip install torch transformers numpy psutil

# With CUDA (recommended for faster compression)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy psutil
```

### Compress a model

```bash
python tcfds.py --eps 0.25 --save compressed.pt
```

### Load and chat

```bash
python tcfds.py --load compressed.pt --chat-only
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--eps` | `0.25` | Compression tolerance (lower = less compression, better quality) |
| `--save` | None | Save compressed model to `.pt` file |
| `--load` | None | Load a previously compressed model |
| `--chat-only` | `False` | Skip verification, go straight to interactive chat |

---

## Epsilon Guide

Epsilon (`eps`) controls the compression-quality tradeoff. It sets the spectral energy threshold: retain enough singular values to capture `(1 - eps^2/2)` of the total energy.

| Epsilon | Compression | Quality | PPL Ratio | Use Case |
|---------|-------------|---------|-----------|----------|
| `0.08` | ~1.2x | Near-lossless | < 1.01 | Research / quality-critical |
| `0.15` | ~1.5x | Excellent | < 1.05 | Production deployment |
| `0.25` | ~2-3x | Good | < 1.10 | Balanced (default) |
| `0.35` | ~3-5x | Acceptable | < 1.20 | Memory-constrained |
| `0.50` | ~5x+ | Degraded | > 1.20 | Experimental only |

PPL ratio = perplexity_after / perplexity_before. A ratio of 1.05 means 5% quality loss.

---

## Benchmarks

Tested on **TinyLlama-1.1B-Chat-v1.0**:

| Epsilon | Layers Compressed | Compression Ratio | PPL Before | PPL After | PPL Ratio | Time (RTX 2060) |
|---------|-------------------|-------------------|------------|-----------|-----------|-----------------|
| 0.08 | 12 | ~1.2x | 35.59 | 35.42 | 0.995 | ~8 min |
| 0.25 | 32 | ~2.5x | 35.59 | 36.82 | 1.035 | ~12 min |
| 0.35 | 39 | ~3.5x | 35.59 | 35.24 | 0.990 | ~15 min |

Layer-level observations:
- **Attention projections** (`q_proj`, `k_proj`) are highly compressible (low spectral entropy)
- **Value/output projections** (`v_proj`, `o_proj`) are moderately compressible
- **MLP layers** (`gate_proj`, `up_proj`, `down_proj`) are harder to compress with standard SVD, but benefit significantly from data-aware weighting

---

## Architecture

### Key Components

**`TCFDSLinear`** (`nn.Module`) — Drop-in replacement for `nn.Linear`. Stores factored weights `(U, S, V)` instead of dense `W`. Forward pass computes `x @ V * S @ U^T` without ever materializing the full matrix.

**`TCFDSFwd`** (`torch.autograd.Function`) — Custom autograd for efficient forward and backward passes. Gradients are computed in the k-dimensional factored space, never allocating an (m x n) matrix.

**`data_aware_svd(W, Cov, eps)`** — Core algorithm. Computes covariance-weighted SVD with adaptive rank selection. Uses a two-phase approach: fast probe with `torch.svd_lowrank` to estimate rank, then targeted decomposition.

**`safe_eigh(M)`** — Eigendecomposition with NumPy fallback. Works around Intel MKL crashes on large matrices (observed on 5632x5632).

**`compress_streaming()`** — Block-by-block compression pipeline. Processes one transformer block at a time to limit peak VRAM to ~230 MB.

### Memory Layout

```
Before compression:          After compression:
┌──────────────────────┐     ┌──────────┐ ┌───┐ ┌──────────┐
│    W (m x n)         │     │ U (m x k)│ │S k│ │ V (n x k)│
│    e.g. 2048 x 5632  │     │          │ │   │ │          │
│   = 11,534,336 params│     │          │ │   │ │          │
└──────────────────────┘     └──────────┘ └───┘ └──────────┘
                              Total: k*(m+n+1) params
                              e.g. k=200 → 1,536,200 params (7.5x)
```

---

## Verification

Compression is verified with 3 structural checks:

1. **Modules replaced** — Every eligible `nn.Linear` is replaced with `TCFDSLinear`. No dense weight matrices remain.
2. **Memory reduced** — Total parameter count after compression is strictly less than before.
3. **Structured SVD** — Singular value ratios exceed 10, confirming the decomposition captures real structure (not random noise).

Standalone verification after loading a compressed model:

```python
# Built into tcfds.py
python tcfds.py --load compressed.pt
# Runs all 3 checks automatically before entering chat
```

---

## Hardware Requirements

| Setup | RAM | GPU | Compression Time | Notes |
|-------|-----|-----|------------------|-------|
| **Recommended** | 16 GB | RTX 2060+ (6 GB VRAM) | ~15 min (1.1B model) | CUDA accelerated SVD + covariance |
| **Minimum** | 8 GB | None (CPU only) | ~2 hours (1.1B model) | Covariance on CPU, MKL fallback |
| **Tested** | 16 GB | RTX 2060 (6 GB) | 15 min | Primary development hardware |
| **Tested** | 8 GB | None | 2 hrs | Intel i7-8565U, Intel i5-3rd gen |

The streaming compression pipeline limits peak memory to ~230 MB per block regardless of model size.

---

## Project Structure

```
tcfds/
├── tcfds.py              # Main compression script (v6.3)
├── paper/
│   ├── tcfds_paper.tex   # Research paper (LaTeX source)
│   └── tcfds_paper.pdf   # Research paper (compiled)
├── pyproject.toml        # Python packaging metadata
├── requirements.txt      # Dependencies
├── LICENSE               # MIT
├── .gitignore
└── README.md
```

---

## Research Paper

The theoretical foundations of TCFDS are detailed in `paper/tcfds_paper.pdf`:

- **Spectral Entropy** and rank selection bounds
- **Main Theorem (TCFDS)** — Formal error bound with spectral foliation and ergodic dynamics
- **Data-Aware Extension** — Covariance-weighted decomposition with adaptive epsilon
- **Hardware Complexity** — Roofline analysis showing TCFDS converts memory-bound operations into compute-bound ones
- **Empirical Validation** — Benchmarks on TinyLlama-1.1B with comparison to GPTQ, AWQ, LoRA

---

## Citation

```bibtex
@software{tcfds2026,
  title   = {TCFDS: Data-Aware Spectral Foliation for LLM Weight Compression},
  author  = {tensorcrush},
  year    = {2026},
  url     = {https://github.com/tensorcrush/tcfds},
  license = {MIT}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
