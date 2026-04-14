# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**TCFDS** (Temporal Compression by Factored Dense Substitution) — compresses LLM weight matrices via data-aware spectral foliation (low-rank SVD factorization). Single-file Python project, no tests.

- **Main file:** `tcfds.py` — v6.4.0 (version string in docstring + 3 places in main())
- **Paper:** `paper/tcfds_paper.tex` (LaTeX source, compile with `pdflatex`)

## Running

```bash
# Compress a model
python tcfds.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eps 0.25 --save compressed.pt

# Load and chat
python tcfds.py --load compressed.pt --chat-only

# Install as package (optional)
pip install -e ".[full]"    # [full] includes datasets for WikiText-2 calibration
```

`HF_ENDPOINT=https://hf-mirror.com` may be needed when accessing HuggingFace from CN/Vast.ai.

## Architecture

```
TCFDSLinear (nn.Module)          ← drop-in replacement for nn.Linear
  └─ U (m×k), S (k), V (n×k)    ← low-rank factored weights
      └─ forward: x@V → *S → @U.T

TCFDSFwd (autograd.Function)     ← custom forward/backward
  └─ backward: grad_U = z.t() @ g, grad_V = x.t() @ gUs

data_aware_svd(W, Cov, eps)       ← core algorithm
  └─ eig(Cov) → W_weighted = W @ Cov^½ → SVD → V_back = V @ Cov^(-½)

compress_streaming()              ← main pipeline
  └─ single-pass covariance collection → per-layer compression

verify()                          ← 3 structural checks
  └─ modules replaced, memory reduced, SV ratios > 10
```

## Key Implementation Notes

- `get_block_groups()` uses `re.search(r'\.layers\.(\d+)\.', name)` — models must use this exact layer naming pattern. Gemma, Mistral, LLaMA use it; GPT-J/GPT-NeoX do not.
- Epsilon scheduling: first/last 2 blocks get `aeps *= 0.5` (tightened further by `max(0.02, ...)`).
- Covariances are stored in RAM as float32 (not VRAM). For large models: `3072² × 4 bytes ≈ 36 MB` per layer, ~1 GB total for 31B models.
- `torch.svd_lowrank` is used (GPU-safe via Lanczos iteration) — not forced to CPU unless MKL/eig crashes in `safe_eigh`.
- Model loading tries 3 strategies in order: `device_map="auto"` (multi-GPU), `low_cpu_mem_usage=True`, then bare CPU.

## Version Bump Checklist

When bumping version (e.g. v6.4.0 → v6.4.1), update ALL of:
1. `tcfds.py` docstring header
2. `tcfds.py` argparse description string
3. `tcfds.py` print banner string
4. `tcfds.py` `save_compressed` meta dict
5. `tcfds.py` `rpt["version"]`
6. `pyproject.toml` `version = "..."`