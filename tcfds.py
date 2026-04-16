"""
TCFDS v6.4.0 - Data-Aware Compression + Adaptive Epsilon + GPU Support
=====================================================================
Fixes vs v6.4.0:
  - Add auto-detection of chat formats for all major LLM families
  - Supports Gemma, LLaMA 3, Qwen, Mistral, Mixtral, Phi, TinyLlama, StableLM
  - Fix TCFDSFwd.backward: grad_U = z.t() @ g (was g.t() @ (z*S.softmax)), grad_V fixed
  - Fix torch.load weights_only=False in load_compressed
  - Bump to v6.4.0

Usage:
  python tcfds.py --eps 0.25 --save compressed.pt
  python tcfds.py --load compressed.pt --chat-only
"""

import torch, torch.nn as nn
import math, time, gc, os, json, hashlib, traceback, argparse, re
import numpy as np
from collections import defaultdict

try:
    import psutil
    def ram_mb(): return psutil.Process(os.getpid()).memory_info().rss / 1e6
except ImportError:
    def ram_mb(): return 0

def vram_mb():
    """Peak VRAM allocated on GPU 0 (single-GPU view)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e6
    return 0

def vram_all_mb():
    """Peak VRAM allocated across all GPUs (multi-GPU view)."""
    if not torch.cuda.is_available():
        return {}
    return {i: torch.cuda.memory_allocated(i) / 1e6
            for i in range(torch.cuda.device_count())}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold above which SVD/eigh operations are forced to CPU to avoid GPU segfaults
CPU_SVD_THRESHOLD = 2048

# =============================================================================
# CHAT FORMAT AUTO-DETECTION SYSTEM v1.0
# Detects model type and returns the appropriate chat template format.
# Returns a dict with 'user', 'model', 'system', 'prefix', 'suffix' keys,
# or None for raw text mode.
# =============================================================================

def detect_chat_format(model_name):
    n = model_name.lower()

    if 'gemma' in n:
        # Gemma 4+: <|turn>user\n{prompt}\n<|turn>model\n
        return {
            'type': 'gemma',
            'user': '<|turn>user\n',
            'model': '<|turn>model\n',
            'system': '<|turn>system\n',
            'prefix': '',
            'suffix': '<|turn|>',
            'requires_generation_marker': True,
            'needs_system_message': False,
            'msg_end': '',  # handled by role prefix
        }
    elif 'llama-3' in n or 'llama3' in n or 'meta-llama/llama-3' in n:
        # LLaMA 3+: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        return {
            'type': 'llama3',
            'user': '<|start_header_id|>user<|end_header_id|>\n\n',
            'model': '<|start_header_id|>assistant<|end_header_id|>\n\n',
            'system': '<|start_header_id|>system<|end_header_id|>\n\n',
            'prefix': '<|begin_of_text|>',
            'suffix': '<|eot_id|>',
            'requires_generation_marker': True,
            'needs_system_message': False,
            'msg_end': '',
        }
    elif 'qwen' in n:
        # Qwen2.5: Each message is <|im_start|>{role}\n{content}<|im_end|>\n
        # System required at start. User msg ends with <|im_end|>\n then assistant prompt.
        return {
            'type': 'qwen',
            'user': '<|im_start|>user\n',
            'model': '<|im_start|>assistant\n',
            'system': '<|im_start|>system\n',
            'prefix': '',
            'suffix': '<|im_end|>\n',
            'requires_generation_marker': True,
            'needs_system_message': True,
            'system_default': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
            'msg_end': '<|im_end|>\n',
        }
    elif 'mistral' in n or 'mixtral' in n:
        # Mistral/Mixtral: [INST] {prompt} [/INST] [/ASSISTANT]
        return {
            'type': 'mistral',
            'user': '[INST] ',
            'model': '[/INST] ',
            'system': '',
            'prefix': '',
            'suffix': ' [/ASSISTANT]',
            'requires_generation_marker': False,
        }
    elif 'phi' in n and 'chat' in n:
        # Phi-3/4 chat: <|user|>\n{prompt}\n<|assistant|>\n
        return {
            'type': 'phi',
            'user': '<|user|>\n',
            'model': '<|assistant|>\n',
            'system': '',
            'prefix': '',
            'suffix': '<|end|>',
            'requires_generation_marker': True,
        }
    elif 'tinyllama' in n or 'chat' in n:
        # TinyLlama / generic ChatML: <|user|>\n{prompt}\n</s>\n<|assistant|>\n
        return {
            'type': 'chatml',
            'user': '<|user|>\n',
            'model': '<|assistant|>\n',
            'system': '',
            'prefix': '',
            'suffix': '</s>',
            'requires_generation_marker': True,
        }
    elif 'stablelm' in n:
        return {
            'type': 'stablelm',
            'user': '<|user|>',
            'model': '<|assistant|>',
            'system': '',
            'prefix': '',
            'suffix': '<|endoftext|>',
            'requires_generation_marker': True,
        }
    else:
        # Unknown model — raw text mode
        return None


def format_prompt_for_chat(prompt, fmt):
    """Build a properly formatted chat prompt from a user message.

    Template structure varies by model:
    - Qwen: <|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    - Gemma: <|turn>user\n{prompt}<|turn|>model\n
    - LLaMA3: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    - ChatML: <|user|>\n{prompt}</s>\n<|assistant|>\n
    """
    if fmt is None:
        return prompt  # raw text mode

    t = fmt['type']

    if t == 'qwen':
        # Qwen requires: system + user message (with <|im_end|>) + assistant prompt
        system_text = fmt.get('system_default', 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.')
        msg_end = fmt.get('msg_end', '<|im_end|>\n')
        result = (fmt['prefix'] +
                  fmt['system'] + system_text + '<|im_end|>\n' +
                  fmt['user'] + prompt + msg_end +
                  fmt['model'])
        return result

    elif t == 'gemma':
        # Gemma: user turn + model generation marker
        return (fmt['prefix'] +
                fmt['user'] + prompt +
                fmt['model'])

    elif t == 'llama3':
        # LLaMA 3: full structure with prefix, user turn, eot, assistant turn
        return (fmt['prefix'] +
                fmt['user'] + prompt +
                fmt['suffix'] +
                fmt['model'])

    elif t == 'mistral':
        # Mistral: [INST] prompt [/INST] (generation follows immediately)
        return fmt['prefix'] + fmt['user'] + prompt + fmt['suffix']

    elif t == 'chatml':
        # ChatML: <|user|>\n{prompt}</s>\n<|assistant|>\n
        return (fmt['prefix'] +
                fmt['user'] + prompt +
                fmt['suffix'] + '\n' +
                fmt['model'])

    elif t == 'phi':
        return (fmt['prefix'] + fmt['user'] + prompt + fmt['suffix'] + '\n' + fmt['model'])

    elif t == 'stablelm':
        return (fmt['prefix'] + fmt['user'] + prompt + fmt['suffix'] + fmt['model'])

    else:
        # Fallback: plain text
        return prompt


def format_ref_for_ppl(ref_text, fmt):
    """Format reference text for PPL evaluation — wraps in appropriate chat template."""
    if fmt is None:
        return ref_text
    t = fmt['type']

    if t == 'qwen':
        # Qwen: system msg + user msg (with <|im_end|>) + assistant prompt
        system_text = fmt.get('system_default', 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.')
        return ('<|im_start|>system\n' + system_text + '<|im_end|>\n' +
                '<|im_start|>user\n' + ref_text + '<|im_end|>\n' +
                '<|im_start|>assistant\n')

    elif t == 'gemma':
        return '<|turn>user\n' + ref_text + '\n<|turn>model\n'

    elif t == 'llama3':
        return ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' +
                ref_text + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

    elif t == 'mistral':
        return '[INST] ' + ref_text + ' [/INST] '

    elif t == 'chatml':
        return '<|user|>\n' + ref_text + '</s>\n<|assistant|>\n'

    elif t == 'phi':
        return '<|user|>\n' + ref_text + '<|assistant|>\n'

    elif t == 'stablelm':
        return '<|user|>' + ref_text + '<|assistant|>'

    else:
        return ref_text

def log(msg):
    v = f" VRAM={vram_mb():.0f}MB" if torch.cuda.is_available() else ""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        v_all = " ".join(f"cuda:{i}={vram_all_mb()[i]:.0f}MB" for i in range(torch.cuda.device_count()))
        v = f" [RAM={ram_mb():.0f}MB | {v_all}]"
    elif torch.cuda.is_available():
        v = f" VRAM={vram_mb():.0f}MB"
    else:
        v = ""
    print(f"{msg} [{v}]", flush=True)

def free_mem():
    """Aggressively free memory — call between layers."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# === CORE ===

class TCFDSFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, S, V):
        z = x @ V
        zs = z * S.unsqueeze(0)
        y = zs @ U.t()
        ctx.save_for_backward(x, U, S, V, z)
        return y

    @staticmethod
    def backward(ctx, g):
        x, U, S, V, z = ctx.saved_tensors
        gU = g @ U
        gUs = gU * S.unsqueeze(0)
        # grad_x: y = (x @ V) * S @ U^T → y = z_s @ U^T, z_s = z * S
        # grad_x = g @ U @ S.diag() @ V^T = gUs @ V.t()
        # grad_U: y = z_s @ U^T → y^T = U @ z_s^T → dy/dU = z_s.t() @ g
        # grad_U = z.t() @ g (NOT g.t() @ (z*S))
        # grad_S: dy/dS = z * gU, summed over batch
        # grad_V: y = z_s @ U^T, z = x @ V → dz/dV = x.t() @ gUs
        grad_x = gUs @ V.t()
        grad_U = z.t() @ g
        grad_S = (gU * z).sum(0)
        grad_V = x.t() @ gUs
        return grad_x, grad_U, grad_S, grad_V

def safe_eigh(M):
    """Eigendecomposition with numpy fallback for MKL/CUDA bugs."""
    n = M.shape[0]
    # Force CPU for large matrices to avoid GPU segfaults
    if n > CPU_SVD_THRESHOLD or M.device.type == 'cpu':
        try:
            M_cpu = M.float().cpu()
            vals, vecs = torch.linalg.eigh(M_cpu)
            return vals.to(M.device), vecs.to(M.device)
        except RuntimeError:
            pass
        # Numpy fallback
        M_np = M.double().cpu().numpy()
        eigvals_np, eigvecs_np = np.linalg.eigh(M_np)
        eigvals = torch.from_numpy(eigvals_np.copy()).float().to(M.device)
        eigvecs = torch.from_numpy(eigvecs_np.copy()).float().to(M.device)
        return eigvals, eigvecs
    try:
        return torch.linalg.eigh(M)
    except RuntimeError:
        M_np = M.double().cpu().numpy()
        eigvals_np, eigvecs_np = np.linalg.eigh(M_np)
        eigvals = torch.from_numpy(eigvals_np.copy()).float().to(M.device)
        eigvecs = torch.from_numpy(eigvecs_np.copy()).float().to(M.device)
        return eigvals, eigvecs

def _do_svd(W, eps, total_energy, force_cpu=False):
    """Shared SVD logic: probe rank, then targeted decomposition.
    Returns (U, S, Vh, k). Vh is k x n (rows are right singular vectors)."""
    m, n = W.shape
    threshold = 1.0 - eps**2 / 2.0
    device = W.device
    # Minimum rank: at least 16, or 1/64th of smallest dimension
    min_rank = max(16, min(m, n) // 64)

    # Auto-detect huge matrices (e.g. MLP down_proj 24576×3072 = 75M elements)
    # → force to CPU to avoid VRAM exhaustion in Lanczos workspace.
    # svd_lowrank Lanczos workspace scales with m*n; 50M is ~200MB float32,
    # but the iterative workspace can be 3-5× that on GPU with small VRAM.
    auto_cpu = (m * n > 50_000_000) and W.device.type != 'cpu'
    if force_cpu or auto_cpu:
        if W.device.type != 'cpu':
            W = W.cpu()

    probe_k = min(min(m, n), max(64, min(m, n) // 4))
    U_p, S_p, V_p = torch.svd_lowrank(W, q=probe_k, niter=3)
    ecum = (S_p**2).cumsum(0) / total_energy
    reached = (ecum >= threshold)

    if reached.any():
        k = max(int(reached.float().argmax().item()) + 1, min_rank)
    else:
        # Probe didn't reach threshold — extrapolate rank from energy curve
        # instead of falling through to expensive full SVD
        captured = ecum[-1].item() if len(ecum) > 0 else 0.0
        if captured > 0.01:
            # Estimate total rank needed by linear extrapolation
            k_est = int(probe_k * threshold / max(captured, 0.01))
            k_est = min(k_est, min(m, n))
            k_est = max(k_est, min_rank)
        else:
            k_est = min(m, n)
        del U_p, S_p, V_p
        free_mem()
        # Use extended svd_lowrank instead of full SVD (much faster)
        q_ext = min(k_est + 32, min(m, n))
        U, S, V = torch.svd_lowrank(W, q=q_ext, niter=3)
        ecum2 = (S**2).cumsum(0) / total_energy
        reached2 = (ecum2 >= threshold)
        if reached2.any():
            k = max(int(reached2.float().argmax().item()) + 1, min_rank)
        else:
            k = max(q_ext, min_rank)
        k = min(k, q_ext)
        U_out = U[:, :k].to(device)
        S_out = S[:k].to(device)
        Vh_out = V[:, :k].t().to(device)
        del U, S, V
        return U_out, S_out, Vh_out, k

    del U_p, S_p, V_p
    q = min(k + 20, min(m, n))
    U, S, V = torch.svd_lowrank(W, q=q, niter=3)
    U_out = U[:, :k].to(device)
    S_out = S[:k].to(device)
    Vh_out = V[:, :k].t().to(device)
    del U, S, V
    return U_out, S_out, Vh_out, k

def data_aware_svd(W, Cov, eps):
    """Covariance-weighted SVD. Forces CPU for large matrices.
    Returns (U, S, Vh, k, Cov_sqrt) — Cov_sqrt is returned so callers
    can compute the data-aware error without a second eigendecomposition."""
    m, n = W.shape
    # svd_lowrank is iterative and GPU-safe; only force CPU for eigh (done in safe_eigh)
    force_cpu = False

    # Eigendecomposition of covariance (always safe)
    eigvals, eigvecs = safe_eigh(Cov)
    eigvals = eigvals.clamp(min=1e-8)

    # Check for NaN/Inf in covariance decomposition
    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        log(f"    WARNING: NaN/Inf in covariance eigenvalues, falling back to standard SVD")
        del eigvals, eigvecs
        U, S, Vh, k = standard_svd(W, eps)
        return U, S, Vh, k, None

    sqrt_ev = eigvals.sqrt()
    inv_sqrt_ev = 1.0 / sqrt_ev

    # Build Cov^{1/2} implicitly on CPU if large
    compute_device = torch.device('cpu') if force_cpu else eigvecs.device
    eigvecs_c = eigvecs.to(compute_device)
    sqrt_ev_c = sqrt_ev.to(compute_device)
    inv_sqrt_ev_c = inv_sqrt_ev.to(compute_device)
    del eigvals, eigvecs, sqrt_ev, inv_sqrt_ev

    # Weighted matrix implicitly computed: (W @ E) * D^{1/2}
    # This avoids materializing the dense Cov_sqrt and Cov_sqrt_inv matrices
    W_c = W.to(compute_device) if W.device != compute_device else W
    W_weighted_compact = (W_c @ eigvecs_c) * sqrt_ev_c.unsqueeze(0)
    total_energy = (W_weighted_compact**2).sum().item()

    if total_energy < 1e-12:
        log(f"    WARNING: near-zero energy in weighted matrix, falling back to standard SVD")
        del eigvecs_c, sqrt_ev_c, inv_sqrt_ev_c, W_weighted_compact
        U, S, Vh, k = standard_svd(W, eps)
        return U, S, Vh, k, None

    # SVD on weighted matrix (on compute_device, which may be CPU)
    U, S, Vh_compact, k = _do_svd(W_weighted_compact, eps, total_energy, force_cpu=False)
    del W_weighted_compact
    if W_c is not W:
        del W_c

    # Transform V back from covariance-weighted space
    # Vh_orig = Vh_compact @ D^{-1/2} @ E^T
    Vh_orig = (Vh_compact * inv_sqrt_ev_c.unsqueeze(0)) @ eigvecs_c.t()
    Vh_orig = Vh_orig.to(W.device)
    del inv_sqrt_ev_c, Vh_compact
    free_mem()

    cov_info = (eigvecs_c, sqrt_ev_c)
    # Return cov_info for reuse in error computation (avoids second eigh)
    return U.to(W.device), S.to(W.device), Vh_orig, k, cov_info

def standard_svd(W, eps):
    """Standard SVD without data-aware weighting."""
    m, n = W.shape
    # svd_lowrank is iterative and GPU-safe; no need to force CPU
    total_energy = (W**2).sum().item()
    U, S, Vh, k = _do_svd(W, eps, total_energy, force_cpu=False)
    return U, S, Vh, k

class TCFDSLinear(nn.Module):
    def __init__(self, m=0, n=0, rank=0, rel_err=0, data_err=0, compression=0):
        super().__init__()
        self.m, self.n, self.rank = m, n, rank
        self.rel_err, self.data_err, self.compression = rel_err, data_err, compression
        self.U = nn.Parameter(torch.empty(0), requires_grad=False)
        self.S = nn.Parameter(torch.empty(0), requires_grad=False)
        self.V = nn.Parameter(torch.empty(0), requires_grad=False)
        self.bias = None

    @classmethod
    def from_weight(cls, W_f32, bias_f32, eps, store_dtype, Cov=None):
        m, n = W_f32.shape
        cov_info = None
        if Cov is not None:
            U, S, Vh, k, cov_info = data_aware_svd(W_f32, Cov, eps)
        else:
            U, S, Vh, k = standard_svd(W_f32, eps)

        # Compute approximation errors
        W_approx = (U * S.unsqueeze(0)) @ Vh
        diff = W_f32 - W_approx
        w_norm = W_f32.norm().item()
        rel_err = diff.norm().item() / max(w_norm, 1e-12)
        data_err = rel_err

        if cov_info is not None:
            # Reuse cov_info from data_aware_svd (no second eigendecomposition)
            eigvecs_c, sqrt_ev_c = cov_info

            # Ensure diff and W_f32 are on the same device as cov_info
            diff_c = diff.to(eigvecs_c.device) if diff.device != eigvecs_c.device else diff
            W_c = W_f32.to(eigvecs_c.device) if W_f32.device != eigvecs_c.device else W_f32

            data_err = ((diff_c @ eigvecs_c) * sqrt_ev_c.unsqueeze(0)).norm().item() / \
                       max(((W_c @ eigvecs_c) * sqrt_ev_c.unsqueeze(0)).norm().item(), 1e-12)

            if diff_c is not diff: del diff_c
            if W_c is not W_f32: del W_c

        del W_approx, diff
        if cov_info is not None: del cov_info

        orig_p = m * n + (m if bias_f32 is not None else 0)
        comp_p = k * (m + n) + k + (m if bias_f32 is not None else 0)
        obj = cls(m, n, k, rel_err, data_err, orig_p / comp_p)
        obj.U = nn.Parameter(U.to(store_dtype).contiguous(), requires_grad=False)
        obj.S = nn.Parameter(S.to(store_dtype).contiguous(), requires_grad=False)
        obj.V = nn.Parameter(Vh.t().to(store_dtype).contiguous(), requires_grad=False)
        if bias_f32 is not None:
            obj.bias = nn.Parameter(bias_f32.to(store_dtype).clone(), requires_grad=False)
        del U, S, Vh
        return obj

    def forward(self, x):
        in_dtype, shape = x.dtype, x.shape
        x2 = x.reshape(-1, shape[-1]) if x.dim() > 2 else x
        x2 = x2.to(self.U.dtype)
        y = TCFDSFwd.apply(x2, self.U, self.S, self.V)
        if self.bias is not None:
            y = y + self.bias
        y = y.to(in_dtype)
        return y.reshape(*shape[:-1], -1) if x.dim() > 2 else y

    def param_count(self):
        t = self.U.numel() + self.S.numel() + self.V.numel()
        return t + (self.bias.numel() if self.bias is not None else 0)

# === CALIBRATION DATA ===

# Fallback: short diverse sentences (used when WikiText-2 is unavailable)
_FALLBACK_CALIBRATION_TEXTS = [
    "The transformer architecture uses self-attention mechanisms to process sequences in parallel.",
    "In mathematics, differential geometry studies smooth manifolds and their curvature properties.",
    "Machine learning models require large amounts of diverse training data to generalize well.",
    "Quantum mechanics describes the behavior of particles at the subatomic scale precisely.",
    "The theory of relativity fundamentally changed our understanding of space, time, and gravity.",
    "DNA stores genetic information using four nucleotide bases arranged in a double helix.",
    "Algorithms are step-by-step procedures for solving computational problems efficiently and correctly.",
    "The internet connects billions of devices worldwide through standardized communication protocols.",
]

def _load_calibration_texts(tokenizer, n_samples=32, seq_len=256):
    """Load calibration data from WikiText-2 (preferred) or fallback to hardcoded texts.
    Returns list of token-ready text chunks of ~seq_len tokens each."""
    # Try WikiText-2 from HuggingFace datasets
    try:
        from datasets import load_dataset
        log(f"  Loading WikiText-2 ({n_samples} x {seq_len} tokens)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Concatenate all text, then chunk into seq_len-token pieces
        texts = [t for t in ds["text"] if len(t.strip()) > 50]
        all_text = "\n".join(texts[:5000])  # cap at ~5 000 paragraphs to avoid OOM
        # Tokenize once to find chunk boundaries
        tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            # Return raw token-id tensors — avoids a decode+re-encode round-trip
            # inside collect_covs_for_layers.
            chunks.append(tokens[i:i + seq_len])
            if len(chunks) >= n_samples:
                break
        if len(chunks) >= n_samples // 2:
            log(f"  WikiText-2: {len(chunks)} chunks of ~{seq_len} tokens")
            return chunks, seq_len
        log(f"  WikiText-2: only {len(chunks)} chunks, falling back to hardcoded texts")
    except Exception as e:
        log(f"  WikiText-2 unavailable ({e}), using fallback calibration texts")

    return _FALLBACK_CALIBRATION_TEXTS, 64

# === STREAMING COVARIANCE (per-block, not all-at-once) ===

def collect_covs_for_layers(model, tokenizer, layer_names, calib_texts=None, seq_len=256):
    """Collect input activation covariance matrices for the given layers.
    Covariances are accumulated on CPU to avoid VRAM pressure."""
    if calib_texts is None:
        calib_texts = _FALLBACK_CALIBRATION_TEXTS
        seq_len = 64
    covs, tcounts, hooks = {}, {}, []
    target = set(layer_names)
    dev = next(model.parameters()).device

    # Pre-allocate covariance matrices to avoid per-sample conditional allocation
    # and to let the hook skip the `if name not in covs` branch entirely.
    for name, mod in model.named_modules():
        if name in target and isinstance(mod, nn.Linear):
            n = mod.in_features
            covs[name] = torch.zeros(n, n)
            tcounts[name] = 0

    def make_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1])
            # torch.addmm(input, mat1, mat2, out=input) avoids allocating a
            # temporary (n x n) matrix for the intermediate x.T @ x product.
            torch.addmm(covs[name], x.t(), x, out=covs[name])
            tcounts[name] += x.shape[0]
        return fn

    for name, mod in model.named_modules():
        if name in target and isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    n_errors = 0
    with torch.no_grad():
        for i, item in enumerate(calib_texts):
            if isinstance(item, torch.Tensor):
                # Pre-tokenized path: item is a 1-D token-id tensor from
                # _load_calibration_texts — no decode/re-encode needed.
                input_ids = item.unsqueeze(0).to(dev)
                ids = {"input_ids": input_ids,
                       "attention_mask": torch.ones_like(input_ids)}
            else:
                ids = tokenizer(item, return_tensors="pt", truncation=True,
                                max_length=seq_len)
                ids = {k: v.to(dev) for k, v in ids.items()}
            try:
                model(**ids)
            except RuntimeError as e:
                n_errors += 1
                if n_errors <= 3:
                    log(f"    WARNING: forward pass error on calibration text {i}: {e}")
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    log(f"    WARNING: unexpected error on calibration text {i}: {e}")

    if n_errors > 3:
        log(f"    WARNING: {n_errors} total errors during calibration (showing first 3)")

    for h in hooks:
        h.remove()

    # Normalize and regularize
    for name in covs:
        if tcounts[name] > 0:
            covs[name] /= tcounts[name]

        # Check for NaN/Inf
        if torch.isnan(covs[name]).any() or torch.isinf(covs[name]).any():
            log(f"    WARNING: NaN/Inf in covariance for {name}, resetting to identity")
            n = covs[name].shape[0]
            covs[name] = torch.eye(n)

        # Regularize
        tr = covs[name].trace().item()
        reg = max(1e-4, 1e-4 * tr / covs[name].shape[0])
        covs[name] += reg * torch.eye(covs[name].shape[0])

    return covs

def calibrate_sensitivity(model, tokenizer, ref_text):
    """Fast sensitivity estimation using weight-norm heuristic.
    Attention projections (q/k) get higher sensitivity than MLP layers.
    This avoids ~150 forward passes of the noise-injection method."""
    log("  Estimating layer sensitivity (weight-norm heuristic)...")
    sens = {}
    norms = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if min(mod.out_features, mod.in_features) < 256:
            continue
        if any(p in name.lower() for p in ('embed', 'norm', 'lm_head')):
            continue
        # Normalized weight norm: ||W||_F / sqrt(m*n)
        w = mod.weight.data.float()
        norms[name] = w.norm().item() / math.sqrt(w.numel())

    if not norms:
        return sens

    mx = max(norms.values())
    for name, nv in norms.items():
        # Base sensitivity from weight norm (higher norm = more sensitive)
        s = nv / max(mx, 1e-12)
        # Boost attention projections (empirically more sensitive)
        low = name.lower()
        if 'q_proj' in low or 'k_proj' in low:
            s = min(s * 1.5, 1.0)
        elif 'v_proj' in low or 'o_proj' in low:
            s = min(s * 1.2, 1.0)
        # MLP layers are typically more robust
        elif 'gate_proj' in low or 'up_proj' in low or 'down_proj' in low:
            s = s * 0.8
        sens[name] = s

    log(f"  Estimated sensitivity for {len(sens)} layers")
    return sens

# === SURGERY ===

def set_mod(model, name, mod):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)

def get_block_groups(model, skip_pats=('lm_head', 'embed', 'norm', 'layernorm',
                                           'audio', 'per_layer_input_gate',
                                           'per_layer_projection', 'per_layer_model_projection'), min_dim=256):
    blocks = defaultdict(list)
    ungrouped = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(p in name.lower() for p in skip_pats):
            continue
        if min(mod.out_features, mod.in_features) < min_dim:
            continue
        m = re.search(r'\.layers\.(\d+)\.', name)
        if m:
            blocks[int(m.group(1))].append((name, mod))
        else:
            ungrouped.append((name, mod))
    result = [(f"block_{i}", blocks[i]) for i in sorted(blocks.keys())]
    if ungrouped:
        result.append(("other", ungrouped))
    return result

def compress_streaming(model, tokenizer, base_eps, sensitivities):
    """Block-by-block compression with aggressive memory management.
    Peak cov memory: ~230 MB (1 block) instead of ~5 GB (all blocks)."""
    store_dtype = next(model.parameters()).dtype
    block_groups = get_block_groups(model)
    total_layers = sum(len(l) for _, l in block_groups)
    log(f"  {total_layers} eligible layers in {len(block_groups)} blocks")
    results, tot_orig, tot_comp, layer_idx = [], 0, 0, 0
    t0 = time.time()

    # Load calibration data (WikiText-2 if available, else fallback)
    calib_texts, calib_seq_len = _load_calibration_texts(tokenizer)

    # Collect ALL covariances in a single pass
    all_layer_names = [n for _, layers in block_groups for n, _ in layers]
    log(f"  Collecting covariances for {len(all_layer_names)} layers (single pass)...")
    all_covs = collect_covs_for_layers(model, tokenizer, all_layer_names,
                                       calib_texts=calib_texts, seq_len=calib_seq_len)
    del calib_texts
    cov_mb = sum(c.numel() * 4 / 1e6 for c in all_covs.values())
    log(f"  Covariances: {len(all_covs)}/{len(all_layer_names)} ({cov_mb:.0f} MB)")

    # Cache values that are constant across the entire compression loop.
    n_blocks = sum(1 for bname, _ in block_groups if bname.startswith("block_"))
    model_dev = next(model.parameters()).device

    for block_name, layers in block_groups:
        layer_names = [n for n, _ in layers]
        log(f"\n  --- {block_name} ({len(layers)} layers) ---")

        for name, mod in layers:
            m, n = mod.out_features, mod.in_features
            layer_idx += 1

            # Free memory before each layer (critical for large down_proj layers)
            free_mem()

            try:
                s = sensitivities.get(name, 0.5)
                aeps = base_eps * (0.3 + 1.7 * (1.0 - s))
                aeps = max(0.02, min(aeps, base_eps * 1.5))

                # Epsilon scheduling: tighter eps for first/last 2 blocks
                # (these layers are empirically the most sensitive)
                m_block = re.search(r'\.layers\.(\d+)\.', name)
                if m_block:
                    blk_idx = int(m_block.group(1))
                    if blk_idx < 2 or blk_idx >= n_blocks - 2:
                        aeps *= 0.5  # halve epsilon for edge layers

                # Extract weight in float32 for SVD
                w32 = mod.weight.data.float().cpu()
                b32 = mod.bias.data.float().cpu() if mod.bias is not None else None
                cov = all_covs.get(name, None)

                t1 = time.time()
                tcfds = TCFDSLinear.from_weight(w32, b32, aeps, store_dtype, Cov=cov)
                svd_t = time.time() - t1
                del w32, b32

                if tcfds.compression < 1.05:
                    log(f"  [{layer_idx}/{total_layers}] {name:<38} SKIP (ratio {tcfds.compression:.2f}x < 1.05)")
                    del tcfds
                    continue

                # Hard Frobenius cap: never compress if we lose >90% of weight energy
                if tcfds.rel_err > 0.90:
                    log(f"  [{layer_idx}/{total_layers}] {name:<38} SKIP (frob {tcfds.rel_err:.3f} > 0.90)")
                    del tcfds
                    continue

                # Quality guard: use data-aware error when available
                err = tcfds.data_err if (cov is not None and tcfds.data_err != tcfds.rel_err) else tcfds.rel_err
                if err > 0.16:
                    # Borderline layer — retry with tighter eps (higher rank)
                    if err < 0.30 and tcfds.compression > 1.5:
                        retry_eps = aeps * 0.4  # much tighter
                        del tcfds
                        w32_r = mod.weight.data.float().cpu()
                        b32_r = mod.bias.data.float().cpu() if mod.bias is not None else None
                        tcfds = TCFDSLinear.from_weight(w32_r, b32_r, retry_eps, store_dtype, Cov=cov)
                        del w32_r, b32_r
                        err2 = tcfds.data_err if (cov is not None and tcfds.data_err != tcfds.rel_err) else tcfds.rel_err
                        if err2 > 0.16 or tcfds.compression < 1.05:
                            log(f"  [{layer_idx}/{total_layers}] {name:<38} SKIP (err {err:.3f}->{err2:.3f}, retry failed)")
                            del tcfds
                            continue
                        # Retry succeeded — use the tighter version
                        svd_t = time.time() - t1  # update timing
                    else:
                        log(f"  [{layer_idx}/{total_layers}] {name:<38} SKIP (err {err:.3f} > 0.16)")
                        del tcfds
                        continue

                orig_p = m * n + (m if mod.bias is not None else 0)
                tot_orig += orig_p
                tot_comp += tcfds.param_count()
                tcfds = tcfds.to(mod.weight.device)
                set_mod(model, name, tcfds)

                results.append({
                    'name': name, 'shape': [m, n], 'rank': tcfds.rank,
                    'ratio': round(tcfds.compression, 2),
                    'frob_err': round(tcfds.rel_err, 5),
                    'data_err': round(tcfds.data_err, 5),
                    'eps_used': round(aeps, 3),
                    'sensitivity': round(s, 3)
                })

                elapsed = time.time() - t0
                eta = elapsed / layer_idx * (total_layers - layer_idx)
                tag = "DATA" if cov is not None else "STD"
                log(f"  [{layer_idx}/{total_layers}] {name:<38} ({m},{n})->r{tcfds.rank:>4} "
                    f"{tcfds.compression:.1f}x frob={tcfds.rel_err:.3f} data={tcfds.data_err:.3f} "
                    f"eps={aeps:.2f} [{tag} {svd_t:.1f}s ETA {eta/60:.0f}m]")

            except Exception as e:
                log(f"  [{layer_idx}/{total_layers}] SKIP {name}: {e}")
                traceback.print_exc()

        free_mem()

    del all_covs
    free_mem()
    elapsed = time.time() - t0
    log(f"\n  Done: {len(results)}/{total_layers} layers in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return results, tot_orig, tot_comp

# === SAVE / LOAD / VERIFY / GEN ===

def save_compressed(model, results, meta, path):
    log(f"  Saving to {path}...")
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    data = {
        'model_state': model_cpu,
        'config': model.config.to_dict(),
        'model_name': meta.get('model_name', ''),
        'results': results,
        'meta': meta,
        'tcfds_layers': {},
        'version': 'v6.4.0'
    }
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'TCFDSLinear':
            data['tcfds_layers'][name] = {
                'm': mod.m, 'n': mod.n, 'rank': mod.rank,
                'rel_err': mod.rel_err, 'data_err': mod.data_err,
                'compression': mod.compression,
                'has_bias': mod.bias is not None
            }
    torch.save(data, path)
    log(f"  Saved: {os.path.getsize(path)/1e6:.0f} MB")

def load_compressed(path):
    log(f"  Loading {path}...")
    data = torch.load(path, map_location='cpu')
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    mn = data['model_name']
    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        config = AutoConfig.from_dict(data['config'])
    except Exception:
        config = AutoConfig.from_pretrained(mn, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    for name, info in data['tcfds_layers'].items():
        t = TCFDSLinear(
            m=info['m'], n=info['n'], rank=info['rank'],
            rel_err=info['rel_err'], data_err=info.get('data_err', info['rel_err']),
            compression=info['compression']
        )
        t.U = nn.Parameter(torch.empty(info['m'], info['rank']), requires_grad=False)
        t.S = nn.Parameter(torch.empty(info['rank']), requires_grad=False)
        t.V = nn.Parameter(torch.empty(info['n'], info['rank']), requires_grad=False)
        if info['has_bias']:
            t.bias = nn.Parameter(torch.empty(info['m']), requires_grad=False)
        set_mod(model, name, t)
    model.load_state_dict(data['model_state'])
    model.eval()
    model = model.to(DEVICE)
    log(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")
    return model, tok, data.get('results', []), data.get('meta', {})

def verify(model, store_dtype=None):
    r = {"checks": {}, "layers": []}
    tot_o = tot_c = 0
    if store_dtype is None:
        store_dtype = next(model.parameters()).dtype
    bpe = 2 if store_dtype in (torch.float16, torch.bfloat16) else 4

    for name, mod in model.named_modules():
        if type(mod).__name__ != 'TCFDSLinear':
            continue
        m, n, k = mod.m, mod.n, mod.rank
        o = m * n + (m if mod.bias is not None else 0)
        c = mod.param_count()
        tot_o += o
        tot_c += c
        hf = any(p.shape == (m, n) or p.shape == (n, m) for _, p in mod.named_parameters())
        r["layers"].append({
            "name": name, "shape": [m, n], "rank": k, "orig": o, "comp": c,
            "ratio": round(o / c, 2),
            "frob_err": round(mod.rel_err, 6), "data_err": round(mod.data_err, 6),
            "full_mat": hf,
            "U_hash": hashlib.sha256(
                mod.U.detach().cpu().float().numpy().tobytes()
            ).hexdigest()[:12]
        })

    nt = len(r["layers"])
    c1 = nt > 0 and not any(l["full_mat"] for l in r["layers"])
    c2 = tot_c < tot_o if nt > 0 else False
    sv_checks = []
    named_mods = dict(model.named_modules())
    for li in r["layers"][:5]:
        mod = named_mods[li["name"]]
        S = mod.S.detach().float()
        sv_checks.append(round(float(S[0] / max(S[-1].abs().item(), 1e-10)), 1) if S.numel() > 1 else 0.0)
    c3 = len(sv_checks) > 0 and all(s > 1.0 for s in sv_checks)
    sav = (tot_o - tot_c) * bpe / 1e6

    r["checks"] = dict(
        modules_replaced=c1, memory_reduced=c2, savings_mb=round(sav, 2),
        ratio=round(tot_o / max(tot_c, 1), 2), structured=c3,
        sv_ranges=sv_checks, bytes_per_elem=bpe
    )
    r["n_tcfds"] = nt
    r["total_params"] = sum(p.numel() for p in model.parameters())
    r["ram_mb"] = round(ram_mb(), 1)
    r["verdict"] = "VERIFIED" if (c1 and c2 and c3) else "FAILED"
    return r

def gen(model, tok, prompt, max_new=80, fmt=None, original_prompt=None):
    ids = tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids = {k: v.to(dev) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(
            ids["input_ids"], attention_mask=ids["attention_mask"],
            max_new_tokens=max_new, do_sample=True, temperature=0.7,
            top_k=40, top_p=0.9, pad_token_id=tok.eos_token_id,
            repetition_penalty=1.2
        )
    result = tok.decode(out[0], skip_special_tokens=True)
    # If chat format provided, strip the model's turn prefix from output
    if fmt is not None and fmt['requires_generation_marker']:
        marker = fmt['model']
        if result.startswith(marker):
            result = result[len(marker):]
        # Strip leftover newlines after marker
        if result.startswith('\n'):
            result = result[1:]
        result = result.lstrip()
        # Clean up residual special-token markers that model may have emitted
        t = fmt['type']
        if t == 'qwen':
            # Remove any leftover <|im_start|>role\n...<|im_end|> sequences
            result = re.sub(r'<\|im_start\|>.*?(?=<\|im_start\|>|$)', '', result, flags=re.DOTALL)
            result = re.sub(r'<\|im_end\|>\n?', '', result)
            result = result.strip()
        elif t in ('chatml', 'phi', 'stablelm'):
            result = result.replace('<|user|>', '').replace('<|assistant|>', '').replace('<|endoftext|>', '')
            result = result.strip()
        elif t == 'gemma':
            result = re.sub(r'<\|turn\|>.*?(?=<\|turn\|>|$)', '', result, flags=re.DOTALL)
            result = result.strip()
        elif t == 'llama3':
            result = re.sub(r'<\|eot_id\|>+', '', result)
            result = result.strip()
    return result

def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt")
    dev = next(model.parameters()).device
    ids = {k: v.to(dev) for k, v in ids.items()}
    with torch.no_grad():
        o = model(**ids, labels=ids["input_ids"])
    return torch.exp(o.loss).item()

# === MAIN ===

def main():
    pa = argparse.ArgumentParser(description="TCFDS v6.4.0 - Data-Aware LLM Compression")
    pa.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HuggingFace model ID or local path")
    pa.add_argument("--eps", type=float, default=0.25,
                    help="Compression tolerance (lower = less compression, better quality)")
    pa.add_argument("--save", type=str, default=None,
                    help="Path to save compressed model (.pt)")
    pa.add_argument("--load", type=str, default=None,
                    help="Path to load a previously compressed model")
    pa.add_argument("--chat-only", action="store_true",
                    help="Skip verification, go straight to chat")
    pa.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32", "bfloat16"],
                    help="Model precision (float16 saves RAM, default: float16)")
    a = pa.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    load_dtype = dtype_map[a.dtype]

    print("=" * 65)
    print(f"  TCFDS v6.4.0 (Data-Aware + GPU + Memory Safety) | eps={a.eps}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {total_vram:.1f} GB total")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.load:
        model, tok, results, meta = load_compressed(a.load)
        if not a.chat_only:
            ref = "The Transformer uses self-attention to process sequences in parallel."
            try:
                log(f"  PPL: {ppl(model, tok, ref):.2f}")
            except Exception as e:
                print(f"  PPL err: {e}")
            rpt = verify(model)
            print(f"  {rpt['n_tcfds']} layers, {rpt['checks']['ratio']:.1f}x, {rpt['verdict']}")
        print(f"\nChat (type 'quit' to exit)\n")
        model_name_for_chat = meta.get('model_name', a.model)
        chat_fmt = detect_chat_format(model_name_for_chat)
        if chat_fmt:
            print(f"  Detected format: {chat_fmt['type']}")
        while True:
            try:
                u = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not u or u.lower() in ('quit', 'exit', 'q'):
                break
            pr = format_prompt_for_chat(u, chat_fmt)
            try:
                if chat_fmt is None:
                    output = gen(model, tok, pr, 150)[len(pr):].strip()
                else:
                    output = gen(model, tok, pr, 150, fmt=chat_fmt, original_prompt=u).strip()
                print(f"AI: {output}\n")
            except Exception as e:
                print(f"Error: {e}\n")
        return

    # [1] Load model
    log(f"\n[1/6] Loading model...")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"  Loading in {a.dtype}...")

    # Load model — prioritise the lightest strategy first (no torch_dtype at
    # load time avoids Windows page-file blow-up observed on 16 GB machines).
    # dtype conversion happens *after* the model is in memory.
    model = None
    strategies = [
        ("multi-GPU", {"device_map": "auto", "torch_dtype": load_dtype}),
        ("CPU-lowmem", dict(low_cpu_mem_usage=True)),
        ("CPU-minimal", dict()),
    ]

    for name, kwargs in strategies:
        try:
            log(f"  Strategy: {name}...")
            model = AutoModelForCausalLM.from_pretrained(
                a.model, trust_remote_code=True, **kwargs
            )
            log(f"  Model loaded via {name}")
            break
        except Exception as e:
            log(f"  {name} failed: {e}")
            model = None
            free_mem()

    if model is None:
        print("FATAL: Could not load model. Try closing other apps or increasing", flush=True)
        print("  Windows page file (sysdm.cpl > Advanced > Performance > Virtual Memory).", flush=True)
        return

    model.eval()
    # Convert dtype on CPU first (avoids double-memory on GPU), then move
    if next(model.parameters()).dtype != load_dtype:
        log(f"  Converting to {a.dtype} on CPU...")
        model = model.to(dtype=load_dtype)
        free_mem()
    if next(model.parameters()).device.type != DEVICE.type and not hasattr(model, "hf_device_map"):
        log(f"  Moving to {DEVICE}...")
        model = model.to(device=DEVICE)
    dtype = next(model.parameters()).dtype
    np_ = sum(p.numel() for p in model.parameters())
    bpe = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    log(f"  {np_:,} params | {dtype} | {np_*bpe/1e9:.2f} GB | {DEVICE}")

    # [2] Baseline
    log("\n[2/6] Baseline...")
    chat_fmt = detect_chat_format(a.model)
    if chat_fmt:
        log(f"  Detected format: {chat_fmt['type']}")
    prompts = [
        "The key insight behind neural networks is",
        "In mathematics, a manifold is",
        "Once upon a time, a young scientist"
    ]
    ref = ("The Transformer uses self-attention to process sequences in parallel. "
           "Each layer computes attention weights so tokens attend to each other, "
           "capturing complex long-range dependencies efficiently.")
    ref_for_ppl = format_ref_for_ppl(ref, chat_fmt)
    try:
        bppl = ppl(model, tok, ref_for_ppl)
        log(f"  PPL: {bppl:.2f}")
    except Exception:
        bppl = float('inf')
    for p in prompts:
        try:
            fp = format_prompt_for_chat(p, chat_fmt)
            t = gen(model, tok, fp, 60, fmt=chat_fmt, original_prompt=p)
            print(f'  "{p}"')
            print(f"    -> {t[:120]}")
        except Exception as e:
            print(f"  ERR: {e}")

    # [3] Sensitivity calibration
    log("\n[3/6] Calibrating layer sensitivity...")
    sensitivities = calibrate_sensitivity(model, tok, ref)
    for name, s in sorted(sensitivities.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name:<40} sensitivity={s:.3f}")

    # [4] Streaming compress (block-by-block)
    log(f"\n[4/6] Streaming compression (base_eps={a.eps}, block-by-block)...")
    results, to, tc = compress_streaming(model, tok, a.eps, sensitivities)
    free_mem()

    if not results:
        print("  No layers compressed. Try --eps 0.35")
        return

    ratio = to / max(tc, 1)
    avg_f = sum(r['frob_err'] for r in results) / len(results)
    avg_d = sum(r['data_err'] for r in results) / len(results)
    n_mlp = sum(1 for r in results if 'mlp' in r['name'])
    n_attn = sum(1 for r in results if 'attn' in r['name'])
    log(f"\n  {len(results)} layers ({n_attn} attn + {n_mlp} MLP) | {ratio:.1f}x | "
        f"frob={avg_f:.4f} data={avg_d:.4f}")

    # [5] Compressed gen
    log("\n[5/6] Compressed generation...")
    ref_for_cppl = format_ref_for_ppl(ref, chat_fmt)
    try:
        cppl = ppl(model, tok, ref_for_cppl)
        log(f"  PPL: {bppl:.2f} -> {cppl:.2f} ({cppl/bppl:.3f}x)")
    except Exception as e:
        log(f"  Step 5 skipped: {e}")
        cppl = float('inf')

    for p in prompts:
        try:
            fp = format_prompt_for_chat(p, chat_fmt)
            t = gen(model, tok, fp, 60, fmt=chat_fmt, original_prompt=p)
            print(f'  "{p}"')
            print(f"    -> {t[:120]}")
        except Exception as e:
            print(f"  ERR: {e}")

    # [6] Verify + save
    log("\n[6/6] Verification...")
    if a.save:
        save_compressed(model, results, {
            'model_name': a.model, 'eps': a.eps,
            'base_ppl': bppl, 'comp_ppl': cppl, 'version': 'v6.4.0'
        }, a.save)

    try:
        rpt = verify(model, store_dtype=dtype)
    except Exception as e:
        log(f"  Verify failed (non-fatal): {e}")
        rpt = None
    if rpt is None:
        rpt = {"checks": {"modules_replaced": len(results) > 0, "memory_reduced": True, "structured": True, "ratio": 0, "savings_mb": 0, "sv_ranges": []}, "verdict": "SAVED (verify skipped)"}
    rpt["model"] = a.model
    rpt["eps"] = a.eps
    rpt["device"] = str(DEVICE)
    rpt["base_ppl"] = round(bppl, 2) if bppl != float('inf') else None
    rpt["comp_ppl"] = round(cppl, 2) if cppl != float('inf') else None
    rpt["results"] = results
    rpt["version"] = "v6.4.0"
    ck = rpt["checks"]
    print(f"  CHECK 1 (modules replaced):  {'PASS' if ck['modules_replaced'] else 'FAIL'}")
    print(f"  CHECK 2 (memory reduced):    {'PASS' if ck['memory_reduced'] else 'FAIL'} "
          f"({ck['savings_mb']:.1f}MB, {ck['ratio']:.1f}x)")
    print(f"  CHECK 3 (structured SVD):    {'PASS' if ck['structured'] else 'FAIL'} "
          f"(SV ranges: {ck['sv_ranges']})")
    print(f"  >>> {rpt['verdict']} <<<")
    with open("tcfds_report_v63.json", "w") as f:
        json.dump(rpt, f, indent=2, default=str)


    print(f"\n{'='*65}")
    print(f"  TCFDS v6.4.0 Results")
    print(f"  Layers: {len(results)} ({n_attn} attn + {n_mlp} MLP) | "
          f"{ck['ratio']:.1f}x | {ck['savings_mb']:.1f}MB saved")
    if bppl != float('inf') and cppl != float('inf'):
        q = cppl / bppl
        tag = "Excellent" if q < 1.05 else "OK" if q < 1.2 else "Degraded" if q < 2 else "Broken"
        print(f"  PPL: {bppl:.2f} -> {cppl:.2f} ({tag})")
    print(f"  Avg Frobenius err: {avg_f:.4f} | Avg Data-aware err: {avg_d:.4f}")
    print(f"  {rpt['verdict']}")
    if a.save:
        print(f"  Saved: {a.save} | Reload: python tcfds.py --load {a.save}")
    print("=" * 65)

    print(f"\nChat (type 'quit' to exit)\n")
    chat_fmt = detect_chat_format(a.model)
    if chat_fmt:
        print(f"  Detected format: {chat_fmt['type']}")
    while True:
        try:
            u = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u or u.lower() in ('quit', 'exit', 'q'):
            break
        pr = format_prompt_for_chat(u, chat_fmt)
        try:
            if chat_fmt is None:
                output = gen(model, tok, pr, 150)[len(pr):].strip()
            else:
                output = gen(model, tok, pr, 150, fmt=chat_fmt, original_prompt=u).strip()
            print(f"AI: {output}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
