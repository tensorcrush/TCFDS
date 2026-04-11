"""
TCFDS v6.3 - Data-Aware Compression + Adaptive Epsilon + GPU Support
=====================================================================
Fixes vs v6.2:
  - Fix GPU segfault on last layer: force CPU for SVD on large matrices (n>2048)
  - Aggressive memory cleanup between each layer (gc + CUDA cache clear)
  - NaN/Inf guard on covariance matrices
  - VRAM monitoring alongside RAM
  - Better error logging in covariance collection
  - Minimum rank guard (rank >= 2) to prevent degenerate factorizations
  - Intermediate cleanup in data_aware_svd to reduce peak memory

Usage:
  python tcfds.py --eps 0.25 --save compressed.pt
  python tcfds.py --load compressed.pt --chat-only
"""

import torch, torch.nn as nn
import math, time, gc, sys, os, json, hashlib, traceback, argparse, re
import numpy as np
from collections import defaultdict

try:
    import psutil
    def ram_mb(): return psutil.Process(os.getpid()).memory_info().rss / 1e6
except ImportError:
    def ram_mb(): return 0

def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold above which SVD/eigh operations are forced to CPU to avoid GPU segfaults
CPU_SVD_THRESHOLD = 2048

def log(msg):
    v = f" VRAM={vram_mb():.0f}MB" if torch.cuda.is_available() else ""
    print(f"{msg} [RAM={ram_mb():.0f}MB{v}]", flush=True)

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
        return gUs @ V.t(), g.t() @ (z * S.unsqueeze(0)), (gU * z).sum(0), x.t() @ gUs

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

    # For large matrices, force CPU to avoid GPU segfaults
    if force_cpu and W.device.type != 'cpu':
        W = W.cpu()

    probe_k = min(min(m, n), max(64, min(m, n) // 4))
    U_p, S_p, V_p = torch.svd_lowrank(W, q=probe_k, niter=3)
    ecum = (S_p**2).cumsum(0) / total_energy
    reached = (ecum >= threshold)

    if reached.any():
        k = max(int(reached.float().argmax().item()) + 1, 2)  # minimum rank 2
    else:
        # Probe didn't reach threshold — full SVD needed
        del U_p, S_p, V_p
        free_mem()
        U_f, S_f, Vh_f = torch.linalg.svd(W, full_matrices=False)
        ecum_f = (S_f**2).cumsum(0) / total_energy
        reached_f = (ecum_f >= threshold)
        k = int(reached_f.float().argmax().item()) + 1 if reached_f.any() else min(m, n)
        k = max(k, 2)
        U_out = U_f[:, :k].to(device)
        S_out = S_f[:k].to(device)
        Vh_out = Vh_f[:k].to(device)
        del U_f, S_f, Vh_f
        return U_out, S_out, Vh_out, k

    del U_p, S_p, V_p
    q = min(k + 20, min(m, n))
    U, S, V = torch.svd_lowrank(W, q=q, niter=5)
    U_out = U[:, :k].to(device)
    S_out = S[:k].to(device)
    Vh_out = V[:, :k].t().to(device)  # V from svd_lowrank is n x q, we want k x n
    del U, S, V
    return U_out, S_out, Vh_out, k

def data_aware_svd(W, Cov, eps):
    """Covariance-weighted SVD. Forces CPU for large matrices."""
    m, n = W.shape
    force_cpu = (n > CPU_SVD_THRESHOLD)

    # Eigendecomposition of covariance (always safe)
    eigvals, eigvecs = safe_eigh(Cov)
    eigvals = eigvals.clamp(min=1e-8)

    # Check for NaN/Inf in covariance decomposition
    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        log(f"    WARNING: NaN/Inf in covariance eigenvalues, falling back to standard SVD")
        del eigvals, eigvecs
        return standard_svd(W, eps)

    sqrt_ev = eigvals.sqrt()
    inv_sqrt_ev = 1.0 / sqrt_ev

    # Build Cov^{1/2} and Cov^{-1/2} on CPU if large
    compute_device = torch.device('cpu') if force_cpu else eigvecs.device
    eigvecs_c = eigvecs.to(compute_device)
    sqrt_ev_c = sqrt_ev.to(compute_device)
    inv_sqrt_ev_c = inv_sqrt_ev.to(compute_device)

    Cov_sqrt = (eigvecs_c * sqrt_ev_c.unsqueeze(0)) @ eigvecs_c.t()
    Cov_sqrt_inv = (eigvecs_c * inv_sqrt_ev_c.unsqueeze(0)) @ eigvecs_c.t()
    del eigvals, eigvecs, sqrt_ev, inv_sqrt_ev, eigvecs_c, sqrt_ev_c, inv_sqrt_ev_c

    # Weighted matrix
    W_c = W.to(compute_device) if W.device != compute_device else W
    W_weighted = W_c @ Cov_sqrt
    total_energy = (W_weighted**2).sum().item()

    if total_energy < 1e-12:
        log(f"    WARNING: near-zero energy in weighted matrix, falling back to standard SVD")
        del Cov_sqrt, Cov_sqrt_inv, W_weighted
        return standard_svd(W, eps)

    # SVD on weighted matrix (on compute_device, which may be CPU)
    U, S, Vh, k = _do_svd(W_weighted, eps, total_energy, force_cpu=False)
    del W_weighted
    if W_c is not W:
        del W_c

    # Transform V back from covariance-weighted space
    Vh_orig = Vh @ Cov_sqrt_inv.to(Vh.device)
    del Cov_sqrt, Cov_sqrt_inv, Vh
    free_mem()

    return U.to(W.device), S.to(W.device), Vh_orig.to(W.device), k

def standard_svd(W, eps):
    """Standard SVD without data-aware weighting."""
    m, n = W.shape
    force_cpu = (max(m, n) > CPU_SVD_THRESHOLD)
    total_energy = (W**2).sum().item()
    U, S, Vh, k = _do_svd(W, eps, total_energy, force_cpu=force_cpu)
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
        if Cov is not None:
            U, S, Vh, k = data_aware_svd(W_f32, Cov, eps)
        else:
            U, S, Vh, k = standard_svd(W_f32, eps)

        # Compute approximation error
        W_approx = (U * S.unsqueeze(0)) @ Vh
        diff = W_f32 - W_approx
        w_norm = W_f32.norm().item()
        rel_err = diff.norm().item() / max(w_norm, 1e-12)
        data_err = rel_err

        if Cov is not None:
            ev, evec = safe_eigh(Cov)
            ev = ev.clamp(min=1e-8)
            Cs = (evec * ev.sqrt().unsqueeze(0)) @ evec.t()
            data_err = (diff @ Cs).norm().item() / max((W_f32 @ Cs).norm().item(), 1e-12)
            del ev, evec, Cs

        del W_approx, diff

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

# === CALIBRATION TEXTS (64 diverse sentences) ===

CALIBRATION_TEXTS = [
    "The transformer architecture uses self-attention mechanisms to process sequences.",
    "In mathematics, differential geometry studies smooth manifolds and their properties.",
    "Machine learning models require large amounts of training data to generalize well.",
    "Neural networks consist of layers of interconnected nodes that transform data.",
    "The history of computing spans from mechanical calculators to quantum processors.",
    "Natural language processing enables computers to understand human communication.",
    "Quantum mechanics describes the behavior of particles at the atomic scale.",
    "The theory of relativity changed our understanding of space and time fundamentally.",
    "Photosynthesis converts sunlight into chemical energy in plant cells efficiently.",
    "DNA stores genetic information using four nucleotide bases in a double helix.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Entropy measures the disorder or randomness in a thermodynamic system.",
    "Black holes are regions where gravity is so strong that nothing can escape.",
    "Algorithms are step-by-step procedures for solving computational problems efficiently.",
    "The internet connects billions of devices through standardized communication protocols.",
    "Artificial intelligence aims to create systems that can perform human-like reasoning.",
    "The Renaissance marked a period of cultural rebirth in Europe during the fourteenth century.",
    "Democracy originated in ancient Athens where citizens voted on laws directly.",
    "The Industrial Revolution transformed manufacturing through mechanization and steam power.",
    "World War Two was the deadliest conflict in human history claiming millions of lives.",
    "The Roman Empire dominated the Mediterranean world for several centuries before declining.",
    "The French Revolution abolished the monarchy and established republican government.",
    "Ancient Egypt built massive pyramids as tombs for their pharaohs thousands of years ago.",
    "The Silk Road connected East and West through trade routes across Central Asia.",
    "Shakespeare wrote many plays and sonnets that remain widely performed today worldwide.",
    "Philosophy examines fundamental questions about existence, knowledge, and morality.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Once upon a time in a distant kingdom there lived a wise old wizard.",
    "Books have been the primary medium for preserving and transmitting knowledge for centuries.",
    "Poetry uses rhythm, imagery, and figurative language to evoke emotions in readers.",
    "Socrates taught his students through questions rather than direct instruction.",
    "The printing press revolutionized the spread of information across European societies.",
    "Cooking a good meal requires fresh ingredients and careful attention to timing.",
    "Regular exercise improves cardiovascular health and reduces the risk of chronic diseases.",
    "Climate change threatens ecosystems worldwide through rising temperatures and sea levels.",
    "Education provides individuals with skills and knowledge needed for productive careers.",
    "Music has the power to influence emotions and bring people together across cultures.",
    "Architecture combines artistic vision with engineering principles to create functional spaces.",
    "Transportation systems move people and goods efficiently across cities and countries.",
    "Agriculture feeds the global population through crop cultivation and animal husbandry.",
    "The function takes an input tensor and returns the compressed representation efficiently.",
    "Memory management is critical in systems programming to prevent leaks and corruption.",
    "Databases store structured information that can be queried using specialized languages.",
    "Cloud computing provides scalable resources for processing large datasets remotely.",
    "Version control systems track changes to source code across multiple developers.",
    "Operating systems manage hardware resources and provide services to application software.",
    "Compilers translate high-level programming languages into machine-executable instructions.",
    "Encryption protects sensitive data by transforming it into unreadable ciphertext.",
    "A matrix is a rectangular array of numbers arranged in rows and columns.",
    "The eigenvalues of a symmetric matrix are always real numbers by the spectral theorem.",
    "Integration calculates the area under a curve defined by a continuous function.",
    "Probability theory provides the mathematical framework for reasoning about uncertain events.",
    "Graph theory studies the properties of networks consisting of nodes and edges.",
    "Topology studies properties of shapes that are preserved under continuous deformations.",
    "Linear algebra provides tools for solving systems of equations and analyzing transformations.",
    "Statistics enables inference about populations based on samples of observed data.",
    "The cat sat on the mat and watched the birds fly across the garden.",
    "In the year two thousand twenty-six, many technological advances were made globally.",
    "Paris is the capital of France, known for the Eiffel Tower and its cuisine.",
    "The ocean covers more than seventy percent of the Earth's surface with saltwater.",
    "Mountains are formed by tectonic plate collisions that push rock upward over time.",
    "Rivers carry water and sediment from highlands to the sea through erosion.",
    "Forests provide habitat for wildlife and play a crucial role in carbon sequestration.",
    "Cities are complex systems that require infrastructure for water, energy, and transport.",
]

# === STREAMING COVARIANCE (per-block, not all-at-once) ===

def collect_covs_for_layers(model, tokenizer, layer_names, seq_len=64):
    """Collect input activation covariance matrices for the given layers.
    Covariances are accumulated on CPU to avoid VRAM pressure."""
    covs, tcounts, hooks = {}, {}, []
    target = set(layer_names)
    dev = next(model.parameters()).device

    def make_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1])
            if name not in covs:
                covs[name] = torch.zeros(x.shape[1], x.shape[1])
                tcounts[name] = 0
            covs[name] += x.t() @ x
            tcounts[name] += x.shape[0]
        return fn

    for name, mod in model.named_modules():
        if name in target and isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    n_errors = 0
    with torch.no_grad():
        for i, text in enumerate(CALIBRATION_TEXTS):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
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
    """Measure per-layer sensitivity by noise injection."""
    log("  Calibrating layer sensitivity...")
    dev = next(model.parameters()).device
    ids = tokenizer(ref_text, return_tensors="pt")
    ids = {k: v.to(dev) for k, v in ids.items()}
    with torch.no_grad():
        base_loss = model(**ids, labels=ids["input_ids"]).loss.item()

    sens = {}
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if min(mod.out_features, mod.in_features) < 256:
            continue
        if any(p in name.lower() for p in ('embed', 'norm', 'lm_head')):
            continue
        orig_w = mod.weight.data.clone()
        noise = 0.01 * orig_w.norm() * torch.randn_like(orig_w) / math.sqrt(orig_w.numel())
        mod.weight.data += noise
        with torch.no_grad():
            try:
                nl = model(**ids, labels=ids["input_ids"]).loss.item()
            except RuntimeError:
                nl = base_loss
        mod.weight.data = orig_w
        del orig_w, noise
        sens[name] = abs(nl - base_loss) / 0.01

    if sens:
        mx = max(sens.values())
        if mx > 0:
            sens = {k: v / mx for k, v in sens.items()}
    log(f"  Calibrated {len(sens)} layers")
    return sens

# === SURGERY ===

def set_mod(model, name, mod):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)

def get_block_groups(model, skip_pats=('lm_head', 'embed', 'norm', 'layernorm'), min_dim=256):
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

    for block_name, layers in block_groups:
        layer_names = [n for n, _ in layers]
        log(f"\n  --- {block_name} ({len(layers)} layers) ---")

        # Collect covariances for this block
        covs = collect_covs_for_layers(model, tokenizer, layer_names)
        n_cov = sum(1 for n in layer_names if n in covs)
        cov_mb = sum(c.numel() * 4 / 1e6 for c in covs.values())
        log(f"  Covariances: {n_cov}/{len(layers)} ({cov_mb:.0f} MB)")

        for name, mod in layers:
            m, n = mod.out_features, mod.in_features
            layer_idx += 1

            # Free memory before each layer (critical for large down_proj layers)
            free_mem()

            try:
                s = sensitivities.get(name, 0.5)
                aeps = base_eps * (0.3 + 1.7 * (1.0 - s))
                aeps = max(0.02, min(aeps, base_eps * 2.5))

                # Extract weight to CPU for SVD
                w32 = mod.weight.data.float().cpu()
                b32 = mod.bias.data.float().cpu() if mod.bias is not None else None
                cov = covs.get(name, None)

                t1 = time.time()
                tcfds = TCFDSLinear.from_weight(w32, b32, aeps, store_dtype, Cov=cov)
                svd_t = time.time() - t1
                del w32, b32

                if tcfds.compression < 1.05:
                    log(f"  [{layer_idx}/{total_layers}] {name:<38} SKIP (ratio {tcfds.compression:.2f}x < 1.05)")
                    del tcfds
                    continue

                orig_p = m * n + (m if mod.bias is not None else 0)
                tot_orig += orig_p
                tot_comp += tcfds.param_count()
                tcfds = tcfds.to(next(model.parameters()).device)
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

        # Free block covariances
        del covs
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
        'version': 'v6.3'
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
    data = torch.load(path, map_location='cpu', weights_only=False)
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
    for li in r["layers"][:5]:
        mod = dict(model.named_modules())[li["name"]]
        S = mod.S.detach().float()
        sv_checks.append(round((S[0] / S[-1]).item(), 1) if S.numel() > 1 else 0.0)
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

def gen(model, tok, prompt, max_new=80):
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
    return tok.decode(out[0], skip_special_tokens=True)

def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt")
    dev = next(model.parameters()).device
    ids = {k: v.to(dev) for k, v in ids.items()}
    with torch.no_grad():
        o = model(**ids, labels=ids["input_ids"])
    return torch.exp(o.loss).item()

# === MAIN ===

def main():
    pa = argparse.ArgumentParser(description="TCFDS v6.3 - Data-Aware LLM Compression")
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
    a = pa.parse_args()

    print("=" * 65)
    print(f"  TCFDS v6.3 (Data-Aware + GPU + Memory Safety) | eps={a.eps}")
    print(f"  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {total_vram:.1f} GB total")
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
        while True:
            try:
                u = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not u or u.lower() in ('quit', 'exit', 'q'):
                break
            pr = f"<|user|>\n{u}</s>\n<|assistant|>\n" if "chat" in a.model.lower() else u
            try:
                print(f"AI: {gen(model, tok, pr, 150)[len(pr):].strip()}\n")
            except Exception as e:
                print(f"Error: {e}\n")
        return

    # [1] Load model
    log(f"\n[1/6] Loading model...")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()
    model = model.to(DEVICE)
    dtype = next(model.parameters()).dtype
    np_ = sum(p.numel() for p in model.parameters())
    bpe = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    log(f"  {np_:,} params | {dtype} | {np_*bpe/1e9:.2f} GB | {DEVICE}")

    # [2] Baseline
    log("\n[2/6] Baseline...")
    prompts = [
        "The key insight behind neural networks is",
        "In mathematics, a manifold is",
        "Once upon a time, a young scientist"
    ]
    ref = ("The Transformer uses self-attention to process sequences in parallel. "
           "Each layer computes attention weights so tokens attend to each other, "
           "capturing complex long-range dependencies efficiently.")
    try:
        bppl = ppl(model, tok, ref)
        log(f"  PPL: {bppl:.2f}")
    except Exception:
        bppl = float('inf')
    for p in prompts:
        try:
            t = gen(model, tok, p, 60)
            print(f'  "{p}"')
            print(f"    -> {t[len(p):len(p)+120]}")
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
    try:
        cppl = ppl(model, tok, ref)
        log(f"  PPL: {bppl:.2f} -> {cppl:.2f} ({cppl/bppl:.3f}x)")
    except Exception:
        cppl = float('inf')
    for p in prompts:
        try:
            t = gen(model, tok, p, 60)
            print(f'  "{p}"')
            print(f"    -> {t[len(p):len(p)+120]}")
        except Exception as e:
            print(f"  ERR: {e}")

    # [6] Verify + save
    log("\n[6/6] Verification...")
    rpt = verify(model, store_dtype=dtype)
    rpt["model"] = a.model
    rpt["eps"] = a.eps
    rpt["device"] = str(DEVICE)
    rpt["base_ppl"] = round(bppl, 2) if bppl != float('inf') else None
    rpt["comp_ppl"] = round(cppl, 2) if cppl != float('inf') else None
    rpt["results"] = results
    rpt["version"] = "v6.3"
    ck = rpt["checks"]
    print(f"  CHECK 1 (modules replaced):  {'PASS' if ck['modules_replaced'] else 'FAIL'}")
    print(f"  CHECK 2 (memory reduced):    {'PASS' if ck['memory_reduced'] else 'FAIL'} "
          f"({ck['savings_mb']:.1f}MB, {ck['ratio']:.1f}x)")
    print(f"  CHECK 3 (structured SVD):    {'PASS' if ck['structured'] else 'FAIL'} "
          f"(SV ranges: {ck['sv_ranges']})")
    print(f"  >>> {rpt['verdict']} <<<")
    with open("tcfds_report_v63.json", "w") as f:
        json.dump(rpt, f, indent=2, default=str)

    if a.save:
        save_compressed(model, results, {
            'model_name': a.model, 'eps': a.eps,
            'base_ppl': bppl, 'comp_ppl': cppl, 'version': 'v6.3'
        }, a.save)

    print(f"\n{'='*65}")
    print(f"  TCFDS v6.3 Results")
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
    while True:
        try:
            u = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u or u.lower() in ('quit', 'exit', 'q'):
            break
        pr = f"<|user|>\n{u}</s>\n<|assistant|>\n" if "chat" in a.model.lower() else u
        try:
            print(f"AI: {gen(model, tok, pr, 150)[len(pr):].strip()}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        input("Enter to exit...")
