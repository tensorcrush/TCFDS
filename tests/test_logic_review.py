import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location("tcfds_module", Path(__file__).resolve().parents[1] / "tcfds.py")
tcfds = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(tcfds)


def test_tcfdsfwd_backward_matches_explicit_formula():
    torch.manual_seed(0)
    b, n, k, m = 4, 7, 3, 5
    x = torch.randn(b, n, dtype=torch.float64, requires_grad=True)
    U = torch.randn(m, k, dtype=torch.float64, requires_grad=True)
    S = torch.randn(k, dtype=torch.float64, requires_grad=True)
    V = torch.randn(n, k, dtype=torch.float64, requires_grad=True)

    # Custom autograd path
    y_custom = tcfds.TCFDSFwd.apply(x, U, S, V)
    loss_custom = (y_custom ** 2).sum()
    grads_custom = torch.autograd.grad(loss_custom, (x, U, S, V), retain_graph=False)

    # Explicit baseline path (PyTorch autograd)
    y_ref = ((x @ V) * S.unsqueeze(0)) @ U.t()
    loss_ref = (y_ref ** 2).sum()
    grads_ref = torch.autograd.grad(loss_ref, (x, U, S, V), retain_graph=False)

    for gc, gr in zip(grads_custom, grads_ref):
        assert torch.allclose(gc, gr, rtol=1e-8, atol=1e-8)


def test_do_svd_rank_is_consistent_on_small_matrix():
    torch.manual_seed(0)
    W = torch.randn(8, 6)
    total_energy = float((W ** 2).sum().item())
    U, S, Vh, k = tcfds._do_svd(W, eps=0.25, total_energy=total_energy)

    assert 1 <= k <= min(W.shape)
    assert U.shape[1] == len(S) == Vh.shape[0] == k


def test_torch_load_compat_passes_weights_only_false(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs.copy())
        return {"ok": True}

    monkeypatch.setattr(tcfds.torch, "load", fake_load)
    out = tcfds.torch_load_compat("dummy.pt", map_location="cpu")

    assert out["ok"] is True
    assert len(calls) == 1
    assert calls[0].get("weights_only") is False


def test_torch_load_compat_fallback_when_weights_only_unsupported(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs.copy())
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"fallback": True}

    monkeypatch.setattr(tcfds.torch, "load", fake_load)
    out = tcfds.torch_load_compat("dummy.pt", map_location="cpu")

    assert out["fallback"] is True
    assert len(calls) == 2
    assert "weights_only" in calls[0]
    assert "weights_only" not in calls[1]


def test_detect_chat_format_specificity_basics():
    assert tcfds.detect_chat_format("meta-llama/Llama-3-8B")['type'] == 'llama3'
    assert tcfds.detect_chat_format("TinyLlama/TinyLlama-1.1B-Chat-v1.0")['type'] == 'chatml'
