"""The fused Triton attention-residual mix matches the eager mix (CUDA, or CPU under the Triton interpreter)."""

import os
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if _CUDA else "cpu"
if not _CUDA:
    # Interpret the kernels on CPU. triton.jit (incl. triton.language's own helpers) picks the interpreter when
    # triton is imported, so the flag must be set before the first triton import of this process.
    if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("triton was imported without TRITON_INTERPRET=1 and there is no GPU", allow_module_level=True)
    # kept for the whole process: kernel launches check the flag too, and without a GPU no kernel compiles anyway
    os.environ["TRITON_INTERPRET"] = "1"
pytest.importorskip("triton")
from miles_plugins.models.kimi_k3 import attn_res_fused  # noqa: E402

from miles_plugins.models.kimi_k3 import ops  # noqa: E402
from miles_plugins.models.kimi_k3.ops import KimiRMSNorm, _attn_res_mix  # noqa: E402

EPS = 1e-5


def _inputs(tokens, nbank, hidden, dtype, seed=0, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    # rows with different magnitudes so the softmax is not uniform
    scale = torch.exp(spread * torch.randn(tokens, 1, nbank + 1, 1, generator=g))
    rows = torch.randn(tokens, 1, nbank + 1, hidden, generator=g) * scale
    prefix = rows[..., nbank, :].to(dtype)
    bank = rows[..., :nbank, :].contiguous().to(dtype)
    w = (1.0 + 0.5 * torch.randn(hidden, generator=g)) * hidden**-0.5 * 8
    go = torch.randn(tokens, 1, hidden, generator=g).to(dtype)
    return prefix, bank, w, go


def _grads(fn, prefix, bank, w, go, device=DEVICE, cast=None):
    prefix = prefix.detach().to(device, cast or prefix.dtype, copy=True).requires_grad_()
    bank = bank.detach().to(device, cast or bank.dtype, copy=True).requires_grad_()
    w = w.detach().to(device, torch.float64 if cast == torch.float64 else torch.float32, copy=True).requires_grad_()
    out = fn(prefix, bank, w, EPS)
    out.backward(go.to(device, out.dtype))
    return [t.detach().cpu() for t in (out, prefix.grad, bank.grad, w.grad)]


NAMES = ("out", "d_prefix_sum", "d_block_residual", "d_score_weight")


def _rel(a, b):
    """max |a - b| / max |b| (0 when empty)."""
    if b.numel() == 0:
        return 0.0
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("nbank", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("tokens,hidden", [(5, 64), (33, 100), (3, 7168)])
def test_fp32_matches_eager(tokens, nbank, hidden):
    args = _inputs(tokens, nbank, hidden, torch.float32, seed=nbank * 7 + tokens)
    eager = _grads(_attn_res_mix, *args)
    fused = _grads(attn_res_fused.attn_res_mix_fused, *args)
    ref = _grads(_attn_res_mix, *args, device="cpu", cast=torch.float64)
    for name, e, f, r in zip(NAMES, eager, fused, ref, strict=True):
        assert f.shape == e.shape and f.dtype == e.dtype, name
        err, err_eager = _rel(f, e), _rel(e, r)
        print(
            f"fp32 T={tokens} H={hidden} bank={nbank} {name}: fused-eager {err:.2e}, "
            f"fused-fp64 {_rel(f, r):.2e}, eager-fp64 {err_eager:.2e}"
        )
        # fp32 rounding only: the sums over H (and over tokens for the weight grad) run in another order
        assert err < 2e-5, name


@pytest.mark.parametrize("nbank", [1, 8])
def test_bf16_matches_eager(nbank):
    tokens, hidden = 64, 256
    args = _inputs(tokens, nbank, hidden, torch.bfloat16, seed=3)
    eager = _grads(_attn_res_mix, *args)
    fused = _grads(attn_res_fused.attn_res_mix_fused, *args)
    ref = _grads(_attn_res_mix, *args, device="cpu", cast=torch.float64)
    for name, e, f, r in zip(NAMES, eager, fused, ref, strict=True):
        assert f.dtype == e.dtype, name
        exact = (f == e).float().mean().item()
        err_f, err_e = _rel(f, r.to(f.dtype)), _rel(e, r.to(e.dtype))
        print(
            f"bf16 bank={nbank} {name}: bitwise-equal {exact:.4f}, fused-eager {_rel(f, e):.2e}, "
            f"fused-fp64 {err_f:.2e}, eager-fp64 {err_e:.2e}"
        )
        # both round the same fp32 value to bf16; an fp32-ulp difference flips at most one bf16 ulp
        assert _rel(f, e) < 1.6e-2, name
        assert err_f <= err_e * 1.5 + 1e-6, name


def _modules(hidden, device):
    torch.manual_seed(0)
    proj = torch.nn.Linear(hidden, 1, bias=False, device=device)
    norm = KimiRMSNorm(hidden, eps=EPS, device=device)
    out_norm = KimiRMSNorm(hidden, eps=EPS, device=device)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        out_norm.weight.uniform_(0.5, 1.5)
    return proj, norm, out_norm


def _aggregate(fused, wrap=None, tokens=40, nbank=3, hidden=96, chunk=16):
    prefix, bank, _, go = _inputs(tokens, nbank, hidden, torch.float32, seed=11)
    prefix = prefix.to(DEVICE).requires_grad_()
    bank = bank.to(DEVICE).requires_grad_()
    mods = _modules(hidden, DEVICE)

    def f(p, b):
        return ops.attn_res_aggregate(p, b, *mods, chunk=chunk, fused=fused)

    out = f(prefix, bank) if wrap is None else wrap(f, prefix, bank)
    out.backward(go.to(DEVICE))
    params = [m.weight.grad for m in mods]
    return [t.detach().cpu() for t in (out, prefix.grad, bank.grad, *params)]


def test_aggregate_fused_matches_eager_with_param_grads():
    eager = _aggregate(False)
    fused = _aggregate(True)
    for name, e, f in zip(NAMES[:3] + ("d_score_proj", "d_score_norm", "d_output_norm"), eager, fused, strict=True):
        print(f"aggregate {name}: fused-eager {_rel(f, e):.2e}")
        assert _rel(f, e) < 2e-5, name


@pytest.mark.parametrize("reentrant", [False, True])
def test_fused_under_recompute_is_bitwise_stable(reentrant):
    """A full-layer recompute (Megatron checkpoint) re-runs the fused forward; grads must not change."""
    plain = _aggregate(True)
    again = _aggregate(True)
    recomputed = _aggregate(True, wrap=lambda f, p, b: checkpoint(f, p, b, use_reentrant=reentrant))
    for a, b, c in zip(plain, again, recomputed, strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_env_switch_default_off(monkeypatch):
    monkeypatch.setattr(ops, "ATTN_RES_FUSED", False)
    t = torch.zeros(2, 1, 8)
    assert not ops._use_fused_mix(t, t.unsqueeze(-2), None)
    monkeypatch.setattr(ops, "ATTN_RES_FUSED", True)
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert ops._use_fused_mix(t, t.unsqueeze(-2), None) == t.is_cuda
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    assert ops._use_fused_mix(t, t.unsqueeze(-2), None)
    assert not ops._use_fused_mix(t, t.unsqueeze(-2).bfloat16(), None)


def test_fused_saves_only_inputs():
    prefix, bank, w, _ = _inputs(64, 8, 64, torch.bfloat16)
    prefix, bank, w = (t.to(DEVICE).requires_grad_() for t in (prefix, bank, w))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        attn_res_fused.attn_res_mix_fused(prefix, bank, w, EPS)
    assert {s.data_ptr() for s in saved} <= {prefix.data_ptr(), bank.data_ptr(), w.data_ptr()}


@pytest.mark.parametrize("pad", [1, 9])
def test_zero_bank_rows_match_eager(pad):
    """All-zero bank rows (MILES_K3_ATTN_RES_BANK_PAD_ROWS): rsqrt(eps) * 0 scores 0 in both paths; rows > 16."""
    prefix, bank, w, go = _inputs(9, 8, 96, torch.float32, seed=5)
    bank = torch.cat((torch.zeros(9, 1, pad, 96), bank), dim=-2)
    eager = _grads(_attn_res_mix, prefix, bank, w, go)
    fused = _grads(attn_res_fused.attn_res_mix_fused, prefix, bank, w, go)
    for name, e, f in zip(NAMES, eager, fused, strict=True):
        print(f"zero pad={pad} {name}: fused-eager {_rel(f, e):.2e}")
        assert _rel(f, e) < 2e-5, name
