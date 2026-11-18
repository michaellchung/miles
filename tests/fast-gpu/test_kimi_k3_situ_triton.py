"""MILES_K3_SITU_TRITON's fused SiTU-and-mul equals the eager fp32 chain bit for bit: forward output and input
gradient, with and without the router-prob scale (and the prob gradient), over a wide input range, empty and
3-D inputs; and it saves only its BF16 input for backward."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models.kimi_k3 import ops  # noqa: E402
from miles_plugins.models.kimi_k3.situ_triton import fused_situ_and_mul  # noqa: E402

N = 3072


def _eager(x):
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    linear = 25.0 * torch.tanh(linear / 25.0)
    return (gate * linear).to(x.dtype)


def _inputs(rows, scale, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(rows, 2 * N, device="cuda") * scale).to(torch.bfloat16)
    return x, torch.randn(rows, N, device="cuda").to(torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("scale", [0.1, 3.0, 40.0])
def test_matches_eager_bitwise(scale):
    x, dy = _inputs(4096, scale)
    xe, xf = x.clone().requires_grad_(), x.clone().requires_grad_()
    ye, yf = _eager(xe), fused_situ_and_mul(xf)
    ye.backward(dy)
    yf.backward(dy)
    assert not (msg := _where_differ("y", yf, ye, x[:, :N], x[:, N:])), msg
    assert not (msg := _where_differ("dx", xf.grad, xe.grad, x, x)), msg


def _where_differ(name, got, want, gate, linear) -> str:
    """'' if bitwise equal, else the count and the input range of the differing elements."""
    differ = _bits(got) != _bits(want)
    if not differ.any():
        return ""
    g, lin, w = gate.float()[differ], linear.float()[differ], want.float()[differ]
    return (
        f"{name}: {int(differ.sum())}/{differ.numel()} differ; gate in [{g.min().item():.4g}, {g.max().item():.4g}], "
        f"linear |.| in [{lin.abs().min().item():.4g}, {lin.abs().max().item():.4g}], eager |{name}| max "
        f"{w.abs().max().item():.4g}"
    )


@pytest.mark.parametrize("probs_shape", ["rows", "rows1"])
def test_with_probs_matches_megatron_bitwise(probs_shape):
    x, dy = _inputs(4096, 3.0, seed=1)
    probs = torch.rand(4096, device="cuda")
    if probs_shape == "rows1":
        probs = probs.view(-1, 1)
    xe, pe = x.clone().requires_grad_(), probs.clone().requires_grad_()
    xf, pf = x.clone().requires_grad_(), probs.clone().requires_grad_()
    ye = _eager(xe)
    ye = (ye * pe.reshape(-1, 1)).to(ye.dtype)  # TEGroupedMLP.bias_act_func
    yf = fused_situ_and_mul(xf, pf)
    ye.backward(dy)
    yf.backward(dy)
    assert torch.equal(_bits(yf), _bits(ye))
    assert torch.equal(_bits(xf.grad), _bits(xe.grad))
    assert pf.grad.shape == pe.grad.shape
    assert torch.equal(pf.grad, pe.grad)


def _bf16_values(device="cuda"):
    """Every BF16 value in 2**-40 <= |v| <= 40, and 0: no fp32 intermediate of the chain gets subnormal (where
    Triton's denormal handling is not asserted to match ATen; it only touches results below 1e-38)."""
    bits = torch.arange(65536, dtype=torch.int32, device=device)
    v = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16).view(torch.bfloat16)
    a = v.float().abs()
    return v[(a <= 40) & ((a >= 2.0**-40) | (a == 0))]


def _same(a, b):
    """Bitwise equal, NaN only where the other is NaN too (payloads may differ)."""
    nan = torch.isnan(a.float())
    return torch.equal(nan, torch.isnan(b.float())) and torch.equal(_bits(a)[~nan], _bits(b)[~nan])


@pytest.mark.parametrize("with_probs", [False, True])
def test_every_bf16_input_matches_eager_bitwise(with_probs):
    # every gate value meets every cancellation point of d(gate) (e.g. g = -1.21875, where its tanh and
    # sigmoid terms cancel and any contracted multiply-add shows up as a BF16 ulp); Megatron / TE imported
    # as in the trainer
    import megatron.core.transformer.moe.experts  # noqa: F401
    import transformer_engine.pytorch  # noqa: F401

    values = _bf16_values()
    torch.manual_seed(3)
    reps = 8
    gate = values.repeat(reps)
    linear = values[torch.randperm(values.numel(), device="cuda")].repeat(reps)
    pad = (-gate.numel()) % N
    gate = torch.cat([gate, gate[:pad]]).view(-1, N)
    linear = torch.cat([linear, linear[:pad]]).view(-1, N)
    x = torch.cat([gate, linear], dim=1).contiguous()
    rows = x.shape[0]
    dy = torch.randn(rows, N, device="cuda").to(torch.bfloat16)
    probs = torch.rand(rows, device="cuda") if with_probs else None
    xe, xf = x.clone().requires_grad_(), x.clone().requires_grad_()
    if with_probs:
        pe, pf = probs.clone().requires_grad_(), probs.clone().requires_grad_()
        ye = _eager(xe)
        ye = (ye * pe.reshape(-1, 1)).to(ye.dtype)
        yf = fused_situ_and_mul(xf, pf)
    else:
        ye, yf = _eager(xe), fused_situ_and_mul(xf)
    ye.backward(dy)
    yf.backward(dy)
    assert _same(yf, ye)
    differ = (_bits(xf.grad) != _bits(xe.grad)) & ~torch.isnan(xe.grad.float())
    assert _same(xf.grad, xe.grad), f"{int(differ.sum())} dx differ, gate values {x[differ][:8].float().tolist()}"
    if with_probs:
        assert _same(pf.grad, pe.grad)


def test_switch_routes_and_keeps_shape(monkeypatch):
    monkeypatch.setattr(ops, "SITU_TRITON", True)
    x = (torch.randn(2, 64, 2 * N, device="cuda") * 3).to(torch.bfloat16)
    assert ops.situ_and_mul(x).shape == (2, 64, N)
    assert torch.equal(_bits(ops.situ_and_mul(x)), _bits(_eager(x)))
    empty = torch.empty(0, 2 * N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = ops.situ_and_mul(empty)
    out.sum().backward()
    assert out.shape == (0, N) and empty.grad.shape == empty.shape


def test_saves_only_the_input():
    x, _ = _inputs(8192, 3.0)
    x.requires_grad_()

    def saved_bytes(fn):
        total = 0

        def pack(t):
            nonlocal total
            total += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fn(x * 1)
        return total

    eager, fused = saved_bytes(_eager), saved_bytes(fused_situ_and_mul)
    assert fused <= x.numel() * x.element_size() + 4096
    assert eager > 5 * fused
