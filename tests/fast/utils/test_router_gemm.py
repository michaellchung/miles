"""MILES_K31_ROUTER_GEMM: unset installs nothing; fp32 mode is Megatron's torch-mm router GEMM bit for bit and
skips a frozen router's weight gradient; install_router_gemm swaps the gating of every Megatron Router."""

import types

import pytest
import torch
from megatron.core.transformer.moe.moe_utils import router_gating_linear
from megatron.core.transformer.moe.router import TopKRouter

from miles.utils import router_gemm
from miles.utils.router_gemm import RouterGemm, install_router_gemm, router_gemm_mode

T, H, E = 64, 256, 32


def _inputs(weight_grad):
    torch.manual_seed(0)
    x = torch.randn(T, 1, H).to(torch.bfloat16)
    w = (torch.randn(E, H) * 0.02).to(torch.bfloat16)
    return x.requires_grad_(), w.requires_grad_(weight_grad)


def test_unset_is_off_and_bad_value_raises(monkeypatch):
    monkeypatch.delenv("MILES_K31_ROUTER_GEMM", raising=False)
    assert router_gemm_mode() is None
    assert install_router_gemm(torch.nn.Linear(2, 2)) == 0
    monkeypatch.setenv("MILES_K31_ROUTER_GEMM", "fp16")
    with pytest.raises(ValueError, match="MILES_K31_ROUTER_GEMM"):
        router_gemm_mode()


@pytest.mark.parametrize("weight_grad", [True, False])
@pytest.mark.parametrize("mode", ["fp32", "tf32"])  # tf32 is a CUDA matmul flag: on CPU it is fp32
def test_matches_megatron_torch_mm_bitwise(mode, weight_grad):
    x, w = _inputs(weight_grad)
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_(weight_grad)
    out = RouterGemm.apply(x, w, None, mode)
    ref = router_gating_linear(xr, wr, None, torch.float32, True)
    assert out.dtype == torch.float32 and torch.equal(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.equal(x.grad, xr.grad)
    if weight_grad:
        assert torch.equal(w.grad, wr.grad)
    else:
        assert w.grad is None


def test_frozen_router_skips_the_weight_gemm(monkeypatch):
    calls = []
    real = router_gemm._mm
    monkeypatch.setattr(router_gemm, "_mm", lambda a, b, mode: calls.append(tuple(b.shape)) or real(a, b, mode))
    x, w = _inputs(weight_grad=False)
    RouterGemm.apply(x, w, None, "fp32").sum().backward()
    assert calls == [(H, E), (E, H)]  # forward, input gradient; no weight gradient


def _router(dtype="fp32"):
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter((torch.randn(E, H) * 0.02).to(torch.bfloat16), requires_grad=False)
    router.bias = None
    router.config = types.SimpleNamespace(moe_router_dtype=dtype)
    return router


def test_install_swaps_every_router_once():
    model = torch.nn.ModuleList([_router(), torch.nn.Linear(2, 2), _router()])
    assert install_router_gemm(model, "fp32") == 2
    assert install_router_gemm(model, "fp32") == 0
    x, _ = _inputs(weight_grad=False)
    ref = router_gating_linear(x, model[0].weight, None, torch.float32, True)
    assert torch.equal(model[0].gating(x), ref)


def test_install_needs_fp32_router_dtype():
    with pytest.raises(ValueError, match="moe-router-dtype fp32"):
        install_router_gemm(torch.nn.ModuleList([_router(dtype="bf16")]), "tf32")
