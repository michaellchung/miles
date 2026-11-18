"""MILES_K31_ROUTER_GEMM on CUDA at the K3.1 router shape (hidden 7168, 896 experts): fp32 mode is Megatron's
torch-mm router bit for bit; tf32 / bf16 give logits equal to fp32 up to accumulation (BF16 operands are exact in
TF32 and BF16 products are exact in fp32; the tensor-core adders accumulate less exactly than the SIMT sgemm:
on B300 max |logit error| / max |logit| is 2.8e-6 fp32, 8.3e-6 bf16, 1.6e-5 tf32), so the R3 gate weights
(sigmoid of the logits at the replayed experts, renormalized) move by ~5e-6 relative, far below the BF16
rounding (2**-9) of the expert outputs they scale."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core.transformer.moe.moe_utils import router_gating_linear  # noqa: E402

from miles.utils.router_gemm import RouterGemm  # noqa: E402

T, H, E, TOPK = 4096, 7168, 896, 16


def _inputs():
    torch.manual_seed(0)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16).requires_grad_()
    w = (torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16)
    return x, w


def _gate_weights(logits, top):
    scores = torch.sigmoid(logits.float()).gather(1, top)
    return scores / (scores.sum(-1, keepdim=True) + 1e-20)


def test_fp32_mode_is_megatron_bitwise():
    x, w = _inputs()
    xr = x.detach().clone().requires_grad_()
    out, ref = RouterGemm.apply(x, w, None, "fp32"), router_gating_linear(xr, w, None, torch.float32, True)
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.equal(x.grad, xr.grad)


@pytest.mark.parametrize("mode", ["tf32", "bf16"])
def test_tensor_core_modes_move_gate_weights_below_bf16(mode):
    x, w = _inputs()
    exact = torch.mm(x.detach().double(), w.double().t())
    ref = router_gating_linear(x.detach(), w, None, torch.float32, True)
    out = RouterGemm.apply(x, w, None, mode)
    scale = exact.abs().max()
    ref_err = ((ref.double() - exact).abs().max() / scale).item()
    err = ((out.double() - exact).abs().max() / scale).item()
    assert ref_err < 1e-5, ref_err
    assert err < 1e-4, err
    top = torch.topk(ref, TOPK, dim=-1).indices  # the replayed experts
    gw_ref, gw = _gate_weights(ref, top), _gate_weights(out, top)
    gw_err = (((gw - gw_ref).abs() / gw_ref).max()).item()
    assert gw_err < 2.0**-12, gw_err
    g = torch.randn_like(out)
    out.backward(g)
    gi_exact = torch.mm(g.double(), w.double())
    gi_err = ((x.grad.double() - gi_exact).norm() / gi_exact.norm()).item()
    assert gi_err < 1e-2, gi_err
