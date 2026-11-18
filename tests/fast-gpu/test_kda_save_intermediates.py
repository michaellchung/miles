"""fla's chunk_kda with disable_recompute (MILES_K31_KDA_SAVE_INTERMEDIATES=1) gives the same output and
gradients as its default backward, which recomputes the gate cumsum, w/u/qg/kg and the state h, bit for bit,
on the K3.1 call (gate in kernel, safe gate, q/k L2 norm, V-first state, packed sequences)."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("fla.ops.kda")

from miles_plugins.models import linear_attn  # noqa: E402


def _run(monkeypatch, flag):
    monkeypatch.setenv("MILES_K31_KDA_SAVE_INTERMEDIATES", flag)
    torch.manual_seed(0)
    T, H, D = 4096, 4, 128
    q, k, v, decay = (torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(4))
    beta_logits = torch.randn(1, T, H, device="cuda", dtype=torch.bfloat16).requires_grad_()
    A_log = (torch.randn(H, device="cuda") * 0.5).requires_grad_()
    dt_bias = (torch.randn(H * D, device="cuda") * 0.5).requires_grad_()
    cu_seqlens = torch.tensor([0, 1000, 2944, T], device="cuda", dtype=torch.long)
    out = linear_attn.kda_recurrence(
        q,
        k,
        v,
        beta_logits,
        decay.flatten(2),
        A_log,
        dt_bias,
        gate_lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        cp_context=None,
    )
    out.backward(torch.randn_like(out))
    return [out.detach()] + [t.grad for t in (q, k, v, decay, beta_logits, A_log, dt_bias)]


def test_saved_intermediates_match_the_recomputing_backward(monkeypatch):
    for a, b in zip(_run(monkeypatch, "0"), _run(monkeypatch, "1"), strict=True):
        assert torch.equal(a, b)
