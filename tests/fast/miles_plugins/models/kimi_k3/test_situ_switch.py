"""MILES_K3_SITU_TRITON is off by default, and where the fused kernel cannot run (CPU, non-BF16, other betas)
situ_and_mul stays on the eager fp32 chain even with the switch on. The fused kernel's bit equality with the
eager chain is checked on CUDA in tests/fast-gpu/test_kimi_k3_situ_triton.py."""

import torch

from miles_plugins.models.kimi_k3 import ops


def _eager(x, beta=4.0, linear_beta=25.0):
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


def test_switch_defaults_off():
    assert ops.SITU_TRITON is False


def test_cpu_and_unsupported_inputs_stay_eager(monkeypatch):
    monkeypatch.setattr(ops, "SITU_TRITON", True)
    torch.manual_seed(0)
    for dtype, kwargs in ((torch.bfloat16, {}), (torch.float32, {}), (torch.bfloat16, {"beta": 2.0})):
        x = (torch.randn(5, 3, 16) * 3).to(dtype).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        out = ops.situ_and_mul(x, **kwargs)
        expected = _eager(ref, **kwargs)
        assert torch.equal(out, expected)
        out.sum().backward()
        expected.sum().backward()
        assert torch.equal(x.grad, ref.grad)
