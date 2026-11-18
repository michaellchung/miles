"""MoE router gating GEMM precision (Megatron backend only) (``MILES_K31_ROUTER_GEMM``; unset keeps Megatron's own path).

Miles forces ``moe_router_use_torch_mm`` (megatron_utils/arguments.py), so with ``--moe-router-dtype fp32``
Megatron's ``RouterGatingLinearFunction`` runs ``torch.mm(inp.float(), weight.float().t())`` with TF32 off:
a CUDA-core SIMT SGEMM (``cutlass3x_sm100_simt_sgemm_f32``), forward, recompute and backward, and its backward
also computes the router's weight gradient even when the router is frozen.

Modes (each installed on every Megatron ``Router`` by ``install_router_gemm``):

* ``fp32``: the same fp32 ``torch.mm`` calls as Megatron, so logits and input gradients are bit for bit
  Megatron's; only the weight gradient of a frozen router (``requires_grad=False``) is skipped.
* ``tf32``: the same calls with TF32 tensor cores. The router input and weight are BF16, whose 8-bit
  significands TF32 holds exactly, so every forward product is exact and the logits differ from fp32 only
  by accumulation order. The backward rounds the fp32 logit gradient to TF32 (2**-11 relative) before a
  GEMM whose result is cast to the BF16 input dtype (2**-8).
* ``bf16``: BF16 tensor-core GEMMs with fp32 output (``torch.mm(..., out_dtype=torch.float32)``). Forward as
  exact as ``tf32``; the backward rounds the logit gradient to BF16.

Under rollout routing replay (R3) the top-k comes from the rollout, but the gate weights are still the
local ``sigmoid(logits)`` gathered at the replayed experts and renormalized (Megatron
``topk_routing_with_score_function``), so these modes reach the gate weights and the routed-expert outputs.
"""

from __future__ import annotations

import contextlib
import os
import types

import torch

ROUTER_GEMM_ENV = "MILES_K31_ROUTER_GEMM"
MODES = ("fp32", "tf32", "bf16")
_INSTALLED = "_miles_router_gemm_mode"


def router_gemm_mode() -> str | None:
    mode = os.environ.get(ROUTER_GEMM_ENV, "") or None
    if mode is not None and mode not in MODES:
        raise ValueError(f"{ROUTER_GEMM_ENV}={mode!r}: expected one of {MODES} or unset")
    return mode


@contextlib.contextmanager
def _tf32_matmul():
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):  # config-access-exempt: torch API probe, only newer torch has it
        previous = matmul.fp32_precision
        matmul.fp32_precision = "tf32"
        try:
            yield
        finally:
            matmul.fp32_precision = previous
    else:
        previous = matmul.allow_tf32
        matmul.allow_tf32 = True
        try:
            yield
        finally:
            matmul.allow_tf32 = previous


def _mm(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """fp32 [m, n] product of a [m, k] and b [k, n] in the mode's precision."""
    if mode == "bf16":
        return torch.mm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)
    if mode == "tf32":
        with _tf32_matmul():
            return torch.mm(a.float(), b.float())
    return torch.mm(a.float(), b.float())


class RouterGemm(torch.autograd.Function):
    """``RouterGatingLinearFunction`` (torch-mm branch, fp32 router dtype) with a selectable GEMM precision."""

    @staticmethod
    def forward(ctx, inp, weight, bias, mode):
        ctx.save_for_backward(inp, weight, bias)
        ctx.mode = mode
        x = inp.view(-1, inp.shape[-1])
        if bias is not None and mode == "fp32":
            out = torch.addmm(bias.float(), x.float(), weight.float().t())
        else:
            out = _mm(x, weight.t(), mode)
            if bias is not None:
                out = out + bias.float()
        return out.view(*inp.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        g = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = _mm(g, weight, ctx.mode).to(inp.dtype).view(inp.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = _mm(g.t(), inp.view(-1, inp.shape[-1]), ctx.mode).to(weight.dtype)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = g.sum(dim=0).to(weight.dtype)
        return grad_input, grad_weight, grad_bias, None


def _gating(self, input: torch.Tensor) -> torch.Tensor:
    """``Router.gating`` for ``--moe-router-dtype fp32`` through ``RouterGemm`` (training: the router already
    lives on the device; Megatron's CPU-to-device move for inference is not needed here)."""
    return RouterGemm.apply(
        input,
        self.weight,
        self.bias,
        getattr(self, _INSTALLED),  # config-access-exempt: mode set by install_router_gemm
    )


def install_router_gemm(model: torch.nn.Module, mode: str | None = None) -> int:
    """Route every Megatron MoE router's gating GEMM through ``RouterGemm``; returns how many.

    ``mode`` defaults to ``MILES_K31_ROUTER_GEMM``; nothing is installed when it is unset.
    """
    mode = router_gemm_mode() if mode is None else mode
    if mode is None:
        return 0
    assert mode in MODES, mode
    from megatron.core.transformer.moe.router import Router

    count = 0
    for module in model.modules():
        if (
            not isinstance(module, Router)
            or getattr(module, _INSTALLED, None) is not None  # config-access-exempt: install marker on the Router
        ):
            continue
        if module.config.moe_router_dtype != "fp32":
            raise ValueError(
                f"{ROUTER_GEMM_ENV}={mode} needs --moe-router-dtype fp32, got {module.config.moe_router_dtype!r}"
            )
        setattr(module, _INSTALLED, mode)
        module.gating = types.MethodType(_gating, module)
        count += 1
    return count


__all__ = ["MODES", "ROUTER_GEMM_ENV", "RouterGemm", "install_router_gemm", "router_gemm_mode"]
