"""Fused Triton SiTU-and-mul for Kimi K3 (``MILES_K3_SITU_TRITON=1``), optionally with the router-prob scale.

Reference (miles_plugins/models/kimi_k3/ops.py:situ_and_mul, eager, fp32 chain):

    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)

followed in Megatron TEGroupedMLP.bias_act_func by ``(out * permuted_probs).to(bf16)``.

The kernels below replay that chain op for op so the forward is meant to be bit-identical to eager:
* ``t / scalar`` on CUDA is ``t * (1 / scalar)`` (ATen div_true with a CPU scalar), so /4 -> *0.25 and
  /25 -> *fp32(1/25);
* sigmoid is ``1 / (1 + exp(-x))`` with an IEEE division (div_rn);
* tanh / exp are libdevice (__nv_tanhf / __nv_expf), the same CUDA math library ATen calls;
* every fp32 multiply / add / subtract is explicit PTX ``mul/add/sub.rn.f32`` (no ``.ftz``), which no
  compiler pass contracts and which keeps subnormals as ATen does: ``enable_fp_fusion=False`` alone was not honored
  in every process (inside a Megatron/TE trainer process ``d_s * (1 - s) * s + d_g1 / 4`` came out fused,
  one BF16 ulp off where the two terms cancel, g = -1.21875); the only FMA is ATen's own, tanh_backward's
  ``a * (1 - b*b)`` as nvcc contracts it (``fma_rn(-b, b, 1)``; the plain form differs on B300).
The backward reproduces autograd's per-op gradients (MulBackward, TanhBackward, SigmoidBackward,
DivBackward with a scalar, ToCopyBackward) and their dtype casts; d(probs) is reduced by the same torch
``sum_to_size`` autograd uses, so it can be bit-identical too.

Saves only the BF16 input (and probs) for backward instead of six fp32 [rows, N] intermediates.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # older triton
    from triton.language.extra import libdevice

if os.environ.get("TRITON_INTERPRET") == "1":  # CPU logic check only: libdevice is CUDA-only

    @triton.jit
    def _tanh(x):
        e = tl.exp(2.0 * x)
        return tl.where(x > 15.0, 1.0, tl.where(x < -15.0, -1.0, (e - 1.0) / (e + 1.0)))

    @triton.jit
    def _exp(x):
        return tl.exp(x)

    @triton.jit
    def _div_rn(a, b):
        return a / b

    @triton.jit
    def _fma(a, b, c):
        return a * b + c

    @triton.jit
    def _mul(a, b):
        return a * b

    @triton.jit
    def _add(a, b):
        return a + b

    @triton.jit
    def _sub(a, b):
        return a - b

else:

    @triton.jit
    def _tanh(x):
        return libdevice.tanh(x)

    @triton.jit
    def _exp(x):
        return libdevice.exp(x)

    @triton.jit
    def _div_rn(a, b):
        return tl.math.div_rn(a, b)

    # PTX with an explicit rounding modifier is never contracted, and without .ftz keeps subnormals like ATen
    # (libdevice's *_rn flush them: Triton links libdevice with __CUDA_FTZ=1)
    @triton.jit
    def _fma(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _sub(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )


BETA = 4.0
LINEAR_BETA = 25.0
INV_BETA = float(np.float32(1.0) / np.float32(BETA))  # 0.25
INV_LINEAR_BETA = float(np.float32(1.0) / np.float32(LINEAR_BETA))  # fp32(1/25), as ATen's host reciprocal


@triton.jit
def _situ_fwd_kernel(
    x_ptr,
    p_ptr,
    out_ptr,
    total,
    N: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    LBETA: tl.constexpr,
    INV_LBETA: tl.constexpr,
    HAS_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    row = offs // N
    col = offs - row * N
    base = row * (2 * N) + col
    g = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    lin = tl.load(x_ptr + base + N, mask=mask, other=0.0).to(tl.float32)
    one = tl.full([BLOCK], 1.0, tl.float32)
    c_beta = tl.full([BLOCK], BETA, tl.float32)
    c_inv_beta = tl.full([BLOCK], INV_BETA, tl.float32)
    c_lbeta = tl.full([BLOCK], LBETA, tl.float32)
    c_inv_lbeta = tl.full([BLOCK], INV_LBETA, tl.float32)
    t1 = _tanh(_mul(g, c_inv_beta))
    a = _mul(t1, c_beta)
    s = _div_rn(one, _add(one, _exp(-g)))
    go = _mul(a, s)
    t2 = _tanh(_mul(lin, c_inv_lbeta))
    lo = _mul(t2, c_lbeta)
    y = _mul(go, lo).to(tl.bfloat16)
    if HAS_P:
        p = tl.load(p_ptr + row, mask=mask, other=0.0).to(tl.float32)
        y = _mul(y.to(tl.float32), p).to(tl.bfloat16)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _situ_bwd_kernel(
    x_ptr,
    p_ptr,
    dy_ptr,
    dx_ptr,
    y_ptr,
    total,
    N: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    LBETA: tl.constexpr,
    INV_LBETA: tl.constexpr,
    HAS_P: tl.constexpr,
    WRITE_Y: tl.constexpr,
    TANH_BWD_FMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total
    row = offs // N
    col = offs - row * N
    base = row * (2 * N) + col
    g = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    lin = tl.load(x_ptr + base + N, mask=mask, other=0.0).to(tl.float32)
    one = tl.full([BLOCK], 1.0, tl.float32)
    c_beta = tl.full([BLOCK], BETA, tl.float32)
    c_inv_beta = tl.full([BLOCK], INV_BETA, tl.float32)
    c_lbeta = tl.full([BLOCK], LBETA, tl.float32)
    c_inv_lbeta = tl.full([BLOCK], INV_LBETA, tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_P:
        p = tl.load(p_ptr + row, mask=mask, other=0.0).to(tl.float32)
        # MulBackward0 grad_self = grad * probs (fp32), cast to the bf16 input dtype; ToCopyBackward -> fp32
        gy = _mul(dy, p).to(tl.bfloat16).to(tl.float32)
    else:
        gy = dy
    t1 = _tanh(_mul(g, c_inv_beta))
    a = _mul(t1, c_beta)
    s = _div_rn(one, _add(one, _exp(-g)))
    go = _mul(a, s)
    t2 = _tanh(_mul(lin, c_inv_lbeta))
    lo = _mul(t2, c_lbeta)
    d_go = _mul(gy, lo)
    d_lo = _mul(gy, go)
    d_t2 = _mul(d_lo, c_lbeta)
    if TANH_BWD_FMA:
        d_l1 = _mul(d_t2, _fma(-t2, t2, one))
    else:
        d_l1 = _mul(d_t2, _sub(one, _mul(t2, t2)))
    d_lin = _mul(d_l1, c_inv_lbeta)
    d_a = _mul(d_go, s)
    d_s = _mul(d_go, a)
    d_g_sig = _mul(_mul(d_s, _sub(one, s)), s)  # sigmoid_backward a * (1 - b) * b
    d_t1 = _mul(d_a, c_beta)
    if TANH_BWD_FMA:
        d_g1 = _mul(d_t1, _fma(-t1, t1, one))
    else:
        d_g1 = _mul(d_t1, _sub(one, _mul(t1, t1)))
    d_g = _add(_mul(d_g1, c_inv_beta), d_g_sig)
    tl.store(dx_ptr + base, d_g.to(tl.bfloat16), mask=mask)
    tl.store(dx_ptr + base + N, d_lin.to(tl.bfloat16), mask=mask)
    if WRITE_Y:
        tl.store(y_ptr + offs, _mul(go, lo).to(tl.bfloat16), mask=mask)


_BLOCK = 1024
TANH_BWD_FMA = True  # ATen tanh_backward a * (1 - b*b) as nvcc contracts it (B300 bench: fma 0 / plain 13682 dx diffs)


def _launch_fwd(x, probs):
    rows, two_n = x.shape
    n = two_n // 2
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=x.device)
    total = rows * n
    if total == 0:
        return out
    grid = (triton.cdiv(total, _BLOCK),)
    _situ_fwd_kernel[grid](
        x,
        probs if probs is not None else x,
        out,
        total,
        n,
        BETA,
        INV_BETA,
        LINEAR_BETA,
        INV_LINEAR_BETA,
        probs is not None,
        _BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


class _FusedSitu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.dim() == 2
        ctx.probs_shape = None if probs is None else probs.shape
        if probs is not None:
            probs = probs.reshape(-1).contiguous()
        ctx.save_for_backward(x, probs)
        return _launch_fwd(x, probs)

    @staticmethod
    def backward(ctx, dy):
        x, probs = ctx.saved_tensors
        dy = dy.contiguous()
        rows, two_n = x.shape
        n = two_n // 2
        dx = torch.empty_like(x)
        need_dp = probs is not None and ctx.needs_input_grad[1]
        y = torch.empty(rows, n, dtype=torch.bfloat16, device=x.device) if need_dp else dx
        total = rows * n
        if total:
            grid = (triton.cdiv(total, _BLOCK),)
            _situ_bwd_kernel[grid](
                x,
                probs if probs is not None else x,
                dy,
                dx,
                y,
                total,
                n,
                BETA,
                INV_BETA,
                LINEAR_BETA,
                INV_LINEAR_BETA,
                probs is not None,
                need_dp,
                TANH_BWD_FMA,
                _BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
        dp = None
        if need_dp:
            promoted = torch.result_type(y, probs)
            dp = torch.mul(dy.to(promoted), y).sum_to_size(rows, 1).reshape(ctx.probs_shape)
        return dx, dp


def fused_situ_and_mul(x: torch.Tensor, probs: torch.Tensor | None = None) -> torch.Tensor:
    """x [rows, 2N] bf16 -> [rows, N] bf16; with probs [rows] or [rows, 1], also scales by them."""
    return _FusedSitu.apply(x, probs)
