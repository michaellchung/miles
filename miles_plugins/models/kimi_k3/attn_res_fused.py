"""Fused Triton attention-residual mix for Kimi K3 (``MILES_K3_ATTN_RES_FUSED=1``).

Same math as ``ops._attn_res_mix`` for one token with rows ``x_j`` (the bank rows, then ``prefix_sum``)::

    r_j = rsqrt(mean_h(x_j^2) + eps)      s_j = sum_h (x_j * r_j) * w      p = softmax_j(s)
    out = sum_j p_j * x_j                                                   (fp32, cast to the row dtype)

Eager materializes ``[T, rows, H]`` fp32 temporaries in ~11 kernels per pass; here every pass reads the bf16
rows directly and keeps the per-row statistics in registers:

* forward: one program per token, two sweeps over H (stats, then the weighted sum; the second sweep re-reads
  the rows the program has just read).
* backward: ``_bwd_stats`` (one program per token) recomputes the stats with ``dp_j = sum_h g * x_j`` and writes
  the per-row coefficients (4 fp32 per row); ``_bwd_grads`` (token-block x H-block grid) writes the row grads
  and per-token-block partial weight grads, summed with one ``torch.sum`` (deterministic, no atomics).

Nothing but the inputs is saved for backward, so the token-block checkpoint of the eager path is not needed.
All accumulation is fp32 like eager; reductions over H run in another order, so results match eager to fp32
rounding (not bitwise).
"""

import torch
import triton
import triton.language as tl

# fp32 lanes of one [ROWS_PAD, BLOCK_H] tile; bounds registers per program
_TILE_ELEMS = 8192
_NUM_WARPS = 8
_GRAD_TOKENS_PER_PROGRAM = 32


@triton.jit
def _load_rows(prefix_ptr, bank_ptr, t, stride_pt, stride_bt, stride_br, cols, cmask, rows, R: tl.constexpr):
    """fp32 [ROWS_PAD, BLOCK_H] tile of token t: rows < R from the bank, row R = prefix_sum, others 0."""
    is_bank = rows < R
    bank_row = tl.where(is_bank, rows, 0).to(tl.int64)
    x = tl.load(
        bank_ptr + t * stride_bt + bank_row[:, None] * stride_br + cols[None, :],
        mask=is_bank[:, None] & cmask[None, :],
        other=0.0,
    ).to(tl.float32)
    xp = tl.load(prefix_ptr + t * stride_pt + cols, mask=cmask, other=0.0).to(tl.float32)
    return tl.where((rows == R)[:, None], xp[None, :], x)


@triton.jit
def _cast(x, dtype: tl.constexpr):
    """fp32 -> dtype with round-to-nearest-even like torch's .to().

    bf16 is rounded on the fp32 bits first: same result as the GPU's cvt.rn, and the CPU interpreter (which
    truncates in its plain fp32 -> bf16 cast) then matches it too.
    """
    if dtype == tl.float32:
        return x
    elif dtype == tl.bfloat16:
        u = x.to(tl.uint32, bitcast=True)
        u = (u + (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000
        return tl.where(x != x, x, u.to(tl.float32, bitcast=True)).to(dtype)
    else:
        return x.to(dtype)


@triton.jit
def _row_stats(sumsq, dot, valid, eps, inv_h):
    """rsqrt, softmax probabilities from the per-row sum of squares and dot with the score weight."""
    r = tl.math.rsqrt(sumsq * inv_h + eps)
    s = tl.where(valid, r * dot, float("-inf"))
    e = tl.exp(s - tl.max(s, axis=0))
    e = tl.where(valid, e, 0.0)
    return r, e / tl.sum(e, axis=0)


@triton.jit
def _attn_res_fwd_kernel(
    prefix_ptr,
    bank_ptr,
    w_ptr,
    out_ptr,
    H,
    stride_pt,
    stride_bt,
    stride_br,
    stride_ot,
    eps,
    inv_h,
    R: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, ROWS_PAD)
    valid = rows <= R
    sumsq = tl.zeros([ROWS_PAD], dtype=tl.float32)
    dot = tl.zeros([ROWS_PAD], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        cmask = cols < H
        x = _load_rows(prefix_ptr, bank_ptr, t, stride_pt, stride_bt, stride_br, cols, cmask, rows, R)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
        sumsq += tl.sum(x * x, axis=1)
        dot += tl.sum(x * w[None, :], axis=1)
    _, p = _row_stats(sumsq, dot, valid, eps, inv_h)
    for h0 in range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        cmask = cols < H
        x = _load_rows(prefix_ptr, bank_ptr, t, stride_pt, stride_bt, stride_br, cols, cmask, rows, R)
        out = tl.sum(p[:, None] * x, axis=0)
        tl.store(out_ptr + t * stride_ot + cols, _cast(out, out_ptr.dtype.element_ty), mask=cmask)


@triton.jit
def _attn_res_bwd_stats_kernel(
    prefix_ptr,
    bank_ptr,
    w_ptr,
    go_ptr,
    coef_ptr,
    H,
    stride_pt,
    stride_bt,
    stride_br,
    stride_gt,
    eps,
    inv_h,
    R: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """coef[t] = [p, ds, r, c] per row: ds = softmax grad of the scores, c = d(loss)/d(mean x^2) / H."""
    t = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, ROWS_PAD)
    valid = rows <= R
    sumsq = tl.zeros([ROWS_PAD], dtype=tl.float32)
    dot = tl.zeros([ROWS_PAD], dtype=tl.float32)
    dp = tl.zeros([ROWS_PAD], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        cmask = cols < H
        x = _load_rows(prefix_ptr, bank_ptr, t, stride_pt, stride_bt, stride_br, cols, cmask, rows, R)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
        g = tl.load(go_ptr + t * stride_gt + cols, mask=cmask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=1)
        dot += tl.sum(x * w[None, :], axis=1)
        dp += tl.sum(x * g[None, :], axis=1)
    r, p = _row_stats(sumsq, dot, valid, eps, inv_h)
    ds = p * (dp - tl.sum(p * dp, axis=0))
    # d(loss)/d r_j = sum_h ds_j w_h x_jh = ds_j * dot_j; rsqrt' = -0.5 r^3; mean' = 1 / H
    c = (-0.5 * (ds * dot)) * (r * r * r) * inv_h
    base = coef_ptr + t * (4 * ROWS_PAD) + rows
    tl.store(base, p)
    tl.store(base + ROWS_PAD, ds)
    tl.store(base + 2 * ROWS_PAD, r)
    tl.store(base + 3 * ROWS_PAD, c)


@triton.jit
def _attn_res_bwd_grads_kernel(
    prefix_ptr,
    bank_ptr,
    w_ptr,
    go_ptr,
    coef_ptr,
    gprefix_ptr,
    gbank_ptr,
    gw_partial_ptr,
    T,
    H,
    stride_pt,
    stride_bt,
    stride_br,
    stride_gt,
    R: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKENS: tl.constexpr,
):
    """d x_j = g p_j + (ds_j w) r_j + c_j (2 x_j);  partial d w = sum over this token block of ds_j (x_j r_j)."""
    tb = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < H
    rows = tl.arange(0, ROWS_PAD)
    is_bank = rows < R
    bank_row = tl.where(is_bank, rows, 0).to(tl.int64)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
    gw = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in range(TOKENS):
        t = tb.to(tl.int64) * TOKENS + i
        if t < T:
            x = _load_rows(prefix_ptr, bank_ptr, t, stride_pt, stride_bt, stride_br, cols, cmask, rows, R)
            g = tl.load(go_ptr + t * stride_gt + cols, mask=cmask, other=0.0).to(tl.float32)
            base = coef_ptr + t * (4 * ROWS_PAD) + rows
            p = tl.load(base)
            ds = tl.load(base + ROWS_PAD)
            r = tl.load(base + 2 * ROWS_PAD)
            c = tl.load(base + 3 * ROWS_PAD)
            dx = g[None, :] * p[:, None] + (ds[:, None] * w[None, :]) * r[:, None] + c[:, None] * (2.0 * x)
            gw += tl.sum(ds[:, None] * (x * r[:, None]), axis=0)
            tl.store(
                gbank_ptr + t * stride_bt + bank_row[:, None] * stride_br + cols[None, :],
                _cast(dx, gbank_ptr.dtype.element_ty),
                mask=is_bank[:, None] & cmask[None, :],
            )
            dxp = tl.sum(tl.where((rows == R)[:, None], dx, 0.0), axis=0)
            tl.store(gprefix_ptr + t * stride_pt + cols, _cast(dxp, gprefix_ptr.dtype.element_ty), mask=cmask)
    tl.store(gw_partial_ptr + tb.to(tl.int64) * H + cols, gw, mask=cmask)


def _config(rows: int, hidden: int) -> tuple[int, int]:
    rows_pad = triton.next_power_of_2(rows)
    block_h = min(triton.next_power_of_2(hidden), max(32, _TILE_ELEMS // rows_pad))
    return rows_pad, block_h


def _flat(prefix_sum: torch.Tensor, block_residual: torch.Tensor):
    hidden = prefix_sum.shape[-1]
    prefix = prefix_sum.contiguous().view(-1, hidden)
    bank = block_residual.contiguous().view(prefix.shape[0], block_residual.shape[-2], hidden)
    return prefix, bank


class AttnResMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix_sum, block_residual, score_weight, eps):
        prefix, bank = _flat(prefix_sum, block_residual)
        tokens, nbank, hidden = bank.shape
        w = score_weight.contiguous().float()
        rows_pad, block_h = _config(nbank + 1, hidden)
        out = torch.empty_like(prefix)
        if tokens > 0:
            _attn_res_fwd_kernel[(tokens,)](
                prefix,
                bank,
                w,
                out,
                hidden,
                prefix.stride(0),
                bank.stride(0),
                bank.stride(1),
                out.stride(0),
                float(eps),
                1.0 / hidden,
                R=nbank,
                ROWS_PAD=rows_pad,
                BLOCK_H=block_h,
                num_warps=_NUM_WARPS,
            )
        ctx.save_for_backward(prefix_sum, block_residual, score_weight)
        ctx.eps = float(eps)
        return out.view(prefix_sum.shape)

    @staticmethod
    def backward(ctx, grad_out):
        prefix_sum, block_residual, score_weight = ctx.saved_tensors
        prefix, bank = _flat(prefix_sum, block_residual)
        tokens, nbank, hidden = bank.shape
        w = score_weight.contiguous().float()
        go = grad_out.contiguous().view(-1, hidden)
        rows_pad, block_h = _config(nbank + 1, hidden)
        gprefix = torch.empty_like(prefix)
        gbank = torch.empty_like(bank)
        n_tb = triton.cdiv(tokens, _GRAD_TOKENS_PER_PROGRAM)
        gw_partial = torch.empty(n_tb, hidden, dtype=torch.float32, device=prefix.device)
        if tokens > 0:
            coef = torch.empty(tokens, 4, rows_pad, dtype=torch.float32, device=prefix.device)
            _attn_res_bwd_stats_kernel[(tokens,)](
                prefix,
                bank,
                w,
                go,
                coef,
                hidden,
                prefix.stride(0),
                bank.stride(0),
                bank.stride(1),
                go.stride(0),
                ctx.eps,
                1.0 / hidden,
                R=nbank,
                ROWS_PAD=rows_pad,
                BLOCK_H=block_h,
                num_warps=_NUM_WARPS,
            )
            _attn_res_bwd_grads_kernel[(n_tb, triton.cdiv(hidden, block_h))](
                prefix,
                bank,
                w,
                go,
                coef,
                gprefix,
                gbank,
                gw_partial,
                tokens,
                hidden,
                prefix.stride(0),
                bank.stride(0),
                bank.stride(1),
                go.stride(0),
                R=nbank,
                ROWS_PAD=rows_pad,
                BLOCK_H=block_h,
                TOKENS=_GRAD_TOKENS_PER_PROGRAM,
                num_warps=_NUM_WARPS,
            )
        gw = gw_partial.sum(dim=0).to(score_weight.dtype)
        return gprefix.view(prefix_sum.shape), gbank.view(block_residual.shape), gw, None


def attn_res_mix_fused(prefix_sum, block_residual, score_weight, eps):
    return AttnResMixFn.apply(prefix_sum, block_residual, score_weight, eps)
