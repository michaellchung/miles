"""Run the K3.1 DSA sparse attention core split over query rows instead of over heads within the TP group.

Why: K3.1 has 96 MLA heads, so TP8 leaves 12 heads per rank, while the sparse attention kernels tile heads in
blocks of 64 (FlashMLA's sparse prefill takes 64 or 128 heads; cuDNN's SM100 DSA backward pads to a 64-head tile
except for exactly 16 heads at d 576). Every rank therefore runs its 12 heads padded to 64, 5.3x the useful
FLOPs, and gathers each query's 2048 top-k latent rows for only 12 heads. The n1 trace (L16, B300) shows it:
31.4 ms per forward call and 143 ms per backward call per DSA layer and microbatch, plus about 27 ms of padding
copies, about 236 ms per DSA layer and microbatch in all.

The latent KV is MQA (one 576-wide group shared by every head) and is already whole on every TP rank, and the
top-k is per query row, shared by all heads. So the attention can just as well run on 1/TP of the query rows with
all 96 heads: an all-to-all over TP turns the head-split [T, H/tp, d] queries into row-split [ceil(T/tp), H, d]
queries, the kernel runs with 96 heads (padded to 128: 1.33x instead of 5.3x), and a second all-to-all brings the
[rows, H, d_v] latent output back to [T, H/tp, d_v]. The autograd backward of each all-to-all is the other one.

Rows are dealt round-robin like the indexer TP split (``indexer_tp_split.tp_rows``: rank r owns rows r, r + tp,
...), so rank r needs exactly its own top-k part; with ``MILES_K31_INDEXER_TP_SPLIT=1`` the all-gather of the top-k
indices disappears too. T is padded up to a multiple of tp: the padded query rows are zero, their single top-k entry
is key 0 (a row with no valid key is a special case in both kernels; this one is an ordinary row), their output is
dropped and their output gradient is zero, so they add nothing to dq (dropped) or dkv (P * 0).

Numerics: every (query, head) row of the forward and of dq is computed from the same q row, the same kv rows and
the same top-k, so it does not depend on which rank runs it; whether the kernel instance gives the same bits for a
row when its 64-row head tile holds 64 real heads instead of 12 real + 52 zero heads is for the GPU test to show
(``MILES_K31_DSA_QUERY_SPLIT=64`` runs the 96 heads as head groups of 64, the forward kernel instance the head
split uses; ``=1`` lets the dispatcher pad 96 to 128 in one call). dkv is a sum over rows and heads; the head
split sums 12 heads x T rows per rank, this sums 96 heads x T/tp rows per rank, and the TP all-reduce of the kv
gradient (``copy_to_tensor_model_parallel_region``) sums the ranks either way, so dkv is equal up to fp32 summation
order (cuDNN accumulates dkv in fp32 with atomics, so it is not bitwise reproducible run to run to begin with).

Torch only (the attention function is passed in), so the CPU tests load it without Megatron or the kernels.
"""

import os

import torch
import torch.distributed as dist

QUERY_SPLIT_ENV = "MILES_K31_DSA_QUERY_SPLIT"

_ON: str | None = None


def query_split_mode() -> str:
    """'' (off, default), '1' (one call over all heads) or '64' (head groups of 64)."""
    value = os.environ.get(QUERY_SPLIT_ENV, "0")
    if value in ("", "0"):
        return ""
    if value not in ("1", "64"):
        raise ValueError(f"{QUERY_SPLIT_ENV}={value!r}: expected 0, 1 or 64")
    return value


def query_split_on(tp_group, device) -> str:
    """This process's mode, decided on its first DSA call; every TP rank must read the same switch (the all-to-all
    of a rank that splits would wait for a rank that does not)."""
    global _ON
    if _ON is None:
        from miles_plugins.models.kimi_k31.indexer_tp_split import check_switch_agrees

        mode = query_split_mode()
        tp = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
        check_switch_agrees(QUERY_SPLIT_ENV, bool(mode), tp_group, device, "TP group")
        _ON = mode if tp > 1 else ""
        if _ON and dist.get_rank(group=tp_group) == 0:
            print(f"[k31] DSA sparse attention split over query rows on tp={tp} (mode {_ON})", flush=True)
    return _ON


def _rows_per_rank(rows: int, tp: int) -> int:
    return -(-rows // tp)


def _a2a(x: torch.Tensor, group) -> torch.Tensor:
    """all_to_all_single over dim 0 of a [tp, ...] contiguous tensor."""
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out


def heads_to_rows(x: torch.Tensor, group) -> torch.Tensor:
    """[T, h, d] (this rank's h heads of every row) -> [ceil(T/tp), tp*h, d] (every head of this rank's rows:
    row p*tp + r on rank r, heads in global order since column-parallel rank s holds heads [s*h, (s+1)*h))."""
    tp = dist.get_world_size(group=group)
    rows, h, d = x.shape
    per = _rows_per_rank(rows, tp)
    if per * tp != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, per * tp - rows))
    send = x.view(per, tp, h, d).transpose(0, 1).contiguous()  # [dest rank, p, h, d]
    recv = _a2a(send, group)  # [src rank (head block), p, h, d]
    return recv.permute(1, 0, 2, 3).reshape(per, tp * h, d)


def rows_to_heads(y: torch.Tensor, rows: int, group) -> torch.Tensor:
    """The inverse of ``heads_to_rows``: [ceil(T/tp), tp*h, d] -> [T, h, d]."""
    tp = dist.get_world_size(group=group)
    per, heads, d = y.shape
    h = heads // tp
    send = y.view(per, tp, h, d).transpose(0, 1).contiguous()  # [dest rank (head block), p, h, d]
    recv = _a2a(send, group)  # [src rank (row owner r), p, h, d]
    return recv.permute(1, 0, 2, 3).reshape(per * tp, h, d)[:rows]


class _HeadsToRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group, ctx.rows = group, x.shape[0]
        return heads_to_rows(x, group)

    @staticmethod
    def backward(ctx, grad):
        return rows_to_heads(grad.contiguous(), ctx.rows, ctx.group), None


class _RowsToHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, rows, group):
        ctx.group = group
        return rows_to_heads(y, rows, group)

    @staticmethod
    def backward(ctx, grad):
        return heads_to_rows(grad.contiguous(), ctx.group), None, None


def own_rows_valid(part: torch.Tensor, rows: int, group) -> torch.Tensor:
    """This rank's padded [ceil(T/tp), topk] top-k part with the padding rows (index >= T) pointed at key 0, so no
    row of the kernel call is empty."""
    tp, r = dist.get_world_size(group=group), dist.get_rank(group=group)
    per = _rows_per_rank(rows, tp)
    assert part.shape[0] == per, (part.shape, rows, tp)
    n_real = len(range(r, rows, tp))
    if n_real == per:
        return part
    part = part.clone()
    part[n_real:] = -1
    part[n_real:, 0] = 0
    return part


def query_split_attention(
    attn_fn, query, kv, part, rows: int, group, mode: str = "1", head_group: int = 64, **kw
) -> torch.Tensor:
    """``attn_fn(q [1, n, H, d], kv [1, S, 1, d], idx [1, n, 1, topk], **kw) -> [1, n, H, d_v]`` (the
    ``sparse_attention`` signature) run on this rank's rows with all heads; ``query`` is [T, h, d] head-split,
    ``part`` this rank's padded top-k rows. Returns the [T, h, d_v] head-split output. Mode '64' calls
    ``attn_fn`` once per ``head_group`` heads (the kernels' 64-head tile)."""
    q_rows = _HeadsToRows.apply(query, group)
    idx = own_rows_valid(part, rows, group).unsqueeze(1).unsqueeze(0)
    kv4 = kv.unsqueeze(0).unsqueeze(2)
    if mode == "64" and q_rows.shape[1] > head_group:
        outs = [
            attn_fn(q_rows[:, g : g + head_group].unsqueeze(0), kv4, idx, **kw).squeeze(0)
            for g in range(0, q_rows.shape[1], head_group)
        ]
        out_rows = torch.cat(outs, dim=1)
    else:
        out_rows = attn_fn(q_rows.unsqueeze(0), kv4, idx, **kw).squeeze(0)
    return _RowsToHeads.apply(out_rows, rows, group)
