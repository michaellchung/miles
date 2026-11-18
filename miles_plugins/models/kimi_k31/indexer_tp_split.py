"""Split the frozen DSA indexer's query rows over the TP group instead of scoring all of them on every rank.

The indexer weights are plain replicated ``nn.Linear`` layers and its inputs (``q_lora`` and the
sequence-parallel gathered hidden states) are identical on every TP rank, so before this split all TP
ranks computed the same [T, topk] key indices. The sparse attention downstream is head-parallel over TP
and every rank needs all T rows of indices, so each rank scores only its rows and the group all-gathers.

Rows are dealt round-robin (rank r owns rows r, r + tp, r + 2 tp, ...): a query's cost grows with its
causal range, so interleaving keeps the ranks balanced whatever the packing or the CP zigzag order. Every
rank pads its rows to ceil(T / tp) with -1 so one equal-size all-gather suffices; the padding is sliced
off after the gather. Rows are independent (each has its own [ks, ke) range and the key indices are
absolute), so the merged result is exactly the single-call result.

This file imports only torch so the CPU tests can load it without Megatron or the DSA kernels.
"""

import os

import torch
import torch.distributed as dist


def indexer_tp_split_enabled() -> bool:
    return os.environ.get("MILES_K31_INDEXER_TP_SPLIT", "0") == "1"


# this process's decision, made on its first DSA layer call (a rank is in exactly one TP group)
_SPLIT_ON: bool | None = None


def check_switch_agrees(name: str, on: bool, group, device, what: str) -> None:
    """Raise on every rank of ``group`` unless all of them read switch ``name`` the same: a rank that runs a
    collective the others skip would hang it until the distributed timeout."""
    if dist.get_world_size(group=group) == 1:
        return
    flags = torch.tensor([int(on), -int(on)], dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    if int(flags[0]) != -int(flags[1]):
        raise RuntimeError(f"rank {dist.get_rank()}: {name} differs across its {what}; set it the same on every rank")


def indexer_tp_split_on(tp_group, device) -> bool:
    """Whether this TP group splits: decided once per process, on its first call. A rank that splits
    all-gathers and a rank that does not would hang that collective until the distributed timeout, so the
    TP ranks first check that they all read the same switch. Logs once per TP group when on."""
    global _SPLIT_ON
    if _SPLIT_ON is None:
        on = indexer_tp_split_enabled()
        tp_size = dist.get_world_size(group=tp_group)
        check_switch_agrees("MILES_K31_INDEXER_TP_SPLIT", on, tp_group, device, "TP group")
        _SPLIT_ON = on and tp_size > 1
        if _SPLIT_ON and dist.get_rank(group=tp_group) == 0:
            print(f"[k31] indexer TP split on tp={tp_size} (rank {dist.get_rank()})", flush=True)
    return _SPLIT_ON


def tp_rows(rows: int, tp_rank: int, tp_size: int) -> slice:
    """The query rows TP rank ``tp_rank`` scores: a strided slice of the ``rows`` packed rows."""
    return slice(tp_rank, rows, tp_size)


def pad_tp_part(part: torch.Tensor, rows: int, tp_size: int) -> torch.Tensor:
    """Pad one rank's [n_r, topk] indices to [ceil(rows / tp), topk] with -1 (the all-gather needs equal sizes)."""
    per_rank = -(-rows // tp_size)
    if part.shape[0] == per_rank:
        return part.contiguous()
    pad = part.new_full((per_rank - part.shape[0], *part.shape[1:]), -1)
    return torch.cat([part, pad])


def merge_tp_parts(gathered: torch.Tensor, rows: int, tp_size: int) -> torch.Tensor:
    """[tp * per_rank, topk] rank-major gather -> [rows, topk] in packed row order."""
    per_rank = gathered.shape[0] // tp_size
    return (
        gathered.view(tp_size, per_rank, *gathered.shape[1:]).transpose(0, 1).reshape(-1, *gathered.shape[1:])[:rows]
    )


def tp_split_part(select, index_q, weights, starts, ends, topk: int, tp_group) -> torch.Tensor:
    """``select(q, w, ks, ke) -> [n, topk]`` on this rank's rows, padded to [ceil(T / tp), topk] with -1."""
    tp_size = dist.get_world_size(group=tp_group)
    rows = index_q.shape[0]
    mine = tp_rows(rows, dist.get_rank(group=tp_group), tp_size)
    if index_q[mine].shape[0] > 0:
        part = select(
            index_q[mine].contiguous(), weights[mine].contiguous(), starts[mine].contiguous(), ends[mine].contiguous()
        )
    else:  # fewer rows than TP ranks
        part = torch.empty((0, topk), dtype=torch.int32, device=index_q.device)
    # every rank must gather the same dtype; the empty part above is int32 like every DSA top-k backend
    assert part.dtype == torch.int32 and part.shape[1:] == (topk,), (part.dtype, part.shape)
    return pad_tp_part(part, rows, tp_size)


def own_tp_part(indices: torch.Tensor, tp_group) -> torch.Tensor:
    """This rank's padded part of the whole [T, topk] indices: what ``tp_split_part`` would have scored."""
    tp_size = dist.get_world_size(group=tp_group)
    if tp_size == 1:
        return indices
    rows = indices.shape[0]
    return pad_tp_part(indices[tp_rows(rows, dist.get_rank(group=tp_group), tp_size)], rows, tp_size)


def gather_tp_parts(part: torch.Tensor, rows: int, tp_group) -> torch.Tensor:
    """Every rank's padded part, all-gathered over ``tp_group`` and merged back to [rows, topk] in row order."""
    tp_size = dist.get_world_size(group=tp_group)
    if tp_size == 1:
        return part
    gathered = part.new_empty((part.shape[0] * tp_size, *part.shape[1:]))
    dist.all_gather_into_tensor(gathered, part, group=tp_group)
    return merge_tp_parts(gathered, rows, tp_size)


def tp_split_topk(select, index_q, weights, starts, ends, topk: int, tp_group) -> torch.Tensor:
    """``select(q, w, ks, ke) -> [n, topk]`` on this rank's rows, all-gathered over ``tp_group`` to [T, topk]."""
    if dist.get_world_size(group=tp_group) == 1:
        return select(index_q, weights, starts, ends)
    part = tp_split_part(select, index_q, weights, starts, ends, topk, tp_group)
    return gather_tp_parts(part, index_q.shape[0], tp_group)
