"""``transformer_engine.pytorch.triton.permutation.make_chunk_sort_map`` in a few torch ops instead of one Triton
program per token (``MILES_K31_CHUNK_SORT_MAP_TORCH=1``, default off).

Megatron's all-to-all dispatcher sorts the received tokens from [source rank, local expert] chunks into
[local expert, source rank] chunks with TE's fused ``moe_sort_chunks_by_index`` (``te_moe::chunk_sort_fwd``), which
first builds a row map with ``_make_chunk_sort_map_kernel``: every program (one per token row) loads all
``num_splits`` split sizes and sorted indices (``IDX_LOAD_WIDTH = next_power_of_2(num_splits)``, 1024 at EP16 x 56
local experts), takes their cumsum and an argmax, and writes one int32. At ~330k routed rows that is ~1.1 ms per call
(n1 trace, B300), four calls per MoE layer and microbatch under full recompute, for what is an O(num_splits) table
plus one gather per row. Here: the per-chunk destination offsets once ([num_splits]), each row's chunk by
``searchsorted`` over the input cumsum, then one gather; no host sync (no data-dependent shapes).

Same map, element for element, as the kernel (``common/triton/permutation.py``, TE 2.17):
  chunk(pid) = #{k : cumsum(split)[k] <= pid}; offset = pid - exclusive_cumsum(split)[chunk];
  out_pos = argmax(sorted_indices == chunk) (the first match; ``sorted_indices`` is a permutation in Megatron);
  dst = sum_{q < out_pos} split[sorted_indices[q]] + offset; rows pid >= sum(split) map to themselves.
Integer arithmetic only, so the sorted tensors are bitwise those of the Triton map.

Torch only: the CPU tests load it without TE.
"""

from __future__ import annotations

import logging
import os

import torch

CHUNK_SORT_MAP_ENV = "MILES_K31_CHUNK_SORT_MAP_TORCH"

logger = logging.getLogger(__name__)


def chunk_sort_map_enabled() -> bool:
    return os.environ.get(CHUNK_SORT_MAP_ENV, "0") == "1"


def make_chunk_sort_map_torch(
    split_sizes: torch.Tensor, sorted_indices: torch.Tensor, num_tokens: int, num_splits: int
) -> torch.Tensor:
    """[num_tokens] int32 destination row of every input row (the signature of TE's ``make_chunk_sort_map``)."""
    device = split_sizes.device
    sizes = split_sizes[:num_splits].to(torch.int64)
    order = sorted_indices[:num_splits].to(torch.int64)
    inclusive = torch.cumsum(sizes, 0)
    in_start = inclusive - sizes
    out_sizes = sizes[order]
    out_start = torch.cumsum(out_sizes, 0) - out_sizes  # [position in the sorted order]
    # position of each chunk in the sorted order: the first q with sorted_indices[q] == chunk (argmax of the match);
    # a chunk that never appears gets position 0, like argmax of an all-zero mask
    pos = torch.arange(num_splits, device=device, dtype=torch.int64)
    first = torch.full((num_splits,), num_splits, device=device, dtype=torch.int64)
    first = first.scatter_reduce(0, order, pos, reduce="amin", include_self=True)
    first = torch.where(first == num_splits, torch.zeros_like(first), first)
    dst_start = out_start[first]  # [chunk]
    rows = torch.arange(num_tokens, device=device, dtype=torch.int64)
    chunk = torch.searchsorted(inclusive, rows, right=True)  # #{k : inclusive[k] <= row}
    valid = chunk < num_splits
    c = chunk.clamp_max(num_splits - 1)
    dst = dst_start[c] + rows - in_start[c]
    return torch.where(valid, dst, rows).to(torch.int32)


_INSTALLED = False


def install_chunk_sort_map(module=None) -> bool:
    """Swap TE's ``make_chunk_sort_map`` for ``make_chunk_sort_map_torch`` (once; ``pytorch/permutation.py`` looks
    it up on the module at every call). Returns whether it is installed."""
    global _INSTALLED
    if module is None:
        if _INSTALLED:
            return True
        try:
            from transformer_engine.pytorch.triton import permutation as module
        except ImportError:
            logger.warning("%s=1 but TE's triton permutation module is not importable: unchanged", CHUNK_SORT_MAP_ENV)
            return False
    if getattr(module.make_chunk_sort_map, "_miles_torch", False):  # config-access-exempt: marker of our patch
        return True
    original = module.make_chunk_sort_map

    def make_chunk_sort_map(split_sizes, sorted_indices, num_tokens, num_splits):
        return make_chunk_sort_map_torch(split_sizes, sorted_indices, num_tokens, num_splits)

    make_chunk_sort_map._miles_torch = True
    make_chunk_sort_map._miles_original = original
    module.make_chunk_sort_map = make_chunk_sort_map
    _INSTALLED = True
    logger.info("%s=1: TE make_chunk_sort_map runs as torch ops", CHUNK_SORT_MAP_ENV)
    return True
