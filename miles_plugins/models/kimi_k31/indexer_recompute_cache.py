"""Reuse the frozen DSA indexer's top-k in the full-recompute pass instead of scoring it again.

Under ``--recompute-granularity full`` every DSA layer runs its indexer twice per training microbatch: in the
forward (inside Megatron's ``CheckpointFunction.forward``, without grad) and again when the backward recomputes
the layer. The indexer is frozen and its inputs are detached, so the recompute scores exactly the keys the forward
already chose, and the indexer is the most expensive part of a DSA layer (A0: 141 TF per layer and pass per GPU at
262k; A1: 18 TF). With ``MILES_K31_INDEXER_RECOMPUTE_CACHE=1`` the forward keeps its top-k and the recompute takes
it back instead of running the indexer projections, the CP all-gather of the index keys and the scoring.

What is kept: this TP rank's rows of the [T_cp, topk] int32 indices (rows r, r + tp, ...; the
``indexer_tp_split`` part), padded to ceil(T_cp / tp) rows: 128 MiB per DSA layer and microbatch at T_cp 131072,
TP8, topk 2048. The recompute all-gathers the parts over TP, as the A1 split does, so its result is the same bytes.

When (``recompute_phase``, from Megatron's ``is_checkpointing()`` and the grad mode):

* store: inside ``CheckpointFunction.forward`` (checkpointing, no grad), the first forward of a recomputed layer.
  Megatron's ``checkpoint`` only enters it with grad enabled, so its backward follows.
* reuse: inside ``CheckpointFunction.backward`` (checkpointing, grad), the recompute.
* plain: everything else. A forward-only pass (log-probs, under ``torch.no_grad``) never checkpoints
  (Megatron eb547f33 "Skip activation checkpointing when grad is disabled"), and neither does a layer that is not
  recomputed, so neither stores anything.

Pairing: each layer keeps its own FIFO, keyed by the identity of the microbatch's ``cu_seqlens_q`` tensor, which the
checkpointed forward and its recompute share (the recompute reruns the same closure on the same
``PackedSeqParams``). The recompute takes the oldest entry with its key; older entries that do not match belong to a
forward whose backward never ran (an aborted step) and are dropped. A layer never holds more entries than microbatches
in flight under 1F1B (the pipeline size); a store past that drops the oldest. A recompute that finds nothing scores
the indexer as before, so a miss costs time, never correctness.

Every TP and CP rank of a layer makes the same store/reuse/miss decision (same schedule, same layers), and the
switch is checked across both groups on the first call: a rank that skipped the CP all-gather the others run would
hang it.

Default off. Torch-only so the CPU tests load it without Megatron or the DSA kernels.
"""

from __future__ import annotations

import enum
import logging
import os
from collections import deque

import torch
import torch.distributed as dist

from miles_plugins.models.kimi_k31.indexer_tp_split import check_switch_agrees

CACHE_ENV = "MILES_K31_INDEXER_RECOMPUTE_CACHE"

logger = logging.getLogger(__name__)


def recompute_cache_enabled() -> bool:
    return os.environ.get(CACHE_ENV, "0") == "1"


class Phase(enum.Enum):
    PLAIN = "plain"
    STORE = "store"
    REUSE = "reuse"


def recompute_phase(checkpointing: bool, grad_enabled: bool) -> Phase:
    """Which pass of a fully recomputed layer runs: its first forward (store), its recompute (reuse), or neither."""
    if not checkpointing:
        return Phase.PLAIN
    return Phase.REUSE if grad_enabled else Phase.STORE


class TopkRecomputeCache:
    """One DSA layer's top-k parts from forwards whose recompute has not run yet, oldest first."""

    def __init__(self, max_pending: int):
        assert max_pending >= 1, max_pending
        self.max_pending = max_pending
        self._pending: deque[tuple[torch.Tensor, torch.Tensor]] = deque()

    def __len__(self) -> int:
        return len(self._pending)

    def put(self, key: torch.Tensor, part: torch.Tensor) -> None:
        if len(self._pending) >= self.max_pending:
            self._pending.popleft()
            logger.warning(
                "%s: more than %d DSA top-k results pending on one layer; dropped the oldest (its recompute will "
                "score the indexer again)",
                CACHE_ENV,
                self.max_pending,
            )
        self._pending.append((key, part))

    def take(self, key: torch.Tensor) -> torch.Tensor | None:
        """The part stored under ``key`` (the same tensor object), removing it and every older entry."""
        for index, (stored, part) in enumerate(self._pending):
            if stored is key:
                for _ in range(index + 1):
                    self._pending.popleft()
                if index:
                    logger.warning("%s: dropped %d DSA top-k results whose recompute never ran", CACHE_ENV, index)
                return part
        return None

    def clear(self) -> None:
        self._pending.clear()


# this process's decision, made on its first DSA layer call
_CACHE_ON: bool | None = None


def indexer_recompute_cache_on(tp_group, cp_group, device) -> bool:
    """Whether the DSA layers cache their top-k for the recompute: read once per process, on the first call, and
    checked across the TP and the CP group (both run collectives the cache skips). Logs once when on."""
    global _CACHE_ON
    if _CACHE_ON is None:
        on = recompute_cache_enabled()
        check_switch_agrees(CACHE_ENV, on, tp_group, device, "TP group")
        check_switch_agrees(CACHE_ENV, on, cp_group, device, "CP group")
        _CACHE_ON = on
        if on and dist.get_rank() == 0:
            print(f"[k31] indexer top-k recompute cache on (rank {dist.get_rank()})", flush=True)
    return _CACHE_ON
