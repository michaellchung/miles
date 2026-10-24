"""Host-sync-free ``_DeepepManager._indices_to_multihot`` for Megatron's flex (DeepEP) dispatcher.

After a DeepEP dispatch, Megatron turns the received ``[tokens, topk]`` local expert
indices (``-1`` = routed to another rank) into a ``[tokens, num_local_experts]``
routing map and probability matrix.  The upstream code (radixark/Megatron-LM
a84b105) does it with boolean-mask indexing (``indices[mask]``, ``probs[mask]``),
``repeat_interleave`` with tensor repeats and an index assignment of the Python
scalar ``1``.  Each of these makes the host wait for the GPU: four or five waits per
MoE layer per micro batch in the forward, one more in the backward of
``probs[mask]``.  On Nemotron-3 Nano 30B-A3B (23 MoE layers, 8 micro batches,
activation recompute) that is 2,024 of the 2,917 waits in one training step.

Megatron has a sync-free Triton kernel (``fused_indices_to_multihot``) but uses it
only under ``--enable-experimental``, which also turns on unrelated code paths.
The replacement here writes both outputs with ``scatter`` into one extra column
that collects the ``-1`` slots and is dropped afterwards:

* forward: every valid ``(row, index)`` gets ``True`` and its probability, exactly
  as the index assignments did (scatter copies values; no arithmetic);
* backward: the gradient of ``probs`` is ``grad.gather`` at the valid slots and
  0 at the ``-1`` slots (the dropped column has zero gradient), exactly what the
  backward of ``probs[mask]`` produced.

Differences only for inputs DeepEP does not produce: a row naming the same expert
twice (both versions then pick one write in an unspecified order), a valid index
equal to ``num_local_experts`` (upstream fails on the out-of-range index, the
replacement drops it with the ``-1`` slots) and negative indices other than ``-1``
(upstream wraps them, the replacement fails).

:func:`install` replaces the method only when the installed Megatron has the exact
a84b105 source of it (SHA-256 of the dedented source).  Any other source, a missing
class or unreadable source logs a WARNING and keeps Megatron's method.  Other
dtypes or shapes than DeepEP's (int64 ``[tokens, topk]`` indices, float32 probs)
call the original method.  ``MILES_HOST_SYNC_FREE=0`` (see :mod:`miles.utils.sync_free`)
skips the install.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch

from miles.utils import sync_free

logger = logging.getLogger(__name__)

# sha256 of textwrap.dedent(inspect.getsource(_DeepepManager._indices_to_multihot)) at
# radixark/Megatron-LM a84b10547 (megatron/core/transformer/moe/token_dispatcher.py:1401-1429).
KNOWN_SOURCE_SHA256 = frozenset({"214c4dc9b4a50e8e0635362bc9cf7a5963321c5a9c590907ef22af3ac3135672"})
_PATCH_MARK = "_miles_host_sync_free"


def indices_to_multihot(
    indices: torch.Tensor, probs: torch.Tensor, *, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(routing_map [tokens, E] bool, probs [tokens, E] float32)`` from ``[tokens, topk]`` indices.

    Same values as Megatron's ``_DeepepManager._indices_to_multihot`` for DeepEP
    output (see the module docstring); no host synchronization.
    """

    num_tokens = indices.shape[0]
    valid = indices != -1
    # -1 slots write into column num_local_experts, which is dropped below.
    columns = torch.where(valid, indices, num_local_experts)
    shape = (num_tokens, num_local_experts + 1)
    routing_map = torch.zeros(shape, dtype=torch.bool, device=indices.device).scatter_(1, columns, valid)
    multihot_probs = torch.zeros(shape, dtype=torch.float, device=indices.device).scatter(1, columns, probs)
    return (
        routing_map[:, :num_local_experts].contiguous(),
        multihot_probs[:, :num_local_experts].contiguous(),
    )


def _handles(indices: torch.Tensor, probs: torch.Tensor) -> bool:
    return (
        indices.dim() == 2
        and indices.dtype == torch.int64
        and probs.dtype == torch.float32
        and probs.shape == indices.shape
        and probs.device == indices.device
    )


def _make_method(original: Callable) -> Callable:
    @functools.wraps(original)
    def _indices_to_multihot(self, indices, probs):
        if not _handles(indices, probs):
            return original(self, indices, probs)
        return indices_to_multihot(indices, probs, num_local_experts=self.num_local_experts)

    setattr(_indices_to_multihot, _PATCH_MARK, True)
    return _indices_to_multihot


def patch_manager_class(manager_cls, *, known_sha256: frozenset[str] = KNOWN_SOURCE_SHA256) -> bool:
    """Replace ``manager_cls._indices_to_multihot`` when its source is a known one.

    Returns True when the class has the sync-free method after the call (patched
    now or before); False, after a WARNING, when the method is missing or unknown.
    """

    original = getattr(manager_cls, "_indices_to_multihot", None)
    if original is None:
        logger.warning("%s has no _indices_to_multihot; keeping Megatron's DeepEP dispatcher as is",
                       manager_cls.__qualname__)
        return False
    if getattr(original, _PATCH_MARK, False):
        return True
    digest = sync_free.source_sha256(original)
    if digest not in known_sha256:
        logger.warning(
            "%s._indices_to_multihot does not match the Megatron source this patch was written for "
            "(sha256 %s); keeping Megatron's version, which makes the host wait for the GPU in every MoE layer",
            manager_cls.__qualname__,
            digest or "unavailable",
        )
        return False
    manager_cls._indices_to_multihot = _make_method(original)
    return True


def install() -> bool:
    """Patch Megatron's ``_DeepepManager`` once per process; False when not patched."""

    if not sync_free.enabled():
        logger.info("%s=0: keeping Megatron's _indices_to_multihot", sync_free.ENV_VAR)
        return False
    try:
        from megatron.core.transformer.moe.token_dispatcher import _DeepepManager
    except ImportError as exc:
        logger.warning("Megatron's _DeepepManager is not importable (%s); multihot patch not installed", exc)
        return False
    patched = patch_manager_class(_DeepepManager)
    if patched:
        logger.info("Megatron _DeepepManager._indices_to_multihot: host-sync-free version installed")
    return patched
