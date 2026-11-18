"""Frozen MXFP4 experts: save the packed bytes, not the dequantized BF16 weights, for the backward.

``MXFP4*GroupedLinear._get_weight_tensors()`` dequantizes every local frozen expert into a fresh BF16 tensor
for each GEMM, and TE's ``_GroupedLinear.forward`` (release_v2.17, ``pytorch/module/grouped_linear.py``) hands
exactly those tensors to ``ctx.save_for_backward`` for the dgrad GEMM: ``cast_if_needed`` returns its input when
the dtype already matches (legacy path ``weights_fp8`` at :574, grouped-tensor path ``weights_for_gemm`` at :204
and :337), and ``prepare_for_saving`` passes plain tensors through (``quantized_tensor.py:136``). The input rows
are not saved (no wgrad: ``weights[0].requires_grad`` is False). So without recompute every MoE layer keeps its
frozen experts in BF16 from the forward until its backward: ``n_frozen_local x 33.0M x 2 B``, 2.95 GiB per layer
per GPU at EP16 with 128 trainable experts (48 frozen local experts), 6.4 GiB at EP8 with 64.

With ``MILES_K31_MXFP4_REDEQUANT_BACKWARD=1`` the module forward runs under
``torch.autograd.graph.saved_tensors_hooks``: a saved tensor that is one of this call's dequantized weights is
packed as a reference to its uint8 parameter (and the parameter's version counter); every other saved tensor
passes through untouched. The unpack hook dequantizes the parameter again. ``dequantize_mxfp4_rows`` is exact and
deterministic, so the backward GEMM reads bitwise the same weights and every gradient is unchanged. The BF16
copy then lives only around the forward GEMM and the backward GEMM of one projection; the cost is one more
dequantization per projection in the backward (elementwise over the 0.53 B/param bytes).

Under ``--recompute-granularity full`` the first forward runs without grad (nothing is saved, the hooks are not
installed) and the backward recompute dequantizes once more, so a recomputed layer dequantizes three times
instead of twice; its peak still drops by the experts the backward is not currently using.

Every weight fetch dequantizes with the fp32 chain of ``dequantize_mxfp4`` (about 77 B of memory traffic per
weight: 122 GB per MoE layer and fetch at EP16, two fetches per layer under full recompute, three with this switch);
``MILES_K31_MXFP4_LUT_DEQUANT=1`` uses ``dequantize_mxfp4_lut`` instead, the same bits through a byte table at
about 9 B per weight.

``MILES_K31_MXFP4_BATCHED_DEQUANT=1`` dequantizes all the experts of one fetch together
(``dequantize_mxfp4_rows_batched``: stacked into chunks of ``MILES_K31_MXFP4_BATCHED_DEQUANT_CHUNK`` weights, default
2**27) instead of one expert at a time, in the forward fetch and, with the redequant switch, in the backward unpack
(the first unpacked expert of a forward dequantizes the whole group, the rest take their slices). Same bits. Why: the
48 frozen local experts x 2 projections make 96 dequantizations per MoE layer and pass, 7 (LUT) to 19 (fp32) small
kernels each, and the a1d trace shows the host launching them as the bottleneck: ~0.2-0.66 ms of CPU per expert
against tens of microseconds of GPU, 12 ms of GPU idle per MoE layer and pass (24 per layer and microbatch).

``MILES_K31_MXFP4_TRITON_DEQUANT=1`` (wins over the batched and LUT switches) dequantizes all the same-shape experts of
a fetch in one Triton kernel (``miles.utils.mxfp4_triton``: 0.53 B read + 2 B written per weight, the same bits; the
n1 trace measured the LUT path at 9.2 ms per MoE layer and fetch, ~0.6 ms is the HBM bound). It goes through
``dequantize_expert_rows`` like the batched path, so the forward fetch, the a2a-overlap prefetch and the redequant
backward's group unpack all use it. CUDA only (or ``TRITON_INTERPRET=1``); elsewhere the batched/per-expert path.

Default off: the module forward is TE's, unchanged. Torch-only (plus ``miles.utils.mxfp4``) so the CPU tests load
it without Megatron or TE.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator, Sequence

import torch

from miles.utils.mxfp4 import (
    BATCHED_DEQUANT_CHUNK_ELEMENTS,
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4_rows,
    dequantize_mxfp4_rows_batched,
)
from miles.utils.mxfp4_triton import dequantize_mxfp4_rows_triton, triton_dequant_available

REDEQUANT_ENV = "MILES_K31_MXFP4_REDEQUANT_BACKWARD"
# dequantize through the byte table (miles.utils.mxfp4.dequantize_mxfp4_lut): the same bits, about 1/8 of the traffic
LUT_DEQUANT_ENV = "MILES_K31_MXFP4_LUT_DEQUANT"
# dequantize all the experts of a fetch in a few stacked chunks (dequantize_mxfp4_rows_batched): the same bits
BATCHED_DEQUANT_ENV = "MILES_K31_MXFP4_BATCHED_DEQUANT"
BATCHED_DEQUANT_CHUNK_ENV = "MILES_K31_MXFP4_BATCHED_DEQUANT_CHUNK"
# one Triton kernel per fetch (miles.utils.mxfp4_triton): the same bits, 2.53 B per weight, no per-expert launches
TRITON_DEQUANT_ENV = "MILES_K31_MXFP4_TRITON_DEQUANT"

logger = logging.getLogger(__name__)
_reported = False


def redequant_enabled() -> bool:
    return os.environ.get(REDEQUANT_ENV, "0") == "1"


def lut_dequant_enabled() -> bool:
    return os.environ.get(LUT_DEQUANT_ENV, "0") == "1"


def batched_dequant_enabled() -> bool:
    return os.environ.get(BATCHED_DEQUANT_ENV, "0") == "1"


def triton_dequant_enabled() -> bool:
    return os.environ.get(TRITON_DEQUANT_ENV, "0") == "1"


def _triton_path(rows: Sequence[torch.Tensor]) -> bool:
    return triton_dequant_enabled() and len(rows) > 0 and triton_dequant_available(rows[0].device)


def dequant_mode() -> tuple[bool, bool, bool]:
    """(LUT, batched, Triton) switches: keys the prefetch's freshness check (every mode gives the same bits)."""
    return lut_dequant_enabled(), batched_dequant_enabled(), triton_dequant_enabled()


def dequantize_expert_rows(rows: Sequence[torch.Tensor], group_size: int = MXFP4_GROUP_SIZE) -> list[torch.Tensor]:
    """The exact BF16 weights of a grouped linear's packed experts, per the Triton, batched and LUT switches."""
    if _triton_path(rows):
        return list(dequantize_mxfp4_rows_triton(list(rows), group_size))
    lut = lut_dequant_enabled()
    if batched_dequant_enabled():
        chunk = int(os.environ.get(BATCHED_DEQUANT_CHUNK_ENV, BATCHED_DEQUANT_CHUNK_ELEMENTS))
        return dequantize_mxfp4_rows_batched(list(rows), group_size, lut=lut, max_elements=chunk)
    return [dequantize_mxfp4_rows(packed, group_size, lut=lut) for packed in rows]


class _PackedGroup:
    """The packed experts of one forward fetch: the first unpack in the backward dequantizes them all together."""

    __slots__ = ("rows", "group_size", "_weights")

    def __init__(self, rows: Sequence[torch.Tensor], group_size: int):
        self.rows = list(rows)
        self.group_size = group_size
        self._weights: list[torch.Tensor | None] | None = None

    def prefetch(self) -> None:
        """Dequantize the group now (``moe_a2a_overlap``: inside the combine-backward all-to-all's window); the
        backward's unpacks then take these weights. A no-op once any of them is dequantized."""
        if self._weights is None:
            self._weights = dequantize_expert_rows(self.rows, self.group_size)

    def take(self, index: int) -> torch.Tensor:
        if self._weights is None or self._weights[index] is None:
            # first unpack (or a second read of the saved tensors): dequantize the whole group again
            self._weights = dequantize_expert_rows(self.rows, self.group_size)
        weight, self._weights[index] = self._weights[index], None
        if all(w is None for w in self._weights):
            self._weights = None
        return weight


_FetchGroup = _PackedGroup  # the name the Triton switch tests use


def _key(tensor: torch.Tensor) -> tuple:
    # identity of a dequantized weight while its forward runs: a fresh allocation, so (address, geometry) is unique
    return (tensor.device, tensor.dtype, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))


class PackedWeightRef:
    """What the autograd graph holds instead of a dequantized expert: its packed parameter."""

    __slots__ = ("rows", "version", "shape", "dtype", "group_size", "group", "index")

    def __init__(
        self,
        rows: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        group_size: int,
        group: _PackedGroup | None = None,
        index: int = 0,
    ):
        self.rows = rows
        self.version = rows._version
        self.shape = shape
        self.dtype = dtype
        self.group_size = group_size
        self.group = group
        self.index = index

    def dequantize(self) -> torch.Tensor:
        assert (
            self.rows._version == self.version
        ), "a frozen MXFP4 expert's packed bytes changed between its forward and its backward"
        if self.group is not None:
            weight = self.group.take(self.index)
        else:
            weight = dequantize_mxfp4_rows(self.rows, self.group_size, lut=lut_dequant_enabled())
        assert weight.shape == self.shape and weight.dtype == self.dtype, (weight.shape, self.shape, weight.dtype)
        return weight


class RedequantScope:
    """The dequantized weights of one module forward, and the pack/unpack hooks that swap them for their bytes."""

    def __init__(self, group_size: int = MXFP4_GROUP_SIZE):
        self.group_size = group_size
        self._packed: dict[tuple, tuple[torch.Tensor, _PackedGroup | None, int]] = {}
        self.registered = 0
        self.packed = 0

    def register(self, weights: Sequence[torch.Tensor], rows: Sequence[torch.Tensor]) -> _PackedGroup | None:
        """``weights[i]`` was dequantized from ``rows[i]`` for this forward. Returns the group the backward will
        dequantize them through (None off the grouped paths)."""
        assert len(weights) == len(rows)
        grouped = batched_dequant_enabled() or _triton_path(rows)
        group = _PackedGroup(rows, self.group_size) if grouped and len(rows) > 1 else None
        for index, (weight, packed) in enumerate(zip(weights, rows, strict=True)):
            self._packed[_key(weight)] = (packed, group, index)
        self.registered += len(weights)
        return group

    def pack(self, tensor: torch.Tensor):
        entry = self._packed.get(_key(tensor)) if isinstance(tensor, torch.Tensor) else None
        if entry is None:
            return tensor
        packed, group, index = entry
        self.packed += 1
        return PackedWeightRef(packed, tensor.shape, tensor.dtype, self.group_size, group, index)

    @staticmethod
    def unpack(saved):
        return saved.dequantize() if isinstance(saved, PackedWeightRef) else saved

    def close(self) -> None:
        # drop the address table: the weights it named are freed after the forward
        self._packed.clear()


@contextlib.contextmanager
def redequant_saved_weights(group_size: int = MXFP4_GROUP_SIZE) -> Iterator[RedequantScope]:
    """Run a forward whose ``RedequantScope.register``-ed weights the graph keeps as packed bytes."""
    scope = RedequantScope(group_size)
    try:
        with torch.autograd.graph.saved_tensors_hooks(scope.pack, scope.unpack):
            yield scope
    finally:
        scope.close()
        _report_once(scope)


def _report_once(scope: RedequantScope) -> None:
    global _reported
    if _reported or scope.registered == 0:
        return
    _reported = True
    if scope.packed == 0:
        logger.warning(
            "%s=1 but the grouped linear saved none of the %d dequantized MXFP4 experts it was given: this TE "
            "keeps other tensors for the backward, so nothing is saved and nothing changes",
            REDEQUANT_ENV,
            scope.registered,
        )
    else:
        logger.info(
            "%s=1: frozen MXFP4 experts are re-dequantized in the backward (%d of %d packed on the first call)",
            REDEQUANT_ENV,
            scope.packed,
            scope.registered,
        )


class RedequantFrozenExperts:
    """Mixin over a grouped linear whose ``weight{i}`` are packed MXFP4 rows: ``_get_weight_tensors`` returns their
    exact BF16 dequantization, and with the switch on the forward saves the packed rows, not the BF16 copies.

    Listed before the TE grouped linear in the bases: the host provides ``num_gemms`` and ``weight0..weightN-1``,
    and the next class in the MRO the forward that calls ``_get_weight_tensors()`` and saves what it returns.
    """

    _redequant_scope: RedequantScope | None = None
    # (weights, the rows they came from, the rows' versions, lut): set by prefetch_weights, taken by the next fetch
    _prefetched: tuple | None = None
    # the packed group the last grad-enabled forward saved for the backward (moe_a2a_overlap's backward prefetch)
    _saved_group: _PackedGroup | None = None

    def _packed_rows(self) -> list[torch.Tensor]:
        return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]  # config-access-exempt: TE weight0..N-1

    @staticmethod
    def _dequantize(rows: list[torch.Tensor], lut) -> list[torch.Tensor]:
        # dequantize_expert_rows reads the Triton, batched and LUT switches itself (same bits either way); ``lut``
        # (``dequant_mode()``) keys the prefetch's freshness check, read from the same switches at the same moment
        return dequantize_expert_rows(rows, MXFP4_GROUP_SIZE)

    def prefetch_weights(self) -> None:
        """Dequantize now, for the next ``_get_weight_tensors`` (``moe_a2a_overlap``: during the dispatch a2a)."""
        rows = self._packed_rows()
        lut = dequant_mode()
        self._prefetched = (self._dequantize(rows, lut), rows, [r._version for r in rows], lut)

    def _take_prefetched(self, rows: list[torch.Tensor], lut: bool) -> list[torch.Tensor] | None:
        prefetched, self._prefetched = self._prefetched, None
        if prefetched is None:
            return None
        weights, prefetched_rows, versions, prefetched_lut = prefetched
        fresh = (
            prefetched_lut == lut
            and len(prefetched_rows) == len(rows)
            and all(a is b and a._version == v for a, b, v in zip(prefetched_rows, rows, versions, strict=True))
        )
        # stale (bytes reloaded since): dequantize again; the values would be the new bytes' anyway
        return weights if fresh else None

    def _get_weight_tensors(self):
        rows = self._packed_rows()
        lut = dequant_mode()
        weights = self._take_prefetched(rows, lut)
        if weights is None:
            weights = self._dequantize(rows, lut)
        if self._redequant_scope is not None:
            self._saved_group = self._redequant_scope.register(weights, rows)
        return weights

    def take_saved_group(self) -> _PackedGroup | None:
        """The packed group of the forward that just ran (once), for an early backward dequantization."""
        group, self._saved_group = self._saved_group, None
        return group

    def forward(self, *args, **kwargs):
        if not redequant_enabled() or not torch.is_grad_enabled():
            return super().forward(*args, **kwargs)
        assert self._redequant_scope is None, "re-entrant forward of one frozen MXFP4 grouped linear"
        with redequant_saved_weights(MXFP4_GROUP_SIZE) as scope:
            self._redequant_scope = scope
            try:
                return super().forward(*args, **kwargs)
            finally:
                self._redequant_scope = None
