"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn


try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(
        module, "cp_group", None
    )  # config-access-exempt: context parallel groups are attached only to participating model modules
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _zigzag_maps(cu_seqlens, cp_size, total):
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding). Returns ``(zigzag_position, packed_position)``:
    ``zigzag_position[p]`` is the row of packed position ``p`` in the rank-concatenated zigzag shards,
    and ``packed_position`` is its inverse."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    zigzag_position = owner * shard + start // cp_size + offset % chunk + mirrored * chunk
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = position
    return zigzag_position, packed_position


def _relayout_indices(cu_seqlens, cp_rank, cp_size, total):
    """``(to_packed, to_zigzag)``: ``to_packed`` selects this rank's contiguous shard from the
    rank-concatenated zigzag shards, ``to_zigzag`` selects this rank's zigzag shard from the
    rank-concatenated contiguous shards (the packed stream itself)."""
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, total)
    rows = slice(cp_rank * (total // cp_size), (cp_rank + 1) * (total // cp_size))
    return zigzag_position[rows], packed_position[rows]


def zigzag_gather_layout(cu_seqlens, cp_rank, cp_size, local_len):
    """For attention over the whole packed stream from a rank's zigzag shard of ``local_len`` rows:
    ``(gather_order, positions)``. ``gather_order`` puts the rank-concatenated zigzag shards (a CP
    all-gather) in packed order; ``positions`` are this rank's rows' packed positions."""
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, local_len * cp_size)
    return zigzag_position, packed_position[cp_rank * local_len : (cp_rank + 1) * local_len]


def _gather_select(x, index, cp_group):
    gathered = x.new_empty((x.shape[0] * dist.get_world_size(group=cp_group), *x.shape[1:]))
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=cp_group)
    return gathered.index_select(0, index)


class _Relayout(torch.autograd.Function):
    """All-gather over CP, then select this rank's rows; the gradient is the same with the inverse map."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group):
        ctx.cp_group = cp_group
        ctx.save_for_backward(inverse_index)
        return _gather_select(x, index, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)


def tp_slice_all_gather(x: torch.Tensor, cp_group, tp_group) -> torch.Tensor:
    """``all_gather_into_tensor`` of ``x`` over ``cp_group`` (rank-concatenated rows) for an ``x`` identical on
    every rank of ``tp_group``: each TP rank moves only its 1 / TP slice over CP, the TP ranks all-gather the
    slices. Same bytes out; 1 / TP of the cross-CP traffic. No autograd: the caller owns the backward."""
    cp_size = dist.get_world_size(group=cp_group)
    tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
    rows, rest = x.shape[0], x.shape[1:]
    assert rows % tp_size == 0, (rows, tp_size)
    part = rows // tp_size
    slices = x.new_empty((cp_size * part, *rest))  # [cp, part, ...]
    dist.all_gather_into_tensor(slices, x[tp_rank * part : (tp_rank + 1) * part].contiguous(), group=cp_group)
    full = x.new_empty((tp_size * cp_size * part, *rest))  # [tp, cp, part, ...]
    dist.all_gather_into_tensor(full, slices, group=tp_group)
    return full.view(tp_size, cp_size, part, *rest).transpose(0, 1).reshape(cp_size * rows, *rest)


class _TPReplicatedRelayout(torch.autograd.Function):
    """``_Relayout`` of a tensor that is identical on every TP rank (the sequence-parallel-gathered layer input).

    ``_Relayout`` all-gathers the whole [rows, ...] tensor over CP on every TP rank: TP copies of the same bytes
    over the (cross-node) CP link. Here TP rank t sends only its 1 / TP slice of rows over CP, then the TP ranks
    all-gather the peers' slices over the (intra-node) TP link. Pure data movement, so the output is bit-identical.
    The gradient is TP-distinct (the consumers are column-parallel), so the backward is ``_Relayout``'s."""

    @staticmethod
    def forward(ctx, x, index, inverse_index, cp_group, tp_group):
        ctx.cp_group, ctx.tp_group = cp_group, tp_group
        ctx.save_for_backward(inverse_index)
        cp_size, cp_rank = dist.get_world_size(group=cp_group), dist.get_rank(group=cp_group)
        tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
        rows, rest = x.shape[0], x.shape[1:]
        part = rows // tp_size
        # [cp, part, ...]: slice tp_rank of every CP rank's rows
        slices = x.new_empty((cp_size * part, *rest))
        dist.all_gather_into_tensor(slices, x[tp_rank * part : (tp_rank + 1) * part].contiguous(), group=cp_group)
        # the peers' slices only: this CP rank's own rows are x itself
        peers = slices.view(cp_size, part, *rest)[[c for c in range(cp_size) if c != cp_rank]].reshape(-1, *rest)
        remote = x.new_empty((tp_size * (cp_size - 1) * part, *rest))  # [tp, cp - 1, part, ...]
        dist.all_gather_into_tensor(remote, peers, group=tp_group)
        # row g of the rank-concatenated zigzag shards: CP rank g // rows, TP slice (g % rows) // part
        owner, local = index // rows, index % rows
        peer = owner - (owner > cp_rank).to(owner.dtype)
        remote_row = rows + ((local // part) * (cp_size - 1) + peer) * part + local % part
        return torch.cat((x, remote)).index_select(0, torch.where(owner == cp_rank, local, remote_row))

    @staticmethod
    def backward(ctx, grad_output):
        (inverse_index,) = ctx.saved_tensors
        if relayout_bwd_reduce_on():
            grad = _tp_reduced_gather_select(grad_output, inverse_index, ctx.cp_group, ctx.tp_group)
            return grad, None, None, None, None
        return _gather_select(grad_output, inverse_index, ctx.cp_group), None, None, None, None


RELAYOUT_BWD_REDUCE_ENV = "MILES_K31_KDA_RELAYOUT_BWD_REDUCE"


def relayout_bwd_reduce_on() -> bool:
    """MILES_K31_KDA_RELAYOUT_BWD_REDUCE=1 (default off; set it the same on every rank, the backward's collectives
    differ): ``_TPReplicatedRelayout``'s backward sums the gradient over TP before it crosses CP. Only that class
    reads it: under MILES_K31_KDA_CP_RELAYOUT_A2A the layer uses ``_A2ARelayout`` (bit-identical, half the CP bytes
    of the all-gather at CP2) and this switch has no effect; the layer logs that once."""
    return os.environ.get(RELAYOUT_BWD_REDUCE_ENV, "0") == "1"


def _reduce_scatter_rows(x: torch.Tensor, group) -> torch.Tensor:
    """Sum ``x`` over ``group`` and keep this rank's 1 / size block of rows (gloo, the CPU tests' backend, has no
    reduce-scatter: all-reduce and slice there)."""
    size, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
    part = x.shape[0] // size
    if dist.get_backend(group) == "gloo":
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x[rank * part : (rank + 1) * part].contiguous()
    out = x.new_empty((part, *x.shape[1:]))
    dist.reduce_scatter_tensor(out, x, group=group)
    return out


def _tp_reduced_gather_select(grad, inverse_index, cp_group, tp_group):
    """The gradient of ``_TPReplicatedRelayout`` with 1 / TP of the cross-CP bytes.

    ``_gather_select`` all-gathers each TP rank's whole TP-distinct partial gradient over CP (the n1 trace: 19.8 ms
    per KDA layer and microbatch for [131072, 7168] bf16, against 2.6-3.2 ms for the forward's TP-slice gather).
    The layer input is the sequence-parallel gather (``tensor_parallel_output_grad=True``), whose backward sums its
    gradient over TP, so any split of the true gradient over the TP ranks is as good as the partials. Here the TP
    ranks first reduce-scatter the packed-layout partials (rank t gets packed rows [t * part, (t + 1) * part) summed,
    over the intra-node link), the CP ranks all-gather those sums (1 / TP of today's cross-CP bytes), and rank t
    returns, for every zigzag row, the summed gradient if its packed row is in slice t, else zero: the TP sum of the
    returned tensors is the summed gradient, exactly once per row. The values differ from the default only in the
    order of the TP sum (bf16 reduce-scatter in packed instead of zigzag row blocks), not in what is summed."""
    cp_size = dist.get_world_size(group=cp_group)
    tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)
    rows, rest = grad.shape[0], grad.shape[1:]
    part = rows // tp_size
    summed = _reduce_scatter_rows(grad.contiguous(), tp_group)  # [part, ...] packed rows of slice tp_rank
    slices = grad.new_empty((cp_size * part, *rest))  # [cp, part, ...]
    dist.all_gather_into_tensor(slices, summed, group=cp_group)
    # zigzag row i takes packed row inverse_index[i] of the rank-concatenated packed shards
    owner, local = inverse_index // rows, inverse_index % rows
    mine = local // part == tp_rank
    src = torch.where(mine, owner * part + local - tp_rank * part, torch.zeros_like(local))
    picked = slices.index_select(0, src)
    return torch.where(mine.view(-1, *([1] * len(rest))), picked, torch.zeros_like(picked))


def zigzag_to_packed_shard_tp_replicated(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, tp_group):
    """``zigzag_to_packed_shard`` of a tensor identical on every rank of ``tp_group``: the same result and
    gradient, with 1 / TP of its cross-CP forward bytes. Rows must split evenly over TP (they do under SP)."""
    assert hidden_states.size(0) % tp_group.size() == 0, (hidden_states.size(0), tp_group.size())
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _TPReplicatedRelayout.apply(hidden_states, to_packed, to_zigzag, cp_group, tp_group)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)


# ---------------------------------------------------------------------------------------------------------------
# MILES_K31_KDA_CP_RELAYOUT_A2A: the same relayouts by an all-to-all of only the rows each rank needs.
#
# A relayout is a permutation of the rank-concatenated rows: every source row lands on exactly one rank. The
# all-gather above moves every rank's whole shard to every other rank and then keeps the rows it needs. At
# CP2 that is half of the peer's shard (a rank keeps half its own zigzag rows in its contiguous shard), so
# half the bytes are thrown away, and more at higher CP. Sending only the needed rows moves the same values
# into the same places. It is a pure permutation, so it is bit-identical, forward and backward (the backward
# is the inverse permutation).
#
# The all-to-all split sizes must be known on the host. They depend only on ``cu_seqlens`` and are computed
# once per microbatch (one small DtoH copy) and cached with the index tensors. Every KDA layer, the recompute
# and the backward reuse them: ``cu_seqlens`` is the same tensor object for all of a microbatch's layers.
# ---------------------------------------------------------------------------------------------------------------

CP_RELAYOUT_A2A_ENV = "MILES_K31_KDA_CP_RELAYOUT_A2A"
_PLAN_CACHE: list = []
_PLAN_CACHE_SIZE = 8


def cp_relayout_a2a_on() -> bool:
    return os.environ.get(CP_RELAYOUT_A2A_ENV, "0") == "1"


class _CPPermutation:
    """One rank's side of a permutation of the rank-concatenated rows, ``out_global[g] = in_global[src_of[g]]``.

    ``own_pos`` / ``own_src``: output rows taken from this rank's own input rows. ``recv_pos``: the output rows
    filled by the all-to-all, grouped by source rank (ascending), each group in output order. ``send_idx``: this
    rank's input rows the others need, grouped by destination rank, each group in the destination's order.
    ``counts[r][s]``: rows rank ``r`` takes from rank ``s`` (host ints, the same table on every rank)."""

    def __init__(self, src_of: torch.Tensor, cp_rank: int, cp_size: int, counts: list[list[int]]):
        shard = src_of.numel() // cp_size
        self.cp_rank, self.cp_size, self.shard, self.counts = cp_rank, cp_size, shard, counts
        mine = src_of[cp_rank * shard : (cp_rank + 1) * shard]
        owner = mine // shard
        self.own_pos = torch.nonzero(owner == cp_rank).flatten()
        self.own_src = mine[self.own_pos] - cp_rank * shard
        others = [s for s in range(cp_size) if s != cp_rank]
        self.recv_pos = torch.cat([torch.nonzero(owner == s).flatten() for s in others])
        sends = []
        for r in others:
            theirs = src_of[r * shard : (r + 1) * shard]
            sends.append(theirs[theirs // shard == cp_rank] - cp_rank * shard)
        self.send_idx = torch.cat(sends)
        self.recv_splits = [0 if s == cp_rank else counts[cp_rank][s] for s in range(cp_size)]
        self.send_splits = [0 if r == cp_rank else counts[r][cp_rank] for r in range(cp_size)]

    def _place(self, own: torch.Tensor, received: torch.Tensor) -> torch.Tensor:
        out = own.new_empty((self.shard, *own.shape[1:]))
        out.index_copy_(0, self.own_pos, own)
        out.index_copy_(0, self.recv_pos, received)
        return out

    def forward(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        received = x.new_empty((sum(self.recv_splits), *x.shape[1:]))
        dist.all_to_all_single(
            received,
            x.index_select(0, self.send_idx),
            output_split_sizes=self.recv_splits,
            input_split_sizes=self.send_splits,
            group=cp_group,
        )
        return self._place(x.index_select(0, self.own_src), received)

    def backward(self, grad: torch.Tensor, cp_group) -> torch.Tensor:
        returned = grad.new_empty((sum(self.send_splits), *grad.shape[1:]))
        dist.all_to_all_single(
            returned,
            grad.index_select(0, self.recv_pos),
            output_split_sizes=self.send_splits,
            input_split_sizes=self.recv_splits,
            group=cp_group,
        )
        grad_x = grad.new_empty((self.shard, *grad.shape[1:]))
        grad_x.index_copy_(0, self.own_src, grad.index_select(0, self.own_pos))
        grad_x.index_copy_(0, self.send_idx, returned)
        return grad_x

    def forward_tp_replicated(self, x: torch.Tensor, cp_group, tp_group) -> torch.Tensor:
        """``forward`` of an ``x`` identical on every rank of ``tp_group``: TP rank ``t`` carries the ``t``-th
        ``1 / TP`` of every rank pair's rows over CP, then the TP ranks all-gather what they received."""
        tp_size, tp_rank = dist.get_world_size(group=tp_group), dist.get_rank(group=tp_group)

        def piece(n: int, t: int) -> tuple[int, int]:  # rows [lo, hi) of n that TP rank t carries
            k = -(-n // tp_size)
            return min(n, t * k), min(n, (t + 1) * k)

        send_parts, send_splits, recv_splits = [], [], []
        offset = 0
        for r in range(self.cp_size):
            n = 0 if r == self.cp_rank else self.counts[r][self.cp_rank]
            lo, hi = piece(n, tp_rank)
            send_parts.append(self.send_idx[offset + lo : offset + hi])
            send_splits.append(hi - lo)
            offset += n
        widths = []  # per source rank: the padded rows every TP rank holds for it
        for s in range(self.cp_size):
            m = 0 if s == self.cp_rank else self.counts[self.cp_rank][s]
            lo, hi = piece(m, tp_rank)
            recv_splits.append(hi - lo)
            widths.append(-(-m // tp_size))
        received = x.new_empty((sum(recv_splits), *x.shape[1:]))
        dist.all_to_all_single(
            received,
            x.index_select(0, torch.cat(send_parts)),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=cp_group,
        )
        # pad each source's piece to its width so the TP all-gather is uniform: [tp, sum(widths), ...]
        padded = x.new_zeros((sum(widths), *x.shape[1:]))
        index, start, pos = [], 0, 0
        for s in range(self.cp_size):
            padded[start : start + recv_splits[s]] = received[pos : pos + recv_splits[s]]
            start += widths[s]
            pos += recv_splits[s]
        gathered = x.new_empty((tp_size * sum(widths), *x.shape[1:]))
        dist.all_gather_into_tensor(gathered, padded, group=tp_group)
        # back to the plain all-to-all's received order: source by source, TP piece by TP piece
        start = 0
        for s in range(self.cp_size):
            m = 0 if s == self.cp_rank else self.counts[self.cp_rank][s]
            for t in range(tp_size):
                lo, hi = piece(m, t)
                base = t * sum(widths) + start
                index.append(torch.arange(base, base + hi - lo, device=x.device))
            start += widths[s]
        full = gathered.index_select(0, torch.cat(index))
        return self._place(x.index_select(0, self.own_src), full)


def _relayout_plans(cu_seqlens: torch.Tensor, cp_rank: int, cp_size: int, total: int):
    """``(to_packed, to_zigzag)`` :class:`_CPPermutation` of one microbatch, cached on its ``cu_seqlens``."""
    for entry in _PLAN_CACHE:
        cu, version, key, plans = entry
        if cu is cu_seqlens and version == cu_seqlens._version and key == (cp_rank, cp_size, total):
            return plans
    zigzag_position, packed_position = _zigzag_maps(cu_seqlens, cp_size, total)
    shard = total // cp_size
    dest = torch.arange(total, device=zigzag_position.device) // shard
    # one DtoH for both directions: counts[r][s] of rank r's rows taken from rank s
    both = torch.stack(
        [
            torch.bincount(dest * cp_size + src // shard, minlength=cp_size * cp_size)
            for src in (zigzag_position, packed_position)
        ]
    ).tolist()
    counts = [[row[r * cp_size : (r + 1) * cp_size] for r in range(cp_size)] for row in both]
    plans = (
        _CPPermutation(zigzag_position, cp_rank, cp_size, counts[0]),
        _CPPermutation(packed_position, cp_rank, cp_size, counts[1]),
    )
    _PLAN_CACHE.insert(0, (cu_seqlens, cu_seqlens._version, (cp_rank, cp_size, total), plans))
    del _PLAN_CACHE[_PLAN_CACHE_SIZE:]
    return plans


class _A2ARelayout(torch.autograd.Function):
    """``_Relayout`` / ``_TPReplicatedRelayout`` by all-to-all; the gradient is the inverse permutation."""

    @staticmethod
    def forward(ctx, x, plan, cp_group, tp_group):
        ctx.plan, ctx.cp_group = plan, cp_group
        if tp_group is not None:
            return plan.forward_tp_replicated(x, cp_group, tp_group)
        return plan.forward(x, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.plan.backward(grad_output.contiguous(), ctx.cp_group), None, None, None


def zigzag_to_packed_shard_a2a(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size, tp_group=None):
    """``zigzag_to_packed_shard`` (``tp_group`` given: ``zigzag_to_packed_shard_tp_replicated``) by
    all-to-all. Same result and gradient, bit for bit."""
    to_packed, _ = _relayout_plans(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _A2ARelayout.apply(hidden_states, to_packed, cp_group, tp_group)


def packed_shard_to_zigzag_a2a(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """``packed_shard_to_zigzag`` by all-to-all. Same result and gradient, bit for bit."""
    _, to_zigzag = _relayout_plans(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _A2ARelayout.apply(hidden_states, to_zigzag, cp_group, None)


# ---------------------------------------------------------------------------------------------------------------
# MILES_K31_KDA_SP_RELAYOUT_FUSED: the sequence-parallel all-gather and the zigzag -> contiguous CP relayout of the
# KDA layer input as one operation, with a backward that sums over TP before anything crosses CP.
#
# Under sequence parallelism the layer input arrives as rank (c, t)'s 1 / TP slice of CP rank c's zigzag shard.
# Today it is all-gathered over TP (Megatron's ``gather_from_sequence_parallel_region``), relayouted over CP, and in
# the backward each TP rank sends its own TP-distinct partial gradient of the whole packed shard across CP
# (the largest KDA collective: [shard, hidden] bf16 per rank, 896 MiB at 262k / CP2 / TP8, 10.5-11 ms), before
# Megatron's reduce-scatter sums the partials over TP.
#
# Here the forward routes every row from its sequence-parallel owner (c, t) to a "carrier" (c, x) over TP, from the
# carrier to (c', x) over CP when its packed shard c' is the other one, and the TP ranks of c' all-gather what they
# carry. The carrier x splits each (packed shard, source CP rank) class of rows evenly over the TP ranks, so every
# rank sends and receives the same cross-CP bytes whatever the sequence layout (one long sequence or many short
# ones), and the TP all-gather blocks are of equal size. It is pure data movement: the output is bit-identical.
#
# The backward is the mirror image: the TP ranks reduce-scatter the packed-shard gradient over their carried rows
# (the sum the SP all-gather's backward did, now before the CP hop), then the summed rows go back over CP and TP to
# their sequence-parallel owner. Cross-CP bytes drop by TP (the summed rows instead of every TP rank's partials) and
# there is no second reduce-scatter. It sums the same TP partials as before; only which TP rank reduces a row (so
# the bf16 ring order of NCCL's reduce-scatter) changes. Forward values (and log-probs) are unchanged bit for bit.
# ---------------------------------------------------------------------------------------------------------------

SP_RELAYOUT_FUSED_ENV = "MILES_K31_KDA_SP_RELAYOUT_FUSED"
_SP_PLAN_CACHE: list = []


def sp_relayout_fused_on() -> bool:
    """MILES_K31_KDA_SP_RELAYOUT_FUSED=1 (default off; set it the same on every rank)."""
    return os.environ.get(SP_RELAYOUT_FUSED_ENV, "0") == "1"


class _Hop:
    """One ``all_to_all_single`` that sends every input row exactly once. ``send_idx``: the input rows in send
    order (grouped by destination rank, ascending); the output holds the received rows grouped by source rank,
    ascending (the own block included: a rank also sends to itself)."""

    def __init__(self, send_idx: torch.Tensor, send_splits: list[int], recv_splits: list[int]):
        assert send_idx.numel() == sum(send_splits), (send_idx.numel(), send_splits)
        self.send_idx, self.send_splits, self.recv_splits = send_idx, send_splits, recv_splits

    def forward(self, x: torch.Tensor, group) -> torch.Tensor:
        out = x.new_empty((sum(self.recv_splits), *x.shape[1:]))
        dist.all_to_all_single(
            out,
            x.index_select(0, self.send_idx),
            output_split_sizes=self.recv_splits,
            input_split_sizes=self.send_splits,
            group=group,
        )
        return out

    def backward(self, grad: torch.Tensor, group) -> torch.Tensor:
        back = grad.new_empty((sum(self.send_splits), *grad.shape[1:]))
        dist.all_to_all_single(
            back,
            grad.contiguous(),
            output_split_sizes=self.send_splits,
            input_split_sizes=self.recv_splits,
            group=group,
        )
        grad_x = grad.new_empty((self.send_idx.numel(), *grad.shape[1:]))
        grad_x.index_copy_(0, self.send_idx, back)
        return grad_x


class _SPRelayoutPlan:
    """Rank (``cp_rank``, ``tp_rank``)'s side of the fused SP all-gather + zigzag -> contiguous relayout of one
    microbatch: ``hop_tp`` (SP owner -> carrier, over TP), ``hop_cp`` (carrier -> packed shard, over CP), then
    ``pad_idx`` places the carried rows in this rank's ``width``-row all-gather block and ``final_idx`` picks the
    packed shard out of the TP all-gather of the blocks."""

    def __init__(self, cu_seqlens: torch.Tensor, cp_rank: int, cp_size: int, tp_rank: int, tp_size: int, total: int):
        C, T, N = cp_size, tp_size, total
        R = N // C
        S = R // T
        assert R * C == N and S * T == R, (N, C, T)
        zigzag_position, _ = _zigzag_maps(cu_seqlens, C, N)
        dev = zigzag_position.device
        p = torch.arange(N, device=dev)
        z = zigzag_position  # zigzag row of packed position p
        pc, zc = p // R, z // R  # packed shard (destination CP rank), zigzag shard (source CP rank)
        zt = (z % R) // S  # source SP owner
        cls = pc * C + zc
        # rank of each row within its (packed shard, source CP rank) class, rows in packed order
        n_cls = torch.bincount(cls, minlength=C * C)
        starts = torch.cumsum(n_cls, 0) - n_cls
        order = torch.argsort(cls * N + p)
        rank_in_cls = torch.empty_like(p)
        rank_in_cls[order] = p - starts[cls[order]]
        k_cls = (n_cls + T - 1) // T  # rows of a class each carrier takes (the last one may take fewer)
        x = rank_in_cls // k_cls.clamp(min=1)[cls]  # carrier TP rank
        # one DtoH: class sizes, [zc, zt, x] (TP hop) and [zc, x, pc] (CP hop) row counts
        host = torch.cat(
            [
                n_cls,
                torch.bincount((zc * T + zt) * T + x, minlength=C * T * T),
                torch.bincount((zc * T + x) * C + pc, minlength=C * T * C),
            ]
        ).tolist()
        n_host = host[: C * C]
        tp_table = host[C * C : C * C + C * T * T]
        cp_table = host[C * C + C * T * T :]
        c, t = cp_rank, tp_rank
        # TP hop: rank (c, t) sends its SP rows to their carriers (c, x), in packed order within each carrier
        mine = torch.nonzero((zc == c) & (zt == t)).flatten()
        mine = mine[torch.argsort(x[mine] * N + mine)]
        self.hop_tp = _Hop(
            (z[mine] % R) - t * S,
            [tp_table[(c * T + t) * T + d] for d in range(T)],
            [tp_table[(c * T + s) * T + t] for s in range(T)],
        )
        # what (c, t) carries after the TP hop: rows with zc == c and x == t, by source SP owner then packed order
        carried = torch.nonzero((zc == c) & (x == t)).flatten()
        carried = carried[torch.argsort(zt[carried] * N + carried)]
        # CP hop: send them to their packed shard (c', t), in packed order within each destination
        by_dest = torch.argsort(pc[carried] * N + carried)
        self.hop_cp = _Hop(
            by_dest,
            [cp_table[(c * T + t) * C + d] for d in range(C)],
            [cp_table[(s * T + t) * C + c] for s in range(C)],
        )
        # received rows: packed shard c, carrier t, by source CP rank then packed order = piece t of each class
        k_host = [-(-n // T) for n in n_host]
        offsets, width = [], 0
        for s in range(C):
            offsets.append(width)
            width += k_host[c * C + s]
        self.width = width
        off_dev = torch.tensor(offsets, device=dev, dtype=p.dtype)
        got = torch.nonzero((pc == c) & (x == t)).flatten()
        got = got[torch.argsort(zc[got] * N + got)]
        self.pad_idx = off_dev[zc[got]] + rank_in_cls[got] - t * k_cls[cls[got]]
        shard = p[c * R : (c + 1) * R]
        self.final_idx = x[shard] * width + off_dev[zc[shard]] + rank_in_cls[shard] - x[shard] * k_cls[cls[shard]]
        self.rows_out = R
        self.cross_cp_rows = sum(self.hop_cp.send_splits) - self.hop_cp.send_splits[c]


def _sp_relayout_plan(cu_seqlens, cp_rank, cp_size, tp_rank, tp_size, total) -> _SPRelayoutPlan:
    key = (cp_rank, cp_size, tp_rank, tp_size, total)
    for cu, version, k, plan in _SP_PLAN_CACHE:
        if cu is cu_seqlens and version == cu_seqlens._version and k == key:
            return plan
    plan = _SPRelayoutPlan(cu_seqlens, *key)
    _SP_PLAN_CACHE.insert(0, (cu_seqlens, cu_seqlens._version, key, plan))
    del _SP_PLAN_CACHE[_PLAN_CACHE_SIZE:]
    return plan


class _SPRelayout(torch.autograd.Function):
    """``gather_from_sequence_parallel_region(tensor_parallel_output_grad=True)`` then ``zigzag_to_packed_shard``,
    fused: SP rows -> carriers (TP all-to-all) -> packed shard (CP all-to-all) -> TP all-gather. Backward: TP
    reduce-scatter, then the two all-to-alls reversed."""

    @staticmethod
    def forward(ctx, x_sp, plan, cp_group, tp_group):
        ctx.plan, ctx.cp_group, ctx.tp_group = plan, cp_group, tp_group
        carried = plan.hop_cp.forward(plan.hop_tp.forward(x_sp, tp_group), cp_group)
        block = carried.new_zeros((plan.width, *carried.shape[1:]))
        block.index_copy_(0, plan.pad_idx, carried)
        gathered = block.new_empty((plan.width * dist.get_world_size(group=tp_group), *block.shape[1:]))
        dist.all_gather_into_tensor(gathered, block, group=tp_group)
        return gathered.index_select(0, plan.final_idx)

    @staticmethod
    def backward(ctx, grad):
        plan = ctx.plan
        spread = grad.new_zeros((plan.width * dist.get_world_size(group=ctx.tp_group), *grad.shape[1:]))
        spread.index_copy_(0, plan.final_idx, grad.contiguous())
        block = _reduce_scatter_rows(spread, ctx.tp_group)
        carried = plan.hop_cp.backward(block.index_select(0, plan.pad_idx), ctx.cp_group)
        return plan.hop_tp.backward(carried, ctx.tp_group), None, None, None


def sp_gather_zigzag_to_packed_shard(x_sp, cu_seqlens, cp_group, tp_group):
    """``zigzag_to_packed_shard(gather_from_sequence_parallel_region(x_sp))``: this CP rank's contiguous packed
    shard ``[shard, ...]`` (the same on every TP rank) from this rank's sequence-parallel slice ``x_sp`` of its
    zigzag shard. Forward bit-identical; see the MILES_K31_KDA_SP_RELAYOUT_FUSED note for the backward."""
    cp_size, tp_size = dist.get_world_size(group=cp_group), dist.get_world_size(group=tp_group)
    total = x_sp.size(0) * tp_size * cp_size
    plan = _sp_relayout_plan(
        cu_seqlens, dist.get_rank(group=cp_group), cp_size, dist.get_rank(group=tp_group), tp_size, total
    )
    return _SPRelayout.apply(x_sp, plan, cp_group, tp_group)


# ---------------------------------------------------------------------------------------------------------------
# MILES_K31_KDA_CORE_RELAYOUT_OVERLAP: the KDA core output's contiguous -> zigzag relayout and ``out_proj`` as one
# function, whose backward queues the relayout's all-to-all (on a side stream) between out_proj's data-gradient
# and weight-gradient GEMMs, so the weight-gradient GEMM runs while the gradient crosses CP. The same aten ops on
# the same tensors as autograd's ``linear`` backward (``mm_mat1_backward`` / ``mm_mat2_backward`` for a row-major
# input and the transposed weight): bit-identical, forward and backward. Requires MILES_K31_KDA_CP_RELAYOUT_A2A.
# ---------------------------------------------------------------------------------------------------------------

CORE_RELAYOUT_OVERLAP_ENV = "MILES_K31_KDA_CORE_RELAYOUT_OVERLAP"
_side_streams: dict[int, "torch.cuda.Stream"] = {}


def core_relayout_overlap_on() -> bool:
    """MILES_K31_KDA_CORE_RELAYOUT_OVERLAP=1 (default off)."""
    return os.environ.get(CORE_RELAYOUT_OVERLAP_ENV, "0") == "1"


def _side_stream(device: torch.device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _side_streams:
        # high priority: the NCCL CTAs get SMs ahead of the GEMM they overlap
        _side_streams[index] = torch.cuda.Stream(device=index, priority=-1)
    return _side_streams[index]


class _A2ARelayoutLinear(torch.autograd.Function):
    """``F.linear(packed_shard_to_zigzag_a2a(core), weight)``; see MILES_K31_KDA_CORE_RELAYOUT_OVERLAP."""

    @staticmethod
    def forward(ctx, core, weight, plan, cp_group):
        ctx.plan, ctx.cp_group = plan, cp_group
        zigzag = plan.forward(core, cp_group)
        ctx.save_for_backward(zigzag, weight)
        return torch.nn.functional.linear(zigzag, weight)

    @staticmethod
    def backward(ctx, grad_output):
        zigzag, weight = ctx.saved_tensors
        grad2d = grad_output.reshape(-1, grad_output.shape[-1])
        input2d = zigzag.reshape(-1, zigzag.shape[-1])
        grad_zigzag = grad2d.mm(weight).view(zigzag.shape)  # mm_mat1_backward: grad.mm(mat2.t()), mat2 = weight.t()
        if grad_zigzag.is_cuda:
            current, side = torch.cuda.current_stream(), _side_stream(grad_zigzag.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                grad_core = ctx.plan.backward(grad_zigzag, ctx.cp_group)
            grad_zigzag.record_stream(side)
        else:
            grad_core = ctx.plan.backward(grad_zigzag, ctx.cp_group)
        # mm_mat2_backward of the column-major mat2 = weight.t(): grad.t().mm(mat1).t(), then .t() for the weight
        grad_weight = grad2d.t().mm(input2d)
        if grad_zigzag.is_cuda:
            current.wait_stream(side)
            grad_core.record_stream(current)
        return grad_core, grad_weight, None, None


def packed_shard_to_zigzag_a2a_linear(core, weight, cu_seqlens, cp_group, cp_rank, cp_size):
    """``F.linear(packed_shard_to_zigzag_a2a(core, ...), weight)``, the backward's all-to-all overlapping the
    weight-gradient GEMM. Same values and gradients, bit for bit."""
    _, to_zigzag = _relayout_plans(cu_seqlens, cp_rank, cp_size, core.size(0) * cp_size)
    return _A2ARelayoutLinear.apply(core, weight, to_zigzag, cp_group)


# ---------------------------------------------------------------------------------------------------------------
# MILES_K31_KDA_OUT_PROJ_RS_CHUNKS=n (default 1: off): the KDA ``out_proj`` and the sequence-parallel
# reduce-scatter after it, pipelined over n row chunks in the forward: chunk i+1's GEMM runs on the current stream
# while chunk i's reduce-scatter runs on a side stream. Chunk i takes, from every TP rank's destination block, its
# i-th slice of rows, so its reduce-scatter writes rows [i*c, (i+1)*c) of this rank's output: the same rows and
# the same TP partials as the unchunked ``reduce_scatter_to_sequence_parallel_region(F.linear(x, w))``. Bit-exact
# as long as (a) a GEMM over a row subset gives the same bits as those rows of the full GEMM and (b) NCCL sums a
# reduce-scatter element in the same order whatever the message size; both hold for the K3 shapes on B300 TP8
# (work/overlap2 tp_overlap_bench: n = 2, 4, 8 bit-equal) but are properties of the kernels, so the GPU check
# re-tests them on the module. The backward is not chunked: Megatron's all-gather of the gradient, then the same
# aten ops as autograd's ``linear`` backward (dgrad ``grad.mm(w)``, wgrad ``grad.t().mm(x)`` on the full rows), so it
# is the unchunked backward. With MILES_K31_KDA_CORE_RELAYOUT_OVERLAP the output relayout is folded in as in
# ``_A2ARelayoutLinear`` (its all-to-all overlaps the wgrad GEMM).
# ---------------------------------------------------------------------------------------------------------------

OUT_PROJ_RS_CHUNKS_ENV = "MILES_K31_KDA_OUT_PROJ_RS_CHUNKS"


def out_proj_rs_chunks() -> int:
    """MILES_K31_KDA_OUT_PROJ_RS_CHUNKS (default 1 = off)."""
    value = int(os.environ.get(OUT_PROJ_RS_CHUNKS_ENV, "1"))
    if value < 1:
        raise ValueError(f"{OUT_PROJ_RS_CHUNKS_ENV} must be >= 1, got {value}")
    return value


def out_proj_rs_chunks_apply(rows: int, tp_size: int, chunks: int) -> bool:
    """Whether ``rows`` (this CP rank's out_proj rows, before the reduce-scatter) split into the chunks."""
    return chunks > 1 and tp_size > 1 and rows % (tp_size * chunks) == 0


def _reduce_scatter_tensor(output: torch.Tensor, input_: torch.Tensor, group) -> None:
    """Megatron's ``_reduce_scatter_along_first_dim`` collective (``dist_reduce_scatter_func``)."""
    dist.reduce_scatter_tensor(output, input_, group=group)


def _all_gather_tensor(output: torch.Tensor, input_: torch.Tensor, group) -> None:
    """Megatron's ``_gather_along_first_dim`` collective (``dist_all_gather_func``)."""
    dist.all_gather_into_tensor(output, input_, group=group)


class _ChunkedLinearReduceScatter(torch.autograd.Function):
    """``reduce_scatter_to_sequence_parallel_region(F.linear(relayout(core), weight))``, the forward pipelined over
    row chunks; see MILES_K31_KDA_OUT_PROJ_RS_CHUNKS. ``plan``: the output relayout to fold in (or None)."""

    @staticmethod
    def forward(ctx, core, weight, plan, cp_group, tp_group, chunks):
        ctx.plan, ctx.cp_group, ctx.tp_group = plan, cp_group, tp_group
        x = plan.forward(core, cp_group) if plan is not None else core
        ctx.save_for_backward(x, weight)
        tp = tp_group.size()
        rows = x.shape[0] // tp
        c = rows // chunks
        out = x.new_empty((rows, *x.shape[1:-1], weight.shape[0]))
        blocks = x.view(tp, chunks, c, *x.shape[1:])
        current = side = None
        if x.is_cuda:
            current, side = torch.cuda.current_stream(), _side_stream(x.device)
            out.record_stream(side)
        for i in range(chunks):
            # every TP rank's i-th slice of its destination block: the reduce-scatter lands on rows [i*c, (i+1)*c)
            part = torch.nn.functional.linear(blocks[:, i].reshape(tp * c, *x.shape[1:]), weight)
            if side is None:
                _reduce_scatter_tensor(out[i * c : (i + 1) * c], part, tp_group)
                continue
            side.wait_stream(current)
            with torch.cuda.stream(side):
                _reduce_scatter_tensor(out[i * c : (i + 1) * c], part, tp_group)
            part.record_stream(side)
        if side is not None:
            current.wait_stream(side)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        tp_group = ctx.tp_group
        grad_full = grad_output.new_empty((grad_output.shape[0] * tp_group.size(), *grad_output.shape[1:]))
        _all_gather_tensor(grad_full, grad_output.contiguous(), tp_group)
        grad2d = grad_full.reshape(-1, grad_full.shape[-1])
        input2d = x.reshape(-1, x.shape[-1])
        grad_core = grad_weight = None
        side = None
        if ctx.needs_input_grad[0]:
            grad_x = grad2d.mm(weight).view(x.shape)  # mm_mat1_backward: grad.mm(mat2.t()), mat2 = weight.t()
            if ctx.plan is None:
                grad_core = grad_x
            elif grad_x.is_cuda:
                current, side = torch.cuda.current_stream(), _side_stream(grad_x.device)
                side.wait_stream(current)
                with torch.cuda.stream(side):
                    grad_core = ctx.plan.backward(grad_x, ctx.cp_group)
                grad_x.record_stream(side)
            else:
                grad_core = ctx.plan.backward(grad_x, ctx.cp_group)
        if ctx.needs_input_grad[1]:
            # mm_mat2_backward of the column-major mat2 = weight.t(): grad.t().mm(mat1).t(), then .t() for the weight
            grad_weight = grad2d.t().mm(input2d)
        if side is not None:
            current.wait_stream(side)
            grad_core.record_stream(current)
        return grad_core, grad_weight, None, None, None, None


def linear_reduce_scatter_chunked(core, weight, tp_group, chunks: int, cp_group=None, relayout=None):
    """``reduce_scatter_to_sequence_parallel_region(F.linear(core, weight), group=tp_group)`` with the forward
    pipelined over ``chunks`` row chunks. ``relayout = (cu_seqlens, cp_rank, cp_size)`` folds the output
    contiguous -> zigzag all-to-all in first (MILES_K31_KDA_CORE_RELAYOUT_OVERLAP)."""
    plan = None
    if relayout is not None:
        cu_seqlens, cp_rank, cp_size = relayout
        _, plan = _relayout_plans(cu_seqlens, cp_rank, cp_size, core.size(0) * cp_size)
    return _ChunkedLinearReduceScatter.apply(core, weight, plan, cp_group, tp_group, chunks)
