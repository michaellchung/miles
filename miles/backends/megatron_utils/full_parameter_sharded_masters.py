"""Full-parameter FP32 masters under Megatron's distributed optimizer at DP>1.

Megatron's distributed optimizer splits every gradient bucket contiguously
across the bucket's data-parallel group.  Each rank keeps the FP32 master (and
the Adam moments) of one flat range of each parameter, possibly an empty one,
and steps only that range.  The BF16 model parameter stays whole on every rank:
after each step the optimizer copies its ranges into the model and all-gathers
the model.  Exactly one rank of the group therefore holds each FP32 master
element.

The full-parameter sync keeps its layout, owners and fragment plans
(``full_parameter_state._replica_owners``), so the layout hash, the plan hashes
and the exported bytes do not change.  What changes is where an owner reads its
FP32 bytes and where an apply writes:

* export: an owner's fragment is its own master ranges plus the ranges the other
  ranks of the parameter's group hold.  Those ranks first store their ranges in
  Ray (:func:`contribute`; no collective, so a rank call never waits for another
  rank), and the owner checks that its ranges and the contributions tile every
  fragment exactly once (:func:`fragment_sources`).
* apply: every rank already reads the whole new value of each parameter it
  holds (its own fragments, or its owners' chunks).  :meth:`MasterRange.write`
  writes the whole value into the model parameter and the part inside the
  rank's range into its FP32 master: the values Megatron's main-to-model copy
  and parameter all-gather would produce, without a collective.
* checks: model == master over each rank's own range (:func:`check_model`).

This module does not import ``full_parameter_state`` at run time (that module
imports this one); the caller passes the tracking state and the Ray ``put``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from miles.backends.megatron_utils.full_parameter_state import (
        FullParameterFragmentPlan,
        FullParameterReplication,
        FullParameterTopology,
    )

# ``(installed policy version, local-step generation, scheduler steps)``: the
# optimizer state a contribution is read at and may be used at.
Tracking = tuple[int, int, int]

_DEVICE_CAST_ENV = "MILES_FULL_PARAMETER_COMMIT_DEVICE_CAST"
# Elements of every device-cast write compared bit for bit with the host cast.
_CAST_SAMPLE_NUMEL = 4096


def _require_host_cast_bits(model: torch.Tensor, values: torch.Tensor) -> None:
    """The first ``_CAST_SAMPLE_NUMEL`` model elements hold the host cast of
    ``values`` bit for bit (what the host-cast write stored there)."""

    count = min(model.numel(), _CAST_SAMPLE_NUMEL)
    if not count:
        return
    device_bits = model[:count].detach().to("cpu").view(torch.uint8)
    host_bits = values[:count].to(model.dtype).view(torch.uint8)
    if not torch.equal(device_bits, host_bits):
        raise RuntimeError("full-parameter device cast of the master differs from the host cast")



@dataclass(frozen=True, eq=False)
class MasterRange:
    """This rank's part of one parameter's FP32 master.

    ``[start, stop)`` is the flat range of ``parameter`` the base optimizer
    steps on this rank (``start == stop``: none of it).  ``owned`` is the tensor
    in the optimizer's parameter groups, ``fp32`` the flat FP32 values of the
    range: Megatron's ``main_param``, or for a native-FP32 parameter the
    ``shard_fp32_groups`` view of the parameter itself (``fp32_is_model``)."""

    parameter: torch.nn.Parameter
    start: int
    stop: int
    owned: torch.Tensor | None
    fp32: torch.Tensor | None
    fp32_is_model: bool

    @property
    def numel(self) -> int:
        return self.stop - self.start

    def overlap(self, start: int, stop: int) -> tuple[int, int] | None:
        """``[start, stop) ∩ [self.start, self.stop)`` in parameter offsets, or None."""

        low, high = max(start, self.start), min(stop, self.stop)
        return (low, high) if low < high else None

    def values(self, start: int, stop: int) -> torch.Tensor:
        """The FP32 master of parameter offsets ``[start, stop)`` (inside the range)."""

        return self.fp32.detach()[start - self.start : stop - self.start]

    @torch.no_grad()
    def write(self, start: int, values: torch.Tensor) -> None:
        """Set parameter offsets ``[start, start + len(values))`` to ``values`` (FP32).

        The model gets every value (cast to its dtype, as Megatron's
        main-to-model copy casts), the master the values inside the range.

        Inside the range the values cross to the master's device once and the
        model takes the master's cast there: Megatron's main-to-model copy, and
        exactly what :meth:`matches_model` compares.  A host-side cast instead
        (``copy_`` of host FP32 into a device BF16 tensor casts on the host)
        reads the host bytes twice and allocates a host BF16 copy.  Both casts
        round to nearest even; a sample of every write is compared bit for bit
        with the host cast (``_require_host_cast_bits``), so a disagreement
        stops the commit as :func:`check_model` stopped the host-cast write.
        ``MILES_FULL_PARAMETER_COMMIT_DEVICE_CAST=0`` keeps the host cast."""

        stop = start + values.numel()
        model = self.parameter.detach().view(-1)
        inside = self.overlap(start, stop)
        if inside is None or self.fp32_is_model or os.environ.get(_DEVICE_CAST_ENV, "1") == "0":
            model[start:stop].copy_(values, non_blocking=False)
            if inside is not None and not self.fp32_is_model:
                low, high = inside
                self.fp32.detach()[low - self.start : high - self.start].copy_(
                    values[low - start : high - start], non_blocking=False
                )
            return
        low, high = inside
        master = self.fp32.detach()[low - self.start : high - self.start]
        master.copy_(values[low - start : high - start], non_blocking=False)
        model[low:high].copy_(master, non_blocking=False)
        _require_host_cast_bits(model[low:high], values[low - start : high - start])
        if start < low:
            model[start:low].copy_(values[: low - start], non_blocking=False)
        if high < stop:
            model[high:stop].copy_(values[high - start :], non_blocking=False)

    def matches_model(self) -> bool:
        if not self.numel:
            return True
        model = self.parameter.detach().view(-1)[self.start : self.stop]
        return torch.equal(model, self.fp32.detach().to(device=model.device, dtype=model.dtype))


# -- ranges ----------------------------------------------------------------------


def master_ranges(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    replication: FullParameterReplication,
    optimizer_owned_parameter_ids: frozenset[int] | None,
) -> dict[str, MasterRange] | None:
    """``{name: MasterRange}`` of every local parameter, or None when masters are whole.

    Masters are sharded when the optimizer is Megatron's distributed optimizer
    and some parameter is replicated on more than one rank.  Every rank of a
    replica group sees the same groups, so all of them decide alike."""

    views = getattr(optimizer_owned_parameter_ids, "distributed_views", None) or {}
    if not views and not any(getattr(parameter, "main_param_sharded", False) for _name, parameter in named_parameters):
        return None  # the non-distributed optimizer: whole masters
    group_sizes = {name: len(replication.ranks_for(name, parameter)) for name, parameter in named_parameters}
    if all(size == 1 for size in group_sizes.values()):
        return None  # DP=1: every range is the whole parameter
    if getattr(optimizer_owned_parameter_ids, "hybrid_copies", False):
        raise RuntimeError(
            "full-parameter sync at DP>1 with the distributed optimizer needs its FP32 masters on the "
            "device: no optimizer CPU offload (HybridDeviceOptimizer)"
        )
    return {
        name: _master_range(name, parameter, views.get(id(parameter)), whole=group_sizes[name] == 1)
        for name, parameter in named_parameters
    }


def _master_range(name: str, parameter: torch.nn.Parameter, view_entry, *, whole: bool) -> MasterRange:
    if not parameter.is_contiguous():
        raise RuntimeError(f"{name!r} model parameter is not contiguous")
    if view_entry is None:
        if whole:
            raise RuntimeError(f"{name!r} has no FP32 optimizer master on the only rank that holds it")
        return MasterRange(parameter, 0, 0, None, None, False)
    view, owned, fp32 = view_entry
    if view is None:
        raise RuntimeError(f"{name!r} distributed-optimizer range is not a view of the model (quantized parameter)")
    start = _view_start(name, parameter, view)
    stop = start + view.numel()
    if not isinstance(fp32, torch.Tensor) or fp32.dtype != torch.float32 or fp32.numel() != view.numel():
        raise RuntimeError(
            f"{name!r} has no FP32 master range: full-parameter sync at DP>1 with the distributed optimizer "
            "needs FP32 main params (no precision-aware optimizer)"
        )
    if fp32.dim() != 1 or not fp32.is_contiguous():
        raise RuntimeError(f"{name!r} distributed-optimizer master is not a flat contiguous range")
    if whole and (start, stop) != (0, parameter.numel()):
        raise RuntimeError(f"{name!r} is held by one rank but its FP32 master is a partial range")
    return MasterRange(parameter, start, stop, owned, fp32, _shares_storage(fp32, parameter))


def _view_start(name: str, parameter: torch.Tensor, view: torch.Tensor) -> int:
    """Flat offset of ``view`` (Megatron's ``param.view(-1)[start:end]``) in ``parameter``."""

    start = view.storage_offset() - parameter.storage_offset()
    if (
        view.dim() != 1
        or not view.is_contiguous()
        or not _shares_storage(view, parameter)
        or not 0 <= start <= start + view.numel() <= parameter.numel()
    ):
        raise RuntimeError(f"{name!r} distributed-optimizer range is not a flat view of the parameter")
    return start


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def refuse_unsupported_args(args) -> None:
    """Refuse the Megatron options sharded masters cannot serve.

    * ``overlap_param_gather``: a parameter all-gather still pending at an
      apply would overwrite the model values the apply wrote;
    * ``num_distributed_optimizer_instances`` > 1: each instance keeps its own
      copy of the ranges, so two ranks of a replica group would hold one
      FP32 element."""

    unsupported = []
    if getattr(args, "overlap_param_gather", False):
        unsupported.append("overlap_param_gather")
    if int(getattr(args, "num_distributed_optimizer_instances", 1) or 1) > 1:
        unsupported.append("num_distributed_optimizer_instances > 1")
    if unsupported:
        raise RuntimeError(
            f"full-parameter sync at DP>1 with the distributed optimizer does not support {', '.join(unsupported)}"
        )


def check_model(rows: Iterable[tuple[str, torch.nn.Parameter]], ranges: Mapping[str, MasterRange]) -> int:
    """Model == master over this rank's range of every row; returns the rows checked."""

    count = 0
    for name, _parameter in rows:
        if not ranges[name].matches_model():
            raise RuntimeError(f"{name!r} model value differs from its optimizer master")
        count += 1
    if count < 1:
        raise RuntimeError("full-parameter model/master check is empty")
    return count


def owned_tensor_names(ranges: Mapping[str, MasterRange]) -> dict[int, str]:
    """``id(optimizer-owned tensor) -> parameter name`` for the ranges this rank steps."""

    return {id(master.owned): name for name, master in ranges.items() if master.owned is not None}


def range_writer(ranges: Mapping[str, MasterRange]) -> Callable[[str, int, torch.Tensor], None]:
    """``write(key, start, values)`` into ``ranges[key]`` (see :meth:`MasterRange.write`)."""

    return lambda key, start, values: ranges[key].write(start, values)


# -- contributions -----------------------------------------------------------------


@dataclass(frozen=True)
class FullParameterMasterContribution:
    """The FP32 master ranges one rank holds of one fragment another rank owns.

    ``spans`` are ``(fragment flat offset, numel)``, ascending and disjoint;
    ``refs`` are Ray objects of CPU FP32 tensors whose concatenation is the
    spans' values in order.  The tracking fields name the optimizer state the
    values were read at; the owner uses them only at that same state."""

    contributor: FullParameterTopology
    owner: FullParameterTopology
    fragment_id: int
    policy_version: int
    local_step_generation: int
    scheduler_num_steps: int
    spans: tuple[tuple[int, int], ...]
    refs: tuple[object, ...]

    def __post_init__(self) -> None:
        for name in ("fragment_id", "policy_version", "local_step_generation", "scheduler_num_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"full-parameter contribution {name.replace('_', ' ')} is malformed")
        if self.contributor == self.owner:
            raise ValueError("a full-parameter rank does not contribute to its own fragment")
        cursor = 0
        for item in self.spans:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
                or item[0] < cursor
                or item[1] < 1
            ):
                raise ValueError("full-parameter contribution spans are malformed")
            cursor = item[0] + item[1]
        if not self.spans or not self.refs:
            raise ValueError("full-parameter contribution is empty")

    @property
    def numel(self) -> int:
        return sum(numel for _offset, numel in self.spans)

    @property
    def tracking(self) -> Tracking:
        return self.policy_version, self.local_step_generation, self.scheduler_num_steps


def contribute(
    *,
    topology: FullParameterTopology,
    tracking: Tracking,
    replica_plan,
    ranges: Mapping[str, MasterRange],
    requests: Sequence[tuple[FullParameterTopology, tuple[int, ...] | None]],
    resident: Callable[[FullParameterTopology, int], bool] | None,
    max_chunk_numel: int,
    put: Callable[[torch.Tensor], object],
) -> tuple[FullParameterMasterContribution, ...]:
    """Store this rank's master ranges of the requested owner fragments in Ray.

    ``requests``: ``(owner topology, fragment IDs)``, None for every fragment
    of that owner this rank replicates; a requested fragment this rank holds
    nothing of is skipped.  ``resident``: when given, every contributed
    fragment must be one this rank's replicas still hold since the last export
    note or commit (a materialize reads the bytes of an earlier export)."""

    sources = {} if replica_plan is None else {source.topology: source for source in replica_plan.sources}
    policy_version, generation, scheduler_steps = tracking
    contributions = []
    for owner, fragment_ids in requests:
        source = sources.get(owner)
        if source is None:
            raise RuntimeError(f"full-parameter rank holds no replica of owner {owner.shard_id}")
        by_id = {fragment.fragment_id: fragment for fragment in source.fragments}
        wanted = source.fragments if fragment_ids is None else [by_id[f] for f in fragment_ids if f in by_id]
        for fragment in wanted:
            pieces = _local_pieces(fragment, ranges)
            if not pieces:
                continue
            if resident is not None and not resident(owner, fragment.fragment_id):
                raise RuntimeError(
                    f"full-parameter fragment {fragment.fragment_id} (owner {owner.shard_id}) is no longer "
                    "held by this rank's master ranges"
                )
            contributions.append(
                FullParameterMasterContribution(
                    contributor=topology,
                    owner=owner,
                    fragment_id=fragment.fragment_id,
                    policy_version=policy_version,
                    local_step_generation=generation,
                    scheduler_num_steps=scheduler_steps,
                    spans=tuple((offset, values.numel()) for offset, values in pieces),
                    refs=_store_pieces(pieces, max_chunk_numel, put),
                )
            )
    return tuple(contributions)


def _local_pieces(fragment, ranges: Mapping[str, MasterRange]) -> list[tuple[int, torch.Tensor]]:
    """``(fragment offset, FP32 values)`` of the master ranges this rank holds of a replica fragment."""

    pieces = []
    for copy in fragment.copies:
        if not getattr(copy, "contributes", True):
            continue  # another tensor-parallel rank's group holds the owner's master ranges
        inside = ranges[copy.name].overlap(copy.start, copy.stop)
        if inside is not None:
            low, high = inside
            pieces.append((copy.fragment_offset + low - copy.start, ranges[copy.name].values(low, high)))
    return pieces


@torch.no_grad()
def _store_pieces(
    pieces: Sequence[tuple[int, torch.Tensor]], max_chunk_numel: int, put: Callable[[torch.Tensor], object]
) -> tuple[object, ...]:
    payload = torch.empty(sum(values.numel() for _offset, values in pieces), dtype=torch.float32)
    cursor = 0
    for _offset, values in pieces:
        payload[cursor : cursor + values.numel()].copy_(values, non_blocking=False)
        cursor += values.numel()
    refs = tuple(put(part) for part in payload.split(max_chunk_numel))
    if any(reference is None for reference in refs):
        raise RuntimeError("Ray returned an empty full-parameter contribution reference")
    return refs


def contributions_by_fragment(
    contributions: Sequence[FullParameterMasterContribution],
    *,
    owner: FullParameterTopology,
    fragment_ids: Iterable[int],
    tracking: Tracking,
) -> dict[int, tuple[FullParameterMasterContribution, ...]]:
    """``contributions`` grouped by fragment, refused unless they are for
    ``owner``'s ``fragment_ids`` at the owner's ``tracking`` state."""

    expected = set(fragment_ids)
    grouped: dict[int, list[FullParameterMasterContribution]] = {}
    for contribution in contributions:
        if not isinstance(contribution, FullParameterMasterContribution):
            raise TypeError("full-parameter contribution has the wrong type")
        if contribution.owner != owner or contribution.fragment_id not in expected:
            raise RuntimeError("full-parameter contribution is for another owner or fragment")
        if contribution.tracking != tracking:
            raise RuntimeError("full-parameter contribution was read at another optimizer state")
        grouped.setdefault(contribution.fragment_id, []).append(contribution)
    return {fragment_id: tuple(values) for fragment_id, values in grouped.items()}


def fragment_sources(
    fragment: FullParameterFragmentPlan,
    *,
    numel_by_wire: Mapping[str, int],
    ranges_by_wire: Mapping[str, MasterRange],
    contributions: Sequence[FullParameterMasterContribution],
    ray_module,
) -> list[torch.Tensor]:
    """Flat FP32 tensors whose concatenation is ``fragment``'s payload.

    The owner's own master ranges and the contributed ranges must tile the
    fragment exactly once; anything else (a missing rank, two ranks claiming
    one element) is refused."""

    pieces: list[tuple[int, torch.Tensor]] = []
    offset = 0
    for wire_name, start, stop in fragment.members(numel_by_wire):
        inside = ranges_by_wire[wire_name].overlap(start, stop)
        if inside is not None:
            low, high = inside
            pieces.append((offset + low - start, ranges_by_wire[wire_name].values(low, high)))
        offset += stop - start
    for contribution in contributions:
        pieces.extend(_contributed_pieces(contribution, ray_module))
    pieces.sort(key=lambda piece: piece[0])
    cursor = 0
    for piece_offset, values in pieces:
        if piece_offset != cursor:
            raise RuntimeError(
                f"full-parameter fragment {fragment.fragment_id} is not covered exactly once by the "
                "data-parallel master ranges"
            )
        cursor += values.numel()
    if cursor != fragment.numel:
        raise RuntimeError(
            f"full-parameter fragment {fragment.fragment_id} is not covered exactly once by the data-parallel master ranges"
        )
    return [values for _offset, values in pieces]


def _contributed_pieces(contribution: FullParameterMasterContribution, ray_module) -> list[tuple[int, torch.Tensor]]:
    parts = [ray_module.get(reference) for reference in contribution.refs]
    if any(
        not isinstance(part, torch.Tensor)
        or part.device.type != "cpu"
        or part.dtype != torch.float32
        or part.ndim != 1
        or not part.is_contiguous()
        for part in parts
    ) or sum(part.numel() for part in parts) != contribution.numel:
        raise ValueError("full-parameter contribution payload is malformed")
    pieces, part_index, part_offset = [], 0, 0
    for span_offset, numel in contribution.spans:
        done = 0
        while done < numel:
            part = parts[part_index]
            count = min(numel - done, part.numel() - part_offset)
            pieces.append((span_offset + done, part[part_offset : part_offset + count]))
            done += count
            part_offset += count
            if part_offset == part.numel():
                part_index += 1
                part_offset = 0
    return pieces
