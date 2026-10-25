"""Data-parallel replicas of full-parameter FP32 masters.

At DP>1 every data-parallel rank holds each of its parameters: with the
non-distributed optimizer as a whole FP32 master, with Megatron's distributed
optimizer as the whole model parameter plus the FP32 master of one range
(``full_parameter_sharded_masters``).  Exactly one rank of each replica group
owns a parameter: only it lists the parameter in its manifest and exports it
(``full_parameter_state._replica_owners``).  Every other rank of the group
holds a replica.  This module gives a replica rank:

* a replica plan: for each owner it replicates, which members of the owner's
  fragments land in which flat ranges of its local parameters.  It is built
  once, at plan install, from the owner's manifest and fragment plan;
* a replica resident record: which owner fragments its replicas hold, by
  fragment payload hash.  An export note sets it (the replicas hold what the
  owner exported: Megatron keeps data-parallel replicas identical), a commit
  sets it (the replicas hold the committed bytes), and everything that drops
  the owner resident record drops it too;
* the prepare check and the commit copy of the owner chunks it needs.

On every apply each replica copies the same hash-checked bytes its owner
copies, so all replicas of a parameter hold the same values afterwards.
``MILES_FULL_PARAMETER_VERIFY_REPLICAS=1`` also hashes the replicas at every
export note of a content export (not after a lazy export, whose descriptors are
labels) and refuses to continue when one differs from its owner.

``full_parameter_state`` imports this module inside its functions (this module
imports ``full_parameter_state`` at the top; a top-level import both ways
would be a cycle).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterFragmentDescriptor,
    FullParameterHolding,
    FullParameterOwnerFragmentPlan,
    FullParameterShardManifest,
    FullParameterTopology,
    _REPLICA_RESIDENT_ATTRIBUTE,
    _chunk_hash,
    _hash_pool,
    _parallel_map,
    _require_full_parameter_tracking,
    _resident_skip_enabled,
    _run_now,
    _scheduler_num_steps,
    _validate_cpu_chunk,
    _validate_fragment_plan_against_manifest,
    is_deferred_full_parameter_chunk,
    validate_full_parameter_chunked_shard_state,
)

VERIFY_REPLICAS_ENV = "MILES_FULL_PARAMETER_VERIFY_REPLICAS"

_PLAN_ATTRIBUTE = "_external_full_parameter_replica_plan"
_RESIDENT_ATTRIBUTE = _REPLICA_RESIDENT_ATTRIBUTE

ReplicaFragmentKey = tuple[FullParameterTopology, int]


@dataclass(frozen=True)
class ReplicaCopy:
    """One owner-fragment member that lands in a local replica."""

    fragment_offset: int  # flat element offset of the member in the fragment payload
    name: str  # local parameter name
    start: int  # flat element range of the member inside the parameter
    stop: int
    # False: this rank writes the owner's bytes but its FP32 master range is
    # not part of the owner's (a parameter tensor parallelism does not split,
    # held by another tensor-parallel rank's data-parallel group under a
    # shape-free layout), so it contributes nothing to the owner's export.
    contributes: bool = True

    @property
    def numel(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ReplicaFragment:
    fragment_id: int
    numel: int
    copies: tuple[ReplicaCopy, ...]  # in payload order


@dataclass(frozen=True)
class ReplicaSource:
    """The fragments of one owner rank that hold members this rank replicates."""

    topology: FullParameterTopology
    owner_rank: int
    plan_hash: str
    fragments: tuple[ReplicaFragment, ...]  # by fragment ID


@dataclass(frozen=True)
class ReplicaPlan:
    parameter_layout_hash: str
    holdings: tuple[FullParameterHolding, ...]  # the local holdings the plan was built for
    sources: tuple[ReplicaSource, ...]  # by owner topology

    @property
    def topologies(self) -> tuple[FullParameterTopology, ...]:
        return tuple(source.topology for source in self.sources)


@dataclass(frozen=True)
class ReplicaResidentRecord:
    """Which owner fragments the replicas hold, valid while the tracking state is unchanged."""

    parameter_layout_hash: str
    policy_version: int
    local_step_generation: int
    scheduler_num_steps: int
    payload_hashes: Mapping[ReplicaFragmentKey, tuple[int, str]]


# -- plan ----------------------------------------------------------------------


def build_replica_plan(
    manifest: FullParameterShardManifest,
    plan: FullParameterOwnerFragmentPlan,
    replica_sources: Sequence[tuple[FullParameterShardManifest, FullParameterOwnerFragmentPlan]],
    master_ranks: Mapping[str, tuple[int, ...]] | None = None,
) -> ReplicaPlan | None:
    """The replica plan of the rank ``manifest`` describes; None when it holds no replica.

    ``master_ranks``: this rank's data-parallel group of each parameter
    (``FullParameterReplication.ranks_for``).  A replica whose owner is outside
    that group writes the owner's bytes but contributes no master range
    (``ReplicaCopy.contributes``); None: every owner is inside it."""

    replicated = manifest.replicated_holdings()
    by_owner: dict[int, dict[str, FullParameterHolding]] = {}
    for holding in replicated:
        by_owner.setdefault(holding.owner_rank, {})[holding.name] = holding
    source_ranks = [owner_manifest.global_rank for owner_manifest, _owner_plan in replica_sources]
    if sorted(source_ranks) != sorted(by_owner) or len(set(source_ranks)) != len(source_ranks):
        raise RuntimeError("full-parameter replica sources do not match the owners of this rank's replicas")
    if not replicated:
        return None
    sources = tuple(
        sorted(
            (
                _replica_source(
                    owner_manifest,
                    owner_plan,
                    by_owner[owner_manifest.global_rank],
                    plan,
                    contributing=frozenset(
                        name
                        for name in by_owner[owner_manifest.global_rank]
                        if master_ranks is None or owner_manifest.global_rank in master_ranks[name]
                    ),
                )
                for owner_manifest, owner_plan in replica_sources
            ),
            key=lambda source: source.topology,
        )
    )
    return ReplicaPlan(parameter_layout_hash=plan.parameter_layout_hash, holdings=manifest.holdings, sources=sources)


def _replica_source(
    owner_manifest: FullParameterShardManifest,
    owner_plan: FullParameterOwnerFragmentPlan,
    replicas: Mapping[str, FullParameterHolding],
    plan: FullParameterOwnerFragmentPlan,
    *,
    contributing: frozenset[str] | None = None,
) -> ReplicaSource:
    _validate_fragment_plan_against_manifest(owner_plan, owner_manifest)
    if owner_plan.parameter_layout_hash != plan.parameter_layout_hash:
        raise RuntimeError("full-parameter replica source has another global layout")
    owner_holdings = {holding.name: holding for holding in owner_manifest.holdings}
    for name, holding in replicas.items():
        if owner_holdings.get(name) != holding:
            raise RuntimeError(f"full-parameter replica {name!r} disagrees with its owner's holding")
    spec_names = {spec.wire_name: spec.name for spec in owner_manifest.specs}
    numel_by_wire = {spec.wire_name: spec.numel for spec in owner_manifest.specs}
    fragments, covered = [], {name: [] for name in replicas}
    for fragment in owner_plan.fragments:
        copies, offset = [], 0
        for wire_name, start, stop in fragment.members(numel_by_wire):
            name = spec_names[wire_name]
            if name in replicas:
                covered[name].append((start, stop))
                copies.append(
                    ReplicaCopy(
                        fragment_offset=offset,
                        name=name,
                        start=start,
                        stop=stop,
                        contributes=contributing is None or name in contributing,
                    )
                )
            offset += stop - start
        if copies:
            fragments.append(
                ReplicaFragment(fragment_id=fragment.fragment_id, numel=fragment.numel, copies=tuple(copies))
            )
    for name, holding in replicas.items():
        # The owner plan covers each parameter exactly once (checked above);
        # a canonical-order plan in order, a free-order plan in any order.
        spans = covered[name]
        cursor = 0
        for start, stop in sorted(spans) if owner_plan.free_member_order else spans:
            if start != cursor:
                raise RuntimeError(f"full-parameter replica {name!r} ranges do not tile the parameter")
            cursor = stop
        if cursor != _numel(holding.shape):
            raise RuntimeError(f"full-parameter owner plan does not cover replica {name!r}")
    return ReplicaSource(
        topology=owner_manifest.topology,
        owner_rank=owner_manifest.global_rank,
        plan_hash=owner_plan.plan_hash,
        fragments=tuple(fragments),
    )


def _numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return numel


def install_replica_plan(actor, replica_plan: ReplicaPlan | None) -> None:
    installed = getattr(actor, _PLAN_ATTRIBUTE, None)
    if installed is None:
        setattr(actor, _PLAN_ATTRIBUTE, replica_plan)
    elif installed != replica_plan:
        raise RuntimeError("full-parameter replica plan is already installed")


def require_replica_plan(actor, manifest: FullParameterShardManifest) -> ReplicaPlan | None:
    """The installed replica plan, refused when the local holdings changed since install."""

    replica_plan = getattr(actor, _PLAN_ATTRIBUTE, None)
    if replica_plan is None:
        if manifest.replicated_holdings():
            raise RuntimeError("full-parameter replica plan is not installed")
        return None
    if not isinstance(replica_plan, ReplicaPlan) or replica_plan.holdings != manifest.holdings:
        raise RuntimeError("full-parameter replica holdings changed after plan installation")
    return replica_plan


def match_replica_states(
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    policy_version: int,
    local_step_generation: int,
) -> tuple[FullParameterChunkedShardState, ...]:
    """``states`` of exactly the plan's owners, in plan order, checked against the plan."""

    expected = () if replica_plan is None else replica_plan.topologies
    by_topology = {}
    for state in states:
        validate_full_parameter_chunked_shard_state(state)
        if state.topology in by_topology:
            raise RuntimeError("full-parameter replica states repeat an owner")
        by_topology[state.topology] = state
    if set(by_topology) != set(expected):
        raise RuntimeError("full-parameter replica states do not match this rank's replica owners")
    ordered = tuple(by_topology[topology] for topology in expected)
    for source, state in zip(() if replica_plan is None else replica_plan.sources, ordered, strict=True):
        if (
            state.parameter_layout_hash != replica_plan.parameter_layout_hash
            or state.plan_hash != source.plan_hash
            or state.policy_version != policy_version
            or state.local_step_generation != local_step_generation
        ):
            raise RuntimeError("full-parameter replica state identity changed")
        descriptors = {descriptor.fragment_id: descriptor for descriptor in state.fragments}
        for fragment in source.fragments:
            descriptor = descriptors.get(fragment.fragment_id)
            if descriptor is None or descriptor.numel != fragment.numel:
                raise RuntimeError("full-parameter replica fragment identity changed")
    return ordered


# -- resident record -------------------------------------------------------------


def forget_replica_resident(actor) -> None:
    setattr(actor, _RESIDENT_ATTRIBUTE, None)


def record_replica_resident(
    actor,
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
) -> None:
    """The replicas hold every fragment of ``states`` they replicate, at the current tracking state."""

    if replica_plan is None:
        forget_replica_resident(actor)
        return
    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    hashes = {}
    for source, state in zip(replica_plan.sources, states, strict=True):
        descriptors = {descriptor.fragment_id: descriptor for descriptor in state.fragments}
        for fragment in source.fragments:
            descriptor = descriptors[fragment.fragment_id]
            hashes[(source.topology, fragment.fragment_id)] = (descriptor.numel, descriptor.payload_hash)
    setattr(
        actor,
        _RESIDENT_ATTRIBUTE,
        ReplicaResidentRecord(
            parameter_layout_hash=replica_plan.parameter_layout_hash,
            policy_version=installed,
            local_step_generation=generation,
            scheduler_num_steps=_scheduler_num_steps(actor),
            payload_hashes=hashes,
        ),
    )


def resident_replica_fragments(
    actor,
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    skip_resident: bool,
) -> tuple[frozenset[ReplicaFragmentKey], ReplicaResidentRecord | None]:
    """Replica fragments of ``states`` a prepare neither reads nor copies.

    Mirrors the owner rule: with ``skip_resident`` (and the kill switch
    open), every fragment whose payload hash the live record holds; always,
    every deferred fragment, which has no stored bytes and must be resident."""

    if replica_plan is None:
        return frozenset(), None
    record = getattr(actor, _RESIDENT_ATTRIBUTE, None)
    held = _live_resident(actor, replica_plan, record)
    resident, deferred = set(), set()
    for source, state in zip(replica_plan.sources, states, strict=True):
        rows = {
            descriptor.fragment_id: (descriptor, refs)
            for descriptor, refs in zip(state.fragments, state.chunk_refs, strict=True)
        }
        for fragment in source.fragments:
            descriptor, refs = rows[fragment.fragment_id]
            key = (source.topology, fragment.fragment_id)
            is_held = held.get(key) == (descriptor.numel, descriptor.payload_hash)
            if any(is_deferred_full_parameter_chunk(reference) for reference in refs):
                if not is_held:
                    deferred.add(key)
                    continue
                resident.add(key)
            elif is_held and skip_resident and _resident_skip_enabled():
                resident.add(key)
    if deferred:
        missing = sorted((topology.shard_id, fragment_id) for topology, fragment_id in deferred)
        raise RuntimeError(
            f"deferred full-parameter fragments {missing} are not held by this rank's replicas and have no stored bytes"
        )
    return frozenset(resident), record if resident else None


def _live_resident(actor, replica_plan: ReplicaPlan, record: object) -> Mapping[ReplicaFragmentKey, tuple[int, str]]:
    if not isinstance(record, ReplicaResidentRecord):
        return {}
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if (
        record.parameter_layout_hash != replica_plan.parameter_layout_hash
        or record.policy_version != installed
        or record.local_step_generation != generation
        or exported != generation
        or record.scheduler_num_steps != _scheduler_num_steps(actor)
    ):
        return {}
    return record.payload_hashes


def replica_holds(
    actor,
    replica_plan: ReplicaPlan | None,
    expected: Mapping[ReplicaFragmentKey, tuple[int, str]],
) -> Callable[[FullParameterTopology, int], bool]:
    """``holds(owner, fragment_id)``: the live replica record says this rank's
    replicas hold that owner fragment with exactly ``expected``'s ``(numel,
    payload hash)``."""

    live = {} if replica_plan is None else _live_resident(actor, replica_plan, getattr(actor, _RESIDENT_ATTRIBUTE, None))
    return lambda owner, fragment_id: (
        (owner, fragment_id) in expected and live.get((owner, fragment_id)) == expected[(owner, fragment_id)]
    )


def require_live_replica_record(actor, prepared_record: object, resident: frozenset[ReplicaFragmentKey]) -> None:
    if resident and getattr(actor, _RESIDENT_ATTRIBUTE, None) is not prepared_record:
        raise RuntimeError("full-parameter replica resident fragments changed after prepare")


# -- export note -------------------------------------------------------------------


def note_replica_export(
    actor,
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    replica_masters: Mapping[str, torch.Tensor] | None,
    lazy: bool = False,
) -> dict[str, int]:
    """Record that the replicas hold what their owners just exported.

    ``states`` are the owners' exported shards at this rank's tracking state.
    With ``MILES_FULL_PARAMETER_VERIFY_REPLICAS=1`` every owner chunk that lies
    wholly inside this rank's replicas is hashed from the replicas first;
    except after a ``lazy`` export, whose descriptors are labels, not content
    hashes (the replicas are then checked at the next non-lazy export).
    ``replica_masters`` None: the masters are sharded (the distributed
    optimizer at DP>1), so this rank holds no FP32 replica to compare; the
    owners exported its ranges themselves (``full_parameter_sharded_masters``)."""

    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if exported != generation:
        raise RuntimeError("full-parameter replica export note has unexported local progress")
    ordered = match_replica_states(replica_plan, states, policy_version=installed, local_step_generation=generation)
    verified = 0
    if replica_plan is not None and replica_masters is not None and not lazy and os.environ.get(VERIFY_REPLICAS_ENV, "0") == "1":
        verified = sum(
            _parallel_map(
                lambda item: _verify_replica_fragment(
                    *item, replica_masters=replica_masters, layout_hash=replica_plan.parameter_layout_hash
                ),
                [(source, state) for source, state in zip(replica_plan.sources, ordered, strict=True)],
            )
        )
    record_replica_resident(actor, replica_plan, ordered)
    fragments = 0 if replica_plan is None else sum(len(source.fragments) for source in replica_plan.sources)
    stats = {"fragments": fragments, "verified_chunks": verified, "lazy": int(lazy)}
    if replica_masters is None:
        stats["sharded"] = 1
    return stats


def _verify_replica_fragment(
    source: ReplicaSource,
    state: FullParameterChunkedShardState,
    *,
    replica_masters: Mapping[str, torch.Tensor],
    layout_hash: str,
) -> int:
    descriptors = {descriptor.fragment_id: descriptor for descriptor in state.fragments}
    verified = 0
    for fragment in source.fragments:
        for chunk_index, chunk, parts in iter_chunk_overlaps(fragment, descriptors[fragment.fragment_id]):
            if sum(stop - start for _copy, start, stop in parts) != chunk.numel:
                continue  # part of this chunk is a parameter this rank does not hold
            value = torch.empty(chunk.numel, dtype=torch.float32)
            for copy, start, stop in parts:
                value[start - chunk.flat_offset : stop - chunk.flat_offset].copy_(
                    _replica_range(copy, start, stop, replica_masters)
                )
            payload_hash = _chunk_hash(
                fragment.fragment_id,
                chunk_index,
                chunk.flat_offset,
                value,
                parameter_layout_hash=layout_hash,
                plan_hash=source.plan_hash,
            )
            if payload_hash != chunk.payload_hash:
                raise RuntimeError(
                    f"data-parallel replica of fragment {fragment.fragment_id} (owner {source.topology.shard_id}) "
                    "differs from its owner's export"
                )
            verified += 1
    return verified


# -- prepare and commit ----------------------------------------------------------------


def iter_chunk_overlaps(
    fragment: ReplicaFragment,
    descriptor: FullParameterFragmentDescriptor,
) -> Iterator[tuple[int, FullParameterChunkDescriptor, tuple[tuple[ReplicaCopy, int, int], ...]]]:
    """``(chunk index, chunk, ((copy, start, stop), ...))`` for every chunk a copy overlaps.

    ``start``/``stop`` are flat fragment offsets of the overlap; copies and
    chunks are both in payload order, so one forward sweep finds them."""

    copies = fragment.copies
    first = 0
    for chunk_index, chunk in enumerate(descriptor.chunks):
        begin, end = chunk.flat_offset, chunk.flat_offset + chunk.numel
        while first < len(copies) and copies[first].fragment_offset + copies[first].numel <= begin:
            first += 1
        parts = []
        for copy in copies[first:]:
            if copy.fragment_offset >= end:
                break
            start, stop = max(begin, copy.fragment_offset), min(end, copy.fragment_offset + copy.numel)
            if start < stop:
                parts.append((copy, start, stop))
        if parts:
            yield chunk_index, chunk, tuple(parts)


def _replica_range(
    copy: ReplicaCopy, start: int, stop: int, replica_masters: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """The slice of ``copy``'s local master that fragment offsets ``[start, stop)`` address."""

    offset = copy.start - copy.fragment_offset
    return replica_masters[copy.name].detach().reshape(-1)[offset + start : offset + stop]


def _rows_to_read(
    replica_plan: ReplicaPlan,
    states: Sequence[FullParameterChunkedShardState],
    skip: frozenset[ReplicaFragmentKey],
):
    for source, state in zip(replica_plan.sources, states, strict=True):
        rows = {
            descriptor.fragment_id: (descriptor, refs)
            for descriptor, refs in zip(state.fragments, state.chunk_refs, strict=True)
        }
        for fragment in source.fragments:
            if (source.topology, fragment.fragment_id) not in skip:
                descriptor, refs = rows[fragment.fragment_id]
                yield source, state, fragment, descriptor, refs


def verify_replica_chunks(
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    skip: frozenset[ReplicaFragmentKey],
    ray_module,
) -> int:
    """Check every owner chunk a commit will copy into a replica (finite, chunk hash); returns fragments read."""

    if replica_plan is None:
        return 0
    rows = list(_rows_to_read(replica_plan, states, skip))
    overlap_checks = len(rows) <= 1

    def verify(row) -> None:
        source, state, fragment, descriptor, refs = row
        for chunk_index, chunk, _parts in iter_chunk_overlaps(fragment, descriptor):
            (_hash_pool().submit if overlap_checks else _run_now)(
                _validate_cpu_chunk,
                ray_module.get(refs[chunk_index]),
                chunk,
                fragment_id=fragment.fragment_id,
                verify_values=True,
                parameter_layout_hash=state.parameter_layout_hash,
                plan_hash=source.plan_hash,
            ).result()

    _parallel_map(verify, rows)
    return len(rows)


@torch.no_grad()
def replica_names_to_copy(
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    skip: frozenset[ReplicaFragmentKey],
) -> tuple[str, ...]:
    """Local parameter names :func:`copy_replica_chunks` writes for these
    arguments (every member of a non-resident replica fragment), in order."""

    if replica_plan is None:
        return ()
    names: dict[str, None] = {}
    for _source, _state, fragment, _descriptor, _refs in _rows_to_read(replica_plan, states, skip):
        names.update(dict.fromkeys(copy.name for copy in fragment.copies))
    return tuple(names)


def copy_replica_chunks(
    replica_plan: ReplicaPlan | None,
    states: Sequence[FullParameterChunkedShardState],
    *,
    skip: frozenset[ReplicaFragmentKey],
    write: Callable[[str, int, torch.Tensor], None],
    ray_module,
) -> int:
    """Copy the owner bytes of every non-resident replica fragment; returns fragments copied.

    ``write(name, start, values)`` stores ``values`` at flat offsets
    ``[start, start + len(values))`` of local parameter ``name``'s master
    (``full_parameter_state._commit_writer``)."""

    if replica_plan is None:
        return 0
    copied = 0
    for source, state, fragment, descriptor, refs in _rows_to_read(replica_plan, states, skip):
        for chunk_index, chunk, parts in iter_chunk_overlaps(fragment, descriptor):
            if is_deferred_full_parameter_chunk(refs[chunk_index]):
                raise RuntimeError("a deferred full-parameter chunk has no stored bytes to copy")
            value = _validate_cpu_chunk(
                ray_module.get(refs[chunk_index]),
                chunk,
                fragment_id=fragment.fragment_id,
                verify_values=False,
                parameter_layout_hash=state.parameter_layout_hash,
                plan_hash=source.plan_hash,
            )
            for copy, start, stop in parts:
                write(
                    copy.name,
                    copy.start - copy.fragment_offset + start,
                    value[start - chunk.flat_offset : stop - chunk.flat_offset],
                )
            del value
        copied += 1
    return copied
