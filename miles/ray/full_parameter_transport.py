"""Reference-backed Ray transport for topology-owned full-parameter fragments."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from miles.backends.megatron_utils.full_parameter_state import (
    FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterCommitStatus,
    FullParameterFragmentDescriptor,
    FullParameterLocalStepReceipt,
    FullParameterOptimizerState,
    FullParameterOwnerFragmentPlan,
    FullParameterShardManifest,
    FullParameterTopology,
    _chunk_hash,
    _fragment_hasher,
    _resolve_ray_module,
    _tensor_bytes,
    _validate_commit_token,
    _all_finite,
    _run_now,
    _scan_pool,
    _same_fragment_geometry,
    _validate_cpu_chunk,
    is_deferred_full_parameter_chunk,
    put_full_parameter_chunk,
    full_parameter_chunked_shard_identity,
    validate_full_parameter_chunked_shard_state,
)


_CUT_CONTENT_IDENTITY_DOMAIN = b"miles-full-parameter-chunked-cut-content-v1\0"

# What this transport offers beyond the original protocol; callers that reach
# the group through a forwarding handle check this set in their own Miles.
#   skip_resident: ``apply_chunked_cut(skip_resident_fragments=True)``
#   deferred_export: ``export_chunked_cut(defer_payload=True)`` and
#       ``materialize_chunked_fragments``
#   optimizer_state_check_model: ``optimizer_states(check_model=False)``
#   lazy_export: ``export_chunked_cut(lazy=True)``,
#       ``materialize_chunked_fragments(return_descriptors=True)``,
#       ``iter_full_parameter_authoritative_fragment_delta_parts(local_descriptor=...)``
#       and ``record_local_step(check_model=False)``
#   zero_copy_chunks: chunks are stored in Ray as numpy buffers
#       (``put_full_parameter_chunk``); readers receive the same tensors
#   dp_replicas: data-parallel replicas (DP>1, EP spanning DP): one rank owns
#       each parameter, the other ranks of its replica group apply the
#       owner's bytes (``full_parameter_replicas``)
#   sharded_masters: data-parallel replicas under Megatron's distributed
#       optimizer, which keeps one FP32 range of each parameter per rank; the
#       other ranks contribute their ranges to an owner's export
#       (``full_parameter_sharded_masters``)
FULL_PARAMETER_TRANSPORT_FEATURES = frozenset(
    {
        "skip_resident",
        "deferred_export",
        "optimizer_state_check_model",
        "zero_copy_chunks",
        "dp_replicas",
        "sharded_masters",
        # Lazy export (labels, no master read), materialize returning the
        # content descriptors, delta reads through such a descriptor, and a
        # local-step receipt without the whole-model model/master check.
        "lazy_export",
        # An apply may name fragments whose stored chunks the caller already
        # verified (``verify_stored_fragment_chunks``); their prepare reads none.
        "verified_apply",
    }
)

# Set on the group by ``install_fragment_plans``: ``{replica rank topology:
# owner topologies}`` for every rank that holds data-parallel replicas.
REPLICA_OWNERS_ATTRIBUTE = "full_parameter_replica_owners"
# Set on the group by ``install_fragment_plans``: the topology of every rank
# when the ranks keep sharded masters (empty for whole masters).
SHARDED_MASTERS_ATTRIBUTE = "full_parameter_sharded_master_topologies"


@dataclass(frozen=True)
class FullParameterChunkedCut:
    """Complete cut descriptor whose tensor bytes remain in nested ObjectRefs."""

    policy_version: int
    local_step_generation: int
    parameter_layout_hash: str
    shards: tuple[FullParameterChunkedShardState, ...]

    def __post_init__(self) -> None:
        _validate_chunked_cut(self)


class FullParameterCommitInDoubtError(RuntimeError):
    """A policy commit began but did not reach unanimous finalization."""

    def __init__(self, commit_token: str, phase: str) -> None:
        self.commit_token = commit_token
        self.phase = phase
        super().__init__(f"full-parameter commit {commit_token!r} is in doubt during {phase}")


def _validate_chunked_cut(
    cut: FullParameterChunkedCut,
    *,
    require_refs: bool = True,
) -> FullParameterChunkedCut:
    if not isinstance(cut, FullParameterChunkedCut):
        raise TypeError("full-parameter chunked cut has the wrong type")
    if not cut.shards or cut.shards != tuple(sorted(cut.shards, key=lambda state: state.topology)):
        raise ValueError("full-parameter chunked cut shards are not canonical")
    topologies = [state.topology for state in cut.shards]
    if len(set(topologies)) != len(topologies):
        raise ValueError("full-parameter chunked cut has duplicate topologies")
    fragment_ids = []
    for state in cut.shards:
        validate_full_parameter_chunked_shard_state(
            state,
            require_refs=require_refs,
        )
        if state.policy_version != cut.policy_version or state.local_step_generation != cut.local_step_generation or state.parameter_layout_hash != cut.parameter_layout_hash:
            raise ValueError("full-parameter chunked cut mixes policy identities")
        fragment_ids.extend(fragment.fragment_id for fragment in state.fragments)
    if sorted(fragment_ids) != list(range(len(fragment_ids))):
        raise ValueError("full-parameter chunked cut is not a complete fragment sweep")
    return cut


def full_parameter_chunked_cut_identity(cut: FullParameterChunkedCut) -> str:
    """Return a stable transaction identity without resolving payload refs."""

    _validate_chunked_cut(cut)
    digest = hashlib.sha256(b"miles-full-parameter-chunked-cut-v1\0")
    for state in cut.shards:
        digest.update(full_parameter_chunked_shard_identity(state).encode("ascii"))
    return digest.hexdigest()


def full_parameter_chunked_cut_content_identity(
    cut: FullParameterChunkedCut,
) -> str:
    """Hash cut content and owner metadata independently of version labels."""

    _validate_chunked_cut(cut)
    digest = hashlib.sha256(_CUT_CONTENT_IDENTITY_DOMAIN)
    for state in cut.shards:
        normalized = replace(
            state,
            policy_version=0,
            local_step_generation=0,
        )
        digest.update(full_parameter_chunked_shard_identity(normalized).encode("ascii"))
    return digest.hexdigest()


def full_parameter_commit_token(
    cut: FullParameterChunkedCut,
    caller_nonce: str | None = None,
) -> str:
    """Derive a non-reusable actor token from cut identity and caller nonce."""

    identity = full_parameter_chunked_cut_identity(cut)
    nonce = "" if caller_nonce is None else _validate_commit_token(caller_nonce)
    digest = hashlib.sha256(b"miles-full-parameter-commit-token-v1\0")
    digest.update(identity.encode("ascii"))
    digest.update(b"\0")
    digest.update(nonce.encode("ascii"))
    return _validate_commit_token(f"fullparam:{digest.hexdigest()}")


def _fragment_row(
    cut: FullParameterChunkedCut,
    fragment_id: int,
) -> tuple[
    FullParameterChunkedShardState,
    FullParameterFragmentDescriptor,
    list[object | None],
]:
    _validate_chunked_cut(cut)
    if isinstance(fragment_id, bool) or not isinstance(fragment_id, int):
        raise TypeError("full-parameter fragment ID has the wrong type")
    matches = [
        (shard, descriptor, references)
        for shard in cut.shards
        for descriptor, references in zip(
            shard.fragments,
            shard.chunk_refs,
            strict=True,
        )
        if descriptor.fragment_id == fragment_id
    ]
    if len(matches) != 1:
        raise ValueError("full-parameter fragment has no unique owner")
    return matches[0]


def _require_stored(references: Sequence[object | None], fragment_id: int) -> None:
    if any(is_deferred_full_parameter_chunk(reference) for reference in references):
        raise RuntimeError(f"full-parameter fragment {fragment_id} is deferred: materialize it before reading its bytes")


def iter_full_parameter_fragment_parts(
    cut: FullParameterChunkedCut,
    fragment_id: int,
    *,
    ray_module=None,
):
    """Yield one canonical FP32 fragment without driver-side reassembly."""

    shard, descriptor, references = _fragment_row(cut, fragment_id)
    _require_stored(references, fragment_id)
    ray_module = _resolve_ray_module(ray_module)
    for chunk, reference in zip(descriptor.chunks, references, strict=True):
        value = _validate_cpu_chunk(
            ray_module.get(reference),
            chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=shard.parameter_layout_hash,
            plan_hash=shard.plan_hash,
        )
        yield _tensor_bytes(value)
        del value


def iter_full_parameter_fragment_delta_parts(
    anchor: FullParameterChunkedCut,
    local: FullParameterChunkedCut,
    fragment_id: int,
    *,
    ray_module=None,
):
    """Yield bounded canonical FP32 ``local - anchor`` byte parts."""

    if anchor.parameter_layout_hash != local.parameter_layout_hash or anchor.policy_version != local.policy_version or anchor.local_step_generation != 0 or local.local_step_generation != 1:
        raise ValueError("full-parameter delta cuts do not share one local base")
    anchor_shard, anchor_descriptor, anchor_refs = _fragment_row(
        anchor,
        fragment_id,
    )
    local_shard, local_descriptor, local_refs = _fragment_row(local, fragment_id)
    _require_stored(anchor_refs, fragment_id)
    _require_stored(local_refs, fragment_id)
    if anchor_shard.topology != local_shard.topology or anchor_shard.plan_hash != local_shard.plan_hash or anchor_descriptor.fragment_id != local_descriptor.fragment_id or anchor_descriptor.numel != local_descriptor.numel or len(anchor_descriptor.chunks) != len(local_descriptor.chunks):
        raise ValueError("full-parameter delta fragment identity changed")
    ray_module = _resolve_ray_module(ray_module)
    for anchor_chunk, anchor_ref, local_chunk, local_ref in zip(
        anchor_descriptor.chunks,
        anchor_refs,
        local_descriptor.chunks,
        local_refs,
        strict=True,
    ):
        if anchor_chunk.chunk_index != local_chunk.chunk_index or anchor_chunk.flat_offset != local_chunk.flat_offset or anchor_chunk.numel != local_chunk.numel:
            raise ValueError("full-parameter delta chunk identity changed")
        anchor_value = _validate_cpu_chunk(
            ray_module.get(anchor_ref),
            anchor_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=anchor_shard.parameter_layout_hash,
            plan_hash=anchor_shard.plan_hash,
        )
        local_value = _validate_cpu_chunk(
            ray_module.get(local_ref),
            local_chunk,
            fragment_id=fragment_id,
            verify_values=True,
            parameter_layout_hash=local_shard.parameter_layout_hash,
            plan_hash=local_shard.plan_hash,
        )
        delta = local_value.sub(anchor_value)
        if not _all_finite(delta):
            raise ValueError("full-parameter delta contains NaN or Inf")
        yield _tensor_bytes(delta)
        del anchor_value, local_value, delta


def iter_full_parameter_authoritative_fragment_delta_parts(
    local_cut: FullParameterChunkedCut,
    fragment_id: int,
    *,
    authoritative_parameter_layout_hash: str,
    authoritative_topology: FullParameterTopology,
    authoritative_plan_hash: str,
    authoritative_descriptor: FullParameterFragmentDescriptor,
    authoritative_refs: Sequence[object | None],
    ray_module=None,
    local_descriptor: FullParameterFragmentDescriptor | None = None,
):
    """Yield bounded FP32 ``local - authoritative`` parts for any local round.

    ``local_descriptor``: the content descriptor that materializing the
    local row returned, when the row itself carries labels (lazy cut).  It
    must have the row's geometry; the local chunks are checked against it."""

    local_shard, row_descriptor, local_refs = _fragment_row(
        local_cut,
        fragment_id,
    )
    _require_stored(local_refs, fragment_id)
    if local_descriptor is None:
        local_descriptor = row_descriptor
    elif not isinstance(local_descriptor, FullParameterFragmentDescriptor) or not _same_fragment_geometry(local_descriptor, row_descriptor):
        raise ValueError("full-parameter local content descriptor does not match its row")
    if local_cut.local_step_generation < 1:
        raise ValueError("full-parameter local cut has no optimizer progress")
    if authoritative_parameter_layout_hash != local_cut.parameter_layout_hash:
        raise ValueError("full-parameter authoritative layout identity changed")
    if not isinstance(authoritative_topology, FullParameterTopology) or authoritative_topology != local_shard.topology:
        raise ValueError("full-parameter authoritative topology identity changed")
    if authoritative_plan_hash != local_shard.plan_hash:
        raise ValueError("full-parameter authoritative plan identity changed")
    if not isinstance(
        authoritative_descriptor,
        FullParameterFragmentDescriptor,
    ):
        raise TypeError("full-parameter authoritative fragment is malformed")
    if authoritative_descriptor.fragment_id != local_descriptor.fragment_id or authoritative_descriptor.numel != local_descriptor.numel or len(authoritative_descriptor.chunks) != len(local_descriptor.chunks):
        raise ValueError("full-parameter authoritative fragment identity changed")
    if not isinstance(authoritative_refs, Sequence) or isinstance(
        authoritative_refs,
        (str, bytes, bytearray, memoryview),
    ):
        raise TypeError("full-parameter authoritative fragment refs are malformed")
    if len(authoritative_refs) != len(authoritative_descriptor.chunks) or any(reference is None for reference in authoritative_refs):
        raise ValueError("full-parameter authoritative fragment refs are incomplete")

    ray_module = _resolve_ray_module(ray_module)
    local_fragment_hash = _fragment_hasher(
        fragment_id,
        local_shard.parameter_layout_hash,
        local_shard.plan_hash,
    )
    authoritative_fragment_hash = _fragment_hasher(
        fragment_id,
        authoritative_parameter_layout_hash,
        authoritative_plan_hash,
    )
    for local_chunk, local_ref, authoritative_chunk, authoritative_ref in zip(
        local_descriptor.chunks,
        local_refs,
        authoritative_descriptor.chunks,
        authoritative_refs,
        strict=True,
    ):
        if local_chunk.chunk_index != authoritative_chunk.chunk_index or local_chunk.flat_offset != authoritative_chunk.flat_offset or local_chunk.numel != authoritative_chunk.numel:
            raise ValueError("full-parameter authoritative chunk identity changed")
        # Shape checks first; then the two value checks (finite + chunk hash)
        # and the two fragment digests run concurrently with the subtraction
        # (four SHA-256 streams; hashlib releases the GIL).  Every check ends
        # before the part is yielded, in the original order of the errors.
        local_value = _validate_cpu_chunk(
            ray_module.get(local_ref),
            local_chunk,
            fragment_id=fragment_id,
            verify_values=False,
            parameter_layout_hash=local_shard.parameter_layout_hash,
            plan_hash=local_shard.plan_hash,
        )
        authoritative_value = _validate_cpu_chunk(
            ray_module.get(authoritative_ref),
            authoritative_chunk,
            fragment_id=fragment_id,
            verify_values=False,
            parameter_layout_hash=authoritative_parameter_layout_hash,
            plan_hash=authoritative_plan_hash,
        )
        # Small chunks (tests, tiny fragments): a thread handoff costs more
        # than the hashing it would overlap.
        pool = _scan_pool() if local_value.numel() >= _PARALLEL_CHECK_MIN_NUMEL else _INLINE
        checks = [
            pool.submit(
                _validate_cpu_chunk,
                local_value,
                local_chunk,
                fragment_id=fragment_id,
                verify_values=True,
                parameter_layout_hash=local_shard.parameter_layout_hash,
                plan_hash=local_shard.plan_hash,
            ),
            pool.submit(
                _validate_cpu_chunk,
                authoritative_value,
                authoritative_chunk,
                fragment_id=fragment_id,
                verify_values=True,
                parameter_layout_hash=authoritative_parameter_layout_hash,
                plan_hash=authoritative_plan_hash,
            ),
            pool.submit(local_fragment_hash.update, _tensor_bytes(local_value)),
            pool.submit(authoritative_fragment_hash.update, _tensor_bytes(authoritative_value)),
        ]
        delta = local_value.sub(authoritative_value)
        failure = None
        for check in checks:
            try:
                check.result()
            except BaseException as error:  # noqa: BLE001 - raised after every check ended
                failure = failure or error
        if failure is not None:
            raise failure
        if not _all_finite(delta):
            raise ValueError("full-parameter authoritative delta contains NaN or Inf")
        yield _tensor_bytes(delta)
        del local_value, authoritative_value, delta
    if local_fragment_hash.hexdigest() != local_descriptor.payload_hash:
        raise ValueError("full-parameter local fragment hash changed")
    if authoritative_fragment_hash.hexdigest() != authoritative_descriptor.payload_hash:
        raise ValueError("full-parameter authoritative fragment hash changed")


_PARALLEL_CHECK_MIN_NUMEL = 1 << 20  # 4 MiB of f32


class _Inline:
    """``submit`` that runs at once (same interface as the scan pool)."""

    @staticmethod
    def submit(function, *args, **kwargs):
        return _run_now(function, *args, **kwargs)


_INLINE = _Inline()


def store_full_parameter_fragment_payload(
    reference_cut: FullParameterChunkedCut,
    fragment_id: int,
    payload: bytes | bytearray | memoryview,
    *,
    ray_module=None,
) -> tuple[FullParameterFragmentDescriptor, list[object | None]]:
    """Store one bounded authoritative fragment as immutable Ray chunks."""

    shard, reference, _references = _fragment_row(reference_cut, fragment_id)
    try:
        raw = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("full-parameter fragment payload is not contiguous bytes") from exc
    if raw.nbytes != reference.numel * 4:
        raw.release()
        raise ValueError("full-parameter fragment payload byte count changed")
    ray_module = _resolve_ray_module(ray_module)
    fragment_digest = _fragment_hasher(
        fragment_id,
        shard.parameter_layout_hash,
        shard.plan_hash,
    )
    descriptors = []
    references: list[object | None] = []
    try:
        for expected in reference.chunks:
            start = expected.flat_offset * 4
            stop = start + expected.numel * 4
            view = raw[start:stop]
            value = torch.frombuffer(view, dtype=torch.float32).clone().contiguous()
            view.release()
            if not _all_finite(value):
                raise ValueError("full-parameter fragment contains NaN or Inf")
            fragment_digest.update(_tensor_bytes(value))
            descriptor = FullParameterChunkDescriptor(
                chunk_index=expected.chunk_index,
                flat_offset=expected.flat_offset,
                numel=expected.numel,
                payload_hash=_chunk_hash(
                    fragment_id,
                    expected.chunk_index,
                    expected.flat_offset,
                    value,
                    parameter_layout_hash=shard.parameter_layout_hash,
                    plan_hash=shard.plan_hash,
                ),
            )
            stored = put_full_parameter_chunk(ray_module, value)
            if stored is None:
                raise RuntimeError("Ray returned an empty fragment reference")
            descriptors.append(descriptor)
            references.append(stored)
            del value
    finally:
        raw.release()
    return (
        FullParameterFragmentDescriptor(
            fragment_id=fragment_id,
            numel=reference.numel,
            payload_hash=fragment_digest.hexdigest(),
            chunks=tuple(descriptors),
        ),
        references,
    )


def promote_full_parameter_chunked_cut(
    cut: FullParameterChunkedCut,
    target_policy_version: int,
) -> FullParameterChunkedCut:
    """Promote immutable payload refs to a newer committed global version."""

    _validate_chunked_cut(cut)
    if isinstance(target_policy_version, bool) or not isinstance(target_policy_version, int) or target_policy_version <= cut.policy_version:
        raise ValueError("promoted full-parameter cut must advance policy version")
    shards = tuple(
        replace(
            state,
            policy_version=target_policy_version,
            local_step_generation=0,
            chunk_refs=[list(references) for references in state.chunk_refs],
        )
        for state in cut.shards
    )
    return FullParameterChunkedCut(
        policy_version=target_policy_version,
        local_step_generation=0,
        parameter_layout_hash=cut.parameter_layout_hash,
        shards=shards,
    )


def assemble_full_parameter_chunked_target(
    reference_cut: FullParameterChunkedCut,
    *,
    target_policy_version: int,
    authoritative_fragments: Mapping[
        int,
        tuple[FullParameterFragmentDescriptor, Sequence[object | None]],
    ],
) -> FullParameterChunkedCut:
    """Build a target cut by reindexing authoritative payload refs only.

    ``reference_cut`` fixes the global layout, owner topology, owner plan, and
    canonical fragment order.  The authoritative mapping may select each
    fragment descriptor/ref row from a different local cut or synchronizer
    output, without resolving any tensor onto the orchestration driver.
    """

    _validate_chunked_cut(reference_cut)
    if isinstance(target_policy_version, bool) or not isinstance(target_policy_version, int) or target_policy_version <= reference_cut.policy_version:
        raise ValueError("assembled full-parameter cut must advance policy version")
    if not isinstance(authoritative_fragments, Mapping):
        raise TypeError("authoritative full-parameter fragments must be a mapping")
    if any(isinstance(fragment_id, bool) or not isinstance(fragment_id, int) or fragment_id < 0 for fragment_id in authoritative_fragments):
        raise ValueError("authoritative full-parameter fragment IDs are malformed")

    expected_ids = {fragment.fragment_id for shard in reference_cut.shards for fragment in shard.fragments}
    if set(authoritative_fragments) != expected_ids:
        raise ValueError("authoritative full-parameter fragments do not exactly cover the cut")

    target_shards = []
    for shard in reference_cut.shards:
        fragments = []
        chunk_refs = []
        for expected in shard.fragments:
            payload = authoritative_fragments[expected.fragment_id]
            if not isinstance(payload, tuple) or len(payload) != 2 or not isinstance(payload[0], FullParameterFragmentDescriptor) or not isinstance(payload[1], Sequence):
                raise TypeError("authoritative full-parameter fragment is malformed")
            descriptor, references = payload
            if descriptor.fragment_id != expected.fragment_id or descriptor.numel != expected.numel:
                raise ValueError("authoritative full-parameter fragment changed owner-plan identity")
            copied_references = list(references)
            if len(copied_references) != len(descriptor.chunks) or any(reference is None for reference in copied_references):
                raise ValueError("authoritative full-parameter fragment refs are incomplete")
            fragments.append(descriptor)
            chunk_refs.append(copied_references)
        target_shards.append(
            replace(
                shard,
                policy_version=target_policy_version,
                local_step_generation=0,
                fragments=tuple(fragments),
                chunk_refs=chunk_refs,
            )
        )
    return FullParameterChunkedCut(
        policy_version=target_policy_version,
        local_step_generation=0,
        parameter_layout_hash=reference_cut.parameter_layout_hash,
        shards=tuple(target_shards),
    )


def release_full_parameter_chunked_cut(cut: FullParameterChunkedCut) -> int:
    """Drop only this caller's nested refs and let Ray ownership release them."""

    if not isinstance(cut, FullParameterChunkedCut):
        raise TypeError("full-parameter chunked cut has the wrong type")
    released = 0
    for state in cut.shards:
        for references in state.chunk_refs:
            for index, reference in enumerate(references):
                if reference is not None:
                    references[index] = None
                    released += 1
    return released


# -- group layer -------------------------------------------------------------
#
# The trainer controller reaches every rank through ``run_plugin``; one plugin
# call returns one value per rank (an exception when that rank failed), in
# cell then rank order.


async def _rank_call(
    group,
    operation: str,
    *,
    by_topology: Mapping[FullParameterTopology, Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> list[object]:
    from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH

    results = await group.run_plugin(
        PLUGIN_PATH,
        {"operation": operation, "kwargs": kwargs, "by_topology": by_topology},
    )
    if not results:
        raise RuntimeError("full-parameter transport reached no trainer rank")
    return list(results)


async def _rank_values(group, operation: str, expected: type, message: str, **kwargs: Any) -> list[Any]:
    """One healthy ``expected`` value per rank with unique topologies, or raise."""

    results = await _rank_call(group, operation, **kwargs)
    failure = next((result for result in results if isinstance(result, BaseException)), None)
    if failure is not None:
        raise RuntimeError(message) from failure
    if any(not isinstance(result, expected) for result in results):
        raise RuntimeError(message)
    topologies = [result.topology for result in results]
    if len(set(topologies)) != len(topologies):
        raise RuntimeError(f"{message}: duplicate topologies")
    return sorted(results, key=lambda result: result.topology)


async def _group_topologies(group) -> list[FullParameterTopology]:
    topologies = await _rank_call(group, "full_parameter_topology")
    if any(not isinstance(topology, FullParameterTopology) for topology in topologies):
        raise RuntimeError("Megatron returned an invalid full-parameter topology")
    if len(set(topologies)) != len(topologies):
        raise RuntimeError("Megatron returned duplicate full-parameter topologies")
    return topologies


async def shard_manifests(group) -> tuple[FullParameterShardManifest, ...]:
    manifests = tuple(
        await _rank_values(
            group,
            "full_parameter_shard_manifest",
            FullParameterShardManifest,
            "Megatron returned incomplete full-parameter manifests",
        )
    )
    replica_owner_manifests(manifests)
    return manifests


def replica_owner_manifests(
    manifests: Sequence[FullParameterShardManifest],
) -> dict[FullParameterTopology, tuple[FullParameterShardManifest, ...]]:
    """``{replica rank topology: owner manifests}``; checks every parameter has exactly one owner.

    Each holding must name an owner rank of the group that holds the same
    parameter (same shape and replica group) and owns it, and every rank of
    that replica group must hold it with the same owner.  The owners' specs
    then list every parameter of the group exactly once.  Empty when no rank
    holds a data-parallel replica (DP=1)."""

    if not any(manifest.replicated_holdings() for manifest in manifests):
        return {}
    by_rank = {manifest.global_rank: manifest for manifest in manifests}
    if len(by_rank) != len(manifests):
        raise RuntimeError("full-parameter manifests repeat a global rank")
    holdings = {(manifest.global_rank, holding.name): holding for manifest in manifests for holding in manifest.holdings}
    for manifest in manifests:
        for holding in manifest.holdings:
            if holdings.get((holding.owner_rank, holding.name)) != holding:
                raise RuntimeError(f"full-parameter replica {holding.name!r} disagrees with its owner rank {holding.owner_rank}")
            if holding.owner_rank == manifest.global_rank and any(
                holdings.get((rank, holding.name)) != holding for rank in holding.replica_ranks
            ):
                raise RuntimeError(f"a replica rank of {holding.name!r} does not hold it with the same owner")
    return {
        manifest.topology: tuple(
            sorted({by_rank[holding.owner_rank].topology: by_rank[holding.owner_rank] for holding in manifest.replicated_holdings()}.values(), key=lambda owner: owner.topology)
        )
        for manifest in manifests
        if manifest.replicated_holdings()
    }


def _replica_owners(group) -> Mapping[FullParameterTopology, tuple[FullParameterTopology, ...]]:
    return getattr(group, REPLICA_OWNERS_ATTRIBUTE, None) or {}


def _validate_group_plans(
    plans: tuple[FullParameterOwnerFragmentPlan, ...],
) -> dict[FullParameterTopology, FullParameterOwnerFragmentPlan]:
    if not plans or any(not isinstance(plan, FullParameterOwnerFragmentPlan) for plan in plans):
        raise ValueError("full-parameter owner plans are empty or malformed")
    by_topology = {plan.topology: plan for plan in plans}
    if len(by_topology) != len(plans):
        raise ValueError("full-parameter owner plans contain duplicate topologies")
    layout_hashes = {plan.parameter_layout_hash for plan in plans}
    if len(layout_hashes) != 1:
        raise ValueError("full-parameter owner plans mix global layouts")
    fragment_ids = [fragment.fragment_id for plan in plans for fragment in plan.fragments]
    if sorted(fragment_ids) != list(range(len(fragment_ids))):
        raise ValueError("full-parameter owner plans are not a complete fragment sweep")
    return by_topology


async def install_fragment_plans(
    group,
    plans: tuple[FullParameterOwnerFragmentPlan, ...],
) -> int:
    by_topology = _validate_group_plans(tuple(plans))
    manifests = await shard_manifests(group)
    topologies = await _group_topologies(group)
    if set(topologies) != set(by_topology) or {manifest.topology for manifest in manifests} != set(by_topology):
        raise RuntimeError("full-parameter owner plans do not match the actor group")
    owner_manifests = replica_owner_manifests(manifests)
    sharded = _sharded_master_topologies(manifests)
    arguments = {topology: {"plan": plan} for topology, plan in by_topology.items()}
    for topology, owners in owner_manifests.items():
        arguments[topology]["replica_sources"] = tuple((owner, by_topology[owner.topology]) for owner in owners)
    # Cleared first: a failed install leaves no routing to stale replica plans.
    setattr(group, REPLICA_OWNERS_ATTRIBUTE, {})
    setattr(group, SHARDED_MASTERS_ATTRIBUTE, frozenset())
    results = await _rank_call(group, "install_full_parameter_fragment_plan", by_topology=arguments)
    expected = [len(by_topology[topology].fragments) for topology in topologies]
    if any(
        isinstance(result, bool) or not isinstance(result, int) or result != count
        for result, count in zip(results, expected, strict=True)
    ):
        raise RuntimeError("Megatron ranks rejected full-parameter owner plans")
    setattr(
        group,
        REPLICA_OWNERS_ATTRIBUTE,
        {topology: tuple(owner.topology for owner in owners) for topology, owners in owner_manifests.items()},
    )
    setattr(group, SHARDED_MASTERS_ATTRIBUTE, sharded)
    return sum(results)


def _sharded_master_topologies(manifests: Sequence[FullParameterShardManifest]) -> frozenset[FullParameterTopology]:
    """Every rank's topology when the ranks keep sharded masters, else empty.

    All ranks of a group decide alike (``full_parameter_sharded_masters.master_ranges``);
    a mixed group is refused."""

    flags = {manifest.sharded_masters for manifest in manifests}
    if len(flags) != 1:
        raise RuntimeError("full-parameter ranks disagree on sharded FP32 masters")
    return frozenset(manifest.topology for manifest in manifests) if flags == {True} else frozenset()


async def _master_contributions(
    group,
    fragments_by_owner: Mapping[FullParameterTopology, tuple[int, ...] | None],
    *,
    held: Sequence[tuple[FullParameterTopology, int, int, str]] | None = None,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
) -> dict[FullParameterTopology, tuple[object, ...]]:
    """``{owner topology: contributions}``: the master ranges every replica of
    an owner stores in Ray for ``fragments_by_owner[owner]`` (None: all its
    fragments).  Empty for whole masters.  ``held``: see
    ``full_parameter_master_contributions``."""

    topologies = getattr(group, SHARDED_MASTERS_ATTRIBUTE, None) or frozenset()
    if not topologies or not fragments_by_owner:
        return {}
    owners = _replica_owners(group)
    results = await _rank_call(
        group,
        "full_parameter_master_contributions",
        by_topology={
            topology: {
                "requests": tuple(
                    (owner, fragments_by_owner[owner]) for owner in owners.get(topology, ()) if owner in fragments_by_owner
                )
            }
            for topology in topologies
        },
        max_chunk_bytes=max_chunk_bytes,
        **({} if held is None else {"held": tuple(held)}),
    )
    failure = next((result for result in results if isinstance(result, BaseException)), None)
    if failure is not None or any(not isinstance(result, tuple) for result in results):
        raise RuntimeError("Megatron replica ranks refused to contribute full-parameter master ranges") from failure
    by_owner: dict[FullParameterTopology, list[object]] = {}
    for contribution in (item for result in results for item in result):
        by_owner.setdefault(contribution.owner, []).append(contribution)
    return {owner: tuple(values) for owner, values in by_owner.items()}


async def export_chunked_cut(
    group,
    policy_version: int,
    local_step_generation: int = 0,
    *,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    defer_payload: bool = False,
    lazy: bool = False,
) -> FullParameterChunkedCut:
    """``defer_payload``: descriptors only; see ``materialize_chunked_fragments``.

    ``lazy`` (with ``defer_payload``): no rank reads its masters; descriptors
    carry labels (``export_full_parameter_chunked_shard``).

    Sharded masters (the distributed optimizer at DP>1): unless ``lazy``, the
    replicas first contribute their master ranges of every owner fragment and
    each owner exports with them."""

    sharded = () if lazy else getattr(group, SHARDED_MASTERS_ATTRIBUTE, None) or ()
    routed = {}
    if sharded:
        contributions = await _master_contributions(
            group, dict.fromkeys(sharded), max_chunk_bytes=max_chunk_bytes
        )
        routed = {"by_topology": {topology: {"contributions": contributions.get(topology, ())} for topology in sharded}}
    ordered = tuple(
        await _rank_values(
            group,
            "export_full_parameter_chunked_shard",
            FullParameterChunkedShardState,
            "Megatron returned an incomplete chunked cut",
            policy_version=policy_version,
            local_step_generation=local_step_generation,
            max_chunk_bytes=max_chunk_bytes,
            **({"defer_payload": True} if defer_payload else {}),
            **({"lazy": True} if lazy else {}),
            **routed,
        )
    )
    layout_hashes = {state.parameter_layout_hash for state in ordered}
    if len(layout_hashes) != 1:
        raise RuntimeError("Megatron chunked shards mix global layouts")
    await _note_replica_export(group, ordered, lazy=lazy)
    return FullParameterChunkedCut(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        parameter_layout_hash=next(iter(layout_hashes)),
        shards=ordered,
    )


async def _note_replica_export(group, states: Sequence[FullParameterChunkedShardState], *, lazy: bool = False) -> None:
    """Tell every replica rank what its owners exported (see ``note_full_parameter_replica_export``).

    ``lazy``: the export carried labels, so the replicas are not hashed against it."""

    owners = _replica_owners(group)
    if not owners:
        return
    by_topology = {state.topology: state for state in states}
    results = await _rank_call(
        group,
        "note_full_parameter_replica_export",
        by_topology={
            topology: {"states": tuple(by_topology[owner] for owner in owners.get(topology, ()))}
            for topology in by_topology
        },
        **({"lazy": True} if lazy else {}),
    )
    failure = _first_failure(results, expected_positive=False)
    if failure is not None or any(isinstance(result, bool) or not isinstance(result, int) for result in results):
        raise RuntimeError("Megatron replica ranks refused the full-parameter export note") from failure


async def materialize_chunked_fragments(
    group,
    cut: FullParameterChunkedCut,
    fragment_ids: Sequence[int],
    *,
    return_descriptors: bool = False,
) -> dict[int, object]:
    """Store deferred fragments of ``cut`` from the owner ranks' masters.

    Returns ``{fragment_id: chunk refs}`` for exactly ``fragment_ids``.  Each
    owner rank refuses a fragment its masters no longer hold, and checks the
    stored bytes against the cut's descriptor (same hashes).  The caller puts
    the refs into its cut rows; the cut identity does not change.

    ``return_descriptors``: ``{fragment_id: (content descriptor, chunk
    refs)}``.  Required for a lazy cut, whose rows carry labels: the caller
    reads the stored chunks through the returned descriptor."""

    _validate_chunked_cut(cut)
    wanted = list(dict.fromkeys(fragment_ids))
    if len(wanted) != len(fragment_ids) or any(isinstance(f, bool) or not isinstance(f, int) for f in wanted):
        raise ValueError("full-parameter materialize fragment IDs are malformed")
    if not wanted:
        return {}
    rows = {descriptor.fragment_id: (state.topology, descriptor) for state in cut.shards for descriptor in state.fragments}
    if any(fragment_id not in rows for fragment_id in wanted):
        raise ValueError("full-parameter materialize names a fragment outside the cut")
    by_topology = {
        state.topology: {
            "state": state,
            "fragment_ids": tuple(sorted(f for f in wanted if rows[f][0] == state.topology)),
        }
        for state in cut.shards
    }
    topologies = await _group_topologies(group)
    if set(topologies) != set(by_topology):
        raise RuntimeError("full-parameter chunked cut does not match the actor group")
    contributions = await _master_contributions(
        group,
        {topology: arguments["fragment_ids"] for topology, arguments in by_topology.items() if arguments["fragment_ids"]},
        held=tuple(
            (rows[fragment_id][0], fragment_id, rows[fragment_id][1].numel, rows[fragment_id][1].payload_hash)
            for fragment_id in sorted(wanted)
        ),
    )
    for topology, values in contributions.items():
        by_topology[topology]["contributions"] = values
    results = await _rank_call(
        group,
        "materialize_full_parameter_fragments",
        by_topology=by_topology,
        **({"return_descriptors": True} if return_descriptors else {}),
    )
    stored: dict[int, object] = {}
    for topology, result in zip(topologies, results, strict=True):
        if isinstance(result, BaseException):
            raise RuntimeError("Megatron rank refused to materialize full-parameter fragments") from result
        expected = by_topology[topology]["fragment_ids"]
        if not isinstance(result, tuple) or tuple(row[0] for row in result) != expected:
            raise RuntimeError("Megatron materialized the wrong full-parameter fragments")
        for row in result:
            fragment_id, references = row[0], row[1]
            descriptor = rows[fragment_id][1]
            if not isinstance(references, list) or len(references) != len(descriptor.chunks) or any(reference is None for reference in references):
                raise RuntimeError("Megatron materialized incomplete full-parameter chunk refs")
            if not return_descriptors:
                stored[fragment_id] = references
                continue
            content = row[2] if len(row) == 3 else None
            if not isinstance(content, FullParameterFragmentDescriptor) or not _same_fragment_geometry(content, descriptor):
                raise RuntimeError("Megatron returned a full-parameter fragment descriptor of another geometry")
            stored[fragment_id] = (content, references)
    return stored


async def abort_prepared_chunked_cut(group, commit_token: str) -> tuple[object, ...]:
    token = _validate_commit_token(commit_token)
    return tuple(await _rank_call(group, "abort_prepared_full_parameter_shard", commit_token=token))


def _first_failure(results: list[object], *, expected_positive: bool) -> BaseException | None:
    for result in results:
        if isinstance(result, BaseException):
            return result
        if expected_positive and (isinstance(result, bool) or not isinstance(result, int) or result < 1):
            return RuntimeError("Megatron returned an invalid full-parameter count")
    return None


async def apply_chunked_cut(
    group,
    cut: FullParameterChunkedCut,
    *,
    commit_token: str | None = None,
    max_commit_attempts: int = 3,
    skip_resident_fragments: bool = False,
    verified_fragments: Mapping[int, str] | None = None,
) -> int:
    """Prepare, idempotently converge, then finalize every owner rank.

    ``skip_resident_fragments``: ``cut`` is the last exported or committed
    cut with some fragments replaced; each rank reads and copies only the
    fragments whose bytes its masters do not already hold (see
    ``prepare_full_parameter_chunked_shard``).  The applied values are the
    same either way.

    ``verified_fragments``: ``{fragment ID: payload hash}`` of fragments of
    ``cut`` whose stored chunks the caller already checked with
    ``verify_stored_fragment_chunks``; no rank reads them in its prepare.
    The applied values are the same either way."""

    _validate_chunked_cut(cut)
    if cut.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    token = full_parameter_commit_token(cut, commit_token)
    _validate_attempts(max_commit_attempts, "commit")
    by_topology = {state.topology: state for state in cut.shards}
    topologies = await _group_topologies(group)
    if len(by_topology) != len(topologies) or set(topologies) != set(by_topology):
        raise RuntimeError("full-parameter chunked cut does not match the actor group")

    owners = _replica_owners(group)
    prepare_results = await _rank_call(
        group,
        "prepare_full_parameter_chunked_shard",
        by_topology={
            topology: {
                "state": state,
                **({"replica_states": tuple(by_topology[owner] for owner in owners[topology])} if topology in owners else {}),
            }
            for topology, state in by_topology.items()
        },
        commit_token=token,
        **({"skip_resident": True} if skip_resident_fragments else {}),
        **({"verified_fragments": dict(verified_fragments)} if verified_fragments else {}),
    )
    prepare_failure = _first_failure(prepare_results, expected_positive=True)
    if prepare_failure is not None:
        await _abort_after_failed_prepare(group, token, by_topology, prepare_failure)

    commit_results = await _converge(group, "commit_prepared_full_parameter_shard", token, max_commit_attempts, phase="commit")
    if commit_results != prepare_results:
        raise FullParameterCommitInDoubtError(token, "count-reconciliation")
    finalize_results = await _converge(group, "finalize_full_parameter_commit", token, max_commit_attempts, phase="finalize")
    if finalize_results != commit_results:
        raise FullParameterCommitInDoubtError(token, "final-count-reconciliation")
    return sum(finalize_results)


def _validate_attempts(attempts: int, name: str) -> None:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 8:
        raise ValueError(f"full-parameter {name} retry budget is invalid")


async def _abort_after_failed_prepare(
    group,
    token: str,
    by_topology: Mapping[FullParameterTopology, FullParameterChunkedShardState],
    prepare_failure: BaseException,
) -> None:
    """Abort only when every rank is provably ABSENT or PREPARED; never returns."""

    status_results = await _rank_call(
        group,
        "full_parameter_commit_status",
        by_topology={
            topology: {
                "policy_version": state.policy_version,
                "state_identity_hash": full_parameter_chunked_shard_identity(state),
            }
            for topology, state in by_topology.items()
        },
        commit_token=token,
    )
    safe_to_abort = all(
        isinstance(status, FullParameterCommitStatus) and status.status in {"ABSENT", "PREPARED"}
        for status in status_results
    )
    if not safe_to_abort:
        raise FullParameterCommitInDoubtError(token, "prepare-reconciliation")
    abort_results = await abort_prepared_chunked_cut(group, token)
    abort_safe = all(
        type(result) is bool and result == (status.status == "PREPARED")
        for result, status in zip(abort_results, status_results, strict=True)
    )
    if not abort_safe:
        raise FullParameterCommitInDoubtError(token, "prepare-abort-race")
    raise RuntimeError("Megatron rank rejected full-parameter prepare") from prepare_failure


async def _converge(group, operation: str, token: str, attempts: int, *, phase: str) -> list[object]:
    """Retry an idempotent commit phase until every rank succeeds.

    Never abort after commit begins: a successful rank may already expose the
    target, so the exact token must be reconciled or a checkpoint restored.
    """

    for _attempt in range(attempts):
        results = await _rank_call(group, operation, commit_token=token)
        if _first_failure(results, expected_positive=True) is None:
            return results
    raise FullParameterCommitInDoubtError(token, phase)


async def record_local_step(
    group,
    *,
    base_policy_version: int,
    rollout_id: int,
    max_receipt_attempts: int = 3,
    check_model: bool = True,
) -> tuple[FullParameterLocalStepReceipt, ...]:
    """Record exact successful scheduler progress on every rank.

    ``check_model=False``: skip each rank's whole-model model/master check
    (only before a lazy export; see ``record_full_parameter_local_step``)."""

    _validate_attempts(max_receipt_attempts, "receipt")
    for _attempt in range(max_receipt_attempts):
        results = await _rank_call(
            group,
            "record_full_parameter_local_step",
            base_policy_version=base_policy_version,
            rollout_id=rollout_id,
            **({} if check_model else {"check_model": False}),
        )
        if all(isinstance(receipt, FullParameterLocalStepReceipt) for receipt in results):
            break
    else:
        raise RuntimeError("Megatron local-step receipt is in doubt; learner must fail stop")
    if len({receipt.topology for receipt in results}) != len(results):
        raise RuntimeError("Megatron returned duplicate local-step receipts")
    progress = {
        (
            receipt.role,
            receipt.base_policy_version,
            receipt.local_step_generation,
            receipt.rollout_id,
            receipt.optimizer_steps,
            receipt.scheduler_start_steps,
            receipt.scheduler_end_steps,
        )
        for receipt in results
    }
    if len(progress) != 1:
        raise RuntimeError("Megatron ranks disagree on local optimizer progress")
    return tuple(sorted(results, key=lambda receipt: receipt.topology))


async def optimizer_states(group, *, check_model: bool = True) -> tuple[FullParameterOptimizerState, ...]:
    """One bounded optimizer-progress proof from every rank.

    ``check_model=False``: skip each rank's whole-model model/master
    comparison (see ``full_parameter_optimizer_state``)."""

    results = await _rank_values(
        group,
        "full_parameter_optimizer_state",
        FullParameterOptimizerState,
        "Megatron returned incomplete optimizer-state proofs",
        **({} if check_model else {"check_model": False}),
    )
    progress = {
        (state.installed_policy_version, state.last_rollout_id, state.scheduler_num_steps)
        for state in results
    }
    if len(progress) != 1:
        raise RuntimeError("Megatron ranks disagree on optimizer progress")
    return tuple(results)
