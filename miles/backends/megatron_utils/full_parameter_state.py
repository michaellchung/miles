"""Topology-bound FP32 parameter shards for external DiLoCo synchronization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import torch

from miles.backends.megatron_utils import full_parameter_sharded_masters as sharded
from miles.backends.megatron_utils.full_parameter_chunk_reader import CommitChunkReader, prefetch_enabled


_ROLE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_PARAMETER_NAME = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,511}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}\Z")

FULL_PARAMETER_DEFAULT_CHUNK_BYTES = 256 << 20
FULL_PARAMETER_MAX_CHUNK_BYTES = 1 << 30
_FP32_BYTES = 4

_FRAGMENT_HASH_DOMAIN = b"miles-full-parameter-fragment-v1\0"
_CHUNK_HASH_DOMAIN = b"miles-full-parameter-chunk-v1\0"
# Labels of a lazy export (see ``_lazy_fragment_descriptor``).  Their own
# domains keep a label from ever equalling a content hash.
_LAZY_CHUNK_DOMAIN = b"miles-full-parameter-lazy-chunk-v1\0"
_LAZY_FRAGMENT_DOMAIN = b"miles-full-parameter-lazy-fragment-v1\0"
_SHARD_IDENTITY_DOMAIN = b"miles-full-parameter-chunked-shard-identity-v1\0"


@dataclass(frozen=True, order=True)
class FullParameterTopology:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    ep_rank: int
    ep_size: int
    cp_rank: int
    cp_size: int
    dp_rank: int
    dp_size: int

    def __post_init__(self) -> None:
        for prefix in ("tp", "pp", "ep", "cp", "dp"):
            rank = getattr(self, f"{prefix}_rank")
            size = getattr(self, f"{prefix}_size")
            if isinstance(rank, bool) or not isinstance(rank, int) or isinstance(size, bool) or not isinstance(size, int) or size < 1 or not 0 <= rank < size:
                raise ValueError(f"invalid {prefix} topology coordinates")

    @property
    def shard_id(self) -> str:
        return ".".join(f"{prefix}{getattr(self, f'{prefix}_rank')}-of-{getattr(self, f'{prefix}_size')}" for prefix in ("tp", "pp", "ep", "cp", "dp"))


@dataclass(frozen=True, order=True)
class FullParameterShardSpec:
    role: str
    shard_id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        if not self.shard_id or "::" in self.shard_id:
            raise ValueError("invalid full-parameter shard identity")
        if not _PARAMETER_NAME.fullmatch(self.name) or "::" in self.name:
            raise ValueError(f"invalid full-parameter name: {self.name!r}")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"invalid shape for {self.wire_name!r}")
        if self.dtype != "float32":
            raise ValueError("full-parameter synchronization requires FP32 masters")
        if math.prod(self.shape) != self.numel:
            raise ValueError(f"shape/numel mismatch for {self.wire_name!r}")

    @property
    def wire_name(self) -> str:
        return f"{self.role}::{self.shard_id}::{self.name}"


@dataclass(frozen=True)
class FullParameterShardState:
    policy_version: int
    topology: FullParameterTopology
    layout_hash: str
    specs: tuple[FullParameterShardSpec, ...]
    tensors: Mapping[str, torch.Tensor]
    local_step_generation: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("full-parameter policy version must be non-negative")
        if isinstance(self.local_step_generation, bool) or not isinstance(self.local_step_generation, int) or self.local_step_generation < 0:
            raise ValueError("full-parameter local-step generation must be non-negative")
        if not _SHA256.fullmatch(self.layout_hash):
            raise ValueError("full-parameter layout hash must be a lowercase SHA256")


@dataclass(frozen=True)
class FullParameterLocalStepReceipt:
    """Rank-local proof that an optimizer step advanced a global-policy base."""

    topology: FullParameterTopology
    role: str
    base_policy_version: int
    local_step_generation: int
    rollout_id: int
    optimizer_steps: int
    scheduler_start_steps: int
    scheduler_end_steps: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        for name in (
            "base_policy_version",
            "local_step_generation",
            "rollout_id",
            "optimizer_steps",
            "scheduler_start_steps",
            "scheduler_end_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid full-parameter local-step field: {name}")
        if self.local_step_generation < 1 or self.optimizer_steps < 1:
            raise ValueError("full-parameter local-step receipt has no optimizer progress")
        if self.scheduler_end_steps <= self.scheduler_start_steps:
            raise ValueError("full-parameter scheduler did not advance")


@dataclass(frozen=True)
class FullParameterOptimizerState:
    """Small, topology-bound proof that Adam state survived a policy apply.

    The full optimizer state is intentionally never copied through Ray.  Each
    rank instead fingerprints one deterministic populated parameter state and
    reports aggregate cardinalities for the complete local optimizer.
    """

    topology: FullParameterTopology
    role: str
    installed_policy_version: int
    local_step_generation: int
    last_rollout_id: int
    scheduler_num_steps: int
    populated_parameter_count: int
    optimizer_state_tensor_count: int
    optimizer_state_scalar_count: int
    selected_wire_name: str
    selected_state_sha256: str
    model_master_parameter_count: int

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        for name in (
            "installed_policy_version",
            "local_step_generation",
            "last_rollout_id",
            "scheduler_num_steps",
            "populated_parameter_count",
            "optimizer_state_tensor_count",
            "optimizer_state_scalar_count",
            "model_master_parameter_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid optimizer-state field: {name}")
        if self.populated_parameter_count < 1:
            raise ValueError("full-parameter optimizer state is empty")
        if self.optimizer_state_tensor_count < 1:
            raise ValueError("full-parameter optimizer has no tensor state")
        if self.optimizer_state_scalar_count < 1:
            raise ValueError("full-parameter optimizer tensor state is empty")
        if self.model_master_parameter_count < 1:
            raise ValueError("full-parameter model/master proof is empty")
        if "::" not in self.selected_wire_name:
            raise ValueError("selected optimizer-state parameter is not canonical")
        if not _SHA256.fullmatch(self.selected_state_sha256):
            raise ValueError("optimizer-state fingerprint must be a lowercase SHA256")


def _validate_rank_tuple(ranks: object, name: str) -> tuple[int, ...]:
    if (
        not isinstance(ranks, tuple)
        or not ranks
        or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks)
        or len(set(ranks)) != len(ranks)
    ):
        raise ValueError(f"full-parameter {name} must be a non-empty tuple of distinct global ranks")
    return ranks


@dataclass(frozen=True)
class FullParameterReplication:
    """The data-parallel groups whose ranks hold identical copies of this rank's parameters.

    Megatron replicates a dense parameter across the data-parallel group with
    context parallelism, and an expert parameter (``allreduce=False``) across
    the expert data-parallel group; ``DistributedDataParallel.broadcast_params``
    syncs the same two groups.  Ranks are global ``torch.distributed`` ranks in
    group order.  ``expert_ranks`` is None when Megatron built no expert groups.
    """

    global_rank: int
    dense_ranks: tuple[int, ...]
    expert_ranks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.global_rank, "global rank")
        for name in ("dense_ranks", "expert_ranks"):
            ranks = getattr(self, name)
            if ranks is None and name == "expert_ranks":
                continue
            if self.global_rank not in _validate_rank_tuple(ranks, name.replace("_", " ")):
                raise ValueError(f"full-parameter {name.replace('_', ' ')} do not contain this rank")

    def ranks_for(self, name: str, parameter: torch.Tensor) -> tuple[int, ...]:
        if getattr(parameter, "allreduce", True):
            return self.dense_ranks
        if self.expert_ranks is None:
            raise RuntimeError(f"{name!r} is an expert parameter but Megatron has no expert data-parallel group")
        return self.expert_ranks


@dataclass(frozen=True, order=True)
class FullParameterHolding:
    """One local trainable parameter and the ranks that hold identical copies of it.

    ``owner_rank`` exports the parameter and is the only rank whose manifest
    lists a spec for it; every other rank of ``replica_ranks`` holds a replica
    and writes the owner's bytes into it on every apply."""

    name: str
    shape: tuple[int, ...]
    owner_rank: int
    replica_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _PARAMETER_NAME.fullmatch(self.name) or "::" in self.name:
            raise ValueError(f"invalid full-parameter name: {self.name!r}")
        if not isinstance(self.shape, tuple) or not self.shape or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError(f"invalid shape for full-parameter holding {self.name!r}")
        _validate_non_negative_integer(self.owner_rank, "owner rank")
        if self.owner_rank not in _validate_rank_tuple(self.replica_ranks, "replica ranks"):
            raise ValueError(f"full-parameter owner of {self.name!r} is not one of its replicas")


@dataclass(frozen=True)
class FullParameterShardManifest:
    """Small immutable description of one topology-owned FP32 master shard.

    ``specs`` are the parameters this rank owns (exports).  ``holdings``
    lists every local parameter, owned or replicated, with its owner; it is
    metadata for the group transport and is not part of the layout hash, so a
    rank without data-parallel replicas keeps its pre-replica identity.
    ``sharded_masters`` (also outside the hash): the rank keeps only a range
    of each FP32 master (Megatron's distributed optimizer at DP>1), so its
    owners export with the ranges the other replicas contribute
    (``full_parameter_sharded_masters``)."""

    topology: FullParameterTopology
    role: str
    layout_hash: str
    specs: tuple[FullParameterShardSpec, ...]
    global_rank: int = 0
    holdings: tuple[FullParameterHolding, ...] = ()
    sharded_masters: bool = False

    def __post_init__(self) -> None:
        if not _ROLE.fullmatch(self.role):
            raise ValueError(f"invalid full-parameter role: {self.role!r}")
        if not _SHA256.fullmatch(self.layout_hash):
            raise ValueError("full-parameter manifest layout hash is malformed")
        if not self.specs or self.specs != tuple(sorted(self.specs)):
            raise ValueError("full-parameter manifest specs are not canonical")
        names = [spec.wire_name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("full-parameter manifest contains duplicate specs")
        if any(spec.role != self.role or spec.shard_id != self.topology.shard_id for spec in self.specs):
            raise ValueError("full-parameter manifest mixes roles or topology shards")
        if self.layout_hash != _layout_hash(self.topology, self.specs):
            raise ValueError("full-parameter manifest layout hash changed")
        _validate_non_negative_integer(self.global_rank, "global rank")
        if not isinstance(self.sharded_masters, bool):
            raise ValueError("full-parameter manifest sharded-masters flag is malformed")
        if self.sharded_masters and not self.replicated_holdings():
            raise ValueError("full-parameter manifest has sharded masters but no data-parallel replica")
        if self.holdings:
            if self.holdings != tuple(sorted(self.holdings)) or len({holding.name for holding in self.holdings}) != len(self.holdings):
                raise ValueError("full-parameter manifest holdings are not canonical")
            if any(self.global_rank not in holding.replica_ranks for holding in self.holdings):
                raise ValueError("full-parameter manifest holds a parameter outside its replica group")
            owned = {holding.name: holding.shape for holding in self.holdings if holding.owner_rank == self.global_rank}
            if owned != {spec.name: spec.shape for spec in self.specs}:
                raise ValueError("full-parameter manifest specs differ from its owned holdings")

    def replicated_holdings(self) -> tuple[FullParameterHolding, ...]:
        """Holdings another rank owns: this rank writes their owner's bytes."""

        return tuple(holding for holding in self.holdings if holding.owner_rank != self.global_rank)


@dataclass(frozen=True, order=True)
class FullParameterFragmentPlan:
    """Canonical wire-name order for one topology-owned logical fragment.

    ``flat_ranges`` is empty when every member is a whole parameter.  A
    parameter larger than the fragment byte bound is split into several
    fragments; then ``flat_ranges`` holds one ``(start, stop)`` flat element
    range per wire name, and a whole member uses ``(0, numel)``.
    """

    fragment_id: int
    wire_names: tuple[str, ...]
    numel: int
    flat_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.fragment_id, bool) or not isinstance(self.fragment_id, int) or self.fragment_id < 0:
            raise ValueError("full-parameter fragment ID must be non-negative")
        if not self.wire_names or len(set(self.wire_names)) != len(self.wire_names) or any(not isinstance(name, str) or name.count("::") != 2 or not name for name in self.wire_names):
            raise ValueError("full-parameter fragment wire names are malformed")
        if isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 1:
            raise ValueError("full-parameter fragment must contain scalars")
        if self.flat_ranges:
            if len(self.flat_ranges) != len(self.wire_names) or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or any(isinstance(bound, bool) or not isinstance(bound, int) for bound in item)
                or not 0 <= item[0] < item[1]
                for item in self.flat_ranges
            ):
                raise ValueError("full-parameter fragment flat ranges are malformed")
            if sum(stop - start for start, stop in self.flat_ranges) != self.numel:
                raise ValueError("full-parameter fragment flat ranges do not match its scalar count")

    def members(self, numel_by_name: Mapping[str, int]) -> tuple[tuple[str, int, int], ...]:
        """``(wire_name, start, stop)`` flat element range of every member."""

        if self.flat_ranges:
            return tuple((name, start, stop) for name, (start, stop) in zip(self.wire_names, self.flat_ranges, strict=True))
        return tuple((name, 0, numel_by_name[name]) for name in self.wire_names)

    def identity(self) -> dict[str, object]:
        """Hash payload; a plan without split parameters keeps its old bytes."""

        payload: dict[str, object] = {"fragment_id": self.fragment_id, "wire_names": list(self.wire_names), "numel": self.numel}
        if self.flat_ranges:
            payload["flat_ranges"] = [list(item) for item in self.flat_ranges]
        return payload


@dataclass(frozen=True)
class FullParameterOwnerFragmentPlan:
    """Deterministic fragment plan installed on exactly one topology owner."""

    topology: FullParameterTopology
    parameter_layout_hash: str
    fragments: tuple[FullParameterFragmentPlan, ...]

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.parameter_layout_hash):
            raise ValueError("full-parameter global layout hash is malformed")
        if not self.fragments or self.fragments != tuple(sorted(self.fragments, key=lambda fragment: fragment.fragment_id)):
            raise ValueError("full-parameter owner fragments are not canonical")
        fragment_ids = [fragment.fragment_id for fragment in self.fragments]
        if len(set(fragment_ids)) != len(fragment_ids):
            raise ValueError("full-parameter owner plan has duplicate fragment IDs")
        # A split parameter repeats its wire name only across consecutive
        # members; the manifest check proves the ranges tile the parameter.
        names = [name for fragment in self.fragments for name in fragment.wire_names]
        runs = [name for index, name in enumerate(names) if index == 0 or names[index - 1] != name]
        if len(set(runs)) != len(runs):
            raise ValueError("full-parameter owner plan has duplicate parameters")

    @property
    def plan_hash(self) -> str:
        payload = {
            "schema": 1,
            "topology": asdict(self.topology),
            "parameter_layout_hash": self.parameter_layout_hash,
            "fragments": [fragment.identity() for fragment in self.fragments],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class FullParameterChunkDescriptor:
    chunk_index: int
    flat_offset: int
    numel: int
    payload_hash: str

    def __post_init__(self) -> None:
        for name in ("chunk_index", "flat_offset", "numel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "numel" else 0):
                raise ValueError(f"invalid full-parameter chunk field: {name}")
        if not _SHA256.fullmatch(self.payload_hash):
            raise ValueError("full-parameter chunk hash is malformed")
        if self.numel * _FP32_BYTES >= FULL_PARAMETER_MAX_CHUNK_BYTES:
            raise ValueError("full-parameter chunk exceeds the Ray transport bound")


@dataclass(frozen=True, order=True)
class FullParameterFragmentDescriptor:
    fragment_id: int
    numel: int
    payload_hash: str
    chunks: tuple[FullParameterChunkDescriptor, ...]

    def __post_init__(self) -> None:
        if isinstance(self.fragment_id, bool) or not isinstance(self.fragment_id, int) or self.fragment_id < 0 or isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 1:
            raise ValueError("full-parameter fragment descriptor is malformed")
        if not _SHA256.fullmatch(self.payload_hash):
            raise ValueError("full-parameter fragment hash is malformed")
        if not self.chunks:
            raise ValueError("full-parameter fragment has no chunks")
        offset = 0
        for index, chunk in enumerate(self.chunks):
            if chunk.chunk_index != index or chunk.flat_offset != offset:
                raise ValueError("full-parameter fragment chunks are not contiguous")
            offset += chunk.numel
        if offset != self.numel:
            raise ValueError("full-parameter fragment chunk coverage is incomplete")


@dataclass(frozen=True)
class FullParameterChunkedShardState:
    """Small cut descriptor whose payload stays in nested immutable Ray objects."""

    policy_version: int
    local_step_generation: int
    topology: FullParameterTopology
    parameter_layout_hash: str
    plan_hash: str
    fragments: tuple[FullParameterFragmentDescriptor, ...]
    chunk_refs: list[list[object | None]]

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.policy_version, "policy version")
        _validate_non_negative_integer(
            self.local_step_generation,
            "local-step generation",
        )
        if not _SHA256.fullmatch(self.parameter_layout_hash):
            raise ValueError("full-parameter global layout hash is malformed")
        if not _SHA256.fullmatch(self.plan_hash):
            raise ValueError("full-parameter owner plan hash is malformed")
        if not self.fragments or self.fragments != tuple(sorted(self.fragments, key=lambda fragment: fragment.fragment_id)):
            raise ValueError("full-parameter shard fragments are not canonical")
        if len({fragment.fragment_id for fragment in self.fragments}) != len(self.fragments):
            raise ValueError("full-parameter shard has duplicate fragments")
        _validate_chunk_ref_shape(self, require_refs=True)


@dataclass(frozen=True)
class _PreparedFullParameterChunkedShard:
    commit_token: str
    state: FullParameterChunkedShardState
    parameter_tensor_count: int
    base_policy_version: int
    base_local_step_generation: int
    base_exported_local_step_generation: int
    scheduler_num_steps: int
    state_identity_hash: str
    commit_started: bool
    # Fragments whose target bytes the optimizer masters already hold (same
    # fragment payload hash as the last export or commit): prepare does not
    # read them and commit does not copy them.
    resident_fragment_ids: frozenset[int] = frozenset()
    resident_record: object | None = None
    # Data-parallel replicas (``full_parameter_replicas``): the owners' shards
    # of this cut, in replica-plan order, and the owner fragments the replicas
    # already hold.
    replica_states: tuple[FullParameterChunkedShardState, ...] = ()
    replica_resident_ids: frozenset = frozenset()
    replica_resident_record: object | None = None
    # Resident fragments whose descriptor is a lazy-export label; the commit
    # keeps them labelled in the record it writes.
    lazy_fragment_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class _ResidentFullParameterFragments:
    """Which fragment bytes the optimizer masters hold, by fragment payload hash.

    Written by a chunked export (the bytes it hashed are the masters) and by a
    chunked commit (the masters now hold the target).  Valid only while the
    tracking state it was taken at is unchanged: a local step, a restore, a
    legacy apply or the start of a commit drops it.

    ``lazy_fragment_ids``: fragments whose recorded payload hash is a label of
    a lazy export, not a content hash.  The label is fresh for every lazy
    export and only this record vouches for it, so a row carrying it is
    either resident or unusable (its stored bytes cannot be checked)."""

    parameter_layout_hash: str
    plan_hash: str
    policy_version: int
    local_step_generation: int
    scheduler_num_steps: int
    payload_hashes: Mapping[int, tuple[int, str]]
    lazy_fragment_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class _FullParameterCommitOutcome:
    """Actor-local journal for an acknowledged or in-doubt commit."""

    commit_token: str
    policy_version: int
    parameter_tensor_count: int
    state_identity_hash: str


@dataclass(frozen=True)
class FullParameterCommitStatus:
    commit_token: str
    status: str
    policy_version: int
    parameter_tensor_count: int
    state_identity_hash: str

    def __post_init__(self) -> None:
        _validate_commit_token(self.commit_token)
        if self.status not in {"ABSENT", "PREPARED", "COMMITTING", "COMMITTED", "FINALIZED"}:
            raise ValueError("full-parameter commit status is invalid")
        _validate_non_negative_integer(self.policy_version, "policy version")
        if isinstance(self.parameter_tensor_count, bool) or not isinstance(self.parameter_tensor_count, int) or self.parameter_tensor_count < 0:
            raise ValueError("full-parameter commit parameter count is invalid")
        if not _SHA256.fullmatch(self.state_identity_hash):
            raise ValueError("full-parameter commit state identity is malformed")


class _OptimizerOwnedParameters(frozenset):
    """Identities of the tensors the base optimizers step, plus distributed-optimizer masters.

    ``distributed_masters`` maps ``id(model_param)`` to ``(owned, fp32)`` for
    Megatron's distributed optimizer when the parameter has no
    ``main_param`` tensor to read:

    * a native-FP32 parameter is stepped through a flat view
      (``shard_fp32_groups``); ``owned`` and ``fp32`` are that view;
    * under the precision-aware optimizer with ``--optimizer-cpu-offload``
      the base optimizer owns the BF16 model view (``shard_float16_groups``)
      and the HybridDeviceOptimizer steps an FP32 copy of it (host-pinned
      when offloaded); ``fp32`` is that copy.

    Empty for the non-distributed optimizer.

    ``distributed_views`` maps ``id(model_param)`` to ``(view, owned, fp32)``
    for every parameter the distributed optimizer steps a range of on this
    rank: ``view`` is Megatron's ``shard_*_groups`` view of the model parameter
    at that range (None for a quantized parameter), ``fp32`` its FP32 values
    (None under the precision-aware optimizer without CPU offload).  Read only
    when masters are sharded (``full_parameter_sharded_masters``).
    ``hybrid_copies``: a base optimizer keeps HybridDeviceOptimizer copies.
    """

    distributed_masters: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
    distributed_views: Mapping[int, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]]
    hybrid_copies: bool

    def __new__(
        cls,
        identities,
        distributed_masters: Mapping[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        distributed_views: Mapping[int, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]] | None = None,
        hybrid_copies: bool = False,
    ):
        owned = super().__new__(cls, identities)
        owned.distributed_masters = dict(distributed_masters or {})
        owned.distributed_views = dict(distributed_views or {})
        owned.hybrid_copies = bool(hybrid_copies)
        return owned


def _resolve_master(
    parameter: torch.nn.Parameter,
    name: str,
    *,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(owned, master)``: the tensor the base optimizer owns and the FP32 master view.

    The non-distributed optimizer keeps a whole FP32 master per parameter;
    both tensors are the same object. Megatron's distributed optimizer keeps a
    flat FP32 range per parameter (``main_param_sharded``); at DP=1 that range
    is the whole parameter, so ``master`` is a parameter-shaped view of it and
    writes through the view land in the tensor that is stepped. A partial
    range (DP>1, or a rank owning none of the parameter) is refused here: at
    DP>1 ``_local_layout`` reads the distributed optimizer's ranges through
    ``full_parameter_sharded_masters`` instead.
    """

    sharded = bool(getattr(parameter, "main_param_sharded", False))
    owned = getattr(parameter, "main_param", None)
    fp32 = owned
    if owned is None:
        distributed = getattr(optimizer_owned_parameter_ids, "distributed_masters", {})
        if id(parameter) in distributed:
            owned, fp32 = distributed[id(parameter)]
            sharded = True
        elif parameter.dtype == torch.float32 and optimizer_owned_parameter_ids is not None and id(parameter) in optimizer_owned_parameter_ids:
            # Megatron keeps parameters such as Qwen3.5 ``A_log`` natively in
            # FP32 and places that same object directly in the base optimizer.
            owned = fp32 = parameter
        elif sharded:
            raise RuntimeError(
                f"{name!r} has no local FP32 master (DP-sharded optimizer, or precision-aware without "
                "CPU offload); full-parameter sync needs a whole FP32 master of every parameter at DP=1, "
                "and FP32 main params without CPU offload at DP>1"
            )
        else:
            raise RuntimeError(f"{name!r} has no FP32 optimizer master")
    if not isinstance(fp32, torch.Tensor) or fp32.dtype != torch.float32 or fp32.numel() != parameter.numel():
        if sharded:
            raise RuntimeError(
                f"{name!r} optimizer master is a partial DP shard outside a data-parallel replica group; "
                "full-parameter sync reads partial ranges only at DP>1"
            )
        raise RuntimeError(f"{name!r} has an incomplete FP32 optimizer master")
    if sharded:
        if fp32.dim() != 1 or not fp32.is_contiguous():
            raise RuntimeError(f"{name!r} distributed-optimizer master is not a flat contiguous range")
        master = fp32.view(parameter.shape)
    else:
        if tuple(fp32.shape) != tuple(parameter.shape):
            raise RuntimeError(f"{name!r} has an incomplete FP32 optimizer master")
        master = fp32
    if optimizer_owned_parameter_ids is not None and id(owned) not in optimizer_owned_parameter_ids:
        raise RuntimeError(f"{name!r} FP32 master is not owned by the optimizer")
    return owned, master


def _master_parameter(
    parameter: torch.nn.Parameter,
    name: str,
    *,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    return _resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned_parameter_ids)[1]


def _specs_and_masters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    role: str,
    topology: FullParameterTopology,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[
    tuple[FullParameterShardSpec, ...],
    dict[str, torch.Tensor],
]:
    if topology.dp_size != 1:
        raise RuntimeError("the initial full-parameter DiLoCo profile requires DP=1")
    masters = {}
    specs = []
    for name, parameter in sorted(named_parameters, key=lambda value: value[0]):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{name!r} is not a trainable parameter")
        master = _master_parameter(
            parameter,
            name,
            optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
        )
        spec = FullParameterShardSpec(
            role,
            topology.shard_id,
            name,
            tuple(int(dimension) for dimension in master.shape),
            "float32",
            master.numel(),
        )
        if spec.wire_name in masters:
            raise RuntimeError(f"duplicate full-parameter name: {spec.wire_name!r}")
        specs.append(spec)
        masters[spec.wire_name] = master
    if not specs:
        raise RuntimeError("full-parameter shard is empty")
    return tuple(specs), masters


def full_parameter_replication() -> FullParameterReplication:
    """This rank's data-parallel replica groups (one rank when not distributed)."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return FullParameterReplication(global_rank=0, dense_ranks=(0,), expert_ranks=(0,))
    from megatron.core import mpu

    def ranks(group) -> tuple[int, ...]:
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))

    try:
        expert_ranks = ranks(mpu.get_expert_data_parallel_group())
    except (AssertionError, RuntimeError):  # Megatron built no expert groups
        expert_ranks = None
    return FullParameterReplication(
        global_rank=int(dist.get_rank()),
        dense_ranks=ranks(mpu.get_data_parallel_group(with_context_parallel=True)),
        expert_ranks=expert_ranks,
    )


def _replica_owners(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    replication: FullParameterReplication,
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """``{name: (owner rank, replica ranks)}``, balanced within each replica group.

    Every rank of a group holds the same parameters, so each computes the
    same assignment without communication: largest parameter first (ties by
    name) to the least-loaded rank (ties by group order).  The first
    ``len(group)`` parameters go to distinct ranks, so every rank of a group
    with at least that many parameters owns one.  A group of one rank owns
    everything, which keeps a DP=1 layout unchanged."""

    by_group: dict[tuple[int, ...], list[tuple[str, int]]] = {}
    for name, parameter in named_parameters:
        by_group.setdefault(replication.ranks_for(name, parameter), []).append((name, parameter.numel()))
    owners: dict[str, tuple[int, tuple[int, ...]]] = {}
    for ranks, members in by_group.items():
        load = [0] * len(ranks)
        for name, numel in sorted(members, key=lambda member: (-member[1], member[0])):
            index = min(range(len(ranks)), key=lambda position: (load[position], position))
            load[index] += numel
            owners[name] = (ranks[index], ranks)
    return owners


@dataclass(frozen=True)
class _LocalLayout:
    """This rank's parameters split into the ones it owns and the replicas it holds.

    Whole masters fill ``masters`` and ``replica_masters``.  Sharded masters
    (Megatron's distributed optimizer at DP>1) leave both empty and fill
    ``ranges``: this rank's FP32 range of every local parameter, by name
    (``full_parameter_sharded_masters``)."""

    specs: tuple[FullParameterShardSpec, ...]
    masters: dict[str, torch.Tensor]  # wire name -> FP32 master of an owned parameter
    holdings: tuple[FullParameterHolding, ...]
    replica_masters: dict[str, torch.Tensor]  # parameter name -> FP32 master of a replica
    ranges: Mapping[str, sharded.MasterRange] | None = None  # by parameter name

    def ranges_by_wire_name(self) -> dict[str, sharded.MasterRange]:
        return {spec.wire_name: self.ranges[spec.name] for spec in self.specs}


def _local_layout(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    role: str,
    topology: FullParameterTopology,
    replication: FullParameterReplication,
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> _LocalLayout:
    ordered = tuple(sorted(named_parameters, key=lambda value: value[0]))
    for name, parameter in ordered:
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{name!r} is not a trainable parameter")
    owners = _replica_owners(ordered, replication)
    ranges = sharded.master_ranges(ordered, replication, optimizer_owned_parameter_ids)
    specs, masters, holdings, replica_masters = [], {}, [], {}
    for name, parameter in ordered:
        # A sharded master has the parameter's shape (it is a range of it).
        master = None if ranges is not None else _master_parameter(
            parameter, name, optimizer_owned_parameter_ids=optimizer_owned_parameter_ids
        )
        shape = tuple(int(dimension) for dimension in (parameter if master is None else master).shape)
        owner, ranks = owners[name]
        holdings.append(FullParameterHolding(name=name, shape=shape, owner_rank=owner, replica_ranks=ranks))
        if owner != replication.global_rank:
            if master is not None:
                replica_masters[name] = master
            continue
        spec = FullParameterShardSpec(role, topology.shard_id, name, shape, "float32", math.prod(shape))
        specs.append(spec)
        if master is not None:
            masters[spec.wire_name] = master
    if not specs:
        raise RuntimeError(
            "this rank owns no full-parameter tensor: its data-parallel group has more ranks than parameters"
        )
    return _LocalLayout(
        specs=tuple(specs), masters=masters, holdings=tuple(holdings), replica_masters=replica_masters, ranges=ranges
    )


def _actor_local_layout(actor, *, role: str, topology: FullParameterTopology) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    frozenset[int],
    _LocalLayout,
]:
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    layout = _local_layout(
        named_parameters,
        role=role,
        topology=topology,
        replication=full_parameter_replication(),
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    return named_parameters, optimizer_owned, layout


def _layout_hash(
    topology: FullParameterTopology,
    specs: Sequence[FullParameterShardSpec],
) -> str:
    payload = {
        "schema": 1,
        "topology": asdict(topology),
        "specs": [asdict(spec) for spec in specs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"full-parameter {name} must be a non-negative integer")
    return value


def _validate_commit_token(commit_token: object) -> str:
    if not isinstance(commit_token, str) or not _COMMIT_TOKEN.fullmatch(commit_token):
        raise ValueError("full-parameter commit token is malformed")
    return commit_token


def _validate_chunk_byte_limit(max_chunk_bytes: object) -> int:
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes < _FP32_BYTES or max_chunk_bytes >= FULL_PARAMETER_MAX_CHUNK_BYTES:
        raise ValueError("full-parameter chunk bytes must be in [4, 1 GiB)")
    return max_chunk_bytes // _FP32_BYTES


def _validate_chunk_ref_shape(
    state: FullParameterChunkedShardState,
    *,
    require_refs: bool,
) -> None:
    if not isinstance(state.chunk_refs, list) or len(state.chunk_refs) != len(state.fragments):
        raise ValueError("full-parameter chunk references do not match fragments")
    for fragment, references in zip(
        state.fragments,
        state.chunk_refs,
        strict=True,
    ):
        if not isinstance(references, list) or len(references) != len(fragment.chunks):
            raise ValueError("full-parameter chunk references are incomplete")
        if require_refs and any(reference is None for reference in references):
            raise ValueError("full-parameter chunk reference was released")


def validate_full_parameter_chunked_shard_state(
    state: FullParameterChunkedShardState,
    *,
    require_refs: bool = True,
) -> FullParameterChunkedShardState:
    if not isinstance(state, FullParameterChunkedShardState):
        raise TypeError("full-parameter chunked shard has the wrong type")
    _validate_non_negative_integer(state.policy_version, "policy version")
    _validate_non_negative_integer(
        state.local_step_generation,
        "local-step generation",
    )
    if not _SHA256.fullmatch(state.parameter_layout_hash) or not _SHA256.fullmatch(state.plan_hash):
        raise ValueError("full-parameter chunked shard identity is malformed")
    if not state.fragments or state.fragments != tuple(sorted(state.fragments, key=lambda fragment: fragment.fragment_id)):
        raise ValueError("full-parameter shard fragments are not canonical")
    for fragment in state.fragments:
        FullParameterFragmentDescriptor(
            fragment.fragment_id,
            fragment.numel,
            fragment.payload_hash,
            fragment.chunks,
        )
    _validate_chunk_ref_shape(state, require_refs=require_refs)
    return state


def _fragment_hasher(
    fragment_id: int,
    parameter_layout_hash: str,
    plan_hash: str,
):
    digest = hashlib.sha256()
    digest.update(_FRAGMENT_HASH_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment_id.to_bytes(8, "little"))
    return digest


def _all_finite(value: torch.Tensor) -> bool:
    """``torch.isfinite(value).all()`` without the full-size bool tensor.

    ``amax``/``amin`` propagate NaN, +inf is the maximum and -inf the minimum,
    so both extremes are finite exactly when every value is.  Two vectorized
    reductions instead of an elementwise pass plus an allocation (about 40x
    faster on 256 MiB of f32 with one thread)."""

    if not value.is_floating_point():
        return True
    if value.numel() == 0:
        return True
    return bool(torch.isfinite(value.amax()).item() and torch.isfinite(value.amin()).item())


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    return memoryview(value.numpy()).cast("B")


def _chunk_hasher(
    fragment_id: int,
    chunk_index: int,
    flat_offset: int,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
):
    """The chunk hash before its value bytes (feed them in any split)."""

    digest = hashlib.sha256()
    digest.update(_CHUNK_HASH_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment_id.to_bytes(8, "little"))
    digest.update(chunk_index.to_bytes(8, "little"))
    digest.update(flat_offset.to_bytes(8, "little"))
    return digest


def _chunk_hash(
    fragment_id: int,
    chunk_index: int,
    flat_offset: int,
    value: torch.Tensor,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
) -> str:
    digest = _chunk_hasher(
        fragment_id,
        chunk_index,
        flat_offset,
        parameter_layout_hash=parameter_layout_hash,
        plan_hash=plan_hash,
    )
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


_HASH_POOL: ThreadPoolExecutor | None = None
_HASH_POOL_LOCK = threading.Lock()


def _hash_pool() -> ThreadPoolExecutor:
    """Two hashing threads (hashlib and torch release the GIL on large buffers)."""

    global _HASH_POOL
    with _HASH_POOL_LOCK:
        if _HASH_POOL is None:
            _HASH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="miles-fp-hash")
        return _HASH_POOL


_SCAN_THREADS_ENV = "MILES_FULL_PARAMETER_HASH_THREADS"
_SCAN_POOL: ThreadPoolExecutor | None = None
_SCAN_POOL_THREADS = 0
# The finite check and both hashes of a deferred export read the masters in
# blocks of this many floats (8 MiB): one pass, cache-resident, no copy.
_SCAN_BLOCK_NUMEL = 1 << 21


def _scan_threads() -> int:
    raw = os.environ.get(_SCAN_THREADS_ENV, "8")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{_SCAN_THREADS_ENV} must be an integer, got {raw!r}") from exc
    if not 1 <= threads <= 64:
        raise RuntimeError(f"{_SCAN_THREADS_ENV} must be within 1..64, got {threads}")
    return threads


def _scan_pool() -> ThreadPoolExecutor:
    """Fragment-parallel workers for deferred exports, materialization and
    prepare verification (``MILES_FULL_PARAMETER_HASH_THREADS``, default 8).

    Fragments are independent hash streams, so the result does not depend on
    the number of workers; hashlib and torch release the GIL on large buffers."""

    global _SCAN_POOL, _SCAN_POOL_THREADS
    threads = _scan_threads()
    with _HASH_POOL_LOCK:
        if _SCAN_POOL is None or _SCAN_POOL_THREADS != threads:
            if _SCAN_POOL is not None:
                _SCAN_POOL.shutdown(wait=False)
            _SCAN_POOL = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="miles-fp-scan")
            _SCAN_POOL_THREADS = threads
        return _SCAN_POOL


class _DeferredFullParameterChunk:
    """The ref of a chunk a deferred export hashed but did not store in Ray.

    A deferred export (``defer_payload=True``) describes every fragment from
    the masters (finite check, chunk hashes, fragment digest) but puts no
    bytes into the object store.  Its chunk refs are this marker.  The bytes
    stay readable only while the masters hold them, which the resident record
    proves: ``materialize_full_parameter_fragments`` then stores a fragment on
    demand, and a skip-resident apply never reads a resident one.  Reading a
    marker anywhere else is an error (``ray.get`` refuses a non-ObjectRef)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFERRED_FULL_PARAMETER_CHUNK"

    def __reduce__(self):
        return (_deferred_full_parameter_chunk, ())


DEFERRED_FULL_PARAMETER_CHUNK = _DeferredFullParameterChunk()


def _deferred_full_parameter_chunk() -> _DeferredFullParameterChunk:
    return DEFERRED_FULL_PARAMETER_CHUNK


def is_deferred_full_parameter_chunk(reference: object) -> bool:
    return isinstance(reference, _DeferredFullParameterChunk)


def _deferred_fragment_ids(state: FullParameterChunkedShardState) -> frozenset[int]:
    return frozenset(
        fragment.fragment_id
        for fragment, references in zip(state.fragments, state.chunk_refs, strict=True)
        if any(is_deferred_full_parameter_chunk(reference) for reference in references)
    )


_RESIDENT_SKIP_ENV = "MILES_FULL_PARAMETER_SKIP_RESIDENT"
# The owner fragments a rank's data-parallel replicas hold (``full_parameter_replicas``).
_REPLICA_RESIDENT_ATTRIBUTE = "_external_full_parameter_replica_resident"


def _resident_skip_enabled() -> bool:
    return os.environ.get(_RESIDENT_SKIP_ENV, "1") != "0"


def forget_full_parameter_resident_fragments(actor) -> None:
    """Drop the record of which fragment bytes the masters hold.

    Call after anything writes optimizer masters outside the chunked
    export/commit path (a checkpoint or peer snapshot restore)."""

    actor._external_full_parameter_resident = None
    setattr(actor, _REPLICA_RESIDENT_ATTRIBUTE, None)


def _record_resident_fragments(
    actor,
    *,
    parameter_layout_hash: str,
    plan_hash: str,
    policy_version: int,
    local_step_generation: int,
    fragments: Sequence[FullParameterFragmentDescriptor],
    lazy_fragment_ids: frozenset[int] = frozenset(),
) -> None:
    actor._external_full_parameter_resident = _ResidentFullParameterFragments(
        parameter_layout_hash=parameter_layout_hash,
        plan_hash=plan_hash,
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        scheduler_num_steps=_scheduler_num_steps(actor),
        payload_hashes={fragment.fragment_id: (fragment.numel, fragment.payload_hash) for fragment in fragments},
        lazy_fragment_ids=frozenset(lazy_fragment_ids),
    )


def _resident_fragment_ids(
    actor,
    state: FullParameterChunkedShardState,
    *,
    honor_kill_switch: bool = True,
) -> tuple[frozenset[int], object | None]:
    """Fragments of ``state`` whose bytes the masters provably hold now.

    ``honor_kill_switch=False``: ignore ``MILES_FULL_PARAMETER_SKIP_RESIDENT``
    (deferred fragments have no stored bytes; residency is their only source)."""

    record = getattr(actor, "_external_full_parameter_resident", None)
    if (honor_kill_switch and not _resident_skip_enabled()) or not isinstance(record, _ResidentFullParameterFragments):
        return frozenset(), None
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if (
        record.parameter_layout_hash != state.parameter_layout_hash
        or record.plan_hash != state.plan_hash
        or record.policy_version != installed
        or record.local_step_generation != generation
        or exported != generation
        or record.scheduler_num_steps != _scheduler_num_steps(actor)
    ):
        return frozenset(), None
    resident = frozenset(
        fragment.fragment_id
        for fragment in state.fragments
        if record.payload_hashes.get(fragment.fragment_id) == (fragment.numel, fragment.payload_hash)
    )
    return resident, record


def _lazy_fragment_ids(actor, state: FullParameterChunkedShardState) -> frozenset[int]:
    """Fragments of ``state`` whose descriptor is a label the live record minted.

    Only the label is compared here; whether the record still describes the
    masters is :func:`_resident_fragment_ids`' check."""

    record = getattr(actor, "_external_full_parameter_resident", None)
    if not isinstance(record, _ResidentFullParameterFragments) or not record.lazy_fragment_ids:
        return frozenset()
    return frozenset(
        fragment.fragment_id
        for fragment in state.fragments
        if fragment.fragment_id in record.lazy_fragment_ids
        and record.payload_hashes.get(fragment.fragment_id) == (fragment.numel, fragment.payload_hash)
    )


def _note_apply_stats(actor, phase: str, **values) -> None:
    stats = dict(getattr(actor, "_external_full_parameter_apply_stats", None) or {})
    stats[phase] = values
    actor._external_full_parameter_apply_stats = stats


def full_parameter_chunked_shard_identity(
    state: FullParameterChunkedShardState,
) -> str:
    """Hash immutable metadata that gives every referenced byte its meaning."""

    validate_full_parameter_chunked_shard_state(state, require_refs=False)
    payload = {
        "policy_version": state.policy_version,
        "local_step_generation": state.local_step_generation,
        "topology": asdict(state.topology),
        "parameter_layout_hash": state.parameter_layout_hash,
        "plan_hash": state.plan_hash,
        "fragments": [
            {
                "fragment_id": fragment.fragment_id,
                "numel": fragment.numel,
                "payload_hash": fragment.payload_hash,
                "chunks": [asdict(chunk) for chunk in fragment.chunks],
            }
            for fragment in state.fragments
        ],
    }
    digest = hashlib.sha256(_SHARD_IDENTITY_DOMAIN)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


@torch.no_grad()
def make_full_parameter_shard_state(
    *,
    policy_version: int,
    local_step_generation: int = 0,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> FullParameterShardState:
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    tensors = {}
    for spec in specs:
        value = masters[spec.wire_name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not _all_finite(value):
            raise ValueError(f"{spec.wire_name!r} contains NaN or Inf")
        tensors[spec.wire_name] = value.clone()
    return FullParameterShardState(
        policy_version,
        topology,
        _layout_hash(topology, specs),
        specs,
        tensors,
        local_step_generation,
    )


def validate_full_parameter_shard_state(
    state: FullParameterShardState,
) -> FullParameterShardState:
    """Revalidate a shard after crossing a Ray/object-store boundary."""

    if state.specs != tuple(sorted(state.specs)):
        raise ValueError("full-parameter specs are not canonical")
    expected_names = {spec.wire_name for spec in state.specs}
    if set(state.tensors) != expected_names:
        raise ValueError("full-parameter state is incomplete")
    if state.layout_hash != _layout_hash(state.topology, state.specs):
        raise ValueError("full-parameter state layout hash changed")
    for spec in state.specs:
        tensor = state.tensors[spec.wire_name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous() or tuple(tensor.shape) != spec.shape or not _all_finite(tensor):
            raise ValueError(f"malformed full-parameter tensor {spec.wire_name!r}")
    return state


def _prepare_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    incoming, specs, masters = _validate_full_parameter_shard_metadata(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    staged = []
    for spec in specs:
        target = masters[spec.wire_name]
        source = incoming.tensors[spec.wire_name]
        staged.append((target, source.to(device=target.device, dtype=torch.float32)))
    return staged


def _validate_full_parameter_shard_metadata(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> tuple[
    FullParameterShardState,
    tuple[FullParameterShardSpec, ...],
    dict[str, torch.Tensor],
]:
    incoming = validate_full_parameter_shard_state(state)
    if incoming.topology != topology:
        raise RuntimeError("full-parameter topology identity changed")
    specs, masters = _specs_and_masters(
        named_parameters,
        role=role,
        topology=topology,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    if incoming.specs != specs or incoming.layout_hash != _layout_hash(topology, specs):
        raise RuntimeError("full-parameter local layout changed")
    return incoming, specs, masters


def validate_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> int:
    """Validate one complete shard without changing model or optimizer state."""

    _incoming, specs, _masters = _validate_full_parameter_shard_metadata(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    return len(specs)


@torch.no_grad()
def apply_full_parameter_shard_values(
    state: FullParameterShardState,
    *,
    role: str,
    topology: FullParameterTopology,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_owned_parameter_ids: frozenset[int] | None = None,
) -> int:
    """Apply a complete shard to optimizer masters without resetting Adam state."""

    staged = _prepare_full_parameter_shard_values(
        state,
        role=role,
        topology=topology,
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
    )
    for target, source in staged:
        target.copy_(source)
    return len(staged)


def require_full_parameter_actor(args, action: str) -> None:
    """Refuse a LoRA actor unless it opted into the LoRA hybrid profile.

    A hybrid actor streams every ``requires_grad`` parameter (adapters plus
    recipe-unfrozen base parameters); the rest of the profile (a whole FP32
    optimizer master of every trainable parameter on every rank) is unchanged.
    """

    from miles.utils.lora.utils import full_parameter_lora_hybrid_enabled, is_lora_enabled

    hybrid = full_parameter_lora_hybrid_enabled(args)
    if not is_lora_enabled(args):
        if hybrid:
            raise RuntimeError("--full-parameter-lora-hybrid requires a LoRA actor (--lora-rank > 0)")
        return
    if not hybrid:
        raise RuntimeError(
            f"full-parameter DiLoCo cannot {action} a LoRA actor without --full-parameter-lora-hybrid"
        )
    if getattr(args, "lora_base_cpu_backup", False):
        raise RuntimeError("--full-parameter-lora-hybrid is incompatible with --lora-base-cpu-backup")


def full_parameter_topology() -> FullParameterTopology:
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    return FullParameterTopology(
        tp_rank=state.tp.rank,
        tp_size=state.tp.size,
        pp_rank=state.pp.rank,
        pp_size=state.pp.size,
        ep_rank=state.ep.rank,
        ep_size=state.ep.size,
        cp_rank=state.cp.rank,
        cp_size=state.cp.size,
        dp_rank=state.intra_dp_cp.rank,
        dp_size=state.intra_dp_cp.size,
    )


def _named_actor_parameters(actor) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers

    values = tuple(
        (name, value)
        for name, value in named_params_and_buffers(
            actor.args,
            actor.model,
            convert_to_global_name=True,
        )
        if isinstance(value, torch.nn.Parameter) and value.requires_grad
    )
    if len({name for name, _parameter in values}) != len(values):
        raise RuntimeError("Megatron produced duplicate full-parameter names")
    return values


def _optimizer_owned_parameter_ids(optimizer) -> frozenset[int]:
    """Return exact parameter objects owned by the non-distributed optimizer."""

    bases = _optimizer_bases(optimizer)
    owned: set[int] = set()
    for base in bases:
        groups = getattr(base, "param_groups", None)
        if not isinstance(groups, list):
            raise RuntimeError("full-parameter optimizer has no parameter groups")
        for group in groups:
            parameters = group.get("params") if isinstance(group, Mapping) else None
            if not isinstance(parameters, list):
                raise RuntimeError("full-parameter optimizer group is malformed")
            for parameter in parameters:
                if not isinstance(parameter, torch.Tensor):
                    raise RuntimeError("optimizer owns a non-tensor parameter")
                identity = id(parameter)
                if identity in owned:
                    raise RuntimeError("optimizer owns a parameter more than once")
                owned.add(identity)
    if not owned:
        raise RuntimeError("full-parameter optimizer owns no parameters")
    return _OptimizerOwnedParameters(
        owned,
        _distributed_masters(optimizer),
        distributed_views=_distributed_views(optimizer),
        hybrid_copies=any(getattr(base, "param_to_inner_param", None) for base in bases),
    )


def _optimizer_wrappers(optimizer) -> tuple[object, ...]:
    wrappers = getattr(optimizer, "chained_optimizers", None)
    return tuple((optimizer,) if wrappers is None else wrappers)


def _paired_groups(wrapper, model_attribute: str, shard_attribute: str):
    model_groups = getattr(wrapper, model_attribute, None)
    shard_groups = getattr(wrapper, shard_attribute, None)
    if model_groups is None and shard_groups is None:
        return
    if model_groups is None or shard_groups is None:
        raise RuntimeError("distributed optimizer parameter groups are malformed")
    for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
        yield from zip(model_group, shard_group, strict=True)


def _distributed_masters(optimizer) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Index the masters of distributed-optimizer parameters that have no ``main_param``."""

    masters: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def add(model_param, value):
        if id(model_param) in masters:
            raise RuntimeError("distributed optimizer lists a parameter twice")
        masters[id(model_param)] = value

    for wrapper in _optimizer_wrappers(optimizer):
        for model_param, shard in _paired_groups(wrapper, "model_fp32_groups", "shard_fp32_groups"):
            add(model_param, (shard, shard))
        inner = getattr(getattr(wrapper, "optimizer", None), "param_to_inner_param", None)
        for model_param, shard in _paired_groups(wrapper, "model_float16_groups", "shard_float16_groups"):
            if getattr(model_param, "main_param", None) is not None:
                continue  # read through main_param
            fp32 = None if inner is None or shard is None else inner.get(shard)
            if fp32 is not None and fp32.dtype == torch.float32:
                add(model_param, (shard, fp32))
    return masters


def _distributed_views(optimizer) -> dict[int, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]]:
    """``id(model_param) -> (view, owned, fp32)`` of every distributed-optimizer range on this rank.

    Megatron pairs each model parameter with ``param.view(-1)[start:end]``
    (``shard_float16_groups`` / ``shard_fp32_groups``).  The base optimizer
    steps ``main_param`` (an FP32 clone of a BF16 range), the view itself (a
    native-FP32 parameter), or under the precision-aware optimizer the BF16
    view, whose FP32 values are a HybridDeviceOptimizer copy when offloaded."""

    views: dict[int, tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]] = {}
    for wrapper in _optimizer_wrappers(optimizer):
        inner = getattr(getattr(wrapper, "optimizer", None), "param_to_inner_param", None) or {}
        for model_param, view in _paired_groups(wrapper, "model_fp32_groups", "shard_fp32_groups"):
            views[id(model_param)] = (view, view, view)
        for model_param, view in _paired_groups(wrapper, "model_float16_groups", "shard_float16_groups"):
            main = getattr(model_param, "main_param", None)
            if main is not None:
                views[id(model_param)] = (view, main, main)
            else:
                views[id(model_param)] = (view, view, None if view is None else inner.get(view))
    return views


@torch.no_grad()
def _propagate_master_writes(actor, *, owned_ids: frozenset[int] | set[int] | None = None) -> int:
    """Make the tensors Megatron's HybridDeviceOptimizer steps agree with written masters.

    Under ``--optimizer-cpu-offload`` each base-optimizer parameter may have an
    inner FP32 copy (``param_to_inner_param``) that the sub-optimizer steps
    and copies back after every step; a write to only one side is undone by
    the next step. The FP32 side is the master the sync wrote, so:

    * an FP32 owned tensor (device master, or native-FP32 view) is copied
      into its inner copy;
    * a BF16 owned tensor (precision-aware: the model's own view) receives
      its FP32 inner copy, which is the master.

    Adam moments are left as they are (the non-distributed path keeps them too).

    ``owned_ids``: only the pairs whose optimizer-owned tensor is one of these
    (the masters a commit wrote); None propagates every pair.
    """

    pairs = 0
    for base in _optimizer_bases(actor.optimizer):
        inner = getattr(base, "param_to_inner_param", None)
        if inner is None:
            continue
        for owned, copy in inner.items():
            if copy is owned or (owned_ids is not None and id(owned) not in owned_ids):
                continue
            if copy.numel() != owned.numel():
                raise RuntimeError("hybrid optimizer inner copy has a different size")
            if owned.dtype == torch.float32:
                copy.copy_(owned.view(copy.shape), non_blocking=False)
            else:
                owned.copy_(copy.view(owned.shape), non_blocking=False)
            pairs += 1
    return pairs


def _optimizer_bases(optimizer) -> tuple[object, ...]:
    bases = []
    for wrapper in _optimizer_wrappers(optimizer):
        base = getattr(wrapper, "optimizer", wrapper)
        if any(base is existing for existing in bases):
            raise RuntimeError("full-parameter optimizer base is duplicated")
        bases.append(base)
    if not bases:
        raise RuntimeError("full-parameter optimizer has no base optimizer")
    return tuple(bases)


def _actor_parameter_context(actor):
    return _named_actor_parameters(actor), _optimizer_owned_parameter_ids(actor.optimizer)


def full_parameter_shard_manifest(
    actor,
    *,
    role: str = "actor",
) -> FullParameterShardManifest:
    """Describe this actor rank without copying any parameter payload."""

    require_full_parameter_actor(actor.args, "describe")
    topology = full_parameter_topology()
    replication = full_parameter_replication()
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    layout = _local_layout(
        named_parameters,
        role=role,
        topology=topology,
        replication=replication,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if layout.ranges is not None:
        sharded.refuse_unsupported_args(actor.args)
    return _manifest_from_layout(layout, topology=topology, role=role, global_rank=replication.global_rank)


def _manifest_from_layout(
    layout: _LocalLayout,
    *,
    topology: FullParameterTopology,
    role: str,
    global_rank: int,
) -> FullParameterShardManifest:
    return FullParameterShardManifest(
        topology=topology,
        role=role,
        layout_hash=_layout_hash(topology, layout.specs),
        specs=layout.specs,
        global_rank=global_rank,
        holdings=layout.holdings,
        sharded_masters=layout.ranges is not None,
    )


def _validate_fragment_plan_against_manifest(
    plan: FullParameterOwnerFragmentPlan,
    manifest: FullParameterShardManifest,
) -> None:
    if not isinstance(plan, FullParameterOwnerFragmentPlan):
        raise TypeError("full-parameter owner plan has the wrong type")
    if plan.topology != manifest.topology:
        raise RuntimeError("full-parameter owner plan topology changed")
    canonical_names = tuple(spec.wire_name for spec in manifest.specs)
    numel_by_name = {spec.wire_name: spec.numel for spec in manifest.specs}
    planned_names: list[str] = []
    cursor = {}
    for fragment in plan.fragments:
        unknown = [name for name in fragment.wire_names if name not in numel_by_name]
        if unknown:
            raise RuntimeError(f"full-parameter owner plan contains unknown parameter {unknown[0]!r}")
        members = fragment.members(numel_by_name)
        if sum(stop - start for _name, start, stop in members) != fragment.numel:
            raise RuntimeError("full-parameter fragment scalar count changed")
        for name, start, stop in members:
            # Each parameter is covered by consecutive ranges that start at
            # zero and end at its scalar count, in canonical order.
            if start != cursor.get(name, 0) or stop > numel_by_name[name]:
                raise RuntimeError("full-parameter fragment ranges do not tile the parameter")
            if start == 0:
                planned_names.append(name)
            cursor[name] = stop
    if any(cursor.get(name) != numel for name, numel in numel_by_name.items()):
        raise RuntimeError("full-parameter owner plan does not exactly cover the local shard")
    if tuple(planned_names) != canonical_names:
        raise RuntimeError("full-parameter owner plan does not preserve canonical spec order")


def install_full_parameter_fragment_plan(
    actor,
    plan: FullParameterOwnerFragmentPlan,
    *,
    role: str = "actor",
    replica_sources: Sequence[tuple[FullParameterShardManifest, FullParameterOwnerFragmentPlan]] = (),
) -> int:
    """Install one immutable exact-coverage owner plan on an actor rank.

    ``replica_sources``: ``(manifest, plan)`` of every owner of a parameter
    this rank holds a data-parallel replica of (``full_parameter_replicas``)."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    if getattr(actor, "_external_full_parameter_prepared_cut", None) is not None:
        raise RuntimeError("cannot change a full-parameter plan with a prepared cut")
    manifest = full_parameter_shard_manifest(actor, role=role)
    _validate_fragment_plan_against_manifest(plan, manifest)
    replica_plan = replicas.build_replica_plan(manifest, plan, tuple(replica_sources))
    installed = getattr(actor, "_external_full_parameter_fragment_plan", None)
    if installed is not None and installed != plan:
        raise RuntimeError("full-parameter owner plan is already installed")
    replicas.install_replica_plan(actor, replica_plan)
    actor._external_full_parameter_fragment_plan = plan
    return len(plan.fragments)


def _require_installed_layout(actor, *, role: str):
    """``(plan, manifest, replica plan, named parameters, optimizer-owned IDs, local layout)``.

    Refuses when the owned specs or the replica holdings changed since the
    plan was installed."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    plan = getattr(actor, "_external_full_parameter_fragment_plan", None)
    if not isinstance(plan, FullParameterOwnerFragmentPlan):
        raise RuntimeError("full-parameter owner plan is not installed")
    require_full_parameter_actor(actor.args, "describe")
    topology = full_parameter_topology()
    replication = full_parameter_replication()
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    layout = _local_layout(
        named_parameters,
        role=role,
        topology=topology,
        replication=replication,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    manifest = _manifest_from_layout(layout, topology=topology, role=role, global_rank=replication.global_rank)
    _validate_fragment_plan_against_manifest(plan, manifest)
    replica_plan = replicas.require_replica_plan(actor, manifest)
    return plan, manifest, replica_plan, named_parameters, optimizer_owned, layout


# HybridDeviceOptimizer (param_update_in_fp32) mirrors each inner FP32 copy
# into its per-parameter state as ``master_param``: that is the master a
# policy apply overwrites by design (and _validate_model_matches_masters
# checks), not Adam state, so the proof leaves it out.
_HYBRID_MASTER_STATE_KEY = "master_param"


def _adam_state(state: object) -> object:
    if isinstance(state, Mapping) and _HYBRID_MASTER_STATE_KEY in state:
        return {key: value for key, value in state.items() if key != _HYBRID_MASTER_STATE_KEY}
    return state


def _optimizer_state_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"miles-full-parameter-optimizer-state-v1\0")
    if not state:
        raise RuntimeError("selected optimizer parameter has no state")
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise RuntimeError("optimizer state has a non-canonical key")
        encoded_name = name.encode()
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        if isinstance(value, torch.Tensor):
            canonical = value.detach().to(device="cpu").contiguous()
            if canonical.is_floating_point() and not torch.isfinite(canonical).all().item():
                raise RuntimeError("optimizer state contains NaN or Inf")
            dtype = str(canonical.dtype).encode()
            shape = tuple(int(dimension) for dimension in canonical.shape)
            digest.update(b"tensor\0")
            digest.update(len(dtype).to_bytes(4, "little"))
            digest.update(dtype)
            digest.update(len(shape).to_bytes(4, "little"))
            for dimension in shape:
                digest.update(dimension.to_bytes(8, "little"))
            digest.update(memoryview(canonical.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        elif isinstance(value, bool):
            digest.update(b"bool\0" + bytes([value]))
        elif isinstance(value, int):
            digest.update(b"int\0" + struct.pack("<q", value))
        elif isinstance(value, float) and math.isfinite(value):
            digest.update(b"float\0" + struct.pack("<d", value))
        else:
            raise RuntimeError("optimizer state contains an unsupported value")
    return digest.hexdigest()


def full_parameter_optimizer_state(
    actor,
    *,
    role: str = "actor",
    check_model: bool = True,
) -> FullParameterOptimizerState:
    """Return a bounded proof of real optimizer progress on one actor rank.

    ``check_model=False`` skips the whole-model model/master comparison (a
    host-to-device pass over every master).  A caller that brackets a chunked
    apply with two proofs may skip it: the commit ends with the same
    comparison, and it copies every master into the model first.  The
    reported ``model_master_parameter_count`` is the number of compared
    parameters either way."""

    require_full_parameter_actor(actor.args, "inspect")
    assert_full_parameter_boundary_clear(actor)
    installed = getattr(actor, "_external_full_policy_version", None)
    local_step_generation = getattr(
        actor,
        "_external_full_local_step_generation",
        None,
    )
    last_rollout_id = getattr(actor, "_last_rollout_id", None)
    scheduler_num_steps = getattr(actor.opt_param_scheduler, "num_steps", None)
    for name, value in (
        ("installed policy version", installed),
        ("local-step generation", local_step_generation),
        ("last rollout ID", last_rollout_id),
        ("scheduler progress", scheduler_num_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"full-parameter {name} is unavailable")

    topology = full_parameter_topology()
    named_parameters, optimizer_owned, layout = _actor_local_layout(actor, role=role, topology=topology)
    if check_model:
        model_count = _validate_model_matches_masters(
            named_parameters,
            optimizer_owned_parameter_ids=optimizer_owned,
            ranges=layout.ranges,
        )
    else:
        model_count = len(tuple(named_parameters))
        if model_count < 1:
            raise RuntimeError("full-parameter model/master check is empty")
    # A data-parallel replica has no spec on this rank (its owner exports it);
    # its state counts, but the fingerprint is taken from an owned parameter.
    spec_by_name = {spec.name: spec for spec in layout.specs}
    name_by_master = _optimizer_owned_names(named_parameters, optimizer_owned, layout)
    populated: dict[str, Mapping[str, object]] = {}
    tensor_count = 0
    scalar_count = 0
    for base in _optimizer_bases(actor.optimizer):
        states = getattr(base, "state", None)
        if not isinstance(states, Mapping):
            raise RuntimeError("full-parameter optimizer state is unavailable")
        for parameter, state in states.items():
            state = _adam_state(state)
            if not state:
                continue
            name = name_by_master.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer state belongs to an unknown parameter")
            if name in populated or not isinstance(state, Mapping):
                raise RuntimeError("optimizer parameter state is malformed")
            populated[name] = state
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    tensor_count += 1
                    scalar_count += value.numel()
    owned = [(spec_by_name[name], state) for name, state in populated.items() if name in spec_by_name]
    if not owned and layout.ranges is not None:
        # Sharded masters: the Adam state of the parameters this rank exports
        # may all live on other ranks; fingerprint a local range instead.
        owned = [
            (_local_spec(role, topology, name, layout.ranges[name].parameter), state)
            for name, state in populated.items()
        ]
    if not owned:
        raise RuntimeError("full-parameter optimizer has not completed a step")
    selected_spec, selected_state = min(
        owned,
        key=lambda item: (item[0].numel, item[0].wire_name),
    )
    return FullParameterOptimizerState(
        topology=topology,
        role=role,
        installed_policy_version=installed,
        local_step_generation=local_step_generation,
        last_rollout_id=last_rollout_id,
        scheduler_num_steps=scheduler_num_steps,
        populated_parameter_count=len(populated),
        optimizer_state_tensor_count=tensor_count,
        optimizer_state_scalar_count=scalar_count,
        selected_wire_name=selected_spec.wire_name,
        selected_state_sha256=_optimizer_state_sha256(selected_state),
        model_master_parameter_count=model_count,
    )


def _master_ranges(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_owned: frozenset[int],
) -> Mapping[str, sharded.MasterRange] | None:
    """This rank's master ranges by name when masters are sharded, else None (see ``_local_layout``)."""

    ordered = tuple(sorted(named_parameters, key=lambda value: value[0]))
    return sharded.master_ranges(ordered, full_parameter_replication(), optimizer_owned)


def _optimizer_owned_names(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_owned: frozenset[int],
    layout: _LocalLayout,
) -> dict[int, str]:
    """``id(optimizer-owned tensor) -> parameter name`` of every local parameter."""

    if layout.ranges is not None:
        return sharded.owned_tensor_names(layout.ranges)
    return {
        id(_resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned)[0]): name
        for name, parameter in named_parameters
    }


def _local_spec(role: str, topology: FullParameterTopology, name: str, parameter: torch.Tensor) -> FullParameterShardSpec:
    shape = tuple(int(dimension) for dimension in parameter.shape)
    return FullParameterShardSpec(role, topology.shard_id, name, shape, "float32", math.prod(shape))


def _validate_model_matches_masters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    optimizer_owned_parameter_ids: frozenset[int],
    ranges: Mapping[str, sharded.MasterRange] | None = None,
) -> int:
    """Model == master for every row; ``ranges`` (sharded masters): over each
    row's range on this rank only.  Returns the rows checked."""

    if ranges is not None:
        return sharded.check_model(named_parameters, ranges)
    count = 0
    for name, parameter in named_parameters:
        master = _master_parameter(
            parameter,
            name,
            optimizer_owned_parameter_ids=optimizer_owned_parameter_ids,
        )
        expected = master.detach().to(device=parameter.device, dtype=parameter.dtype)
        if not torch.equal(parameter.detach(), expected):
            raise RuntimeError(f"{name!r} model value differs from its optimizer master")
        count += 1
    if count < 1:
        raise RuntimeError("full-parameter model/master check is empty")
    return count


def _scheduler_num_steps(actor) -> int:
    value = getattr(actor.opt_param_scheduler, "num_steps", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("full-parameter scheduler progress is unavailable")
    return value


def _optimizer_steps_per_local_round(
    actor,
    *,
    role: str = "actor",
) -> tuple[int, int]:
    optimizer_steps = getattr(actor.args, "num_steps_per_rollout", None)
    global_batch_size = getattr(actor.args, "global_batch_size", None)
    for name, value in (
        ("optimizer steps per rollout", optimizer_steps),
        ("global batch size", global_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"full-parameter {name} is invalid")
    if role == "critic":
        critic_epochs = getattr(actor.args, "num_critic_epochs", 1)
        if (
            isinstance(critic_epochs, bool)
            or not isinstance(critic_epochs, int)
            or critic_epochs < 1
        ):
            raise RuntimeError("full-parameter critic epochs is invalid")
        optimizer_steps *= critic_epochs
    return optimizer_steps, optimizer_steps * global_batch_size


def _validate_scheduler_tracking_congruence(
    actor,
    *,
    role: str = "actor",
) -> None:
    _optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(
        actor,
        role=role,
    )
    _installed, generation, _exported, anchor = _require_full_parameter_tracking(actor)
    if _scheduler_num_steps(actor) != anchor + generation * scheduler_increment:
        raise RuntimeError("full-parameter scheduler progress has no local-step receipt")


def full_parameter_initial_policy_version(args, loaded_rollout_id: int) -> int:
    """Match the external policy identity to Miles' effective train-loop start."""

    if isinstance(loaded_rollout_id, bool) or not isinstance(loaded_rollout_id, int) or loaded_rollout_id < -1:
        raise RuntimeError("full-parameter loaded rollout ID is invalid")
    start = getattr(args, "start_rollout_id", None)
    if start is None:
        return loaded_rollout_id + 1
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise RuntimeError("full-parameter configured rollout start is invalid")
    return start


def initialize_full_parameter_tracking(actor, policy_version: int) -> None:
    """Initialize the global-version boundary from Miles checkpoint progress."""

    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise RuntimeError("full-parameter initial policy version is invalid")
    installed = getattr(actor, "_external_full_policy_version", None)
    if installed is None:
        anchor = _scheduler_num_steps(actor)
        actor._external_full_policy_version = policy_version
        actor._external_full_local_step_generation = 0
        actor._external_full_exported_local_step_generation = 0
        actor._external_full_scheduler_anchor_steps = anchor
        actor._external_full_last_local_rollout_id = None
        actor._external_full_last_local_step_receipt = None
        forget_full_parameter_resident_fragments(actor)
        return
    installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != policy_version:
        raise RuntimeError("full-parameter policy tracking is already initialized")


def rebase_full_parameter_scheduler_anchor(actor, *, reason: str) -> int:
    """Re-anchor scheduler tracking after an external whole-state restore.

    A peer snapshot restore loads the donor's LR scheduler progress into a
    trainer that has taken no local step since tracking began, so the old
    anchor no longer describes it. The restore itself is the proof of the new
    progress (the caller checks it against the donor's digest); this only
    moves the anchor, and only when no local step or policy commit is pending.
    The normal path keeps requiring a local-step receipt for any progress.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("full-parameter scheduler rebase needs an explicit reason")
    assert_full_parameter_boundary_clear(actor)
    _installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if generation != 0 or exported != 0:
        raise RuntimeError("full-parameter scheduler rebase would discard local-step progress")
    if getattr(actor, "_external_full_last_local_step_receipt", None) is not None:
        raise RuntimeError("full-parameter scheduler rebase would discard a local-step receipt")
    anchor = _scheduler_num_steps(actor)
    actor._external_full_scheduler_anchor_steps = anchor
    actor._external_full_scheduler_rebase_reason = reason
    # The restore that needs this rebase wrote the masters.
    forget_full_parameter_resident_fragments(actor)
    return anchor


def _require_full_parameter_tracking(actor) -> tuple[int, int, int, int]:
    """Return initialized tracking state without silently creating it."""

    installed = getattr(actor, "_external_full_policy_version", None)
    generation = getattr(actor, "_external_full_local_step_generation", None)
    exported = getattr(
        actor,
        "_external_full_exported_local_step_generation",
        None,
    )
    anchor = getattr(actor, "_external_full_scheduler_anchor_steps", None)
    for name, value in (
        ("installed policy version", installed),
        ("local-step generation", generation),
        ("exported local-step generation", exported),
        ("scheduler anchor", anchor),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"full-parameter {name} is unavailable")
    if exported > generation:
        raise RuntimeError("full-parameter exported generation is ahead of local progress")
    last_rollout = getattr(actor, "_external_full_last_local_rollout_id", None)
    if last_rollout is not None and (isinstance(last_rollout, bool) or not isinstance(last_rollout, int) or last_rollout < 0):
        raise RuntimeError("full-parameter local rollout progress is unavailable")
    return installed, generation, exported, anchor


def assert_full_parameter_boundary_clear(actor) -> None:
    """Reject any operation that could observe an unresolved policy commit."""

    if getattr(actor, "_external_full_parameter_prepared_cut", None) is not None:
        raise RuntimeError("a prepared full-parameter cut blocks boundary mutation until commit or abort")
    if getattr(actor, "_external_full_parameter_committed_cut", None) is not None:
        raise RuntimeError("an unfinalized full-parameter commit blocks boundary mutation")


def _reject_prepared_full_parameter_boundary_mutation(actor) -> None:
    # Compatibility name for the materialized transport path.
    assert_full_parameter_boundary_clear(actor)


def record_full_parameter_local_step(
    actor,
    *,
    base_policy_version: int,
    rollout_id: int,
    role: str = "actor",
    check_model: bool = True,
) -> FullParameterLocalStepReceipt:
    """Commit one local-step generation after exact scheduler progress.

    ``check_model=False`` skips the whole-model model/master comparison.  A
    caller does so only when its next export is lazy: that export reads no
    master, and every fragment it later materializes compares its own
    parameters with their masters."""

    require_full_parameter_actor(actor.args, "record")
    _reject_prepared_full_parameter_boundary_mutation(actor)
    if not _ROLE.fullmatch(role):
        raise RuntimeError("full-parameter local-step role is invalid")
    installed, generation, _exported, anchor = _require_full_parameter_tracking(actor)
    previous_rollout = actor._external_full_last_local_rollout_id
    if installed != base_policy_version:
        raise RuntimeError("full-parameter local step has a stale global base")
    previous_receipt = getattr(
        actor,
        "_external_full_last_local_step_receipt",
        None,
    )
    if previous_receipt is not None:
        if not isinstance(previous_receipt, FullParameterLocalStepReceipt):
            raise RuntimeError("full-parameter local-step journal is malformed")
        if previous_receipt.base_policy_version == base_policy_version and previous_receipt.rollout_id == rollout_id and previous_receipt.role == role:
            if generation != previous_receipt.local_step_generation or previous_rollout != rollout_id or _scheduler_num_steps(actor) != previous_receipt.scheduler_end_steps:
                raise RuntimeError("full-parameter local-step journal disagrees with actor")
            return previous_receipt
    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0 or getattr(actor, "_last_rollout_id", None) != rollout_id or (previous_rollout is not None and rollout_id <= previous_rollout):
        raise RuntimeError("full-parameter local step has invalid rollout progress")
    optimizer_steps, scheduler_increment = _optimizer_steps_per_local_round(
        actor,
        role=role,
    )
    start = anchor + generation * scheduler_increment
    end = _scheduler_num_steps(actor)
    if end != start + scheduler_increment:
        raise RuntimeError("full-parameter local step lacks exact optimizer progress")
    if check_model:
        named_parameters, optimizer_owned = _actor_parameter_context(actor)
        _validate_model_matches_masters(
            named_parameters,
            optimizer_owned_parameter_ids=optimizer_owned,
            ranges=_master_ranges(named_parameters, optimizer_owned),
        )
    generation += 1
    # The local step changed the masters.
    forget_full_parameter_resident_fragments(actor)
    receipt = FullParameterLocalStepReceipt(
        topology=full_parameter_topology(),
        role=role,
        base_policy_version=base_policy_version,
        local_step_generation=generation,
        rollout_id=rollout_id,
        optimizer_steps=optimizer_steps,
        scheduler_start_steps=start,
        scheduler_end_steps=end,
    )
    actor._external_full_local_step_generation = generation
    actor._external_full_last_local_rollout_id = rollout_id
    actor._external_full_last_local_step_receipt = receipt
    return receipt


def export_full_parameter_shard(
    actor,
    *,
    policy_version: int,
    local_step_generation: int = 0,
    role: str = "actor",
) -> FullParameterShardState:
    require_full_parameter_actor(actor.args, "export")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if policy_version != installed:
        raise RuntimeError("full-parameter export policy version is not installed")
    if local_step_generation != generation:
        raise RuntimeError("full-parameter export local-step generation is stale")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    state = make_full_parameter_shard_state(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    actor._external_full_exported_local_step_generation = local_step_generation
    return state


def validate_full_parameter_shard(
    actor,
    state: FullParameterShardState,
    *,
    role: str = "actor",
) -> int:
    require_full_parameter_actor(actor.args, "validate")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed = getattr(actor, "_external_full_policy_version", -1)
    if getattr(actor, "_external_full_exported_local_step_generation", -1) != getattr(
        actor,
        "_external_full_local_step_generation",
        -2,
    ):
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    return validate_full_parameter_shard_values(
        state,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )


@torch.no_grad()
def apply_full_parameter_shard(
    actor,
    state: FullParameterShardState,
    *,
    role: str = "actor",
) -> int:
    require_full_parameter_actor(actor.args, "apply to")
    _reject_prepared_full_parameter_boundary_mutation(actor)
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed = getattr(actor, "_external_full_policy_version", -1)
    if getattr(actor, "_external_full_exported_local_step_generation", -1) != getattr(
        actor,
        "_external_full_local_step_generation",
        -2,
    ):
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    forget_full_parameter_resident_fragments(actor)
    named_parameters, optimizer_owned = _actor_parameter_context(actor)
    count = apply_full_parameter_shard_values(
        state,
        role=role,
        topology=full_parameter_topology(),
        named_parameters=named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    _propagate_master_writes(actor)
    _copy_masters_to_model(actor)
    model_count = _validate_model_matches_masters(
        named_parameters,
        optimizer_owned_parameter_ids=optimizer_owned,
    )
    if model_count != count:
        raise RuntimeError("full-parameter model/master check is incomplete")
    if role == "actor":
        actor.weights_backuper.backup("actor")
    actor._external_full_policy_version = state.policy_version
    actor._external_full_local_step_generation = 0
    actor._external_full_exported_local_step_generation = 0
    actor._external_full_scheduler_anchor_steps = _scheduler_num_steps(actor)
    actor._external_full_last_local_rollout_id = None
    actor._external_full_last_local_step_receipt = None
    return count


def _resolve_ray_module(ray_module):
    if ray_module is None:
        import ray as ray_module

    for name in ("put", "get"):
        if not callable(getattr(ray_module, name, None)):
            raise TypeError(f"full-parameter Ray module has no {name} operation")
    return ray_module


_ZERO_COPY_ENV = "MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS"
# A chunk read back through Ray's zero-copy path is a tensor over the object
# store's read-only memory; every full-parameter reader only reads it.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)


class _ObjectStoreChunk:
    """The ``ray.put`` form of one CPU chunk: pickled as a numpy array.

    Ray pickles a torch tensor through an in-memory ``torch.save`` on put and
    a ``torch.load`` on get (measured 0.30 and 0.58 GB/s per thread for 256
    MiB of f32).  A numpy array travels as one pickle-5 out-of-band buffer:
    the put is one copy into the object store and the get maps the stored
    bytes, so a reader receives a tensor of the same dtype and shape over the
    object store's memory.  Stored chunks are immutable and every reader only
    reads them (checks, hashes, deltas, copies into the masters)."""

    __slots__ = ("value",)

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def __reduce__(self):
        return (_chunk_from_object_store, (self.value.detach().contiguous().numpy(),))


def _chunk_from_object_store(array) -> torch.Tensor:
    return torch.from_numpy(array)


def put_full_parameter_chunk(ray_module, value: torch.Tensor):
    """``ray_module.put`` of one CPU chunk, in the zero-copy form for the
    real Ray module (``MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS=0`` keeps the
    tensor pickling).  An in-process stand-in stores the tensor itself."""

    if getattr(ray_module, "__name__", None) == "ray" and os.environ.get(_ZERO_COPY_ENV, "1") != "0":
        return ray_module.put(_ObjectStoreChunk(value))
    return ray_module.put(value)


def _validate_cpu_chunk(
    value: object,
    descriptor: FullParameterChunkDescriptor,
    *,
    fragment_id: int,
    verify_values: bool,
    parameter_layout_hash: str,
    plan_hash: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 1 or not value.is_contiguous() or value.numel() != descriptor.numel:
        raise ValueError("full-parameter Ray chunk is malformed")
    if verify_values:
        if not _all_finite(value):
            raise ValueError("full-parameter Ray chunk contains NaN or Inf")
        if descriptor.payload_hash != _chunk_hash(
            fragment_id,
            descriptor.chunk_index,
            descriptor.flat_offset,
            value,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        ):
            raise ValueError("full-parameter Ray chunk hash changed")
    return value


def _fragment_sources(
    fragment: FullParameterFragmentPlan,
    layout: _LocalLayout,
    *,
    contributions: Mapping[int, Sequence[sharded.FullParameterMasterContribution]] | None = None,
    ray_module=None,
) -> list[torch.Tensor]:
    """Flat FP32 tensors whose concatenation is ``fragment``'s payload on its owner.

    Whole masters: the member ranges of the owned masters.  Sharded masters:
    the owner's own ranges plus ``contributions[fragment_id]``, the ranges the
    other replicas stored for it (``full_parameter_sharded_masters``)."""

    if layout.ranges is None:
        masters = layout.masters
        members = fragment.members({name: masters[name].numel() for name in fragment.wire_names})
        return [masters[name].detach().reshape(-1)[start:stop] for name, start, stop in members]
    return sharded.fragment_sources(
        fragment,
        numel_by_wire={spec.wire_name: spec.numel for spec in layout.specs},
        ranges_by_wire=layout.ranges_by_wire_name(),
        contributions=(contributions or {}).get(fragment.fragment_id, ()),
        ray_module=ray_module,
    )


def _plan_fragment_ids(plan: FullParameterOwnerFragmentPlan) -> tuple[int, ...]:
    return tuple(fragment.fragment_id for fragment in plan.fragments)


def _contributions_by_fragment(
    actor,
    layout: _LocalLayout,
    contributions: Sequence[sharded.FullParameterMasterContribution],
    *,
    owner: FullParameterTopology,
    fragment_ids: Sequence[int],
) -> dict[int, tuple[sharded.FullParameterMasterContribution, ...]]:
    """Checked contributions by fragment; none are accepted with whole masters."""

    if layout.ranges is None:
        if contributions:
            raise RuntimeError("full-parameter contributions need sharded masters")
        return {}
    return sharded.contributions_by_fragment(
        contributions, owner=owner, fragment_ids=fragment_ids, tracking=_tracking_identity(actor)
    )


def _tracking_identity(actor) -> sharded.Tracking:
    """``(installed policy version, local-step generation, scheduler steps)``."""

    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    return installed, generation, _scheduler_num_steps(actor)


def _add_lap(spent: dict[str, float], key: str, since: float) -> float:
    """Add the seconds since ``since`` to ``spent[key]``; returns now."""

    now = time.monotonic()
    spent[key] += now - since
    return now


def _export_fragment_chunks(
    fragment: FullParameterFragmentPlan,
    *,
    sources: Sequence[torch.Tensor],
    max_chunk_numel: int,
    parameter_layout_hash: str,
    plan_hash: str,
    ray_module,
    overlap_checks: bool = True,
    timings: dict[str, float] | None = None,
) -> tuple[FullParameterFragmentDescriptor, list[object | None]]:
    """``sources``: flat FP32 tensors whose concatenation is the fragment
    payload (:func:`_fragment_sources`).  ``overlap_checks=False``: run the
    finite check and the chunk hash in this thread (callers that export several
    fragments on the scan pool).  ``timings``: adds this call's seconds of
    copying out of the sources (``copy_out_seconds``), waiting for the finite
    check and hashes (``check_seconds``) and storing (``put_seconds``)."""

    submit = _hash_pool().submit if overlap_checks else _run_now
    spent = {"copy_out_seconds": 0.0, "check_seconds": 0.0, "put_seconds": 0.0}
    source_index = 0
    source_offset = 0
    flat_offset = 0
    chunk_descriptors = []
    chunk_refs: list[object | None] = []
    fragment_digest = _fragment_hasher(
        fragment.fragment_id,
        parameter_layout_hash,
        plan_hash,
    )
    while flat_offset < fragment.numel:
        lap = time.monotonic()
        chunk_numel = min(max_chunk_numel, fragment.numel - flat_offset)
        chunk = torch.empty(chunk_numel, device="cpu", dtype=torch.float32)
        destination_offset = 0
        while destination_offset < chunk_numel:
            if source_index >= len(sources):
                raise RuntimeError("full-parameter fragment source ended early")
            source = sources[source_index]
            count = min(
                chunk_numel - destination_offset,
                source.numel() - source_offset,
            )
            chunk[destination_offset : destination_offset + count].copy_(
                source[source_offset : source_offset + count],
                non_blocking=False,
            )
            destination_offset += count
            source_offset += count
            if source_offset == source.numel():
                source_index += 1
                source_offset = 0
        chunk_index = len(chunk_descriptors)
        lap = _add_lap(spent, "copy_out_seconds", lap)
        # The finite check, the chunk hash and the fragment digest read the
        # same immutable chunk; they run concurrently and give the same bytes.
        finite = submit(_all_finite, chunk)
        chunk_hash = submit(
            _chunk_hash,
            fragment.fragment_id,
            chunk_index,
            flat_offset,
            chunk,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        raw = _tensor_bytes(chunk)
        if not finite.result():
            chunk_hash.result()
            raise ValueError("full-parameter optimizer master contains NaN or Inf")
        fragment_digest.update(raw)
        descriptor = FullParameterChunkDescriptor(
            chunk_index=chunk_index,
            flat_offset=flat_offset,
            numel=chunk_numel,
            payload_hash=chunk_hash.result(),
        )
        lap = _add_lap(spent, "check_seconds", lap)
        reference = put_full_parameter_chunk(ray_module, chunk)
        if reference is None:
            raise RuntimeError("Ray returned an empty full-parameter chunk reference")
        _add_lap(spent, "put_seconds", lap)
        chunk_descriptors.append(descriptor)
        chunk_refs.append(reference)
        flat_offset += chunk_numel
        del chunk, raw
    if source_index != len(sources) or source_offset != 0:
        raise RuntimeError("full-parameter fragment did not consume every source")
    if timings is not None:
        for key, seconds in spent.items():
            timings[key] = timings.get(key, 0.0) + seconds
    return (
        FullParameterFragmentDescriptor(
            fragment_id=fragment.fragment_id,
            numel=fragment.numel,
            payload_hash=fragment_digest.hexdigest(),
            chunks=tuple(chunk_descriptors),
        ),
        chunk_refs,
    )


def _describe_fragment(
    fragment: FullParameterFragmentPlan,
    *,
    sources: Sequence[torch.Tensor],
    max_chunk_numel: int,
    parameter_layout_hash: str,
    plan_hash: str,
) -> FullParameterFragmentDescriptor:
    """The descriptor ``_export_fragment_chunks`` returns, hashed in place.

    Same finite check, chunk hashes and fragment digest over the same bytes in
    the same order, read straight from the masters in cache-sized blocks: no
    chunk copy and no Ray object.  SHA-256 is a stream, so feeding a chunk in
    blocks gives the digest of the whole chunk."""

    source_index = 0
    source_offset = 0
    flat_offset = 0
    chunk_descriptors = []
    fragment_digest = _fragment_hasher(
        fragment.fragment_id,
        parameter_layout_hash,
        plan_hash,
    )
    while flat_offset < fragment.numel:
        chunk_numel = min(max_chunk_numel, fragment.numel - flat_offset)
        chunk_index = len(chunk_descriptors)
        chunk_digest = _chunk_hasher(
            fragment.fragment_id,
            chunk_index,
            flat_offset,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        remaining = chunk_numel
        while remaining:
            if source_index >= len(sources):
                raise RuntimeError("full-parameter fragment source ended early")
            source = sources[source_index]
            count = min(remaining, source.numel() - source_offset, _SCAN_BLOCK_NUMEL)
            block = source[source_offset : source_offset + count]
            if block.device.type != "cpu" or block.dtype != torch.float32 or not block.is_contiguous():
                block = block.to(device="cpu", dtype=torch.float32).contiguous()
            if not _all_finite(block):
                raise ValueError("full-parameter optimizer master contains NaN or Inf")
            raw = _tensor_bytes(block)
            chunk_digest.update(raw)
            fragment_digest.update(raw)
            del raw, block
            remaining -= count
            source_offset += count
            if source_offset == source.numel():
                source_index += 1
                source_offset = 0
        chunk_descriptors.append(
            FullParameterChunkDescriptor(
                chunk_index=chunk_index,
                flat_offset=flat_offset,
                numel=chunk_numel,
                payload_hash=chunk_digest.hexdigest(),
            )
        )
        flat_offset += chunk_numel
    if source_index != len(sources) or source_offset != 0:
        raise RuntimeError("full-parameter fragment did not consume every source")
    return FullParameterFragmentDescriptor(
        fragment_id=fragment.fragment_id,
        numel=fragment.numel,
        payload_hash=fragment_digest.hexdigest(),
        chunks=tuple(chunk_descriptors),
    )


def _lazy_fragment_descriptor(
    fragment: FullParameterFragmentPlan,
    *,
    max_chunk_numel: int,
    parameter_layout_hash: str,
    plan_hash: str,
    nonce: bytes,
) -> FullParameterFragmentDescriptor:
    """``fragment``'s chunk geometry with label hashes; no master is read.

    Each chunk and the fragment get a SHA-256 over their own lazy domain, the
    same identity fields as the content hash and ``nonce``, which is fresh for
    every lazy export.  The label therefore names one export of the masters
    and never equals a content hash; ``materialize_full_parameter_fragments``
    computes the content descriptor when a fragment leaves the actor."""

    chunks = []
    flat_offset = 0
    while flat_offset < fragment.numel:
        chunk_numel = min(max_chunk_numel, fragment.numel - flat_offset)
        digest = hashlib.sha256(_LAZY_CHUNK_DOMAIN)
        digest.update(parameter_layout_hash.encode("ascii"))
        digest.update(plan_hash.encode("ascii"))
        digest.update(fragment.fragment_id.to_bytes(8, "little"))
        digest.update(len(chunks).to_bytes(8, "little"))
        digest.update(flat_offset.to_bytes(8, "little"))
        digest.update(nonce)
        chunks.append(
            FullParameterChunkDescriptor(
                chunk_index=len(chunks),
                flat_offset=flat_offset,
                numel=chunk_numel,
                payload_hash=digest.hexdigest(),
            )
        )
        flat_offset += chunk_numel
    digest = hashlib.sha256(_LAZY_FRAGMENT_DOMAIN)
    digest.update(parameter_layout_hash.encode("ascii"))
    digest.update(plan_hash.encode("ascii"))
    digest.update(fragment.fragment_id.to_bytes(8, "little"))
    digest.update(nonce)
    return FullParameterFragmentDescriptor(
        fragment_id=fragment.fragment_id,
        numel=fragment.numel,
        payload_hash=digest.hexdigest(),
        chunks=tuple(chunks),
    )


def _same_fragment_geometry(
    first: FullParameterFragmentDescriptor,
    second: FullParameterFragmentDescriptor,
) -> bool:
    return (
        first.fragment_id == second.fragment_id
        and first.numel == second.numel
        and len(first.chunks) == len(second.chunks)
        and all(
            (a.chunk_index, a.flat_offset, a.numel) == (b.chunk_index, b.flat_offset, b.numel)
            for a, b in zip(first.chunks, second.chunks, strict=True)
        )
    )


class _Done:
    """A finished future (``_hash_pool().submit`` without the pool)."""

    __slots__ = ("_value", "_error")

    def __init__(self, value=None, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._value


def _run_now(function, *args, **kwargs) -> _Done:
    try:
        return _Done(function(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 - raised by result(), like a future
        return _Done(error=error)


def _parallel_map(function, items: Sequence) -> list:
    """``[function(item) for item in items]`` on the scan pool, in order.

    Every item is awaited before an error propagates, so no worker still
    touches the masters (or the object store) when the caller moves on."""

    if len(items) <= 1:
        return [function(item) for item in items]
    futures = [_scan_pool().submit(function, item) for item in items]
    results, failure = [], None
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as error:  # noqa: BLE001 - re-raised after every worker ended
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return results


def export_full_parameter_chunked_shard(
    actor,
    *,
    policy_version: int,
    local_step_generation: int = 0,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    role: str = "actor",
    ray_module=None,
    defer_payload: bool = False,
    lazy: bool = False,
    contributions: Sequence[sharded.FullParameterMasterContribution] = (),
) -> FullParameterChunkedShardState:
    """Export bounded owner chunks without materializing a rank or global cut.

    ``defer_payload``: describe every fragment exactly as the full export
    does (same descriptors, same hashes, same NaN/Inf refusal) but store no
    bytes in Ray; every chunk ref is ``DEFERRED_FULL_PARAMETER_CHUNK``.
    ``materialize_full_parameter_fragments`` stores the fragments a caller
    needs while the masters still hold them.

    ``lazy`` (with ``defer_payload``): read no master at all.  Every
    descriptor carries the real chunk geometry and label hashes
    (``_lazy_fragment_descriptor``), so the cut is a unique name for this
    export, not a content proof, and no NaN/Inf check runs.  The content
    descriptor, the NaN/Inf check and the model/master check of a fragment
    run when ``materialize_full_parameter_fragments`` stores it.

    ``contributions``: with sharded masters, the master ranges the other
    replicas hold of this rank's fragments (``full_parameter_master_contributions``),
    at this tracking state; a lazy export reads none."""

    require_full_parameter_actor(actor.args, "export")
    if lazy and not defer_payload:
        raise ValueError("a lazy full-parameter export must defer its payload")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    _validate_non_negative_integer(policy_version, "policy version")
    _validate_non_negative_integer(
        local_step_generation,
        "local-step generation",
    )
    installed, generation, _exported, _anchor = _require_full_parameter_tracking(actor)
    if policy_version != installed:
        raise RuntimeError("full-parameter export policy version is not installed")
    if local_step_generation != generation:
        raise RuntimeError("full-parameter export local-step generation is stale")
    started = time.monotonic()
    max_chunk_numel = _validate_chunk_byte_limit(max_chunk_bytes)
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest, _replica_plan, _named, _owned, layout = _require_installed_layout(actor, role=role)
    contributed = _contributions_by_fragment(
        actor, layout, contributions, owner=manifest.topology, fragment_ids=() if lazy else _plan_fragment_ids(plan)
    )

    def sources(fragment: FullParameterFragmentPlan) -> list[torch.Tensor]:
        return _fragment_sources(fragment, layout, contributions=contributed, ray_module=ray_module)

    fragment_descriptors = []
    chunk_refs = []
    lazy_ids: frozenset[int] = frozenset()
    if lazy:
        nonce = os.urandom(32)
        fragment_descriptors = [
            _lazy_fragment_descriptor(
                fragment,
                max_chunk_numel=max_chunk_numel,
                parameter_layout_hash=plan.parameter_layout_hash,
                plan_hash=plan.plan_hash,
                nonce=nonce,
            )
            for fragment in plan.fragments
        ]
        lazy_ids = frozenset(fragment.fragment_id for fragment in plan.fragments)
        chunk_refs = [[DEFERRED_FULL_PARAMETER_CHUNK] * len(descriptor.chunks) for descriptor in fragment_descriptors]
    elif defer_payload:
        fragment_descriptors = _parallel_map(
            lambda fragment: _describe_fragment(
                fragment,
                sources=sources(fragment),
                max_chunk_numel=max_chunk_numel,
                parameter_layout_hash=plan.parameter_layout_hash,
                plan_hash=plan.plan_hash,
            ),
            plan.fragments,
        )
        chunk_refs = [[DEFERRED_FULL_PARAMETER_CHUNK] * len(descriptor.chunks) for descriptor in fragment_descriptors]
    for fragment in () if defer_payload else plan.fragments:
        descriptor, references = _export_fragment_chunks(
            fragment,
            sources=sources(fragment),
            max_chunk_numel=max_chunk_numel,
            parameter_layout_hash=plan.parameter_layout_hash,
            plan_hash=plan.plan_hash,
            ray_module=ray_module,
        )
        fragment_descriptors.append(descriptor)
        chunk_refs.append(references)
    state = FullParameterChunkedShardState(
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        topology=manifest.topology,
        parameter_layout_hash=plan.parameter_layout_hash,
        plan_hash=plan.plan_hash,
        fragments=tuple(fragment_descriptors),
        chunk_refs=chunk_refs,
    )
    actor._external_full_exported_local_step_generation = local_step_generation
    _record_resident_fragments(
        actor,
        parameter_layout_hash=plan.parameter_layout_hash,
        plan_hash=plan.plan_hash,
        policy_version=policy_version,
        local_step_generation=local_step_generation,
        fragments=fragment_descriptors,
        lazy_fragment_ids=lazy_ids,
    )
    _note_apply_stats(
        actor,
        "export",
        fragments=len(fragment_descriptors),
        deferred=bool(defer_payload),
        lazy=bool(lazy),
        seconds=time.monotonic() - started,
    )
    return state


def note_full_parameter_replica_export(
    actor,
    *,
    states: Sequence[FullParameterChunkedShardState],
    role: str = "actor",
    lazy: bool = False,
) -> int:
    """Record that this rank's data-parallel replicas hold what their owners exported.

    ``states``: the exported shards of the owners this rank replicates, at
    this rank's tracking state (the group transport calls this right after
    every chunked export).  A later skip-resident apply then reads only the
    owner fragments that changed, as the owners do; a deferred owner fragment
    is readable only through this record.  ``lazy``: the owners' export was
    lazy (labels, no content hashes), so the replicas are not hashed against
    it.  Returns the replica fragments recorded."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    require_full_parameter_actor(actor.args, "note")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    started = time.monotonic()
    _plan, _manifest, replica_plan, _named, _owned, layout = _require_installed_layout(actor, role=role)
    stats = replicas.note_replica_export(
        actor,
        replica_plan,
        tuple(states),
        replica_masters=None if layout.ranges is not None else layout.replica_masters,
        lazy=lazy,
    )
    _note_apply_stats(actor, "replica_export", **stats, seconds=time.monotonic() - started)
    return stats["fragments"]


def full_parameter_master_contributions(
    actor,
    *,
    requests: Sequence[tuple[FullParameterTopology, tuple[int, ...] | None]],
    held: Sequence[tuple[FullParameterTopology, int, int, str]] | None = None,
    max_chunk_bytes: int = FULL_PARAMETER_DEFAULT_CHUNK_BYTES,
    role: str = "actor",
    ray_module=None,
) -> tuple[sharded.FullParameterMasterContribution, ...]:
    """Store this rank's FP32 master ranges of other ranks' fragments in Ray (sharded masters).

    ``requests``: ``(owner topology, fragment IDs)`` for owners whose
    parameters this rank replicates; None asks for every fragment of that
    owner it holds a range of.  The owners pass the returned contributions to
    their export or materialize (``full_parameter_sharded_masters``).

    ``held`` (for a materialize): ``(owner topology, fragment ID, numel,
    payload hash)`` of every requested fragment in the cut being stored; each
    contributed fragment must be one this rank's live replica record holds with
    exactly that hash (no local step, restore or commit start since).  With
    whole masters there is nothing to contribute and ``requests`` must be empty."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle
    require_full_parameter_actor(actor.args, "contribute")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    started = time.monotonic()
    max_chunk_numel = _validate_chunk_byte_limit(max_chunk_bytes)
    _plan, manifest, replica_plan, _named, _owned, layout = _require_installed_layout(actor, role=role)
    if layout.ranges is None:
        if requests:
            raise RuntimeError("full-parameter contributions need sharded masters")
        return ()
    holds = None
    if held is not None:
        expected = {(owner, fragment_id): (numel, payload_hash) for owner, fragment_id, numel, payload_hash in held}
        holds = replicas.replica_holds(actor, replica_plan, expected)
    ray_module = _resolve_ray_module(ray_module)
    contributions = sharded.contribute(
        topology=manifest.topology,
        tracking=_tracking_identity(actor),
        replica_plan=replica_plan,
        ranges=layout.ranges,
        requests=tuple(requests),
        resident=holds,
        max_chunk_numel=max_chunk_numel,
        put=lambda value: put_full_parameter_chunk(ray_module, value),
    )
    _note_apply_stats(
        actor,
        "contribute",
        contributions=len(contributions),
        scalars=sum(contribution.numel for contribution in contributions),
        seconds=time.monotonic() - started,
    )
    return contributions


def materialize_full_parameter_fragments(
    actor,
    *,
    state: FullParameterChunkedShardState,
    fragment_ids: Sequence[int],
    role: str = "actor",
    ray_module=None,
    return_descriptors: bool = False,
    contributions: Sequence[sharded.FullParameterMasterContribution] = (),
) -> tuple[tuple, ...]:
    """Store the bytes of deferred fragments of ``state`` while the masters hold them.

    Each requested fragment must be resident: its payload hash equals the one
    the live resident record holds (same tracking state: no local step,
    restore or commit start since the export or commit that recorded it).
    The fragment is exported again from the masters with the descriptor's
    chunk geometry and its descriptor must come out identical, so the stored
    bytes are exactly the bytes the deferred export hashed.  Returns
    ``(fragment_id, chunk refs)`` in fragment order; the masters, the record
    and the tracking state do not change.

    A fragment of a lazy export carries labels instead of content hashes:
    the record proves the masters still hold the exported bytes, the export
    here computes the content descriptor (with the NaN/Inf check) and its
    model parameters are compared with their masters.  Its descriptor then
    differs from the label, so the caller must ask for it
    (``return_descriptors``: rows are ``(fragment_id, chunk refs,
    descriptor)``) and read the stored chunks through it.

    ``contributions``: with sharded masters, the master ranges the other
    replicas hold of the requested fragments (``full_parameter_master_contributions``
    with the residency check); the fragments are rebuilt from them."""

    require_full_parameter_actor(actor.args, "materialize")
    assert_full_parameter_boundary_clear(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    requested = []
    for fragment_id in fragment_ids:
        if isinstance(fragment_id, bool) or not isinstance(fragment_id, int) or fragment_id < 0:
            raise ValueError("full-parameter materialize fragment ID is malformed")
        requested.append(fragment_id)
    if len(set(requested)) != len(requested):
        raise ValueError("full-parameter materialize repeats a fragment")
    if not requested:
        return ()
    started = time.monotonic()
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest, _replica_plan, named_parameters, optimizer_owned, layout = _require_installed_layout(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    descriptors = {descriptor.fragment_id: descriptor for descriptor in state.fragments}
    fragments = {fragment.fragment_id: fragment for fragment in plan.fragments}
    if any(fragment_id not in descriptors for fragment_id in requested):
        raise ValueError("full-parameter materialize names a fragment this rank does not own")
    resident, _record = _resident_fragment_ids(actor, state, honor_kill_switch=False)
    stale = sorted(fragment_id for fragment_id in requested if fragment_id not in resident)
    if stale:
        raise RuntimeError(f"full-parameter fragments {stale} are no longer held by the masters; a deferred export cannot be materialized after they changed")
    lazy = _lazy_fragment_ids(actor, state) & frozenset(requested)
    if lazy and not return_descriptors:
        raise RuntimeError(f"full-parameter fragments {sorted(lazy)} come from a lazy export: materialize them with return_descriptors=True")
    contributed = _contributions_by_fragment(
        actor, layout, contributions, owner=manifest.topology, fragment_ids=requested
    )

    timings: list[dict[str, float]] = []

    def store(fragment_id: int) -> tuple[int, list[object], FullParameterFragmentDescriptor]:
        expected = descriptors[fragment_id]
        spent: dict[str, float] = {}
        timings.append(spent)
        began = time.monotonic()
        sources = _fragment_sources(fragments[fragment_id], layout, contributions=contributed, ray_module=ray_module)
        spent["sources_seconds"] = time.monotonic() - began
        descriptor, references = _export_fragment_chunks(
            fragments[fragment_id],
            sources=sources,
            max_chunk_numel=expected.chunks[0].numel,
            parameter_layout_hash=plan.parameter_layout_hash,
            plan_hash=plan.plan_hash,
            ray_module=ray_module,
            overlap_checks=len(requested) <= 1,
            timings=spent,
        )
        if fragment_id in lazy:
            if not _same_fragment_geometry(descriptor, expected):
                raise RuntimeError(f"full-parameter fragment {fragment_id} geometry changed since its lazy export")
        elif descriptor != expected:
            raise RuntimeError(f"full-parameter fragment {fragment_id} bytes changed since its deferred export")
        return fragment_id, references, descriptor

    stored = _parallel_map(store, sorted(requested))
    checked = 0
    model_check_started = time.monotonic()
    if lazy:
        # A lazy export compared no model parameter with its master: the
        # parameters of every fragment that leaves the actor are compared now.
        by_wire_name = _parameters_by_wire_name(named_parameters, layout.specs)
        names = dict.fromkeys(name for fragment_id in sorted(lazy) for name in fragments[fragment_id].wire_names)
        checked = _validate_model_matches_masters(
            (by_wire_name[name] for name in names),
            optimizer_owned_parameter_ids=optimizer_owned,
            ranges=layout.ranges,
        )
    # Per-fragment seconds summed over fragments (they may run concurrently).
    split: dict[str, float] = {}
    for spent in timings:
        for key, seconds in spent.items():
            split[key] = split.get(key, 0.0) + seconds
    _note_apply_stats(
        actor,
        "materialize",
        fragments=len(stored),
        lazy_fragments=len(lazy),
        model_checked_parameters=checked,
        model_check_seconds=time.monotonic() - model_check_started,
        seconds=time.monotonic() - started,
        **split,
    )
    if return_descriptors:
        return tuple(stored)
    return tuple((fragment_id, references) for fragment_id, references, _descriptor in stored)


def _parameters_by_wire_name(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    specs: Sequence[FullParameterShardSpec],
) -> dict[str, tuple[str, torch.nn.Parameter]]:
    """``wire name -> (name, parameter)`` for this rank's ``specs`` (every local
    parameter at DP=1; the owned ones with data-parallel replicas)."""

    by_name = dict(named_parameters)
    if any(spec.name not in by_name for spec in specs):
        raise RuntimeError("full-parameter specs do not match the named parameters")
    return {spec.wire_name: (spec.name, by_name[spec.name]) for spec in specs}


def _validate_chunked_shard_against_plan(
    state: FullParameterChunkedShardState,
    plan: FullParameterOwnerFragmentPlan,
    manifest: FullParameterShardManifest,
) -> None:
    validate_full_parameter_chunked_shard_state(state)
    if state.topology != manifest.topology or state.topology != plan.topology:
        raise RuntimeError("full-parameter chunked shard topology changed")
    if state.parameter_layout_hash != plan.parameter_layout_hash or state.plan_hash != plan.plan_hash:
        raise RuntimeError("full-parameter chunked shard plan identity changed")
    if len(state.fragments) != len(plan.fragments):
        raise RuntimeError("full-parameter chunked shard fragment count changed")
    for descriptor, fragment in zip(state.fragments, plan.fragments, strict=True):
        if descriptor.fragment_id != fragment.fragment_id or descriptor.numel != fragment.numel:
            raise RuntimeError("full-parameter chunked fragment identity changed")


def verify_stored_fragment_chunks(
    fragment: FullParameterFragmentDescriptor,
    references: Sequence[object],
    *,
    parameter_layout_hash: str,
    plan_hash: str,
    ray_module=None,
    overlap_checks: bool = True,
) -> None:
    """Check the stored chunks of one fragment against its descriptor: each
    chunk is a well-formed finite FP32 CPU tensor with its chunk hash, and
    their concatenation has the fragment's payload hash.  Raises ValueError
    otherwise.  A prepare runs it on the trainer; a caller that runs it
    earlier (off the trainer) names the fragment in ``verified_fragments``.
    ``overlap_checks=False``: run the chunk checks in this thread (callers
    that verify several fragments on the scan pool)."""

    ray_module = _resolve_ray_module(ray_module)
    fragment_digest = _fragment_hasher(fragment.fragment_id, parameter_layout_hash, plan_hash)
    for descriptor, reference in zip(fragment.chunks, references, strict=True):
        raw = ray_module.get(reference)
        checked = (_hash_pool().submit if overlap_checks else _run_now)(
            _validate_cpu_chunk,
            raw,
            descriptor,
            fragment_id=fragment.fragment_id,
            verify_values=True,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        if isinstance(raw, torch.Tensor) and raw.device.type == "cpu" and raw.dtype == torch.float32 and raw.ndim == 1 and raw.is_contiguous():
            fragment_digest.update(_tensor_bytes(raw))
            checked.result()
        else:
            fragment_digest.update(_tensor_bytes(checked.result()))
        del raw, checked
    if fragment_digest.hexdigest() != fragment.payload_hash:
        raise ValueError("full-parameter fragment hash changed")


def _verified_fragment_ids(
    verified_fragments: Mapping[int, str] | None,
    fragments: Sequence[FullParameterFragmentDescriptor],
) -> frozenset[int]:
    """IDs of ``fragments`` whose payload hash ``verified_fragments`` names exactly."""

    if not verified_fragments:
        return frozenset()
    if not isinstance(verified_fragments, Mapping) or any(
        isinstance(key, bool) or not isinstance(key, int) or not isinstance(value, str)
        for key, value in verified_fragments.items()
    ):
        raise ValueError("full-parameter verified fragments must map fragment IDs to payload hashes")
    return frozenset(
        fragment.fragment_id
        for fragment in fragments
        if verified_fragments.get(fragment.fragment_id) == fragment.payload_hash
    )


def prepare_full_parameter_chunked_shard(
    actor,
    state: FullParameterChunkedShardState,
    *,
    commit_token: str,
    role: str = "actor",
    ray_module=None,
    skip_resident: bool = False,
    replica_states: Sequence[FullParameterChunkedShardState] = (),
    verified_fragments: Mapping[int, str] | None = None,
) -> int:
    """Validate every immutable chunk and stage only its refs for later commit.

    ``skip_resident``: the caller states that it built ``state`` from the cut
    this actor last exported or committed, changing only some fragments.  A
    fragment whose payload hash equals the one the actor recorded for its
    masters at that export/commit is then neither read nor copied: the masters
    already hold those bytes.  The record is dropped by every local step,
    restore rebase and legacy apply, so the skip never applies across one.

    ``replica_states``: the cut's shards of the owners whose parameters this
    rank replicates (data-parallel replicas); the chunks a commit copies into
    the replicas are checked here the same way, and the same skip applies.

    ``verified_fragments``: ``{fragment ID: fragment payload hash}`` whose
    stored chunks the caller already checked with
    :func:`verify_stored_fragment_chunks` (the same finite check, chunk
    hashes and fragment digest, run off the trainer before this apply).  A
    row whose descriptor carries exactly that payload hash is not read here,
    neither for the owner nor for a replica; any other row is checked as
    usual.  Ray objects are immutable, so the commit copies the bytes that
    were checked."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    require_full_parameter_actor(actor.args, "prepare")
    token = _validate_commit_token(commit_token)
    state_identity_hash = full_parameter_chunked_shard_identity(state)
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    for outcome in (committed, finalized):
        if outcome is not None and not isinstance(outcome, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
    if isinstance(committed, _FullParameterCommitOutcome):
        if committed.commit_token == token and committed.policy_version == state.policy_version and committed.state_identity_hash == state_identity_hash:
            return committed.parameter_tensor_count
        raise RuntimeError("an unfinalized full-parameter commit must be reconciled")
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token and finalized.policy_version == state.policy_version and finalized.state_identity_hash == state_identity_hash):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    if prepared is not None:
        if not isinstance(prepared, _PreparedFullParameterChunkedShard):
            raise RuntimeError("full-parameter prepared context is malformed")
        if prepared.commit_token == token and prepared.state_identity_hash == state_identity_hash:
            return prepared.parameter_tensor_count
        raise RuntimeError("a stale full-parameter prepared cut must be aborted")
    if state.local_step_generation != 0:
        raise RuntimeError("a global full-parameter cut cannot contain local steps")
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    _validate_scheduler_tracking_congruence(actor, role=role)
    if exported != generation:
        raise RuntimeError("full-parameter global apply has unexported local progress")
    if state.policy_version <= installed:
        raise RuntimeError("full-parameter policy version is stale")
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest, replica_plan, _named, _owned, _layout = _require_installed_layout(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    replica_states = replicas.match_replica_states(
        replica_plan,
        tuple(replica_states),
        policy_version=state.policy_version,
        local_step_generation=state.local_step_generation,
    )
    started = time.monotonic()
    # A deferred fragment has no stored bytes and a lazy one has no content
    # hash to check stored bytes against: only the masters hold them, so
    # their residency does not depend on the skip switch.
    deferred = _deferred_fragment_ids(state)
    lazy = _lazy_fragment_ids(actor, state)
    resident, resident_record = (
        _resident_fragment_ids(actor, state) if skip_resident else (frozenset(), None)
    )
    if (deferred | lazy) - resident:
        forced, forced_record = _resident_fragment_ids(actor, state, honor_kill_switch=False)
        missing = sorted((deferred | lazy) - forced)
        if missing:
            raise RuntimeError(f"deferred or lazy full-parameter fragments {missing} are not held by the masters and have no checkable stored bytes")
        if resident_record is None:
            resident_record = forced_record
        resident = resident | deferred | lazy
    verified = _verified_fragment_ids(verified_fragments, state.fragments)
    rows = [
        (fragment, references)
        for fragment, references in zip(state.fragments, state.chunk_refs, strict=True)
        if fragment.fragment_id not in resident and fragment.fragment_id not in verified
    ]
    # One fragment: its finite check and chunk hash overlap its digest on the
    # hash pool.  Several: each scan worker verifies one fragment serially.
    overlap_checks = len(rows) <= 1

    def verify(row) -> None:
        fragment, references = row
        verify_stored_fragment_chunks(
            fragment,
            references,
            parameter_layout_hash=state.parameter_layout_hash,
            plan_hash=state.plan_hash,
            ray_module=ray_module,
            overlap_checks=overlap_checks,
        )

    # Resident fragments: the masters already hold these exact bytes; commit
    # skips them.  The others are independent; verify them concurrently.
    _parallel_map(verify, rows)
    replica_resident, replica_record = replicas.resident_replica_fragments(
        actor, replica_plan, replica_states, skip_resident=skip_resident
    )
    replica_verified = frozenset(
        (replica_state.topology, fragment_id)
        for replica_state in replica_states
        for fragment_id in _verified_fragment_ids(verified_fragments, replica_state.fragments)
    )
    replica_read = replicas.verify_replica_chunks(
        replica_plan, replica_states, skip=replica_resident | replica_verified, ray_module=ray_module
    )
    _note_apply_stats(
        actor,
        "prepare",
        fragments=len(state.fragments),
        resident_fragments=len(resident),
        read_fragments=len(rows),
        verified_ahead_fragments=len(verified - resident),
        replica_resident_fragments=len(replica_resident),
        replica_read_fragments=replica_read,
        seconds=time.monotonic() - started,
    )
    parameter_count = len(manifest.specs)
    actor._external_full_parameter_prepared_cut = _PreparedFullParameterChunkedShard(
        commit_token=token,
        state=state,
        parameter_tensor_count=parameter_count,
        base_policy_version=installed,
        base_local_step_generation=generation,
        base_exported_local_step_generation=exported,
        scheduler_num_steps=_scheduler_num_steps(actor),
        state_identity_hash=state_identity_hash,
        commit_started=False,
        resident_fragment_ids=resident,
        resident_record=resident_record,
        replica_states=replica_states,
        replica_resident_ids=replica_resident,
        replica_resident_record=replica_record,
        lazy_fragment_ids=lazy,
    )
    return parameter_count


MasterWriter = Callable[[str, int, torch.Tensor], None]


def _whole_master_writer(masters: Mapping[str, torch.Tensor]) -> MasterWriter:
    """``write(key, start, values)``: flat offsets ``[start, start + len(values))`` of ``masters[key]``."""

    def write(key: str, start: int, values: torch.Tensor) -> None:
        masters[key].detach().reshape(-1)[start : start + values.numel()].copy_(values, non_blocking=False)

    return write


def _commit_writer(layout: _LocalLayout, *, owned: bool) -> MasterWriter:
    """Where a commit writes: the owned masters by wire name (``owned``) or
    the replica masters by parameter name; with sharded masters the model and
    this rank's range (``MasterRange.write``)."""

    if layout.ranges is None:
        return _whole_master_writer(layout.masters if owned else layout.replica_masters)
    return sharded.range_writer(layout.ranges_by_wire_name() if owned else layout.ranges)


def _copy_fragment_chunks_to_masters(
    fragment: FullParameterFragmentPlan,
    descriptor: FullParameterFragmentDescriptor,
    references: Sequence[object | None],
    *,
    numel_by_wire: Mapping[str, int],
    write: MasterWriter,
    parameter_layout_hash: str,
    plan_hash: str,
    ray_module,
    fetch: Callable[[object], object] | None = None,
) -> int:
    """Copy one fragment's stored chunks into ``write``; returns the
    parameters it completed.  ``fetch``: reads a chunk ref (default
    ``ray_module.get``; the commit passes its :class:`CommitChunkReader`)."""

    fetch = fetch or ray_module.get
    members = fragment.members(numel_by_wire)
    member_index = 0
    member_offset = 0  # elements of the current member already written
    copied_parameters = 0
    for chunk_descriptor, reference in zip(
        descriptor.chunks,
        references,
        strict=True,
    ):
        if is_deferred_full_parameter_chunk(reference):
            raise RuntimeError("a deferred full-parameter chunk has no stored bytes to copy")
        value = _validate_cpu_chunk(
            fetch(reference),
            chunk_descriptor,
            fragment_id=fragment.fragment_id,
            verify_values=False,
            parameter_layout_hash=parameter_layout_hash,
            plan_hash=plan_hash,
        )
        source_offset = 0
        while source_offset < value.numel():
            if member_index >= len(members):
                raise RuntimeError("full-parameter chunk target ended early")
            name, start, stop = members[member_index]
            count = min(value.numel() - source_offset, stop - start - member_offset)
            write(name, start + member_offset, value[source_offset : source_offset + count])
            source_offset += count
            member_offset += count
            if member_offset == stop - start:
                # A split parameter counts once: in the fragment holding its last range.
                copied_parameters += stop == numel_by_wire[name]
                member_index += 1
                member_offset = 0
        del value
    if member_index != len(members) or member_offset != 0:
        raise RuntimeError("full-parameter chunks did not cover every target")
    return copied_parameters


def _fragment_completed_parameters(
    fragment: FullParameterFragmentPlan,
    numel_by_wire: Mapping[str, int],
) -> int:
    """Parameters whose last flat range lies in ``fragment`` (as the copy counts them)."""

    return sum(stop == numel_by_wire[name] for name, _start, stop in fragment.members(numel_by_wire))


@torch.no_grad()
def commit_prepared_full_parameter_shard(
    actor,
    *,
    commit_token: str,
    role: str = "actor",
    ray_module=None,
) -> int:
    """Commit a fully prepared cut while retaining Adam state objects."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    for outcome in (committed, finalized):
        if outcome is not None and not isinstance(outcome, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
    if isinstance(committed, _FullParameterCommitOutcome):
        if committed.commit_token != token:
            raise RuntimeError("another full-parameter commit is unresolved")
        installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != committed.policy_version or generation != 0 or exported != 0:
            raise RuntimeError("committed full-parameter journal disagrees with actor")
        return committed.parameter_tensor_count
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    if not isinstance(prepared, _PreparedFullParameterChunkedShard):
        raise RuntimeError("no full-parameter cut is prepared")
    if prepared.commit_token != token:
        raise RuntimeError("full-parameter commit token does not match prepared state")
    state = prepared.state
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != prepared.base_policy_version or generation != prepared.base_local_step_generation or exported != prepared.base_exported_local_step_generation or exported != generation or _scheduler_num_steps(actor) != prepared.scheduler_num_steps or state.policy_version <= installed:
        raise RuntimeError("prepared full-parameter cut became stale")
    ray_module = _resolve_ray_module(ray_module)
    plan, manifest, replica_plan, named_parameters, optimizer_owned, layout = _require_installed_layout(actor, role=role)
    _validate_chunked_shard_against_plan(state, plan, manifest)
    specs = layout.specs
    numel_by_wire = {spec.wire_name: spec.numel for spec in specs}
    if not prepared.commit_started:
        # Resident fragments are skipped only while the masters provably hold
        # them: the record taken at prepare must still be the live one.  A
        # retried commit (commit_started) wrote only non-resident fragments,
        # which never overlap a resident one, so the skip stays exact.
        if prepared.resident_fragment_ids and (
            getattr(actor, "_external_full_parameter_resident", None) is not prepared.resident_record
        ):
            raise RuntimeError("full-parameter resident fragments changed after prepare")
        replicas.require_live_replica_record(actor, prepared.replica_resident_record, prepared.replica_resident_ids)
        prepared = replace(prepared, commit_started=True)
        actor._external_full_parameter_prepared_cut = prepared
    # From here the masters may be partly written: no record describes them.
    forget_full_parameter_resident_fragments(actor)
    started = time.monotonic()
    count = 0
    copied_fragments = 0
    written: dict[str, None] = {}
    write_seconds = [0.0]
    owned_write = _timed_writer(_commit_writer(layout, owned=True), write_seconds)
    # The hashing pool is idle during a commit: it maps the next chunks while
    # this thread copies.
    reader = CommitChunkReader(
        _commit_chunk_reads(plan, state, prepared, replica_plan),
        ray_module=ray_module,
        pool=_hash_pool() if prefetch_enabled(ray_module) else None,
    )
    try:
        for fragment, descriptor, references in zip(
            plan.fragments,
            state.fragments,
            state.chunk_refs,
            strict=True,
        ):
            if fragment.fragment_id in prepared.resident_fragment_ids:
                count += _fragment_completed_parameters(fragment, numel_by_wire)
                continue
            copied_fragments += 1
            written.update(dict.fromkeys(fragment.wire_names))
            count += _copy_fragment_chunks_to_masters(
                fragment,
                descriptor,
                references,
                numel_by_wire=numel_by_wire,
                write=owned_write,
                parameter_layout_hash=state.parameter_layout_hash,
                plan_hash=state.plan_hash,
                ray_module=ray_module,
                fetch=reader.get,
            )
        if count != prepared.parameter_tensor_count or count != len(specs):
            raise RuntimeError("full-parameter chunked apply is incomplete")
        replica_copied = replicas.copy_replica_chunks(
            replica_plan,
            prepared.replica_states,
            skip=prepared.replica_resident_ids,
            write=_timed_writer(_commit_writer(layout, owned=False), write_seconds),
            ray_module=ray_module,
            fetch=reader.get,
        )
    finally:
        reader.close()
    copied_seconds = time.monotonic() - started
    by_wire_name = _parameters_by_wire_name(named_parameters, specs)
    written_names = dict.fromkeys(by_wire_name[wire_name][0] for wire_name in written)
    written_names.update(
        dict.fromkeys(
            replicas.replica_names_to_copy(replica_plan, prepared.replica_states, skip=prepared.replica_resident_ids)
        )
    )
    finish_stats = _finish_commit_writes(
        actor,
        named_parameters,
        optimizer_owned,
        written=tuple(written_names),
        expected_count=len(layout.holdings),
        role=role,
        ranges=layout.ranges,
    )
    actor._external_full_policy_version = state.policy_version
    actor._external_full_local_step_generation = 0
    actor._external_full_exported_local_step_generation = 0
    actor._external_full_scheduler_anchor_steps = _scheduler_num_steps(actor)
    actor._external_full_last_local_rollout_id = None
    actor._external_full_last_local_step_receipt = None
    actor._external_full_parameter_committed_cut = _FullParameterCommitOutcome(
        commit_token=token,
        policy_version=state.policy_version,
        parameter_tensor_count=count,
        state_identity_hash=prepared.state_identity_hash,
    )
    _record_resident_fragments(
        actor,
        parameter_layout_hash=state.parameter_layout_hash,
        plan_hash=state.plan_hash,
        policy_version=state.policy_version,
        local_step_generation=0,
        fragments=state.fragments,
        # A lazy row was resident, so the commit did not write it: its label
        # still names the masters' bytes.
        lazy_fragment_ids=prepared.lazy_fragment_ids,
    )
    replicas.record_replica_resident(actor, replica_plan, prepared.replica_states)
    _note_apply_stats(
        actor,
        "commit",
        fragments=len(state.fragments),
        copied_fragments=copied_fragments,
        resident_fragments=len(prepared.resident_fragment_ids),
        replica_copied_fragments=replica_copied,
        replica_resident_fragments=len(prepared.replica_resident_ids),
        copy_seconds=copied_seconds,
        write_seconds=write_seconds[0],
        seconds=time.monotonic() - started,
        **reader.stats(),
        **finish_stats,
    )
    del actor._external_full_parameter_prepared_cut
    return count


def _timed_writer(write: MasterWriter, total: list[float]) -> MasterWriter:
    """``write`` that adds its own seconds to ``total[0]`` (the commit's write
    split: host-to-device copies and casts, apart from reading chunks)."""

    def timed(key: str, start: int, values: torch.Tensor) -> None:
        began = time.monotonic()
        write(key, start, values)
        total[0] += time.monotonic() - began

    return timed


def _commit_chunk_reads(
    plan: FullParameterOwnerFragmentPlan,
    state: FullParameterChunkedShardState,
    prepared: _PreparedFullParameterChunkedShard,
    replica_plan,
) -> list[tuple[object, tuple[tuple[int, int], ...] | None]]:
    """The chunks the commit copies, in its copy order: every stored chunk of
    each non-resident owned fragment, then the replica rows' chunks with the
    spans they copy (``replicas.replica_chunk_reads``)."""

    from miles.backends.megatron_utils import full_parameter_replicas as replicas  # import cycle

    reads: list[tuple[object, tuple[tuple[int, int], ...] | None]] = [
        (reference, None)
        for fragment, references in zip(plan.fragments, state.chunk_refs, strict=True)
        if fragment.fragment_id not in prepared.resident_fragment_ids
        for reference in references
        if not is_deferred_full_parameter_chunk(reference)
    ]
    reads.extend(
        replicas.replica_chunk_reads(replica_plan, prepared.replica_states, skip=prepared.replica_resident_ids)
    )
    return reads


_PARTIAL_COMMIT_ENV = "MILES_FULL_PARAMETER_PARTIAL_COMMIT"
_DENSE_MODEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _partial_commit_refusal(
    actor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_owned: frozenset[int],
    written: Sequence[str],
    parameter_count: int,
) -> str | None:
    """Why a commit must finish over the whole model; None when it may finish
    over the parameters it wrote only.

    The per-parameter path copies a master into its model parameter with a
    plain ``copy_`` (what Megatron's main-to-model copy does for a dense
    parameter whose master is whole on this rank: DP=1, or the
    non-distributed optimizer that data-parallel replicas run with) and
    propagates only the hybrid-optimizer copies of
    the written masters, so it needs dense model tensors, no Megatron-FSDP or
    quantized parameter gather, and every hybrid inner copy reachable from a
    named parameter's master."""

    if os.environ.get(_PARTIAL_COMMIT_ENV, "1") == "0":
        return f"{_PARTIAL_COMMIT_ENV}=0"
    if len(written) >= parameter_count:
        return "every parameter written"
    for _name, parameter in named_parameters:
        if type(parameter.data) is not torch.Tensor or parameter.dtype not in _DENSE_MODEL_DTYPES:
            return "a model parameter is not a dense tensor"
    for child in getattr(actor.optimizer, "chained_optimizers", (actor.optimizer,)):
        ddp_config = getattr(child, "ddp_config", None)
        if getattr(ddp_config, "use_megatron_fsdp", False) or getattr(ddp_config, "fp4_param_gather", False):
            return "Megatron-FSDP or a quantized parameter gather"
        if getattr(child, "_layer_wise_non_distopt_child", False):
            return "layer-wise FP8 optimizer"
    owned = {
        id(_resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned)[0])
        for name, parameter in named_parameters
    }
    for base in _optimizer_bases(actor.optimizer):
        inner = getattr(base, "param_to_inner_param", None) or {}
        if any(key is not copy and id(key) not in owned for key, copy in inner.items()):
            return "a hybrid optimizer copy belongs to no named parameter"
    return None


@torch.no_grad()
def _copy_masters_to_written_parameters(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_owned: frozenset[int],
) -> None:
    """The main-to-model copy of Megatron for ``rows`` only (dense, whole masters).

    A parameter whose optimizer owns its model view (precision-aware) got its
    master from the hybrid copy in ``_propagate_master_writes``; a native-FP32
    parameter is its own master."""

    for name, parameter in rows:
        owned, master = _resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned)
        if owned.dtype != torch.float32 or master.data_ptr() == parameter.data_ptr():
            continue
        parameter.data.copy_(master.view(parameter.shape), non_blocking=False)


def _backup_written_actor_weights(actor, written: Sequence[torch.nn.Parameter] | None) -> str:
    """Bring the actor's weight backup up to date after a commit.

    Only when the actor keeps one (Miles backs up after a train step only when
    ``_enable_weight_backup``).  ``written``: the parameters the commit
    changed; a backuper that updates a subset in place takes only those, any
    other backs up every weight.  Returns what was done (for the stats)."""

    if getattr(actor, "_enable_weight_backup", True) is False:
        return "off"
    backuper = actor.weights_backuper
    backup_tensors = getattr(backuper, "backup_tensors", None)
    if written is not None and callable(backup_tensors):
        expected = {id(parameter) for parameter in written}
        if backup_tensors("actor", expected) == len(expected):
            return "written"
    backuper.backup("actor")
    return "all"


def _finish_commit_writes(
    actor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_owned: frozenset[int],
    *,
    written: Sequence[str],
    expected_count: int,
    role: str,
    ranges: Mapping[str, sharded.MasterRange] | None = None,
) -> dict[str, object]:
    """After the commit copied chunks into masters: make the hybrid optimizer
    copies, the model and the weight backup agree with the masters, and check
    model == master.

    ``ranges`` (sharded masters): the commit already wrote the model and this
    rank's master ranges (``MasterRange.write``); only the range check and the
    backup of the written parameters remain.

    The parameters the commit did not write are untouched on both sides, so
    only ``written`` (local parameter names: owned and data-parallel replica
    parameters) needs this when ``_partial_commit_refusal`` allows; otherwise
    the whole model, as before (``expected_count`` local parameters).  Returns
    timing stats."""

    if ranges is not None:
        return _finish_sharded_commit_writes(actor, named_parameters, written=written, ranges=ranges, role=role)
    refusal = _partial_commit_refusal(actor, named_parameters, optimizer_owned, written, len(named_parameters))
    stats: dict[str, object] = {"partial": refusal is None, "written_parameters": len(written)}
    if refusal is not None:
        stats["partial_refusal"] = refusal
    clock = time.monotonic()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.monotonic()
        stats[name] = now - clock
        clock = now

    if refusal is None:
        by_name = dict(named_parameters)
        rows = [(name, by_name[name]) for name in written]
        owned_ids = {
            id(_resolve_master(parameter, name, optimizer_owned_parameter_ids=optimizer_owned)[0])
            for name, parameter in rows
        }
        _propagate_master_writes(actor, owned_ids=owned_ids)
        lap("propagate_seconds")
        _assert_masters_resident(actor)
        _copy_masters_to_written_parameters(rows, optimizer_owned)
        lap("model_copy_seconds")
        if rows:
            _validate_model_matches_masters(rows, optimizer_owned_parameter_ids=optimizer_owned)
        lap("check_seconds")
        backup = _backup_written_actor_weights(actor, [parameter for _name, parameter in rows]) if role == "actor" and rows else "none"
    else:
        _propagate_master_writes(actor)
        lap("propagate_seconds")
        _copy_masters_to_model(actor)
        lap("model_copy_seconds")
        model_count = _validate_model_matches_masters(
            named_parameters,
            optimizer_owned_parameter_ids=optimizer_owned,
        )
        if model_count != expected_count:
            raise RuntimeError("full-parameter model/master check is incomplete")
        lap("check_seconds")
        backup = _backup_written_actor_weights(actor, None) if role == "actor" else "none"
    lap("backup_seconds")
    stats["backup"] = backup
    return stats


def _finish_sharded_commit_writes(
    actor,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    written: Sequence[str],
    ranges: Mapping[str, sharded.MasterRange],
    role: str,
) -> dict[str, object]:
    started = time.monotonic()
    by_name = dict(named_parameters)
    rows = [(name, by_name[name]) for name in written]
    if rows:
        sharded.check_model(rows, ranges)
    checked = time.monotonic()
    backup = _backup_written_actor_weights(actor, [parameter for _name, parameter in rows]) if role == "actor" and rows else "none"
    return {
        "partial": True,
        "sharded": True,
        "written_parameters": len(written),
        "check_seconds": checked - started,
        "backup_seconds": time.monotonic() - checked,
        "backup": backup,
    }


def _assert_masters_resident(actor) -> None:
    """Megatron's precondition of a main-to-model copy (no-op without an
    optimizer-state offloader)."""

    check = getattr(actor.optimizer, "assert_master_weights_resident", None)
    if callable(check):
        check("full-parameter partial commit")


def finalize_full_parameter_commit(actor, *, commit_token: str) -> int:
    """Acknowledge a group-wide commit after every rank has converged."""

    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token):
        installed, _generation, _exported, _anchor = _require_full_parameter_tracking(actor)
        if installed != finalized.policy_version:
            raise RuntimeError("finalized full-parameter commit journal is stale")
        return finalized.parameter_tensor_count
    if not isinstance(committed, _FullParameterCommitOutcome):
        raise RuntimeError("no committed full-parameter cut is pending finalization")
    if committed.commit_token != token:
        raise RuntimeError("full-parameter finalize token does not match journal")
    installed, generation, exported, _anchor = _require_full_parameter_tracking(actor)
    if installed != committed.policy_version or generation != 0 or exported != 0:
        raise RuntimeError("committed full-parameter journal disagrees with actor")
    actor._external_full_parameter_last_finalized_cut = committed
    del actor._external_full_parameter_committed_cut
    return committed.parameter_tensor_count


def full_parameter_commit_status(
    actor,
    *,
    commit_token: str,
    policy_version: int,
    state_identity_hash: str,
) -> FullParameterCommitStatus:
    """Attest whether a transaction is safe to abort or must be reconciled."""

    token = _validate_commit_token(commit_token)
    _validate_non_negative_integer(policy_version, "policy version")
    if not _SHA256.fullmatch(state_identity_hash):
        raise ValueError("full-parameter commit state identity is malformed")
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    finalized = getattr(actor, "_external_full_parameter_last_finalized_cut", None)
    if prepared is not None:
        if not isinstance(prepared, _PreparedFullParameterChunkedShard):
            raise RuntimeError("full-parameter prepared context is malformed")
        if prepared.commit_token != token or prepared.state.policy_version != policy_version or prepared.state_identity_hash != state_identity_hash:
            raise RuntimeError("another full-parameter transaction is active")
        return FullParameterCommitStatus(
            token,
            "COMMITTING" if prepared.commit_started else "PREPARED",
            policy_version,
            prepared.parameter_tensor_count,
            state_identity_hash,
        )
    if committed is not None:
        if not isinstance(committed, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
        if committed.commit_token != token or committed.policy_version != policy_version or committed.state_identity_hash != state_identity_hash:
            raise RuntimeError("another full-parameter transaction is active")
        return FullParameterCommitStatus(
            token,
            "COMMITTED",
            policy_version,
            committed.parameter_tensor_count,
            state_identity_hash,
        )
    if isinstance(finalized, _FullParameterCommitOutcome) and (finalized.commit_token == token and finalized.policy_version == policy_version and finalized.state_identity_hash == state_identity_hash):
        return FullParameterCommitStatus(
            token,
            "FINALIZED",
            policy_version,
            finalized.parameter_tensor_count,
            state_identity_hash,
        )
    return FullParameterCommitStatus(
        token,
        "ABSENT",
        policy_version,
        0,
        state_identity_hash,
    )


def abort_prepared_full_parameter_shard(
    actor,
    *,
    commit_token: str,
) -> bool:
    """Discard actor-local prepare metadata without deleting shared Ray objects."""

    token = _validate_commit_token(commit_token)
    committed = getattr(actor, "_external_full_parameter_committed_cut", None)
    if committed is not None:
        if not isinstance(committed, _FullParameterCommitOutcome):
            raise RuntimeError("full-parameter commit journal is malformed")
        if committed.commit_token == token:
            raise RuntimeError("a committed full-parameter cut cannot be aborted")
        raise RuntimeError("another full-parameter commit is unresolved")
    prepared = getattr(actor, "_external_full_parameter_prepared_cut", None)
    if prepared is None:
        return False
    if not isinstance(prepared, _PreparedFullParameterChunkedShard):
        raise RuntimeError("full-parameter prepared context is malformed")
    if prepared.commit_token != token:
        raise RuntimeError("full-parameter abort token does not match prepared state")
    if prepared.commit_started:
        raise RuntimeError("a committing full-parameter cut cannot be aborted")
    del actor._external_full_parameter_prepared_cut
    return True


def _copy_masters_to_model(actor) -> None:
    """Copy every FP32 optimizer master back into its model parameter."""

    children = getattr(actor.optimizer, "chained_optimizers", (actor.optimizer,))
    for child in children:
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("Megatron optimizer lacks main-to-model copy")
        copy()
