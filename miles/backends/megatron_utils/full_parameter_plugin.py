"""Per-rank entry point of the full-parameter transport (``run_plugin``).

The trainer controller runs ``full_parameter_rank_call`` on every worker.  It
dispatches to one ``full_parameter_state`` operation with this actor's role.
Rank-targeted arguments arrive keyed by topology and each rank picks its own.
Failures are returned, not raised, so the group can reconcile per-rank
outcomes exactly as the transaction protocol requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from miles.backends.megatron_utils import full_parameter_state as state

# Operations that take the actor's role.
_ROLE_OPERATIONS = {
    "full_parameter_shard_manifest": state.full_parameter_shard_manifest,
    "full_parameter_partition_report": state.full_parameter_partition_report,
    "install_full_parameter_owner_map": state.install_full_parameter_owner_map,
    "install_full_parameter_fragment_plan": state.install_full_parameter_fragment_plan,
    "export_full_parameter_chunked_shard": state.export_full_parameter_chunked_shard,
    "note_full_parameter_replica_export": state.note_full_parameter_replica_export,
    "materialize_full_parameter_fragments": state.materialize_full_parameter_fragments,
    "full_parameter_master_contributions": state.full_parameter_master_contributions,
    "prepare_full_parameter_chunked_shard": state.prepare_full_parameter_chunked_shard,
    "commit_prepared_full_parameter_shard": state.commit_prepared_full_parameter_shard,
    "record_full_parameter_local_step": state.record_full_parameter_local_step,
    "full_parameter_optimizer_state": state.full_parameter_optimizer_state,
}
# Operations that do not.
_PLAIN_OPERATIONS = {
    "finalize_full_parameter_commit": state.finalize_full_parameter_commit,
    "full_parameter_commit_status": state.full_parameter_commit_status,
    "abort_prepared_full_parameter_shard": state.abort_prepared_full_parameter_shard,
}

PLUGIN_PATH = "miles.backends.megatron_utils.full_parameter_plugin.full_parameter_rank_call"


def ensure_full_parameter_tracking(actor) -> None:
    """Bind the global-version boundary once, from the configured start."""

    if getattr(actor, "_external_full_policy_version", None) is None:
        state.initialize_full_parameter_tracking(
            actor, state.full_parameter_initial_policy_version(actor.args, -1)
        )


def full_parameter_rank_call(
    actor,
    *,
    operation: str,
    kwargs: Mapping[str, Any] | None = None,
    by_topology: Mapping[Any, Mapping[str, Any]] | None = None,
) -> Any:
    """Run one operation on this rank; return its result or its exception."""

    try:
        if operation == "full_parameter_topology":
            return state.full_parameter_topology()
        ensure_full_parameter_tracking(actor)
        arguments = dict(kwargs or {})
        if by_topology is not None:
            arguments.update(by_topology[state.full_parameter_topology()])
        if operation in _ROLE_OPERATIONS:
            return _ROLE_OPERATIONS[operation](actor, role=actor.role, **arguments)
        return _PLAIN_OPERATIONS[operation](actor, **arguments)
    except Exception as error:  # noqa: BLE001 - reconciled per rank by the group
        return error
