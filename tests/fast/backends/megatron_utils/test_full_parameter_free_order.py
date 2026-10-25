"""Free-order owner plans and installed owner maps (yeto shape-free layouts), on CPU."""

from __future__ import annotations

import pytest
import torch

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterFragmentPlan,
    FullParameterOwnerFragmentPlan,
    FullParameterReplication,
    FullParameterShardSpec,
    FullParameterTopology,
)

_TOPOLOGY = FullParameterTopology(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
_HASH = "7" * 64


def _manifest(shapes):
    specs = tuple(
        sorted(FullParameterShardSpec("actor", _TOPOLOGY.shard_id, name, shape, "float32", shape[0] * shape[1])
               for name, shape in shapes.items())
    )
    return fps.FullParameterShardManifest(
        topology=_TOPOLOGY, role="actor", layout_hash=fps._layout_hash(_TOPOLOGY, specs), specs=specs
    )


def _wire(name: str) -> str:
    return f"actor::{_TOPOLOGY.shard_id}::{name}"


def _plan(fragments, *, free=True):
    return FullParameterOwnerFragmentPlan(
        topology=_TOPOLOGY,
        parameter_layout_hash=_HASH,
        fragments=tuple(
            FullParameterFragmentPlan(
                fragment_id=index,
                wire_names=tuple(_wire(name) for name, _start, _stop in members),
                numel=sum(stop - start for _name, start, stop in members),
                flat_ranges=tuple((start, stop) for _name, start, stop in members),
            )
            for index, members in enumerate(fragments)
        ),
        free_member_order=free,
    )


MANIFEST = _manifest({"a.weight": (2, 4), "b.weight": (2, 2)})
# Column halves of a.weight (row segments), b.weight between them.
COLUMNS = [
    [("a.weight", 0, 2), ("a.weight", 4, 6), ("b.weight", 0, 4)],
    [("a.weight", 2, 4), ("a.weight", 6, 8)],
]


def test_a_free_order_plan_may_read_a_parameter_as_interleaved_segments():
    fps._validate_fragment_plan_against_manifest(_plan(COLUMNS), MANIFEST)


def test_a_free_order_plan_hash_differs_from_the_canonical_order_hash():
    canonical = _plan([[("a.weight", 0, 8)], [("b.weight", 0, 4)]], free=False)
    free = _plan([[("a.weight", 0, 8)], [("b.weight", 0, 4)]], free=True)
    assert canonical.plan_hash != free.plan_hash


def test_the_canonical_order_check_still_refuses_interleaved_segments():
    with pytest.raises(ValueError, match="duplicate parameters"):
        _plan(COLUMNS, free=False)


@pytest.mark.parametrize(
    "fragments",
    [
        [[("a.weight", 0, 2), ("a.weight", 4, 6), ("b.weight", 0, 4)], [("a.weight", 2, 4), ("a.weight", 5, 8)]],
        [[("a.weight", 0, 2), ("a.weight", 4, 6), ("b.weight", 0, 4)], [("a.weight", 2, 4)]],
        [[("a.weight", 0, 8)]],
    ],
    ids=["overlap", "gap", "missing-parameter"],
)
def test_a_free_order_plan_must_cover_every_parameter_exactly_once(fragments):
    with pytest.raises(RuntimeError):
        fps._validate_fragment_plan_against_manifest(_plan(fragments), MANIFEST)


def test_an_owner_map_must_keep_the_replica_group_inside_the_writers():
    replication = FullParameterReplication(global_rank=1, dense_ranks=(0, 1))
    named = (("a.weight", torch.nn.Parameter(torch.zeros(2, 4))),)
    assert fps._installed_owners(named, replication, {"a.weight": (2, (0, 1, 2, 3))}) == {
        "a.weight": (2, (0, 1, 2, 3))
    }
    with pytest.raises(RuntimeError, match="replica group"):
        fps._installed_owners(named, replication, {"a.weight": (0, (0, 2))})
    with pytest.raises(RuntimeError, match="exactly this rank"):
        fps._installed_owners(named, replication, {})


def test_a_gated_fc1_reports_its_gate_and_up_blocks():
    parameter = torch.nn.Parameter(torch.zeros(6, 4))
    parameter.tensor_model_parallel, parameter.partition_dim, parameter.partition_stride = True, 0, 1
    name = "decoder.layers.0.mlp.linear_fc1.weight"
    assert fps._parameter_partition(name, parameter, gated=True).partition_sizes == (3, 3)
    assert fps._parameter_partition(name, parameter).partition_sizes == (6,)
    parameter.allreduce = False  # an expert: whole (expert tensor parallelism 1)
    assert fps._parameter_partition(name, parameter, gated=True).partition_dim is None
