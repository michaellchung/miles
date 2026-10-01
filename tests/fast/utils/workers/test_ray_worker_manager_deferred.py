"""Deferred (declared stopped, unbound) cells: yeto F-R1 role transfer onto GPUs a trainer released."""

from __future__ import annotations

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec
from tests.fast.utils.workers.test_ray_worker_manager_rebind import _BUNDLES, _GPUS, _engine_spec, _pgs, _view

from miles.ray.placement_group import slice_pg_info
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager import BundleInUseError, CellUnboundError

_DEFERRED = compute_cell_id(pool_id="engine", cell_index=2)


def _engine_spec_with_deferred(num_deferred: int = 1):
    spec = _engine_spec()
    return spec.model_copy(
        update=dict(scheduling=spec.scheduling.model_copy(update=dict(num_deferred_cells=num_deferred)))
    )


def _declared_engine_spec(aliases, bindings, *, num_started: int):
    """An engine pool laid out like placement map ``rollout_cells`` (what ``specs_inference_engine`` builds)."""
    spec = _engine_spec()
    return spec.model_copy(
        update=dict(
            scheduling=spec.scheduling.model_copy(
                update=dict(
                    num_cells=num_started,
                    num_deferred_cells=len(aliases) - num_started,
                    cell_aliases=tuple(aliases),
                    initial_bindings=tuple(bindings),
                )
            )
        )
    )


def _trainer_spec(num_workers: int = 2):
    return _make_spec(
        "trainer",
        num_workers_per_cell=num_workers,
        num_gpus_per_worker=0.4,
        num_gpu_slots_per_worker=1,
        pg_name="actor",
    )


class TestDeclaredButNotStarted:
    async def test_a_deferred_cell_is_declared_but_starts_nothing(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert sorted(infos) == sorted(compute_cell_id(pool_id="engine", cell_index=i) for i in range(3))
        assert infos[_DEFERRED].alive is False and infos[_DEFERRED].worker_names == []
        # only the two regular engines created actors
        assert len(fake_ray_cluster.handles) == 2
        assert manager.get_cell_bundles(_DEFERRED) == []
        assert manager.describe_cells()[_DEFERRED] == dict(
            cell_id=_DEFERRED,
            alias=None,
            pool_id="engine",
            deferred=True,
            state="unbound",
            lost_workers=None,
            generation=0,
            pg_name=None,
            pg_slot_offset=None,
            bundles=[],
            gpu_ids=[],
        )

    async def test_the_default_spec_declares_no_deferred_cell(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec()], _pgs())
        described = manager.describe_cells()
        assert [d["deferred"] for d in described.values()] == [False, False]
        assert [d["state"] for d in described.values()] == ["running", "running"]
        assert described[compute_cell_id(pool_id="engine", cell_index=1)]["bundles"] == [_BUNDLES[3]]
        assert described[compute_cell_id(pool_id="engine", cell_index=1)]["gpu_ids"] == [_GPUS[3]]

    async def test_an_unbound_cell_cannot_start(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())
        with pytest.raises(CellUnboundError, match="rebind_cell them first"):
            await manager.start_cells([_DEFERRED])
        assert len(fake_ray_cluster.handles) == 2
        # a whole-pool start (M6 path) skips it instead of failing
        await manager.stop_pools(["engine"])
        await manager.start_pools(["engine"])
        assert manager.describe_cells()[_DEFERRED]["state"] == "unbound"

    async def test_re_pointing_the_rollout_view_ignores_an_unbound_cell(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())
        await manager.stop_pools(["engine"])
        await manager.set_pg_view("rollout", _view([2, 3]))


class TestBindAndUnbind:
    async def test_trainer_gpu_to_rollout_and_back(self, fake_ray_cluster: FakeRayCluster):
        """A9: the trainer shrinks off logical bundle 1, a deferred engine starts there, then gives it back."""
        manager = await _launch([_trainer_spec(2), _engine_spec_with_deferred()], _pgs())
        trainer_cell = compute_cell_id(pool_id="trainer", cell_index=0)

        # the trainer still runs on bundle 1: binding there is refused
        with pytest.raises(BundleInUseError, match="in use by running cell trainer"):
            await manager.rebind_cell(_DEFERRED, pg_name="actor", pg_slot_offset=1)

        # trainer shrinks to logical bundle 0 (M6 path)
        await manager.stop_pools(["trainer"])
        await manager.set_pg_view("actor", _view([0]), replacing_pools=["trainer"])
        await manager.replace_pool_spec(_trainer_spec(1))
        await manager.start_pools(["trainer"])

        # a view of the released bundle, built with the public slicer from a startup view
        await manager.set_pg_view("released", slice_pg_info(_view([0, 1]), [1]))
        await manager.rebind_cell(_DEFERRED, pg_name="released", pg_slot_offset=0)
        assert manager.describe_cells()[_DEFERRED]["state"] == "stopped"
        await manager.start_cells([_DEFERRED])

        described = manager.describe_cells()[_DEFERRED]
        assert described["state"] == "running" and described["pg_name"] == "released"
        assert described["bundles"] == [_BUNDLES[1]] and described["gpu_ids"] == [_GPUS[1]]
        assert fake_ray_cluster.handles[-1].options["scheduling_strategy"].placement_group_bundle_index == _BUNDLES[1]
        assert manager.get_worker_infos(_DEFERRED)[0].gpu_ids == [_GPUS[1]]
        assert described["generation"] == 1

        # reverse: stop, unbind, the trainer grows back over bundle 1
        with pytest.raises(AssertionError, match="stop it before unbinding"):
            await manager.unbind_cell(_DEFERRED)
        await manager.stop_cells([_DEFERRED])
        await manager.unbind_cell(_DEFERRED)
        assert manager.describe_cells()[_DEFERRED]["state"] == "unbound"
        await manager.stop_pools(["trainer"])
        await manager.set_pg_view("actor", _view([0, 1]))
        [trainer_cell] = await manager.replace_pool_spec(_trainer_spec(2))
        await manager.start_pools(["trainer"])
        assert manager.get_cell_bundles(trainer_cell) == sorted([_BUNDLES[0], _BUNDLES[1]])
        with pytest.raises(CellUnboundError):
            await manager.start_cells([_DEFERRED])

    async def test_start_rechecks_the_bundles_of_a_bound_deferred_cell(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())
        await manager.rebind_cell(_DEFERRED, pg_name="standby", pg_slot_offset=0)
        # something else took the bundle after the binding
        engine_0 = compute_cell_id(pool_id="engine", cell_index=0)
        await manager.stop_cells([engine_0])
        await manager.rebind_cell(engine_0, pg_name="standby", pg_slot_offset=0)
        await manager.start_cells([engine_0])

        with pytest.raises(BundleInUseError, match="in use by running cell engine"):
            await manager.start_cells([_DEFERRED])
        assert manager.describe_cells()[_DEFERRED]["state"] == "stopped"

    async def test_bindings_are_validated_like_rebind(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())
        with pytest.raises(AssertionError, match="outside 'standby'"):
            await manager.rebind_cell(_DEFERRED, pg_name="standby", pg_slot_offset=2)
        with pytest.raises(BundleInUseError):
            await manager.rebind_cell(_DEFERRED, pg_name="rollout", pg_slot_offset=0)
        assert manager.describe_cells()[_DEFERRED]["state"] == "unbound"

    async def test_only_deferred_cells_can_be_unbound(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_engine_spec_with_deferred()], _pgs())
        engine_0 = compute_cell_id(pool_id="engine", cell_index=0)
        await manager.stop_cells([engine_0])
        with pytest.raises(AssertionError, match="only deferred cells can be unbound"):
            await manager.unbind_cell(engine_0)
        await manager.unbind_cell(_DEFERRED)  # already unbound: a no-op

    async def test_describe_filters_by_pool(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_trainer_spec(2), _engine_spec_with_deferred()], _pgs())
        assert sorted(manager.describe_cells(pool_ids=["trainer"])) == [
            compute_cell_id(pool_id="trainer", cell_index=0)
        ]


class TestDeclaredCells:
    """Cells the caller declares with its own names, started or stopped, on rollout / standby bundles or unbound."""

    def _spec(self):
        return _declared_engine_spec(
            ["c0", "c1", "c2", "c3"],
            [("rollout", 0), ("rollout", 1), ("standby", 0), None],
            num_started=2,
        )

    async def test_names_bindings_and_states_are_reported(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([self._spec()], _pgs())
        described = manager.describe_cells()
        ids = [compute_cell_id(pool_id="engine", cell_index=i) for i in range(4)]
        assert [described[i]["alias"] for i in ids] == ["c0", "c1", "c2", "c3"]
        assert [described[i]["state"] for i in ids] == ["running", "running", "stopped", "unbound"]
        assert [described[i]["gpu_ids"] for i in ids] == [[_GPUS[2]], [_GPUS[3]], [_GPUS[4]], []]
        assert len(fake_ray_cluster.handles) == 2

    async def test_a_standby_cell_starts_without_a_rebind(self, fake_ray_cluster: FakeRayCluster):
        """A4: scale rollout out onto a standby GPU by starting the cell declared there."""
        manager = await _launch([self._spec()], _pgs())
        standby_cell = compute_cell_id(pool_id="engine", cell_index=2)
        await manager.start_cells([standby_cell])
        assert fake_ray_cluster.handles[-1].options["scheduling_strategy"].placement_group_bundle_index == _BUNDLES[4]
        assert manager.describe_cells()[standby_cell]["state"] == "running"
        # scale in: stop it, it stays bound to its standby bundle and can start again
        await manager.stop_cells([standby_cell])
        assert manager.describe_cells()[standby_cell]["state"] == "stopped"
        await manager.start_cells([standby_cell])
        assert manager.get_worker_infos(standby_cell)[0].gpu_ids == [_GPUS[4]]

    async def test_a_whole_pool_start_leaves_declared_stopped_cells_stopped(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([self._spec()], _pgs())
        await manager.stop_pools(["engine"])
        await manager.start_pools(["engine"])
        states = [d["state"] for d in manager.describe_cells().values()]
        assert states == ["running", "running", "stopped", "unbound"]

    async def test_overlapping_declarations_are_refused_at_startup(self, fake_ray_cluster: FakeRayCluster):
        spec = _declared_engine_spec(
            ["c0", "c1", "c2"], [("rollout", 0), ("rollout", 1), ("rollout", 1)], num_started=2
        )
        with pytest.raises(AssertionError, match="are declared on bundle"):
            await _launch([spec], _pgs())

    async def test_a_declaration_outside_its_view_is_refused_at_startup(self, fake_ray_cluster: FakeRayCluster):
        spec = _declared_engine_spec(
            ["c0", "c1", "c2"], [("rollout", 0), ("rollout", 1), ("standby", 2)], num_started=2
        )
        with pytest.raises(AssertionError, match="outside 'standby'"):
            await _launch([spec], _pgs())


class TestStartRechecksEveryCell:
    async def test_a_regular_cell_cannot_start_on_bundles_a_trainer_took(self, fake_ray_cluster: FakeRayCluster):
        """E3: an engine stops, the trainer grows over its bundle, the engine must not start there again."""
        manager = await _launch([_trainer_spec(2), _engine_spec()], _pgs())
        engine_1 = compute_cell_id(pool_id="engine", cell_index=1)
        await manager.stop_cells([engine_1])
        await manager.stop_pools(["trainer"])
        await manager.set_pg_view("actor", _view([0, 1, 3]))
        await manager.replace_pool_spec(_trainer_spec(3))
        await manager.start_pools(["trainer"])

        with pytest.raises(BundleInUseError, match="in use by running cell trainer"):
            await manager.start_cells([engine_1])
        assert manager.describe_cells()[engine_1]["state"] == "stopped"
