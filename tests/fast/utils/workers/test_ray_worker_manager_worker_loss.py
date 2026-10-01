"""A27: a cell whose workers die is marked ``workers_lost`` -- nothing else goes down.

GPU chain 6r1/6r2 (E1-D d2): SIGKILL of one new engine during the up-transaction. The manager must keep running
with its other cells, say which cell lost which workers (so the caller's watchdog can tell "died" from "stopped"),
accept an idempotent stop of that cell, and start it again on request.
"""

from __future__ import annotations

import pytest
import ray
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import EVENT_KILL, READINESS_METHOD, FakeRayCluster

from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec


class DemoWorker:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


_WORKER_CLASS_PATH = f"{DemoWorker.__module__}.{DemoWorker.__qualname__}"


def _make_spec(name: str, *, num_cells: int = 1, num_workers_per_cell: int = 1) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=num_workers_per_cell, num_gpus_per_worker=0
        ),
        worker_class=_WORKER_CLASS_PATH,
        ctor_kwargs=lambda _ctx: {},
    )


async def _launch(specs: list[ServeWorkerSpec]) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, {}, comm_backend=WorkerCommBackend.RPC)
    return manager


async def _scan_all_live_cells(manager: RayWorkerManager) -> None:
    for cell in manager._all_cells():
        if cell.alive:
            await cell._scan_liveness_once()


def _kill_worker_process(cluster: FakeRayCluster, *, handle_index: int) -> None:
    cluster.handles[handle_index].failing_methods[READINESS_METHOD] = ray.exceptions.RayActorError()


@pytest.fixture
async def manager_with_a_lost_cell(fake_ray_cluster: FakeRayCluster) -> tuple[RayWorkerManager, FakeRayCluster]:
    manager = await _launch([_make_spec("engine", num_cells=3)])
    _kill_worker_process(fake_ray_cluster, handle_index=1)
    await _scan_all_live_cells(manager)
    return manager, fake_ray_cluster


class TestWorkerLossIsLocalToTheCell:
    async def test_the_manager_and_the_other_cells_survive(self, manager_with_a_lost_cell):
        """6r2: the whole run died of a stall after one engine was killed; the manager itself never went down and
        must not -- the other cells keep serving and the manager keeps answering."""
        manager, cluster = manager_with_a_lost_cell

        described = manager.describe_cells()
        assert [described[f"engine-0000{i}"]["state"] for i in range(3)] == ["running", "workers_lost", "running"]
        assert [h.killed for h in cluster.handles] == [False, True, False]
        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert [infos[f"engine-0000{i}"].alive for i in range(3)] == [True, False, True]
        # the manager still serves requests about every cell (the lost one included)
        assert manager.get_worker_infos("engine-00001") == []
        assert len(manager.get_worker_infos("engine-00000")) == 1

    async def test_the_lost_workers_are_named(self, manager_with_a_lost_cell):
        """The caller's watchdog must be able to tell 'its engine died' from 'it was stopped on purpose'."""
        manager, _ = manager_with_a_lost_cell

        described = manager.describe_cells()["engine-00001"]

        assert described["state"] == "workers_lost"
        assert described["lost_workers"] == ["engine-00001-00000"]
        assert manager.describe_cells()["engine-00000"]["lost_workers"] is None

    async def test_every_dead_worker_of_a_cell_is_named(self, fake_ray_cluster: FakeRayCluster):
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)])
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await _scan_all_live_cells(manager)

        assert manager.describe_cells()["engine-00000"]["lost_workers"] == ["engine-00000-00000", "engine-00000-00001"]


class TestStopOfALostCellIsIdempotent:
    async def test_stop_cells_succeeds_and_acknowledges_the_loss(self, manager_with_a_lost_cell):
        """REBUILD_OLD stops the target generation; a target that already died must not make that fail."""
        manager, cluster = manager_with_a_lost_cell
        kills_before = cluster.events.count(EVENT_KILL)

        await manager.stop_cells(["engine-00001"])

        described = manager.describe_cells()["engine-00001"]
        assert described["state"] == "stopped" and described["lost_workers"] is None
        assert cluster.events.count(EVENT_KILL) == kills_before, "nothing left to kill"

    async def test_stop_cells_twice_is_fine(self, manager_with_a_lost_cell):
        manager, _ = manager_with_a_lost_cell
        await manager.stop_cells(["engine-00001"])
        await manager.stop_cells(["engine-00001"])
        assert manager.describe_cells()["engine-00001"]["state"] == "stopped"

    async def test_a_mixed_stop_of_lost_and_running_cells_succeeds(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell

        await manager.stop_cells(["engine-00001", "engine-00002"])

        described = manager.describe_cells()
        assert described["engine-00001"]["state"] == "stopped"
        assert described["engine-00002"]["state"] == "stopped"
        assert described["engine-00000"]["state"] == "running"
        assert [h.killed for h in cluster.handles] == [False, True, True]

    async def test_shutdown_still_stops_everything(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell
        await manager.shutdown()
        assert all(d["state"] == "stopped" for d in manager.describe_cells().values())
        assert all(h.killed for h in cluster.handles)


class TestALostCellCanBeStartedAgain:
    async def test_start_cells_clears_the_loss(self, manager_with_a_lost_cell):
        manager, cluster = manager_with_a_lost_cell

        await manager.start_cells(["engine-00001"])

        described = manager.describe_cells()["engine-00001"]
        assert described["state"] == "running" and described["lost_workers"] is None
        assert described["generation"] == 2
        assert len(cluster.handles) == 4

    async def test_an_explicitly_stopped_cell_is_never_reported_lost(self, fake_ray_cluster: FakeRayCluster):
        """Default behaviour unchanged: a stop is a stop."""
        manager = await _launch([_make_spec("engine")])
        await manager.stop_cells(["engine-00000"])
        described = manager.describe_cells()["engine-00000"]
        assert described["state"] == "stopped" and described["lost_workers"] is None
