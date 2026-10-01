from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.http_utils import wrap_ipv6
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.common.metadata import declared_concurrency_groups
from miles.utils.workers.serving.serve_actor import ServeActor
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"

_LIVENESS_SCAN_INTERVAL_SECONDS = 10.0


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(
        args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(args, specs, pgs, comm_backend=comm_backend))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(
        self, args, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo], *, comm_backend: WorkerCommBackend
    ):
        configure_logger(args, source=SimpleProcessIdentity(component="worker_manager"))

        self.port_allocator = PortAllocator(
            dynamic_port_start=getattr(args, "worker_dynamic_port_start", None) or PortAllocator.dynamic_port_start
        )
        self.comm_backend = comm_backend
        self.pgs = pgs
        self._startup_pgs = dict(pgs)
        self._pools = {spec.name: _PoolManager.initial(spec, self) for spec in specs}
        assert len(self._pools) == len(specs)
        self._membership_lock = asyncio.Lock()

        self._validate_initial_bindings()
        # deferred cells are declared stopped: they wait for the caller's start_cells (after rebind_cell if unbound)
        await self.start_cells([c.cell_id for c in self._all_cells() if not c.deferred])

    def _validate_initial_bindings(self) -> None:
        owner: dict[tuple[Any, int], str] = {}
        for cell in self._all_cells():
            if cell.binding is None:
                continue
            (bundles,) = self._validate_binding(cell.spec, [cell.binding])
            for bundle in bundles:
                assert bundle not in owner, f"cells {owner[bundle]} and {cell.cell_id} are declared on bundle {bundle}"
                owner[bundle] = cell.cell_id

    async def start_cells(self, cell_ids: list[str]) -> None:
        async with self._membership_lock:
            cells = [cell for cell_id in cell_ids if (cell := self._find_cell(cell_id)).actors is None]
            unbound = [c.cell_id for c in cells if c.unbound]
            if unbound:
                raise CellUnboundError(f"deferred cells {unbound} are bound to no bundle; rebind_cell them first")
            for cell in cells:
                # bindings were checked free when made; check again, a view may have been re-pointed onto them since
                self._assert_bundles_free(cell.bundles(), ignore=cell)
            try:
                await _gather_or_raise([c.launch_actors() for c in cells])
                await _gather_or_raise([c.alloc_ports() for c in cells])
                await _gather_or_raise([c.post_setup() for c in cells])
            except Exception as error:
                logger.error(f"Starting cells {[c.cell_id for c in cells]} failed, rolling back", exc_info=True)
                results = await asyncio.gather(*[c.stop() for c in cells], return_exceptions=True)
                failed = {c.cell_id: r for c, r in zip(cells, results, strict=True) if isinstance(r, BaseException)}
                if failed:
                    raise StartRollbackFailedError(
                        f"starting cells {[c.cell_id for c in cells]} failed ({error!r}) and stopping "
                        f"{sorted(failed)} during the rollback failed too ({list(failed.values())!r}); they may "
                        f"still hold workers, stop them again"
                    ) from error
                raise

    async def stop_cells(self, cell_ids: list[str]) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[self._find_cell(cell_id).stop() for cell_id in cell_ids])

    async def shutdown(self) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for cell in self._all_cells()])

    # -------------------------- rebinding (elastic placement) -----------------------------
    # A stopped cell or pool may be moved onto other bundles of the placement group created at startup; nothing
    # here creates bundles, so every target stays inside the startup placement (map) range.

    async def rebind_cell(self, cell_id: str, *, pg_name: str, pg_slot_offset: int) -> None:
        """Bind a stopped cell's workers to slots ``pg_slot_offset..`` of ``pgs[pg_name]`` for its next start."""
        async with self._membership_lock:
            cell = self._find_cell(cell_id)
            assert not cell.alive, f"cell {cell_id} is running; stop it before rebinding it"
            binding = _CellBinding(pg_name=pg_name, pg_slot_offset=pg_slot_offset)
            bundles = self._validate_binding(cell.spec, [binding])[0]
            self._assert_bundles_free(bundles, ignore=cell)
            cell.binding = binding
            logger.info(f"Cell {cell_id} rebound to {pg_name}[{pg_slot_offset}:] (bundles {sorted(bundles)})")

    async def unbind_cell(self, cell_id: str) -> None:
        """Return a stopped deferred cell to the unbound state, so it holds no bundle (e.g. before a trainer grows).

        Only deferred cells can be unbound: a cell declared with a bundle layout always keeps one. Unbinding an
        already unbound cell does nothing.
        """
        async with self._membership_lock:
            cell = self._find_cell(cell_id)
            assert cell.deferred, f"cell {cell_id} was not declared deferred; only deferred cells can be unbound"
            assert not cell.alive, f"cell {cell_id} is running; stop it before unbinding it"
            cell.binding = None
            logger.info(f"Cell {cell_id} unbound")

    def describe_cells(self, *, pool_ids: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Every declared cell (of ``pool_ids``, default all) with its state and binding, read-only.

        ``alias`` is the caller's name for the cell (placement map ``rollout_cells``) or None; ``state`` is
        ``unbound`` (a deferred cell with no binding), ``stopped``, ``running`` or ``workers_lost`` (the liveness
        scan found its workers dead and tore the cell down; ``lost_workers`` names them; only ``stop_cells`` on it,
        which succeeds idempotently, or a new ``start_cells`` leaves that state -- the manager itself and its other
        cells are not affected by a loss). ``pg_name`` /
        ``pg_slot_offset`` are the current binding (``pg_slot_offset`` is None for a spec layout, see
        ``bundles``); ``bundles`` are the reordered bundle indices of the startup placement group the cell uses (or
        would use at its next start) and ``gpu_ids`` the matching GPU ids of that view, parallel to ``bundles``.
        """
        pools = self._pools if pool_ids is None else {name: self._pools[name] for name in pool_ids}
        return {c.cell_id: c.describe() for pool in pools.values() for c in pool.cells}

    async def stop_pools(self, pool_ids: list[str]) -> None:
        async with self._membership_lock:
            await asyncio.gather(*[cell.stop() for pool_id in pool_ids for cell in self._pools[pool_id].cells])

    async def start_pools(self, pool_ids: list[str]) -> None:
        """Start the stopped cells of the pools, except deferred cells (declared stopped): those only start through
        an explicit ``start_cells``."""
        await self.start_cells(
            [cell.cell_id for pool_id in pool_ids for cell in self._pools[pool_id].cells if not cell.deferred]
        )

    async def set_pg_view(
        self, pg_name: str, info: PlacementGroupInfo, *, replacing_pools: list[str] | tuple[str, ...] = ()
    ) -> None:
        """Point ``pg_name`` at another slice of the startup placement group (e.g. a new trainer bundle set).

        Stopped cells bound to ``pg_name`` must still fit the new view (otherwise their next start would reach
        outside it); cells of ``replacing_pools`` are exempt because the caller replaces their spec next.
        """
        async with self._membership_lock:
            users = [c.cell_id for c in self._all_cells() if c.alive and c.pg_name == pg_name]
            assert not users, f"cells {users} run on {pg_name!r}; stop them before re-pointing it"
            known = {
                (pg_key(pgi.pg), b) for pgi in self._startup_pgs.values() for b in pgi.pg_reordered_bundle_indices
            }
            wanted = {(pg_key(info.pg), b) for b in info.pg_reordered_bundle_indices}
            outside = sorted(b for _, b in wanted - known)
            assert not outside, f"bundles {outside} are outside the placement group created at startup"
            assert len(wanted) == len(info.pg_reordered_bundle_indices), f"{pg_name!r} view repeats bundles"
            self._assert_bundles_free(wanted, ignore=None)
            num_slots = len(info.pg_reordered_bundle_indices)
            unknown_pools = sorted(set(replacing_pools) - set(self._pools))
            assert not unknown_pools, f"unknown pools {unknown_pools}; known: {sorted(self._pools)}"
            overflowing = {
                c.cell_id: [slot for slot in c.all_slots() if slot >= num_slots]
                for pool_id, pool in self._pools.items()
                if pool_id not in replacing_pools
                for c in pool.cells
                if c.pg_name == pg_name and c.spec.scheduling.num_gpu_slots_per_worker > 0
            }
            overflowing = {cell_id: slots for cell_id, slots in overflowing.items() if slots}
            assert not overflowing, (
                f"stopped cells {overflowing} are bound to slots beyond the {num_slots} bundles of the new "
                f"{pg_name!r} view; rebind them or replace their pool (replacing_pools) first"
            )
            self.pgs = {**self.pgs, pg_name: info}

    def get_pg_view(self, pg_name: str) -> PlacementGroupInfo:
        return self.pgs[pg_name]

    async def replace_pool_spec(self, spec: BaseWorkerSpec) -> list[str]:
        """Swap the spec of a fully stopped pool (e.g. a trainer with a new GPU count); returns its new cell ids."""
        async with self._membership_lock:
            old = self._pools.get(spec.name)
            assert old is not None, f"unknown pool {spec.name!r}; known: {sorted(self._pools)}"
            running = [c.cell_id for c in old.cells if c.alive]
            assert not running, f"cells {running} of {spec.name!r} are running; stop the pool first"
            new = _PoolManager.initial(spec, self)
            self._validate_binding(spec, [None] * len(new.cells), cells=new.cells)
            # a new spec is a new generation even before launch, so nothing can mistake it for the old workers
            for cell in new.cells:
                cell.generation = max((c.generation for c in old.cells), default=0) + 1
            self._pools[spec.name] = new
            return [c.cell_id for c in new.cells]

    def get_cell_bundles(self, cell_id: str) -> list[int]:
        return sorted(b for _, b in self._find_cell(cell_id).bundles())

    def _validate_binding(
        self,
        spec: BaseWorkerSpec,
        bindings: list[_CellBinding | None],
        *,
        cells: list[_CellManager] | None = None,
    ) -> list[set[tuple[Any, int]]]:
        scheduling = spec.scheduling
        results = []
        for index, binding in enumerate(bindings):
            probe = (
                cells[index]
                if cells is not None
                else _CellManager(manager=self, cell_index=0, spec=spec, actors=None, binding=binding)
            )
            if probe.pg_name is None:
                assert binding is None, f"{spec.name!r} owns no GPU slots and cannot be bound to a placement group"
                results.append(set())
                continue
            if scheduling.num_gpu_slots_per_worker == 0:
                assert binding is None, f"{spec.name!r} owns no GPU slots to rebind"
                results.append(set())
                continue
            assert (
                probe.pg_name in self.pgs
            ), f"unknown placement group view {probe.pg_name!r}; known: {sorted(self.pgs)}"
            num_slots = len(self.pgs[probe.pg_name].pg_reordered_bundle_indices)
            slots = probe.all_slots()
            bad = [slot for slot in slots if not 0 <= slot < num_slots]
            assert not bad, f"slots {bad} are outside {probe.pg_name!r}, which has {num_slots} bundles"
            results.append(probe.bundles())
        return results

    def _assert_bundles_free(self, bundles: set[tuple[Any, int]], *, ignore: _CellManager | None) -> None:
        for other in self._all_cells():
            if other is ignore or not other.alive:
                continue
            if overlap := bundles & other.bundles():
                raise BundleInUseError(
                    f"bundles {sorted(b for _, b in overlap)} are in use by running cell {other.cell_id}"
                )

    def inject_fault(self, cell_id: str, *, mode: str, worker_in_cell_index: int) -> None:
        cell = self._find_cell(cell_id)
        if not cell.alive:
            raise RuntimeError(f"Cell {cell_id} is not alive, cannot inject fault")
        if not 0 <= worker_in_cell_index < len(cell.actors):
            raise IndexError(
                f"worker_in_cell_index {worker_in_cell_index} out of range for cell {cell_id} "
                f"(has {len(cell.actors)} workers)"
            )
        cell.actors[worker_in_cell_index].actor_handle.inject_fault.remote(mode)

    def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._find_actor(worker_name).self_addrs
        assert addrs is not None, (
            f"{worker_name} has not been given its ports yet; a caller reading them now would take the "
            f"endpoints it cannot find for endpoints the worker does not have"
        )
        return addrs

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        return {
            name: [a.self_addrs or {} for c in g.cells if c.alive for a in c.actors] for name, g in self._pools.items()
        }

    def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        cell = self._find_cell(cell_id)
        return [self._compute_worker_info(actor) for actor in (cell.actors if cell.actors is not None else [])]

    def get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        unknown = set(pool_ids) - set(self._pools)
        assert not unknown, f"{unknown=} {sorted(self._pools)=}"
        infos = [c.get_info() for name in pool_ids for c in self._pools[name].cells]
        return {info.cell_id: info for info in infos}

    def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> ray.actor.ActorHandle:
        actor = self._find_actor(worker_name)
        assert actor.generation == expected_generation, (
            f"{worker_name} is now generation {actor.generation}, not the {expected_generation} it was described as; "
            f"ask for its worker infos again"
        )
        return actor.actor_handle

    def _compute_worker_info(self, actor: _BaseActorManager) -> WorkerInfo:
        served_over_rpc = isinstance(actor.spec, ServeWorkerSpec) and self.comm_backend == WorkerCommBackend.RPC
        return WorkerInfo(
            name=actor.name,
            generation=actor.generation,
            self_addrs=actor.self_addrs or {},
            gpu_ids=actor.gpu_ids,
            worker_class=actor.spec.worker_class if served_over_rpc else None,
        )

    def _find_actor(self, worker_name: str) -> _BaseActorManager:
        matches = [a for c in self._all_cells() if c.alive for a in c.actors if a.name == worker_name]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    def _find_cell(self, cell_id: str) -> _CellManager:
        matches = [c for c in self._all_cells() if c.cell_id == cell_id]
        assert len(matches) == 1, f"{cell_id=} {matches=}"
        return matches[0]

    def _all_cells(self) -> list[_CellManager]:
        return [c for g in self._pools.values() for c in g.cells]


@dataclass(kw_only=True)
class _PoolManager:
    spec: BaseWorkerSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseWorkerSpec, manager: RayWorkerManager) -> _PoolManager:
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                    deferred=cell_index >= spec.scheduling.num_cells,
                    binding=(
                        _CellBinding(*b)
                        if cell_index < len(spec.scheduling.initial_bindings)
                        and (b := spec.scheduling.initial_bindings[cell_index]) is not None
                        else None
                    ),
                )
                for cell_index in range(spec.scheduling.num_cells + spec.scheduling.num_deferred_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseWorkerSpec)


def _actor_manager_cls(spec: BaseWorkerSpec, *, comm_backend: WorkerCommBackend) -> type[_BaseActorManager]:
    match spec, comm_backend:
        case CommandWorkerSpec(), _:
            return _CommandActorManager
        case ServeWorkerSpec(), WorkerCommBackend.RPC:
            return _ServeActorRpcCommManager
        case ServeWorkerSpec(), WorkerCommBackend.RAY:
            return _ServeActorRayCommManager
    raise AssertionError(f"{spec.name} is neither served nor launched as a command")


class BundleInUseError(RuntimeError):
    pass


class CellUnboundError(RuntimeError):
    """A deferred cell was asked to start before ``rebind_cell`` bound it to bundles."""


class StartRollbackFailedError(RuntimeError):
    """A start failed and the rollback that should have stopped its cells failed as well."""


def pg_key(pg: Any) -> Any:
    return getattr(pg, "id", pg)


@dataclass(frozen=True)
class _CellBinding:
    pg_name: str
    pg_slot_offset: int


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    manager: RayWorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None
    generation: int = 0
    liveness_scan_task: asyncio.Task | None = None
    # set by RayWorkerManager.rebind_cell; None keeps the spec's own slot layout
    binding: _CellBinding | None = None
    # declared beyond the spec's num_cells: starts unbound (no bundle) and only a binding makes it startable
    deferred: bool = False
    # workers the liveness scan found dead, which tore the cell down (state ``workers_lost``); None after an
    # explicit stop (acknowledged) or a new start. Only this cell is affected: the manager and its other cells go on.
    lost_worker_names: list[str] | None = None

    @property
    def alias(self) -> str | None:
        aliases = self.spec.scheduling.cell_aliases
        return aliases[self.cell_index] if self.cell_index < len(aliases) else None

    @property
    def unbound(self) -> bool:
        return self.deferred and self.binding is None

    @property
    def pg_name(self) -> str | None:
        if self.unbound:
            return None
        return self.binding.pg_name if self.binding is not None else self.spec.scheduling.pg_name

    def slot_of(self, worker_in_cell_index: int) -> int | None:
        scheduling = self.spec.scheduling
        if self.pg_name is None:
            return None
        if (binding := self.binding) is not None:
            return binding.pg_slot_offset + worker_in_cell_index * scheduling.num_gpu_slots_per_worker
        return (
            scheduling.pg_slot_offset
            + (self.cell_index * scheduling.num_workers_per_cell + worker_in_cell_index)
            * scheduling.num_gpu_slots_per_worker
        )

    def all_slots(self) -> list[int]:
        scheduling = self.spec.scheduling
        if self.unbound:
            return []
        return [
            self.slot_of(w) + k
            for w in range(scheduling.num_workers_per_cell)
            for k in range(max(1, scheduling.num_gpu_slots_per_worker))
        ]

    def bundles(self) -> set[tuple[Any, int]]:
        if self.pg_name is None or self.spec.scheduling.num_gpu_slots_per_worker == 0:
            return set()
        pg = self.manager.pgs[self.pg_name]
        return {(pg_key(pg.pg), pg.pg_reordered_bundle_indices[slot]) for slot in self.all_slots()}

    async def launch_actors(self):
        assert self.actors is None
        self.generation += 1
        self.lost_worker_names = None
        scheduling = self.spec.scheduling
        actor_manager_cls = _actor_manager_cls(self.spec, comm_backend=self.manager.comm_backend)
        self.actors = [
            actor_manager_cls(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
                gpu_slot_index=self.slot_of(worker_in_cell_index),
            )
            for worker_in_cell_index in range(scheduling.num_workers_per_cell)
        ]
        await self._for_all_actors(lambda a: a.launch_actor())
        self.liveness_scan_task = asyncio.create_task(self._scan_liveness_forever(self.generation))

    async def alloc_ports(self) -> None:
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        await self._for_all_actors(lambda a: a.post_setup())

    async def stop(self, *, lost_workers: list[str] | None = None) -> None:
        """Stop the cell's workers. ``lost_workers`` (from the liveness scan) records why the cell went down; an
        explicit stop (``stop_cells``/``shutdown``) acknowledges an earlier loss and is a no-op on a cell already
        down, so stopping a cell whose workers died is idempotent and succeeds."""
        if lost_workers is None:
            self.lost_worker_names = None
        if self.actors is None:
            return
        await self._for_all_actors(lambda a: a.stop())
        self.actors = None
        self.lost_worker_names = list(lost_workers) if lost_workers is not None else None

    async def _scan_liveness_forever(self, generation: int) -> None:
        while self.generation == generation and self.actors is not None:
            await asyncio.sleep(_LIVENESS_SCAN_INTERVAL_SECONDS)
            try:
                await self._scan_liveness_once()
            except Exception:
                logger.error(f"Scanning liveness of cell {self.cell_id} failed, will scan again", exc_info=True)

    async def _scan_liveness_once(self) -> None:
        generation = self.generation
        dead_worker_names = await self._find_dead_worker_names()
        if not dead_worker_names:
            return

        async with self.manager._membership_lock:
            if self.actors is None or self.generation != generation:
                return
            logger.error(
                f"Cell {self.cell_id} lost workers {dead_worker_names} without being stopped, "
                f"so the whole cell is torn down and reported as not alive (state workers_lost); "
                f"the manager and its other cells keep running"
            )
            await self.stop(lost_workers=dead_worker_names)

    async def _find_dead_worker_names(self) -> list[str]:
        if (actors := self.actors) is None:
            return []
        probes = await asyncio.gather(*[a.probe_is_dead() for a in actors])
        return [actor.name for actor, is_dead in zip(actors, probes, strict=True) if is_dead]

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])

    def describe(self) -> dict[str, Any]:
        bundles: list[int] | None = []
        gpu_ids: list[int] | None = []
        if self.pg_name is not None and self.spec.scheduling.num_gpu_slots_per_worker > 0:
            pg = self.manager.pgs[self.pg_name]
            slots = self.all_slots()
            if all(0 <= slot < len(pg.pg_reordered_bundle_indices) for slot in slots):
                bundles = [pg.pg_reordered_bundle_indices[slot] for slot in slots]
                gpu_ids = [pg.pg_reordered_gpu_ids[slot] for slot in slots]
            else:  # a stopped cell left outside a re-pointed view (its pool is being replaced)
                bundles = gpu_ids = None
        return dict(
            cell_id=self.cell_id,
            alias=self.alias,
            pool_id=self.spec.name,
            deferred=self.deferred,
            state=self.state,
            lost_workers=None if self.lost_worker_names is None else list(self.lost_worker_names),
            generation=self.generation,
            pg_name=self.pg_name,
            pg_slot_offset=self.binding.pg_slot_offset if self.binding is not None else None,
            bundles=bundles,
            gpu_ids=gpu_ids,
        )

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            pool_id=self.spec.name,
            alive=self.alive and self._all_workers_have_addrs,
            worker_names=[a.name for a in self.actors] if self.actors is not None else [],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=f(WorkerMetaContext(cell_index=self.cell_index)) if (f := self.spec.meta) is not None else {},
        )

    @property
    def cell_id(self) -> str:
        return compute_cell_id(pool_id=self.spec.name, cell_index=self.cell_index)

    @property
    def alive(self) -> bool:
        return self.actors is not None

    @property
    def state(self) -> str:
        """``unbound`` / ``running`` / ``workers_lost`` (torn down by the liveness scan, not acknowledged by a stop
        or restarted yet) / ``stopped``."""
        if self.unbound:
            return "unbound"
        if self.alive:
            return "running"
        return "workers_lost" if self.lost_worker_names is not None else "stopped"

    @property
    def _all_workers_have_addrs(self) -> bool:
        return all(a.self_addrs is not None for a in self.actors or [])


_SHUTDOWN_TIMEOUT = 30


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    local_gpu_ids: list[int] | None = None
    gpu_slot_index: int | None

    async def launch_actor(self) -> None:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        allocated: NamedHostAndPorts = {}

        node_ip, external_ip, self.local_gpu_ids = await asyncio.gather(
            self.actor_handle._get_node_ip.remote(),
            self.actor_handle._get_node_external_ip.remote(),
            self.actor_handle._to_local_gpu_ids.remote(gpu_ids=self.gpu_ids),
        )
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            if port_info.allow_dynamic:
                port = await self.manager.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
            else:
                port = port_info.static_port + (self.parent.cell_index if port_info.offset_by_cell else 0)
                await self._assert_static_port_is_free(port=port, port_name=port_info.name, node_ip=node_ip)
            allocated[port_info.name] = HostAndPort(
                host=wrap_ipv6(node_ip),
                port=port,
                external_host=wrap_ipv6(external_ip) if external_ip else None,
            )

        self.self_addrs = allocated

    async def _assert_static_port_is_free(self, *, port: int, port_name: str, node_ip: str) -> None:
        free = await self.actor_handle._is_port_available.remote(port=port)
        assert free, (
            f"Port {port} on {node_ip} is already in use, so {self.name} cannot serve its {port_name!r} "
            f"endpoint there; a stale process from an earlier run is the usual cause"
        )

    @property
    def launch_context(self) -> WorkerLaunchContext:
        return WorkerLaunchContext(
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            gpu_ids=self.gpu_ids,
        )

    def _compute_remote_options(self) -> dict:
        return {}

    def _create_actor(self, actor_class: type, **ctor_kwargs) -> ray.actor.ActorHandle:
        scheduling_strategy = None
        if (pg_name := self.parent.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        remote_options = self._compute_remote_options()
        remote_class = ray.remote(**remote_options)(actor_class) if remote_options else ray.remote(actor_class)

        return remote_class.options(
            num_cpus=self.spec.scheduling.num_cpus_per_worker,
            num_gpus=self.spec.scheduling.num_gpus_per_worker,
            **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
            runtime_env={"env_vars": self.spec.env_var(self.launch_context)},
            **(compute_ray_pin_head_options() if self.spec.scheduling.pin_to_head else {}),
        ).remote(**ctor_kwargs)

    async def probe_is_dead(self) -> bool:
        if self.actor_handle is None:
            return False
        return await RayWorkerHandle(self.actor_handle).probe_is_dead()

    async def stop(self) -> None:
        if self.actor_handle is None:
            return

        await self._shutdown_gracefully()

        try:
            ray.kill(self.actor_handle)
            logger.info(f"Killed actor at {self=}")
        except Exception as e:
            logger.warning(f"Failed to kill actor at {self=} ({e})")

    async def _shutdown_gracefully(self) -> None:
        pass

    @property
    def name(self) -> str:
        return compute_worker_name(
            pool_id=self.spec.name,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def generation(self) -> int:
        return self.parent.generation

    @property
    def gpu_ids(self) -> list[int]:
        if (pg_name := self.parent.pg_name) is None:
            return []
        pg = self.manager.pgs[pg_name]
        base_gpu_id = int(pg.pg_reordered_gpu_ids[self.gpu_slot_index])
        return list(range(base_gpu_id, base_gpu_id + self.spec.scheduling.num_gpu_slots_per_worker))

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):
    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(CommandActor)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            **dict(self.launch_context),
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            pool_addrs=self.manager.get_addrs(),
            local_gpu_ids=self.local_gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        # exec lets the command replace /bin/sh, so parent death and Ray's child cleanup kill the
        # command itself rather than a shell that would leave it orphaned with its ports.
        self.actor_handle.run.remote(cmd=f"exec {launch_cmd}", envs={})

    async def _shutdown_gracefully(self) -> None:
        try:
            await asyncio.wait_for(self.actor_handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")


@dataclass
class _ServeActorRayCommManager(_BaseActorManager[ServeWorkerSpec]):
    def _compute_remote_options(self) -> dict:
        groups = self.spec.concurrency_groups
        return {} if groups is None else dict(concurrency_groups=groups)

    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(
            self._compute_actor_class(),
            ctor_kwargs=self.spec.ctor_kwargs,
            context=self.launch_context,
        )

    def _compute_actor_class(self) -> type:
        actor_class = bootstrapped_worker_class(self.spec.worker_class)
        method_groups = self._compute_method_concurrency_groups(actor_class)
        if not method_groups:
            return actor_class
        return type(
            f"{actor_class.__name__}WithConcurrencyGroups",
            (actor_class,),
            {
                name: _route_method_to_concurrency_group(getattr(actor_class, name), group=group)
                for name, group in method_groups.items()
            },
        )

    def _compute_method_concurrency_groups(self, actor_class: type) -> dict[str, str]:
        if (groups := self.spec.concurrency_groups) is None:
            return {}

        method_groups = declared_concurrency_groups(actor_class)
        assert method_groups, (
            f"Worker {self.spec.name!r} declares concurrency groups {sorted(groups)} but no method of "
            f"{actor_class.__name__} is annotated with @rpc(concurrency_group=...): threading the actor "
            f"while every method stays in the default group buys nothing"
        )
        undeclared = sorted(set(method_groups.values()) - set(groups))
        assert not undeclared, f"Worker {self.spec.name!r} routes methods to undeclared groups: {undeclared}"
        return method_groups

    async def post_setup(self) -> None:
        pass


def _route_method_to_concurrency_group(method: Callable, *, group: str) -> Callable:
    @functools.wraps(method)
    def routed(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return ray.method(concurrency_group=group)(routed)


@dataclass
class _ServeActorRpcCommManager(_BaseActorManager[ServeWorkerSpec]):
    async def launch_actor(self) -> None:
        self.actor_handle = self._create_actor(
            ServeActor,
            build_worker=partial(
                _build_serve_worker,
                worker_class_path=self.spec.worker_class,
                ctor_kwargs=self.spec.ctor_kwargs,
                context=self.launch_context,
            ),
        )

    async def post_setup(self) -> None:
        await self.actor_handle.start_rpc_server.remote(port=self.self_addrs[RPC_PORT_NAME].port)


def _build_serve_worker(
    *, worker_class_path: str, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
) -> Any:
    return bootstrapped_worker_class(worker_class_path)(ctor_kwargs=ctor_kwargs, context=context)


@functools.cache
def bootstrapped_worker_class(worker_class_path: str) -> type:
    worker_class = load_function(worker_class_path)

    class BootstrappedWorker(worker_class, NodeProbeMixin):
        def __init__(
            self, *, ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]], context: WorkerLaunchContext
        ) -> None:
            super().__init__(**ctor_kwargs(_ctor_context(context)))

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def _ctor_context(launch_context: WorkerLaunchContext) -> WorkerCtorContext:
    return WorkerCtorContext(
        cell_index=launch_context.cell_index,
        worker_in_cell_index=launch_context.worker_in_cell_index,
        gpu_ids=launch_context.gpu_ids,
        capability=DeferredBackendCapability(create=_create_ray_backend_capability),
    )


def _create_ray_backend_capability() -> BackendCapability:
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())


async def _gather_or_raise(coros: list[Coroutine[Any, Any, None]]) -> None:
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
