import socket
import threading
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import timedelta
from contextlib import AbstractContextManager, nullcontext

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group


class UpdateWeightFromDistributed(WeightTransferProtocol):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._model_update_groups = None
        parallel_state = get_parallel_state()
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0,
            )
            if parallel_state.pp.size > 1
            else nullcontext()
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self.group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        with self._engine_lock:
            futures = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.rollout_engines,
                bucket,
                selector=self._selector,
            )
            async_utils.wait_futures(futures)
            bucket.clear()


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[SGLangApiClient],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1
    # --update-weight-group-timeout-s bounds the rendezvous / the group's collectives and each engine's join
    # request; None keeps torch's default_pg_timeout and the HTTP client's own (unbounded read) timeout.
    timeout_s = group_timeout_seconds(args)
    join_kwargs = {} if timeout_s is None else dict(timeout=timeout_s)

    futures = []
    rank_cursor = 1
    for i, api_client in enumerate(rollout_engines):
        futures.append(
            async_utils.submit(
                api_client.init_weights_update_group(
                    master_address,
                    master_port,
                    rank_cursor,
                    world_size,
                    group_name,
                    backend="nccl",
                    **join_kwargs,
                )
            )
        )
        rank_cursor += engine_gpu_counts[i]

    # Rank 0 joins in a thread so the engines' join requests can be watched meanwhile: a rendezvous waits for
    # every rank, so an engine that dies (or refuses) before joining would otherwise block this call for the
    # whole pg timeout (default 30 min) with the caller's locks held (yeto A27: SIGKILLed engine during a
    # member publish -> stall). An engine failure aborts the connect at once; the join thread is left to time
    # out on its own (its port is private to this attempt and never reused).
    join: Future = Future()

    def _join() -> None:
        try:
            join.set_result(
                init_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                    **({} if timeout_s is None else dict(timeout=timedelta(seconds=timeout_s))),
                )
            )
        except BaseException as e:  # noqa: BLE001 - delivered to the waiting caller
            join.set_exception(e)

    threading.Thread(target=_join, name=f"weight-update-join-{group_name}", daemon=True).start()
    pending = set(futures)
    while not join.done():
        done, _ = wait({join, *pending}, return_when=FIRST_COMPLETED)
        for future in done:
            if future is join:
                continue
            pending.discard(future)
            if (error := future.exception()) is not None:
                index = futures.index(future)
                raise RuntimeError(
                    f"engine {index} failed to join weight update group {group_name} while rank 0 waited for "
                    f"the rendezvous; connect aborted instead of waiting for the group timeout ({error!r})"
                ) from error
    model_update_groups = join.result()
    async_utils.wait_futures(futures)
    return model_update_groups


def group_timeout_seconds(args: Namespace) -> float | None:
    """``--update-weight-group-timeout-s`` as a float, None when unset (default: torch's default_pg_timeout)."""
    value = getattr(args, "update_weight_group_timeout_s", None)
    if value is None:
        return None
    value = float(value)
    if not value > 0:
        raise ValueError(f"--update-weight-group-timeout-s must be positive, got {value}")
    return value


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    futures = [async_utils.submit(client.destroy_weights_update_group(group_name)) for client in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        async_utils.wait_futures(futures)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
) -> list[Future]:
    """
    Send metadata (HTTP), broadcast tensors (NCCL rank 0 → engines).
    """
    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
            )
        )
        for client in rollout_engines
    ]

    contiguous_tensors = [
        param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
    ]
    handles = []
    for tensor in contiguous_tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return futures
