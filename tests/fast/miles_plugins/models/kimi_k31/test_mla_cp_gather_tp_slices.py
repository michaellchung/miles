"""The TP-slice CP gather of the MLA latent kv / indexer keys equals Megatron's CP gather, forward and backward
(gloo, TP x CP ranks, a head-parallel-like TP-distinct consumer)."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

model = pytest.importorskip("miles_plugins.models.kimi_k31.model")
from megatron.core.tensor_parallel import mappings  # noqa: E402

ROWS, WIDTH = 24, 10


def _reduce_scatter_gloo(input_, group=None, *args, **kwargs):
    """gloo has no reduce-scatter: all-reduce, keep this rank's rows (both paths call this one function)."""
    out = input_.contiguous().clone()
    dist.all_reduce(out, group=group)
    part = out.shape[0] // group.size()
    return out[group.rank() * part : (group.rank() + 1) * part]


def _worker(rank, world, tp, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mappings._reduce_scatter_along_first_dim = _reduce_scatter_gloo
        torch.cuda.current_device = lambda: "cpu"  # Megatron's gather allocates its output there
        cp = world // tp
        cp_rank, tp_rank = divmod(rank, tp)
        tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
        cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
        tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
        torch.manual_seed(cp_rank)  # TP-replicated input: the same on every TP rank of a CP rank
        source = torch.randn(ROWS, WIDTH)
        torch.manual_seed(100 + rank)  # TP-distinct upstream gradient
        upstream = torch.randn(cp * ROWS, WIDTH)
        order = torch.randperm(cp * ROWS, generator=torch.Generator().manual_seed(5))  # a shared gather order
        outs, grads = [], []
        for switch in ("0", "1"):
            os.environ["MILES_K31_MLA_CP_GATHER_V2"] = switch
            x = source.clone().requires_grad_(True)
            y = model.gather_tp_replicated_over_cp(x, cp_group, tp_group).index_select(0, order)
            (y * upstream).sum().backward()
            outs.append(y.detach())
            grads.append(x.grad)
        results[rank] = (torch.equal(outs[0], outs[1]), torch.equal(grads[0], grads[1]))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(2, 2), (4, 2)])
def test_mla_cp_gather_by_tp_slices_matches_megatron(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp and all(all(v) for v in results.values()), dict(results)
