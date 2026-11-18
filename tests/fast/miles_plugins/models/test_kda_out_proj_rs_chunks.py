"""``MILES_K31_KDA_OUT_PROJ_RS_CHUNKS``: the KDA out_proj + sequence-parallel reduce-scatter pipelined over row
chunks equals ``reduce_scatter_to_sequence_parallel_region(F.linear(x, w))`` -- output, input gradient and weight
gradient -- with and without the folded output relayout (MILES_K31_KDA_CORE_RELAYOUT_OVERLAP), for TP x CP gloo
ranks and 2 / 4 chunks.

What the CPU test pins down is the plumbing: which rows each chunk takes, where its reduce-scatter lands, and that
the backward is the unchunked one. The data are small integers, so every sum is exact and the comparison does not
depend on how gloo or the CPU GEMM associate; the rounding-level properties the switch relies on (a row-chunked
GEMM and a chunked NCCL reduce-scatter give the same bits) are kernel properties, checked on the GPU
(work/overlap2/gpu_check_tpchunk.py)."""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["miles-plugin"])

# by file path: the package __init__ imports Megatron, which CPU CI does not install
_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "cp_utils.py"
_SPEC = importlib.util.spec_from_file_location("_cp_utils_out_proj_rs_chunks_under_test", _PATH)
cp_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_utils)

CORE, OUT = 4, 6


def _gloo_reduce_scatter(output, input_, group):
    """reduce_scatter_tensor on gloo (which has none): all-reduce, keep this rank's block."""
    full = input_.clone()
    dist.all_reduce(full, group=group)
    rows = output.shape[0]
    rank = group.rank()
    output.copy_(full[rank * rows : (rank + 1) * rows])


class _ReduceScatterSP(torch.autograd.Function):
    """reduce_scatter_to_sequence_parallel_region: reduce-scatter, all-gather back (Megatron's two collectives)."""

    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.tp_group = tp_group
        out = x.new_empty((x.shape[0] // tp_group.size(), *x.shape[1:]))
        _gloo_reduce_scatter(out, x.contiguous(), tp_group)
        return out

    @staticmethod
    def backward(ctx, grad):
        full = grad.new_empty((grad.shape[0] * ctx.tp_group.size(), *grad.shape[1:]))
        dist.all_gather_into_tensor(full, grad.contiguous(), group=ctx.tp_group)
        return full, None


def _ints(shape, seed):
    return torch.randint(-3, 4, shape, generator=torch.Generator().manual_seed(seed)).float()


def _worker(rank, world, tp, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        cp_utils._reduce_scatter_tensor = _gloo_reduce_scatter
        cp = world // tp
        cp_rank, tp_rank = divmod(rank, tp)
        tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
        cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
        tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
        q = 2 * cp * tp * 4  # shard rows a multiple of TP x 4 chunks
        cases = {"single": [0, 8 * q], "uneven": [0, q, 6 * q, 8 * q]}
        for name, edges in cases.items():
            cu = torch.tensor(edges, dtype=torch.int32)
            shard = int(cu[-1]) // cp
            core = _ints((shard, 1, CORE), 7 + rank)  # TP-distinct partial rows (row-parallel out_proj)
            weight = _ints((OUT, CORE), 3 + tp_rank)
            g_out = _ints((shard // tp, 1, OUT), 5 + rank)
            for fold in (False, True) if cp > 1 else (False,):
                got = {}
                for path in ("reference", "chunks2", "chunks4"):
                    c = core.clone().requires_grad_(True)
                    w = weight.clone().requires_grad_(True)
                    if path == "reference":
                        if fold:
                            x = cp_utils.packed_shard_to_zigzag_a2a_linear(c, w, cu, cp_group, cp_rank, cp)
                        else:
                            x = F.linear(c, w)
                        o = _ReduceScatterSP.apply(x, tp_group)
                    else:
                        chunks = int(path[-1])
                        assert cp_utils.out_proj_rs_chunks_apply(shard, tp, chunks)
                        o = cp_utils.linear_reduce_scatter_chunked(
                            c, w, tp_group, chunks, cp_group=cp_group, relayout=(cu, cp_rank, cp) if fold else None
                        )
                    o.backward(g_out)
                    got[path] = (o.detach(), c.grad, w.grad)
                for path in ("chunks2", "chunks4"):
                    results[(rank, name, fold, path)] = [
                        torch.equal(a, b) for a, b in zip(got[path], got["reference"], strict=True)
                    ]
        # frozen weight (no wgrad) and a core that needs no grad
        c = _ints((16 * tp, 1, CORE), 1 + rank).requires_grad_(True)
        o = cp_utils.linear_reduce_scatter_chunked(c, weight, tp_group, 2)
        o.backward(_ints((16, 1, OUT), 2 + rank))
        ref_c = c.detach().clone().requires_grad_(True)
        _ReduceScatterSP.apply(F.linear(ref_c, weight), tp_group).backward(_ints((16, 1, OUT), 2 + rank))
        results[(rank, "frozen_weight")] = [torch.equal(c.grad, ref_c.grad)]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(2, 1), (4, 1), (2, 2)])
def test_out_proj_rs_chunks_match(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    expected = tp * cp * (2 * (2 if cp > 1 else 1) * 2 + 1)
    assert len(results) == expected, dict(results)
    for key, flags in results.items():
        assert all(flags), (key, flags)


def test_switch_and_applicability(monkeypatch):
    monkeypatch.delenv(cp_utils.OUT_PROJ_RS_CHUNKS_ENV, raising=False)
    assert cp_utils.out_proj_rs_chunks() == 1
    monkeypatch.setenv(cp_utils.OUT_PROJ_RS_CHUNKS_ENV, "2")
    assert cp_utils.out_proj_rs_chunks() == 2
    monkeypatch.setenv(cp_utils.OUT_PROJ_RS_CHUNKS_ENV, "0")
    with pytest.raises(ValueError):
        cp_utils.out_proj_rs_chunks()
    assert not cp_utils.out_proj_rs_chunks_apply(131072, 8, 1)  # off
    assert not cp_utils.out_proj_rs_chunks_apply(131072, 1, 2)  # no TP: nothing to pipeline
    assert not cp_utils.out_proj_rs_chunks_apply(131072 + 8, 8, 2)  # rows not divisible: unchunked path
    assert cp_utils.out_proj_rs_chunks_apply(131072, 8, 2)
