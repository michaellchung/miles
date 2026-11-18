"""``MILES_K31_KDA_SP_RELAYOUT_FUSED``: the fused sequence-parallel all-gather + zigzag -> contiguous CP relayout
equals Megatron's SP all-gather followed by the all-to-all relayout bit for bit in the forward; its gradient sums
the same TP partials (bit-identical at TP2, where a two-term sum has one order; allclose and as close to an fp64
reference as the default at TP4), and every rank carries the same cross-CP rows. ``MILES_K31_KDA_CORE_RELAYOUT_
OVERLAP``: the fused core relayout + out_proj equals ``F.linear(packed_shard_to_zigzag_a2a(core), weight)`` bit for
bit, forward and backward (gloo, TP x CP ranks)."""

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

register_cpu_ci(est_time=300, suite="stage-b-cpu", labels=["miles-plugin"])

# by file path: the package __init__ imports Megatron, which CPU CI does not install
_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "cp_utils.py"
_SPEC = importlib.util.spec_from_file_location("_cp_utils_sp_fused_under_test", _PATH)
cp_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_utils)

HIDDEN, CORE, OUT = 6, 4, 5


def _cases(cp, tp):
    q = 2 * cp * tp  # every length a multiple of 2 * cp and every shard a multiple of tp
    return {
        "single": [0, 8 * q],  # one long zigzag sequence
        "short": [0, q, 2 * q, 3 * q, 4 * q, 5 * q, 6 * q, 7 * q, 8 * q],  # many short ones
        "uneven": [0, q, 6 * q, 8 * q],
        "padded": [0, 3 * q, 3 * q + cp * tp * 3],  # final length an odd multiple of cp: contiguous, no zigzag
    }


class _SPGather(torch.autograd.Function):
    """gather_from_sequence_parallel_region(tensor_parallel_output_grad=True): all-gather, reduce-scatter back."""

    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.tp_group = tp_group
        out = x.new_empty((x.shape[0] * tp_group.size(), *x.shape[1:]))
        dist.all_gather_into_tensor(out, x.contiguous(), group=tp_group)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.tp_group)  # gloo has no reduce-scatter
        part = grad.shape[0] // ctx.tp_group.size()
        rank = ctx.tp_group.rank()
        return grad[rank * part : (rank + 1) * part], None


def _worker(rank, world, tp, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        cp = world // tp
        cp_rank, tp_rank = divmod(rank, tp)
        tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
        cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
        tp_group, cp_group = tp_groups[cp_rank], cp_groups[tp_rank]
        for name, edges in _cases(cp, tp).items():
            cu = torch.tensor(edges, dtype=torch.int32)
            total = int(cu[-1])
            shard = total // cp
            part = shard // tp
            full = torch.randn(total, 1, HIDDEN, generator=torch.Generator().manual_seed(11))
            _, packed_position = cp_utils._zigzag_maps(cu, cp, total)
            zigzag = full[packed_position[cp_rank * shard : (cp_rank + 1) * shard]]
            torch.manual_seed(100 + rank)  # TP-distinct consumer (column-parallel-like) and upstream gradient
            weight, up = torch.randn(HIDDEN), torch.randn(shard, 1, HIDDEN)
            outs, grads = {}, {}
            for path in ("reference", "fused", "fp64"):
                dtype = torch.float64 if path == "fp64" else torch.float32
                sp = zigzag[tp_rank * part : (tp_rank + 1) * part].to(dtype).clone().requires_grad_(True)
                if path == "fused":
                    y = cp_utils.sp_gather_zigzag_to_packed_shard(sp, cu, cp_group, tp_group)
                else:
                    y = cp_utils.zigzag_to_packed_shard_a2a(
                        _SPGather.apply(sp, tp_group), cu, cp_group, cp_rank, cp, tp_group
                    )
                ((y * weight.to(dtype)) * up.to(dtype)).sum().backward()
                outs[path], grads[path] = y.detach(), sp.grad
            plan = cp_utils._sp_relayout_plan(cu, cp_rank, cp, tp_rank, tp, total)
            err = lambda g, grads=grads: (g.double() - grads["fp64"]).abs().max().item()  # noqa: E731
            checks = {
                "out_is_packed_shard": torch.equal(outs["fused"], full[cp_rank * shard : (cp_rank + 1) * shard]),
                "out_equal": torch.equal(outs["fused"], outs["reference"]),
                "grad_close": torch.allclose(grads["fused"], grads["reference"], rtol=1e-6, atol=1e-6),
                "grad_equal": torch.equal(grads["fused"], grads["reference"]),
                "err_fused": err(grads["fused"]),
                "err_reference": err(grads["reference"]),
                "cached": cp_utils._sp_relayout_plan(cu, cp_rank, cp, tp_rank, tp, total) is plan,
                "cross_cp_rows": plan.cross_cp_rows,
                "width": plan.width,
                "part": part,
            }
            # the core relayout + out_proj: bit-identical, forward and backward
            core = torch.randn(shard, 1, CORE, generator=torch.Generator().manual_seed(7 + rank))
            w = torch.randn(OUT, CORE, generator=torch.Generator().manual_seed(3 + rank))
            g_out = torch.randn(shard, 1, OUT, generator=torch.Generator().manual_seed(5 + rank))
            got = {}
            for path in ("reference", "fused"):
                c = core.clone().requires_grad_(True)
                wt = w.clone().requires_grad_(True)
                if path == "fused":
                    o = cp_utils.packed_shard_to_zigzag_a2a_linear(c, wt, cu, cp_group, cp_rank, cp)
                else:
                    o = F.linear(cp_utils.packed_shard_to_zigzag_a2a(c, cu, cp_group, cp_rank, cp), wt)
                o.backward(g_out)
                got[path] = (o.detach(), c.grad, wt.grad)
            checks["core_linear_equal"] = all(
                torch.equal(a, b) for a, b in zip(got["fused"], got["reference"], strict=True)
            )
            results[(rank, name)] = checks  # a Manager dict holds copies: assign once, complete
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (4, 2), (2, 4)])
def test_sp_relayout_fused(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp * len(_cases(cp, tp)), dict(results)
    for key, c in results.items():
        assert c["out_is_packed_shard"] and c["out_equal"] and c["grad_close"] and c["cached"], (key, c)
        assert c["core_linear_equal"], (key, c)
        if tp <= 2:  # a sum of two terms has one rounding whatever the order
            assert c["grad_equal"], (key, c)
        # the fused backward rounds the same terms in another order, no worse than the default
        assert c["err_fused"] <= 4 * c["err_reference"] + 1e-6, (key, c)
        # all-gather blocks: the shard over TP, at most one padding row per source CP rank
        assert c["width"] <= c["part"] + cp, (key, c)
    # cross-CP rows are balanced over the TP ranks of every CP rank (within one row per class)
    for name in _cases(cp, tp):
        for c_rank in range(cp):
            rows = [results[(c_rank * tp + t, name)]["cross_cp_rows"] for t in range(tp)]
            assert max(rows) - min(rows) <= cp, (name, c_rank, rows)
    if cp == 2:  # one long sequence: each rank moves 1 / TP of the half shard its peer needs
        for t in range(tp):
            c = results[(t, "single")]
            assert c["cross_cp_rows"] * 2 * tp == c["part"] * tp, c


def test_switches(monkeypatch):
    for env, on in (
        (cp_utils.SP_RELAYOUT_FUSED_ENV, cp_utils.sp_relayout_fused_on),
        (cp_utils.CORE_RELAYOUT_OVERLAP_ENV, cp_utils.core_relayout_overlap_on),
    ):
        monkeypatch.delenv(env, raising=False)
        assert not on()
        monkeypatch.setenv(env, "1")
        assert on()
