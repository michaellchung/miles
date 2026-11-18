"""``MILES_K31_KDA_CP_RELAYOUT_A2A``: the zigzag <-> contiguous CP relayouts by an all-to-all of only the needed
rows equal the all-gather ones bit for bit, forward and backward, for the plain and the TP-replicated (TP-slice)
input (gloo, TP x CP ranks). Plans are cached per ``cu_seqlens`` and move half of the all-gather's rows at CP2."""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="stage-b-cpu", labels=["miles-plugin"])

# by file path: the package __init__ imports Megatron, which CPU CI does not install
_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "cp_utils.py"
_SPEC = importlib.util.spec_from_file_location("_cp_utils_a2a_under_test", _PATH)
cp_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_utils)

HIDDEN = 6
CASES = {
    "packed": [0, 40, 56, 120],  # every length a multiple of 2 * cp (cp <= 4): zigzag
    "padded": None,  # odd multiples of cp (built per cp): each rank holds a contiguous 1 / cp, no zigzag
    "single": [0, 64],
    "uneven": [0, 16, 112, 120],  # rank pairs exchange different row counts
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
        for name, edges in CASES.items():
            cu = torch.tensor([0, 13 * cp, 16 * cp] if edges is None else edges, dtype=torch.int32)
            total = int(cu[-1])
            shard = total // cp
            full = torch.randn(total, 1, HIDDEN, generator=torch.Generator().manual_seed(11))
            _, packed_position = cp_utils._zigzag_maps(cu, cp, total)
            zigzag = full[packed_position[cp_rank * shard : (cp_rank + 1) * shard]]
            part = shard // tp
            torch.manual_seed(100 + rank)  # TP-distinct consumer weights and upstream gradients
            weight, up1, up2 = torch.randn(HIDDEN), torch.randn(shard, 1, HIDDEN), torch.randn(shard, 1, HIDDEN)
            outs, grads = {}, {}
            for path in ("gather", "a2a", "gather_tp", "a2a_tp"):
                sp = zigzag[tp_rank * part : (tp_rank + 1) * part].clone().requires_grad_(True)
                x = _SPGather.apply(sp, tp_group)
                args = (cu, cp_group, cp_rank, cp)
                if path == "gather":
                    y = cp_utils.zigzag_to_packed_shard(x, *args)
                elif path == "gather_tp":
                    y = cp_utils.zigzag_to_packed_shard_tp_replicated(x, *args, tp_group)
                else:
                    y = cp_utils.zigzag_to_packed_shard_a2a(x, *args, tp_group if path == "a2a_tp" else None)
                core = y * weight  # TP-distinct, like the column-parallel projections
                back = (
                    cp_utils.packed_shard_to_zigzag_a2a(core, *args)
                    if path.startswith("a2a")
                    else cp_utils.packed_shard_to_zigzag(core, *args)
                )
                ((y * up1).sum() + (back * up2).sum()).backward()
                outs[path] = (y.detach(), back.detach())
                grads[path] = sp.grad
            ref_y, ref_back = outs["gather"]
            to_packed, to_zigzag = cp_utils._relayout_plans(cu, cp_rank, cp, total)
            received = sum(to_packed.recv_splits)
            results[(rank, name)] = {
                "y_is_packed_shard": torch.equal(ref_y, full[cp_rank * shard : (cp_rank + 1) * shard]),
                **{f"{p}_out": torch.equal(outs[p][0], ref_y) and torch.equal(outs[p][1], ref_back) for p in outs},
                **{f"{p}_grad": torch.equal(grads[p], grads["gather"]) for p in grads},
                "cached": cp_utils._relayout_plans(cu, cp_rank, cp, total)[0] is to_packed,
                "permutation": sorted(to_packed.own_src.tolist() + to_packed.send_idx.tolist()) == list(range(shard)),
                "fewer_rows": received <= (cp - 1) * shard,
                "received": received,
                "shard": shard,
            }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (2, 4)])
def test_cp_relayout_a2a_is_bit_identical(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp * len(CASES), dict(results)
    for key, checks in results.items():
        assert all(v for k, v in checks.items() if k not in ("received", "shard")), (key, checks)
    if cp == 2:  # one long zigzag sequence: a rank needs exactly half of its peer's shard
        single = [c for (rank, name), c in results.items() if name == "single"]
        assert all(2 * c["received"] == c["shard"] for c in single), single


def test_switch(monkeypatch):
    monkeypatch.delenv(cp_utils.CP_RELAYOUT_A2A_ENV, raising=False)
    assert not cp_utils.cp_relayout_a2a_on()
    monkeypatch.setenv(cp_utils.CP_RELAYOUT_A2A_ENV, "1")
    assert cp_utils.cp_relayout_a2a_on()
