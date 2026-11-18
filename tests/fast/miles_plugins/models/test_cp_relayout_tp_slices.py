"""The TP-slice CP relayout of a TP-replicated tensor is bit-identical to the plain one, forward and backward
(gloo, TP x CP ranks, sequence-parallel input with a column-parallel-like TP-distinct consumer)."""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# by file path: the package __init__ imports Megatron, which CPU CI does not install
_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "cp_utils.py"
_SPEC = importlib.util.spec_from_file_location("_cp_utils_under_test", _PATH)
cp_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_utils)

HIDDEN = 6
CASES = {
    "packed": [0, 40, 56, 120],  # every length a multiple of 2 * cp (cp <= 4): zigzag
    "padded": None,  # odd multiples of cp (built per cp): each rank holds a contiguous 1 / cp, no zigzag
    "single": [0, 64],
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
            g = torch.Generator().manual_seed(11)
            full = torch.randn(total, HIDDEN, generator=g)
            _, packed_position = cp_utils._zigzag_maps(cu, cp, total)
            zigzag = full[packed_position[cp_rank * shard : (cp_rank + 1) * shard]]
            part = shard // tp
            torch.manual_seed(100 + rank)  # TP-distinct consumer weights and upstream gradient
            weight, upstream = torch.randn(HIDDEN), torch.randn(shard, HIDDEN)
            outs, grads = [], []
            for relayout in ("plain", "tp_slices", "tp_slices_bwd_reduce", "a2a_with_bwd_reduce_env"):
                os.environ[cp_utils.RELAYOUT_BWD_REDUCE_ENV] = "0" if relayout in ("plain", "tp_slices") else "1"
                sp = zigzag[tp_rank * part : (tp_rank + 1) * part].clone().requires_grad_(True)
                x = _SPGather.apply(sp, tp_group)
                if relayout == "plain":
                    y = cp_utils.zigzag_to_packed_shard(x, cu, cp_group, cp_rank, cp)
                elif relayout == "a2a_with_bwd_reduce_env":
                    # MILES_K31_KDA_CP_RELAYOUT_A2A takes precedence: its backward ignores the TP-sum switch
                    y = cp_utils.zigzag_to_packed_shard_a2a(x, cu, cp_group, cp_rank, cp, tp_group)
                else:
                    y = cp_utils.zigzag_to_packed_shard_tp_replicated(x, cu, cp_group, cp_rank, cp, tp_group)
                ((y * weight) * upstream).sum().backward()
                outs.append(y.detach())
                grads.append(sp.grad)
            os.environ[cp_utils.RELAYOUT_BWD_REDUCE_ENV] = "0"
            # the plain CP all-gather of the same TP-replicated tensor, rank-concatenated
            plain = zigzag.new_empty((cp * shard, HIDDEN))
            dist.all_gather_into_tensor(plain, zigzag.contiguous(), group=cp_group)
            sliced = cp_utils.tp_slice_all_gather(zigzag, cp_group, tp_group)
            results[(rank, name)] = (
                torch.equal(outs[0], full[cp_rank * shard : (cp_rank + 1) * shard]),
                torch.equal(outs[0], outs[1]),
                torch.equal(grads[0], grads[1]),
                torch.equal(plain, sliced),
                # MILES_K31_KDA_RELAYOUT_BWD_REDUCE: same forward; the gradient sums the same TP partials in
                # another order (fp32 here: equal up to rounding)
                torch.equal(outs[0], outs[2]),
                torch.allclose(grads[0], grads[2], rtol=1e-6, atol=1e-6),
                torch.equal(outs[0], outs[3]) and torch.equal(grads[0], grads[3]),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(2, 2), (4, 2), (2, 4)])
def test_tp_slice_relayout_is_bit_identical(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp * len(CASES), dict(results)
    assert all(all(v) for v in results.values()), dict(results)
