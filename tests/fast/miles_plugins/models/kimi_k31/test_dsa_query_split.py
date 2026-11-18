"""Running the DSA sparse attention core on 1/TP of the query rows with all heads gives the head-split result."""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Load the module files directly: the package __init__ imports Megatron, which CPU CI does not install.
_DIR = Path(__file__).resolve().parents[5] / "miles_plugins" / "models" / "kimi_k31"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qs = _load("dsa_query_split")
tps = _load("indexer_tp_split")


def _attn(q, kv, indices, sm_scale, d_v):
    """CPU stand-in for sparse_attention: q [1, n, H, d], kv [1, S, 1, d], indices [1, n, 1, topk] (-1 = none).
    Row-wise only (gather + elementwise + last-dim reductions), so a row's bits do not depend on the batch."""
    q, kv, idx = q[0], kv[0, :, 0], indices[0, :, 0].long()
    valid = idx >= 0
    keys = kv[idx.clamp_min(0)]  # [n, topk, d]
    scores = (q[:, :, None, :] * keys[:, None, :, :]).sum(-1) * sm_scale  # [n, H, topk]
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = (p[..., None] * keys[:, None, :, :d_v]).sum(-2)  # [n, H, d_v]
    return out.unsqueeze(0)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(rows, heads, d, kv_rows, topk, seed=0):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(rows, heads, d, generator=g, dtype=torch.float64)
    kv = torch.randn(kv_rows, d, generator=g, dtype=torch.float64)
    idx = torch.randint(0, kv_rows, (rows, topk), generator=g, dtype=torch.int32)
    idx[torch.rand(rows, topk, generator=g) < 0.3] = -1
    idx[:, 0] = torch.arange(rows, dtype=torch.int32) % kv_rows  # every real row has a key
    grad = torch.randn(rows, heads, d - 2, generator=g, dtype=torch.float64)
    return query, kv, idx, grad


def _worker(rank, tp, port, rows, mode, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    try:
        heads_total, d, kv_rows, topk = 4 * tp, 6, 11, 5
        d_v = d - 2
        query_all, kv_all, idx, grad_all = _inputs(rows, heads_total, d, kv_rows, topk)
        h = heads_total // tp
        sl = slice(rank * h, (rank + 1) * h)
        kw = dict(sm_scale=0.37, d_v=d_v)

        # head split (the default path): this rank's heads, every row, the full indices
        q_ref = query_all[:, sl].clone().requires_grad_()
        kv_ref = kv_all.clone().requires_grad_()
        out_ref = _attn(q_ref.unsqueeze(0), kv_ref.unsqueeze(0).unsqueeze(2), idx.unsqueeze(1).unsqueeze(0), **kw)[0]
        out_ref.backward(grad_all[:, sl])
        dkv_ref = kv_ref.grad.clone()
        dist.all_reduce(dkv_ref)

        # query split: this rank's padded top-k part only
        part = tps.pad_tp_part(idx[tps.tp_rows(rows, rank, tp)], rows, tp)
        q = query_all[:, sl].clone().requires_grad_()
        kv = kv_all.clone().requires_grad_()
        out = qs.query_split_attention(_attn, q, kv, part, rows, None, mode=mode, head_group=3, **kw)
        out.backward(grad_all[:, sl])
        dkv = kv.grad.clone()
        dist.all_reduce(dkv)

        out_q.put(
            (
                rank,
                torch.equal(out, out_ref),
                torch.equal(q.grad, q_ref.grad),
                float((dkv - dkv_ref).abs().max()),
                float(dkv_ref.abs().max()),
                tuple(out.shape),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp,rows,mode", [(2, 10, "1"), (2, 9, "1"), (4, 13, "1"), (4, 16, "1"), (4, 7, "64"), (2, 5, "64"), (4, 3, "1")]
)
def test_query_split_matches_head_split(tp, rows, mode):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, tp, port, rows, mode, out_q)) for r in range(tp)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank, out_eq, dq_eq, dkv_err, dkv_max, shape in results:
        assert shape == (rows, 4, 4), (rank, shape)
        assert out_eq, f"rank {rank}: forward differs"
        assert dq_eq, f"rank {rank}: dq differs"
        assert dkv_err <= 1e-12 * max(dkv_max, 1.0), (rank, dkv_err)


def _relayout_worker(rank, tp, port, rows, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    try:
        h, d = 3, 5
        full = torch.arange(rows * h * tp * d, dtype=torch.float32).view(rows, h * tp, d)
        mine = full[:, rank * h : (rank + 1) * h].contiguous()
        y = qs.heads_to_rows(mine, None)
        per = -(-rows // tp)
        want = torch.zeros(per, h * tp, d)
        own = full[rank::tp]
        want[: own.shape[0]] = own
        back = qs.rows_to_heads(y, rows, None)
        out_q.put((rank, torch.equal(y, want), torch.equal(back, mine)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,rows", [(2, 6), (3, 7), (4, 2)])
def test_relayout_round_trip(tp, rows):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_relayout_worker, args=(r, tp, port, rows, out_q)) for r in range(tp)]
    for p in procs:
        p.start()
    results = [out_q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank, rows_ok, back_ok in results:
        assert rows_ok, f"rank {rank}: heads_to_rows put the wrong rows/heads"
        assert back_ok, f"rank {rank}: rows_to_heads is not its inverse"


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv(qs.QUERY_SPLIT_ENV, raising=False)
    assert qs.query_split_mode() == ""
    monkeypatch.setenv(qs.QUERY_SPLIT_ENV, "0")
    assert qs.query_split_mode() == ""
    for v in ("1", "64"):
        monkeypatch.setenv(qs.QUERY_SPLIT_ENV, v)
        assert qs.query_split_mode() == v
    monkeypatch.setenv(qs.QUERY_SPLIT_ENV, "2")
    with pytest.raises(ValueError):
        qs.query_split_mode()
