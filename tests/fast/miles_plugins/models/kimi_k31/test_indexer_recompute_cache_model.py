"""KimiK31DSAAttention._select_topk with MILES_K31_INDEXER_RECOMPUTE_CACHE=1 under full recompute (gloo, TP x CP):
the losses and gradients of every microbatch are bitwise those without the cache, the recompute does not run the
indexer, and nothing stays in the cache, with forward-only passes and a 1F1B order mixed in."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

model = pytest.importorskip("miles_plugins.models.kimi_k31.model")
tps = pytest.importorskip("miles_plugins.models.kimi_k31.indexer_tp_split")
irc = pytest.importorskip("miles_plugins.models.kimi_k31.indexer_recompute_cache")
split_model = pytest.importorskip("tests.fast.miles_plugins.models.kimi_k31.test_indexer_tp_split_model")

TOPK, ATTN_HEADS, ATTN_DIM = split_model.TOPK, split_model.ATTN_HEADS, split_model.ATTN_DIM
HIDDEN, Q_LORA = split_model.HIDDEN, split_model.Q_LORA
EDGES = [[0, 72, 200], [0, 136], [0, 4], [0, 40, 64], [0, 200], [0, 8, 72], [0, 64]]  # one microbatch each

# one pipeline stage under 1F1B: F = training forward, B = its backward, I = a forward-only pass (log-probs, no grad),
# A = a training forward whose backward never runs (an aborted step); its entry is dropped by the next recompute
SCHEDULE = ["A5", "F0", "I1", "F1", "B0", "A6", "F2", "B1", "I2", "F3", "B2", "B3", "I0", "I4"]

_CHECKPOINTING = [False]


class _Checkpoint(torch.autograd.Function):
    """Megatron's CheckpointFunction (tensor_parallel/random.py): flag on, no-grad forward, save the inputs; flag on
    again while the backward reruns the function with grad."""

    @staticmethod
    def forward(ctx, fn, *args):
        _CHECKPOINTING[0] = True
        ctx.fn = fn
        with torch.no_grad():
            out = fn(*args)
        ctx.save_for_backward(*args)
        _CHECKPOINTING[0] = False
        return out

    @staticmethod
    def backward(ctx, grad):
        _CHECKPOINTING[0] = True
        inputs = [x.detach().requires_grad_(x.requires_grad) for x in ctx.saved_tensors]
        with torch.enable_grad():
            out = ctx.fn(*inputs)
        torch.autograd.backward(out, grad)
        _CHECKPOINTING[0] = False
        return (None, *(x.grad for x in inputs))


def _microbatch(edges, cp, cp_rank, seed):
    cu = torch.tensor(edges, dtype=torch.int32)
    g = torch.Generator().manual_seed(seed)
    total = int(cu[-1])
    tensors = {
        "hidden": torch.randn(total, HIDDEN, generator=g).bfloat16(),
        "q_lora": torch.randn(total, Q_LORA, generator=g).bfloat16(),
        "query": torch.randn(total, ATTN_HEADS, ATTN_DIM, generator=g).bfloat16(),
        "kv": torch.randn(total, ATTN_DIM, generator=g).bfloat16(),
        "weight": torch.randn(total, ATTN_HEADS, ATTN_DIM, generator=g),
    }
    if cp > 1:
        layout = model.zigzag_gather_layout(cu, cp_rank, cp, total // cp)
        rows = layout[1]
    else:
        layout, rows = None, torch.arange(total)
    local = {name: t[rows] for name, t in tensors.items() if name != "kv"}
    return cu, layout, local, tensors["kv"]


def _run_schedule(attn, cp, cp_rank, tp_rank, tp):
    """Every SCHEDULE event on this rank: per microbatch (loss, query grad, kv grad, top-k) and the indexer calls."""
    calls = [0]
    inner = attn.indexer.forward

    def counted(*args):
        calls[0] += 1
        return inner(*args)

    attn.indexer.forward = counted
    heads = slice(tp_rank * ATTN_HEADS // tp, (tp_rank + 1) * ATTN_HEADS // tp)
    batches = [_microbatch(edges, cp, cp_rank, seed=11 + i) for i, edges in enumerate(EDGES)]
    pending, out, calls_at = {}, {}, {}
    for event in SCHEDULE:
        kind, i = event[0], int(event[1:])
        cu, layout, local, kv_all = batches[i]
        query = local["query"][:, heads].clone().requires_grad_(kind in "FA")
        kv = kv_all.clone().requires_grad_(kind in "FA")
        seen = []

        def layer(q, k, cu=cu, layout=layout, local=local, seen=seen):
            # cu and layout reach the layer through the closure, as packed_seq_params does in Megatron's
            topk = attn._select_topk(local["q_lora"], local["hidden"], cu, layout)
            seen.append(topk)
            return split_model._sparse_attention(q, k, topk, ATTN_DIM**-0.5)

        before = calls[0]
        if kind == "I":
            with torch.no_grad():
                layer(query, kv)
            out[event] = seen[0]
        elif kind in "FA":
            attention = _Checkpoint.apply(layer, query, kv)
            loss = (attention * local["weight"][:, heads]).sum()
            pending[i] = (loss, query, kv, seen)
        else:
            loss, query, kv, seen = pending.pop(i)
            loss.backward()
            assert len(seen) == 2 and torch.equal(seen[0], seen[1])
            out[event] = (loss.detach(), query.grad, kv.grad, seen[1])
        calls_at[event] = calls[0] - before
    return out, calls_at, len(attn.topk_cache)


def _worker(rank, world, tp, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        split_model._cpu_kernels()
        model.is_checkpointing = lambda: _CHECKPOINTING[0]
        cp = world // tp
        cp_rank, tp_rank = divmod(rank, tp)
        tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
        cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
        for split in ["0", "1"]:
            runs = {}
            for cache in ["0", "1"]:
                os.environ["MILES_K31_INDEXER_TP_SPLIT"], os.environ[irc.CACHE_ENV] = split, cache
                tps._SPLIT_ON, irc._CACHE_ON = None, None
                attn = split_model._Attn(
                    indexer=split_model._indexer(),
                    tp_group=tp_groups[cp_rank],
                    cp_group=cp_groups[tp_rank],
                    index_topk=TOPK,
                    topk_fn=model.get_dsa_topk_fn("torch"),
                    topk_cache=irc.TopkRecomputeCache(max_pending=3),
                )
                runs[cache] = _run_schedule(attn, cp, cp_rank, tp_rank, tp)
            (plain, plain_calls, plain_left), (cached, cached_calls, cached_left) = runs["0"], runs["1"]
            same = plain.keys() == cached.keys() and all(
                (
                    all(torch.equal(a, b) for a, b in zip(plain[e], cached[e], strict=True))
                    if isinstance(plain[e], tuple)
                    else torch.equal(plain[e], cached[e])
                )
                for e in plain
            )
            results[(rank, split)] = (same, plain_calls, cached_calls, plain_left, cached_left)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_recompute_reuses_the_forward_top_k(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp * 2, dict(results)
    for key, (same, plain_calls, cached_calls, plain_left, cached_left) in results.items():
        assert same, key
        # off: every forward and every recompute scores; on: the recompute only takes the forward's rows
        assert all(n == 1 for e, n in plain_calls.items() if e[0] in "FAI"), (key, plain_calls)
        assert all(n == 1 for e, n in plain_calls.items() if e[0] == "B"), (key, plain_calls)
        assert all(n == 1 for e, n in cached_calls.items() if e[0] in "FAI"), (key, cached_calls)
        assert all(n == 0 for e, n in cached_calls.items() if e[0] == "B"), (key, cached_calls)
        # the aborted forward's entry is dropped by the next recompute of the layer; nothing is left over
        assert plain_left == 0 and cached_left == 0, key
