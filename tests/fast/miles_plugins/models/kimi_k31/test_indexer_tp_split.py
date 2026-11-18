"""Splitting the frozen K3.1 indexer's query rows over TP selects exactly the keys of one full call."""

import importlib.util
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Load the module file directly: the package __init__ imports Megatron, which CPU CI does not install.
_PATH = Path(__file__).resolve().parents[5] / "miles_plugins" / "models" / "kimi_k31" / "indexer_tp_split.py"
_SPEC = importlib.util.spec_from_file_location("_indexer_tp_split_under_test", _PATH)
split = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(split)


def _causal_ranges(cu, positions=None):
    # same as miles.kernels.attention.dsa.causal_ranges
    if positions is None:
        positions = torch.arange(int(cu[-1]))
    starts = cu[torch.searchsorted(cu, positions, right=True) - 1]
    return starts.to(torch.int32), (positions + 1).to(torch.int32)


def _indexer(q, k, w, ks, ke, topk):
    """CPU stand-in for lighting_indexer: relu(q . k) weighted over heads, -inf outside [ks, ke), torch top-k."""
    scores = (torch.einsum("thd,sd->ths", q.float(), k.float()).relu() * w[:, :, None]).sum(dim=1)
    cols = torch.arange(k.shape[0])
    inside = (cols[None] >= ks[:, None].long()) & (cols[None] < ke[:, None].long())
    scores = torch.where(inside, scores, torch.tensor(float("-inf")))
    n = min(topk, k.shape[0])
    val, idx = torch.topk(scores, n, dim=-1)
    idx = idx.to(torch.int32).masked_fill(val == float("-inf"), -1)
    return torch.cat([idx, idx.new_full((idx.shape[0], topk - n), -1)], dim=-1)


def _chunked(q, k, w, ks, ke, topk, chunk):
    # mirrors model.chunked_indexer_topk
    if chunk <= 0 or q.shape[0] <= chunk:
        return _indexer(q, k, w, ks, ke, topk)
    return torch.cat(
        [
            _indexer(q[r : r + chunk], k, w[r : r + chunk], ks[r : r + chunk], ke[r : r + chunk], topk)
            for r in range(0, q.shape[0], chunk)
        ]
    )


def _inputs(cu, positions=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = int(cu[-1])
    rows = total if positions is None else positions.numel()
    q = torch.randn(rows, 4, 16, generator=g).bfloat16()
    k = torch.randn(total, 16, generator=g).bfloat16()
    w = torch.rand(rows, 4, generator=g)
    ks, ke = _causal_ranges(cu, positions)
    return q, k, w, ks, ke


def _zigzag_positions(cu, cp_rank, cp_size):
    # each sequence in 2 * cp chunks; rank r holds chunks r and 2 cp - 1 - r
    out = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        c = (b - a) // (2 * cp_size)
        out += [
            torch.arange(a + cp_rank * c, a + (cp_rank + 1) * c),
            torch.arange(b - (cp_rank + 1) * c, b - cp_rank * c),
        ]
    return torch.cat(out)


CASES = {
    "packed": (torch.tensor([0, 37, 50, 211, 253], dtype=torch.int32), None),  # 253 rows: odd, no TP divides it
    "zigzag": (torch.tensor([0, 64, 192], dtype=torch.int32), "cp2r1"),
    "tiny": (torch.tensor([0, 3], dtype=torch.int32), None),  # fewer rows than TP ranks
}


def _case(name):
    cu, zz = CASES[name]
    positions = _zigzag_positions(cu, 1, 2) if zz else None
    return _inputs(cu, positions)


def _split_by_hand(q, k, w, ks, ke, topk, tp, chunk):
    rows = q.shape[0]
    parts = []
    for r in range(tp):
        m = split.tp_rows(rows, r, tp)
        part = (
            _chunked(q[m], k, w[m], ks[m], ke[m], topk, chunk)
            if q[m].shape[0]
            else torch.empty((0, topk), dtype=torch.int32)
        )
        parts.append(split.pad_tp_part(part, rows, tp))
    return split.merge_tp_parts(torch.cat(parts), rows, tp)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("tp", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("chunk", [0, 1, 7, 33])
@pytest.mark.parametrize("topk", [8, 64])
def test_split_and_merge_match_one_call(name, tp, chunk, topk):
    q, k, w, ks, ke = _case(name)
    whole = _indexer(q, k, w, ks, ke, topk)
    assert torch.equal(whole, _split_by_hand(q, k, w, ks, ke, topk, tp, chunk))


def test_rows_cover_each_row_once():
    for rows in [0, 1, 7, 8, 9, 253]:
        for tp in [1, 2, 4, 8]:
            seen = torch.cat([torch.arange(rows)[split.tp_rows(rows, r, tp)] for r in range(tp)])
            assert torch.equal(seen.sort().values, torch.arange(rows))


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MILES_K31_INDEXER_TP_SPLIT", raising=False)
    assert not split.indexer_tp_split_enabled()
    monkeypatch.setenv("MILES_K31_INDEXER_TP_SPLIT", "1")
    assert split.indexer_tp_split_enabled()


def _worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        for name in CASES:
            q, k, w, ks, ke = _case(name)
            for chunk in [0, 5]:
                got = split.tp_split_topk(
                    lambda q_, w_, s_, e_, k=k, chunk=chunk: _chunked(q_, k, w_, s_, e_, 16, chunk),
                    q,
                    w,
                    ks,
                    ke,
                    16,
                    dist.group.WORLD,
                )
                results[(rank, name, chunk)] = bool(torch.equal(got, _indexer(q, k, w, ks, ke, 16)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [2, 3, 4, 8])
def test_gloo_all_gather_matches_one_call(tp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp, port, results), nprocs=tp, start_method="spawn")
    assert len(results) == tp * len(CASES) * 2 and all(results.values()), dict(results)


def _switch_worker(rank, world, port, env, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), MILES_K31_INDEXER_TP_SPLIT=env[rank])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        split._SPLIT_ON = None
        try:
            results[rank] = split.indexer_tp_split_on(dist.group.WORLD, torch.device("cpu"))
        except RuntimeError as e:
            results[rank] = "differs" if "differs across its TP group" in str(e) else repr(e)
        results[(rank, "cached")] = split._SPLIT_ON
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "env,expected",
    [("11", [True, True]), ("00", [False, False]), ("10", ["differs", "differs"]), ("1101", ["differs"] * 4)],
)
def test_switch_is_checked_across_the_tp_group(env, expected):
    # a rank whose switch differs would skip the all-gather the others wait on: every rank raises instead
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_switch_worker, args=(len(env), port, env, results), nprocs=len(env), start_method="spawn")
    assert [results[r] for r in range(len(env))] == expected, dict(results)
