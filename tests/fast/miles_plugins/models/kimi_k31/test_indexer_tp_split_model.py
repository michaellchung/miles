"""KimiK31DSAAttention._select_topk with the TP split on (gloo, TP x CP ranks) selects exactly the indices of the
unsplit path, on every rank, and the head-parallel sparse attention downstream of it is bit-identical."""

import os
import socket
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

model = pytest.importorskip("miles_plugins.models.kimi_k31.model")
tps = pytest.importorskip("miles_plugins.models.kimi_k31.indexer_tp_split")
qs = pytest.importorskip("miles_plugins.models.kimi_k31.dsa_query_split")

HIDDEN, Q_LORA, HEADS, HEAD_DIM, TOPK = 32, 24, 4, 16, 16
ATTN_HEADS, ATTN_DIM = 8, 16
CASES = {
    # two packed sequences; with CP every length is a multiple of 2 * cp, the zigzag layout's requirement
    "packed": [0, 72, 200],
    "single": [0, 136],
    "tiny": [0, 4],  # with CP 2 a rank holds 2 rows: fewer rows than TP ranks
}


def _cpu_kernels():
    """CPU stand-ins for the GPU-only indexer kernel and its backward-only autograd wrapper."""
    import miles.kernels.attention.dsa.indexer as indexer_mod

    def write_logits(q, k, weights, ks, ke, out, clean_logits, config):
        del clean_logits, config
        scores = (torch.einsum("thd,sd->ths", q.float(), k.float()).relu() * weights[:, :, None]).sum(dim=1)
        cols = torch.arange(k.shape[0])
        inside = (cols[None, :] >= ks[:, None].long()) & (cols[None, :] < ke[:, None].long())
        out[:, : k.shape[0]] = torch.where(inside, scores, torch.tensor(float("-inf")))

    indexer_mod._write_logits = write_logits
    indexer_mod._default_config = lambda q: None
    indexer_mod.indexer_topk_scores = lambda q, k, w, logits, idx: indexer_mod.gather_topk_scores(logits, idx)
    model.lighting_indexer = indexer_mod.lighting_indexer


def _indexer():
    torch.cuda.current_device = lambda: "cpu"
    config = types.SimpleNamespace(
        params_dtype=torch.bfloat16,
        kimi_dsa_index_num_heads=HEADS,
        kimi_dsa_index_head_dim=HEAD_DIM,
        layernorm_epsilon=1e-6,
        q_lora_rank=Q_LORA,
        hidden_size=HIDDEN,
    )
    torch.manual_seed(1234)  # the indexer is replicated: identical weights on every rank
    indexer = model.KimiK31Indexer(config)
    with torch.no_grad():
        for p in indexer.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return indexer


class _Attn(model.KimiK31DSAAttention):
    """The production methods over just the attributes ``_select_topk`` reads (no Megatron layer init)."""

    def __init__(self, **attrs):
        torch.nn.Module.__init__(self)
        for name, value in attrs.items():
            setattr(self, name, value)


def _sparse_attention(query, kv, indices, scale):
    """Reference for the absorbed DSA core: query [T, h, D] attends kv [T_kv, D] at its row's indices (-1 = pad)."""
    valid = indices >= 0
    keys = kv[indices.clamp(min=0).long()]  # [T, topk, D]
    logits = torch.einsum("thd,tkd->thk", query.float(), keys.float()) * scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    return torch.einsum("thk,tkd->thd", logits.softmax(dim=-1), keys.float())


def _worker(rank, world, tp, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        _cpu_kernels()
        cp = world // tp
        cp_rank, tp_rank = divmod(rank, tp)
        tp_groups = [dist.new_group(list(range(c * tp, (c + 1) * tp))) for c in range(cp)]
        cp_groups = [dist.new_group(list(range(t, world, tp))) for t in range(tp)]
        attn = _Attn(
            indexer=_indexer(),
            tp_group=tp_groups[cp_rank],
            cp_group=cp_groups[tp_rank],
            index_topk=TOPK,
            topk_fn=model.get_dsa_topk_fn("torch"),
        )
        for name, edges in CASES.items():
            cu = torch.tensor(edges, dtype=torch.int32)
            g = torch.Generator().manual_seed(7)
            hidden_all = torch.randn(int(cu[-1]), HIDDEN, generator=g).bfloat16()
            q_lora_all = torch.randn(int(cu[-1]), Q_LORA, generator=g).bfloat16()
            query_all = torch.randn(int(cu[-1]), ATTN_HEADS, ATTN_DIM, generator=g).bfloat16()
            kv_all = torch.randn(int(cu[-1]), ATTN_DIM, generator=g).bfloat16()
            if cp > 1:
                layout = model.zigzag_gather_layout(cu, cp_rank, cp, int(cu[-1]) // cp)
                rows = layout[1]
            else:
                layout, rows = None, torch.arange(int(cu[-1]))
            hidden, q_lora, query = hidden_all[rows], q_lora_all[rows], query_all[rows]
            for chunk in [0, 5]:
                model.INDEXER_QUERY_CHUNK = chunk
                os.environ["MILES_K31_INDEXER_TP_SPLIT"] = "0"
                tps._SPLIT_ON = None  # decide again (and re-check across the TP ranks) on the next call
                whole = attn._select_topk(q_lora, hidden, cu, layout)
                took_whole = tps._SPLIT_ON is False
                os.environ["MILES_K31_INDEXER_TP_SPLIT"] = "1"
                tps._SPLIT_ON = None
                split = attn._select_topk(q_lora, hidden, cu, layout)
                took_split = tps._SPLIT_ON is True
                same_topk = took_whole and took_split and whole.dtype == split.dtype == torch.int32
                same_topk = same_topk and torch.equal(whole, split)
                # every TP rank must hand the head-parallel attention the same indices
                gathered = [torch.empty_like(split) for _ in range(tp)]
                dist.all_gather(gathered, split, group=attn.tp_group)
                same_on_ranks = all(torch.equal(x, split) for x in gathered)
                # downstream: each TP rank runs its heads on its own copy of the indices, then the heads are gathered
                heads = slice(tp_rank * ATTN_HEADS // tp, (tp_rank + 1) * ATTN_HEADS // tp)
                mine = _sparse_attention(query[:, heads], kv_all, split, ATTN_DIM**-0.5).contiguous()
                parts = [torch.empty_like(mine) for _ in range(tp)]
                dist.all_gather(parts, mine, group=attn.tp_group)
                # this rank's heads are bit-identical to the unsplit path's; all heads match a one-shot reference
                # (whose wider head batch may reduce in another order, hence allclose there)
                mine_whole = _sparse_attention(query[:, heads], kv_all, whole, ATTN_DIM**-0.5).contiguous()
                reference = _sparse_attention(query, kv_all, whole, ATTN_DIM**-0.5)
                same_attention = torch.equal(mine, mine_whole) and torch.allclose(
                    torch.cat(parts, dim=1), reference, rtol=1e-5, atol=1e-6
                )
                # MILES_K31_DSA_QUERY_SPLIT: this rank's padded rows only (no all-gather), split on and off
                own_split = attn._select_topk(q_lora, hidden, cu, layout, own_rows=True)
                os.environ["MILES_K31_INDEXER_TP_SPLIT"] = "0"
                tps._SPLIT_ON = None
                own_whole = attn._select_topk(q_lora, hidden, cu, layout, own_rows=True)
                own = tps.own_tp_part(whole, attn.tp_group)
                same_own = torch.equal(own_split, own) and torch.equal(own_whole, own)
                # and the query-split attention over those rows gives this rank's heads of the head-split one
                qsplit = qs.query_split_attention(
                    lambda q, k, i, sm_scale, d_v: _sparse_attention(q[0], k[0, :, 0], i[0, :, 0], sm_scale)[None],
                    query[:, heads].contiguous(),
                    kv_all,
                    own_split,
                    query.shape[0],
                    attn.tp_group,
                    sm_scale=ATTN_DIM**-0.5,
                    d_v=ATTN_DIM,
                )
                same_own = same_own and torch.allclose(qsplit, mine_whole, rtol=1e-5, atol=1e-6)
                results[(rank, name, chunk)] = (
                    same_topk and same_own,
                    same_on_ranks,
                    same_attention,
                    int((split >= 0).sum()),
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(2, 1), (4, 1), (2, 2), (4, 2)])
def test_select_topk_split_matches_unsplit(tp, cp):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.start_processes(_worker, args=(tp * cp, tp, port, results), nprocs=tp * cp, start_method="spawn")
    assert len(results) == tp * cp * len(CASES) * 2, dict(results)
    for key, (same_topk, same_on_ranks, same_attention, selected) in results.items():
        assert same_topk and same_on_ranks and same_attention, (key, results[key])
        assert selected > 0, key
