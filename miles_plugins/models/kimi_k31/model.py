"""Kimi-K3.1: Kimi K3 with DSA on every MLA layer, a lightning indexer that keeps the top-k keys per query.

The indexer is NoPE, keys use a LayerNorm, and there is no Hadamard rotation. The core attention
runs absorbed over the 512-wide KV latent plus K3's 64-wide unrotated key extra, through the
shared DSA kernels in `miles.kernels.attention.dsa`.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import is_checkpointing

from miles.kernels.attention.dsa import causal_ranges, get_dsa_topk_fn, lighting_indexer, sparse_attention
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.cp_utils import tp_slice_all_gather, zigzag_gather_layout
from miles_plugins.models.kimi_k3.kda_layout import kimi_k3_kda_layers
from miles_plugins.models.kimi_k3.layers import KimiK3Attention, _linear
from miles_plugins.models.kimi_k3.model import get_kimi_k3_spec
from miles_plugins.models.kimi_k31.dsa_layout import checked_dsa_layers
from miles_plugins.models.kimi_k31.dsa_query_split import query_split_attention, query_split_on
from miles_plugins.models.kimi_k31.indexer_recompute_cache import (
    Phase,
    TopkRecomputeCache,
    indexer_recompute_cache_on,
    recompute_phase,
)
from miles_plugins.models.kimi_k31.indexer_tp_split import (
    gather_tp_parts,
    indexer_tp_split_on,
    own_tp_part,
    tp_split_part,
    tp_split_topk,
)

# Query rows the frozen DSA indexer scores per block (MILES_K31_INDEXER_QUERY_CHUNK overrides).
INDEXER_QUERY_CHUNK = int(os.environ.get("MILES_K31_INDEXER_QUERY_CHUNK", "8192"))


def chunked_indexer_topk(
    index_q, index_k, weights, starts, ends, topk: int, topk_fn, chunk: int | None = None
) -> torch.Tensor:
    """Top-k key indices of the frozen DSA indexer, scored one block of ``chunk`` query rows at a time.

    The indexer is frozen (its inputs are detached) and only the selected keys are used, so the
    [rows, T_kv] fp32 score matrix never has to exist at once: the peak is chunk x T_kv x 4 B instead
    of rows x T_kv x 4 B (128 GiB for 131k query rows over 262k keys). The indices are absolute key
    positions and every row's range is its own (``starts`` / ``ends``), so the blocks concatenate to
    exactly the single-call result. ``chunk`` defaults to ``INDEXER_QUERY_CHUNK``; the same
    query-block pattern as DeepSeek-V4.1's indexer (``deepseek_v4_1/ops/indexer.py``)."""
    chunk = INDEXER_QUERY_CHUNK if chunk is None else int(chunk)
    rows = index_q.shape[0]
    if chunk <= 0 or rows <= chunk:
        return lighting_indexer(index_q, index_k, weights, starts, ends, topk, topk_fn=topk_fn)[1]
    return torch.cat(
        [
            lighting_indexer(
                index_q[r : r + chunk],
                index_k,
                weights[r : r + chunk],
                starts[r : r + chunk],
                ends[r : r + chunk],
                topk,
                topk_fn=topk_fn,
            )[1]
            for r in range(0, rows, chunk)
        ]
    )


class _TPSliceGatherFromCP(torch.autograd.Function):
    """``gather_from_sequence_parallel_region(x, group=cp_group)`` of a TP-replicated ``x`` (the MLA latent kv,
    the indexer keys): the forward moves 1 / TP of it over CP (``tp_slice_all_gather``), same bytes out; the
    gradient is TP-distinct (head-parallel attention), so the backward is the same CP reduce-scatter."""

    @staticmethod
    def forward(ctx, x, cp_group, tp_group):
        ctx.cp_group = cp_group
        return tp_slice_all_gather(x, cp_group, tp_group)

    @staticmethod
    def backward(ctx, grad_output):
        return mappings._reduce_scatter_along_first_dim(grad_output, ctx.cp_group), None, None


_MLA_CP_GATHER_V2_LOGGED = False


def gather_tp_replicated_over_cp(x: torch.Tensor, cp_group, tp_group) -> torch.Tensor:
    """CP all-gather of a TP-replicated tensor. MILES_K31_MLA_CP_GATHER_V2=1 (default off) sends it by TP slices."""
    global _MLA_CP_GATHER_V2_LOGGED
    if (
        os.environ.get("MILES_K31_MLA_CP_GATHER_V2", "0") == "1"
        and tp_group.size() > 1
        and x.shape[0] % tp_group.size() == 0
    ):
        if not _MLA_CP_GATHER_V2_LOGGED:
            _MLA_CP_GATHER_V2_LOGGED = True
            if tp_group.rank() == 0:
                print(f"[k31] MLA/DSA CP gather v2 (TP slices) on tp={tp_group.size()}", flush=True)
        return _TPSliceGatherFromCP.apply(x, cp_group, tp_group)
    return gather_from_sequence_parallel_region(x, group=cp_group)


class KimiK31Indexer(nn.Module):
    """Frozen indexer: RL trains no indexer objective, so its weights never change."""

    def __init__(self, config) -> None:
        super().__init__()
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_dsa_index_num_heads
        self.head_dim = config.kimi_dsa_index_head_dim
        self.eps = config.layernorm_epsilon
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=self.eps, device=device, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, device=device, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, q_lora: torch.Tensor, hidden: torch.Tensor):
        """q_lora [T, q_lora_rank], hidden [T, H] -> q [T, heads, head_dim], k [T, head_dim], weights [T, heads]."""
        query = self.wq_b(q_lora).view(-1, self.num_heads, self.head_dim)
        key = F.layer_norm(
            self.wk(hidden).float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.eps,
        ).to(hidden.dtype)
        # bf16 GEMM, then fp32 scaling: SGLang's bf16 head gate for K3.1
        weights = self.weights_proj(hidden).float() * (self.num_heads**-0.5 * self.head_dim**-0.5)
        return query, key, weights


class KimiK31DSAAttention(KimiK3Attention):
    """K3's MLA layer with DSA: the indexer picks each query's top-k keys, the rest run sparse."""

    def __init__(self, config, layer_number: int, *args, **kwargs) -> None:
        super().__init__(config, layer_number, *args, **kwargs)
        self.index_topk = config.kimi_dsa_index_topk
        self.topk_fn = get_dsa_topk_fn(config.miles_dsa_topk_backend)
        self.indexer = KimiK31Indexer(config)
        # SGLang returns one top-k stream per DSA layer, in layer order
        indexer_replay_manager.register_to_module(
            self, "indexer_replay", stream_idx=config.kimi_dsa_layers.index(layer_number), supports_zigzag_cp=True
        )
        # MILES_K31_INDEXER_RECOMPUTE_CACHE: at most one pending top-k per microbatch in flight on this stage (1F1B)
        self.topk_cache = TopkRecomputeCache(max_pending=config.pipeline_model_parallel_size)

    def _indexer_inputs(self, q_lora, hidden, cu_seqlens, cp_layout):
        index_q, index_k, weights = self.indexer(q_lora.detach(), hidden.detach())
        positions = None
        if cp_layout is not None:
            gather_order, positions = cp_layout
            index_k = gather_tp_replicated_over_cp(index_k, self.cp_group, self.tp_group).index_select(0, gather_order)
        starts, ends = causal_ranges(cu_seqlens, positions)
        return index_q, index_k, weights, starts.to(torch.int32), ends.to(torch.int32)

    def _select_topk(self, q_lora, hidden, cu_seqlens, cp_layout, own_rows: bool = False) -> torch.Tensor:
        """[T, topk] indices, or with ``own_rows`` only this TP rank's padded rows of them (``own_tp_part``), which
        the TP-split indexer scores without the all-gather."""
        topk_fn = indexer_replay_manager.get_topk_fn(self.topk_fn, return_probs=False)
        if indexer_replay_manager.enabled:
            # replay hands back one top-k stream per layer call: keep a single call
            index_q, index_k, weights, starts, ends = self._indexer_inputs(q_lora, hidden, cu_seqlens, cp_layout)
            indices = lighting_indexer(index_q, index_k, weights, starts, ends, self.index_topk, topk_fn=topk_fn)[1]
            return own_tp_part(indices, self.tp_group) if own_rows else indices
        phase = Phase.PLAIN
        if indexer_recompute_cache_on(self.tp_group, self.cp_group, q_lora.device):
            phase = recompute_phase(is_checkpointing(), torch.is_grad_enabled())
        if phase is Phase.PLAIN:
            if own_rows:
                return self._score_topk_part(q_lora, hidden, cu_seqlens, cp_layout, topk_fn)[0]
            return self._score_topk(q_lora, hidden, cu_seqlens, cp_layout, topk_fn)
        # a fully recomputed layer: the forward keeps this rank's rows of its top-k, the recompute reuses them
        part = self.topk_cache.take(cu_seqlens) if phase is Phase.REUSE else None
        if part is None:
            part, indices = self._score_topk_part(q_lora, hidden, cu_seqlens, cp_layout, topk_fn)
            if phase is Phase.STORE:
                self.topk_cache.put(cu_seqlens, part)
            if indices is not None and not own_rows:
                return indices
        if own_rows:
            return part
        return gather_tp_parts(part, q_lora.shape[0], self.tp_group)

    def _score_topk(self, q_lora, hidden, cu_seqlens, cp_layout, topk_fn) -> torch.Tensor:
        index_q, index_k, weights, starts, ends = self._indexer_inputs(q_lora, hidden, cu_seqlens, cp_layout)
        if indexer_tp_split_on(self.tp_group, index_q.device):
            # every TP rank holds the same indexer inputs: score 1/TP of the rows, all-gather the indices
            return tp_split_topk(
                lambda q, w, ks, ke: chunked_indexer_topk(q, index_k, w, ks, ke, self.index_topk, topk_fn),
                index_q,
                weights,
                starts,
                ends,
                self.index_topk,
                self.tp_group,
            )
        return chunked_indexer_topk(index_q, index_k, weights, starts, ends, self.index_topk, topk_fn)

    def _score_topk_part(self, q_lora, hidden, cu_seqlens, cp_layout, topk_fn):
        """(this TP rank's padded rows of the top-k, the whole [T, topk] top-k if it was scored whole else None)."""
        index_q, index_k, weights, starts, ends = self._indexer_inputs(q_lora, hidden, cu_seqlens, cp_layout)
        if indexer_tp_split_on(self.tp_group, index_q.device):
            part = tp_split_part(
                lambda q, w, ks, ke: chunked_indexer_topk(q, index_k, w, ks, ke, self.index_topk, topk_fn),
                index_q,
                weights,
                starts,
                ends,
                self.index_topk,
                self.tp_group,
            )
            return part, None
        indices = chunked_indexer_topk(index_q, index_k, weights, starts, ends, self.index_topk, topk_fn)
        return own_tp_part(indices, self.tp_group), indices

    def _forward_mla(self, hidden_states: torch.Tensor, packed_seq_params: PackedSeqParams | None) -> torch.Tensor:
        assert (
            packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        ), "Kimi-K3.1 DSA runs on packed (thd) sequences"
        assert hidden_states.shape[1] == 1
        x = hidden_states.squeeze(1)

        q_lora = self.q_a_layernorm(_linear(self.q_a_proj, x))
        query = _linear(self.q_b_proj, q_lora).view(-1, self.local_num_heads, self.q_head_dim)
        query_nope, query_extra = torch.split(query, [self.qk_nope_head_dim, self.qk_extra_head_dim], dim=-1)

        kv_latent, key_extra = torch.split(
            _linear(self.kv_a_proj_with_mqa, x), [self.kv_lora_rank, self.qk_extra_head_dim], dim=-1
        )
        # the latent bypasses kv_b_proj's input copy, so sum its gradient over TP here
        kv = copy_to_tensor_model_parallel_region(
            torch.cat((self.kv_a_layernorm(kv_latent), key_extra), dim=-1), group=self.tp_group
        )
        cp_layout = None
        if self.cp_size > 1:
            # queries stay on this rank's zigzag rows; keys (and top-k indices) span the whole packed stream
            cp_layout = zigzag_gather_layout(
                packed_seq_params.cu_seqlens_q, self.cp_group.rank(), self.cp_size, x.shape[0]
            )
            kv = gather_tp_replicated_over_cp(kv, self.cp_group, self.tp_group).index_select(0, cp_layout[0])

        w_kc, w_vc = self.kv_b_proj.weight.unflatten(0, (-1, self.qk_nope_head_dim + self.v_head_dim)).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        query = torch.cat((torch.einsum("thd,hdm->thm", query_nope, w_kc), query_extra), dim=-1)

        split_mode = query_split_on(self.tp_group, query.device)
        if split_mode:
            # MILES_K31_DSA_QUERY_SPLIT: all heads of 1/TP of the rows per rank instead of 12 heads padded to 64
            part = self._select_topk(q_lora, x, packed_seq_params.cu_seqlens_q, cp_layout, own_rows=True)
            latent = query_split_attention(
                sparse_attention,
                query,
                kv,
                part,
                x.shape[0],
                self.tp_group,
                mode=split_mode,
                sm_scale=self.q_head_dim**-0.5,
                d_v=self.kv_lora_rank,
            )
        else:
            topk_indices = self._select_topk(q_lora, x, packed_seq_params.cu_seqlens_q, cp_layout)
            latent = sparse_attention(
                query.unsqueeze(0),
                kv.unsqueeze(0).unsqueeze(2),
                topk_indices.unsqueeze(1).unsqueeze(0),
                self.q_head_dim**-0.5,
                d_v=self.kv_lora_rank,
            ).squeeze(0)
        output = torch.einsum("thm,hdm->thd", latent, w_vc).reshape(x.shape[0], 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, hidden_states))
        return _linear(self.o_proj, output)


def configure_kimi_k31_dsa(config, dsa_config: dict, topk_backend: str, kda_layers: tuple[int, ...]) -> None:
    """K3.1 adds a NoPE, LayerNorm-key indexer on every MLA layer; SGLang rejects other variants.
    ``kda_layers``: ``kimi_k3_kda_layers`` of the checkpoint's text config, the table the K3 spec builds from."""
    dsa_layers = checked_dsa_layers(dsa_config["dsa_layers"], num_layers=config.num_layers, kda_layers=kda_layers)
    assert dsa_config.get("index_topk_freq", 1) == 1 and dsa_config.get("index_rope_dim", 0) == 0
    assert dsa_config.get("index_k_norm_type", "layernorm") == "layernorm"
    config.kimi_dsa_layers = dsa_layers
    config.kimi_dsa_index_topk = dsa_config["index_topk"]
    config.kimi_dsa_index_num_heads = dsa_config["index_num_heads"]
    config.kimi_dsa_index_head_dim = dsa_config["index_head_dim"]
    config.miles_dsa_topk_backend = topk_backend


def get_kimi_k31_spec(args, config, vp_stage):
    text_config = load_hf_config(args.hf_checkpoint).text_config
    kda_layers = kimi_k3_kda_layers(text_config, config.num_layers)
    configure_kimi_k31_dsa(config, text_config.dsa_config, args.miles_dsa_topk_backend, kda_layers)
    return get_kimi_k3_spec(args, config, vp_stage, mla_attention=KimiK31DSAAttention, text_config=text_config)
