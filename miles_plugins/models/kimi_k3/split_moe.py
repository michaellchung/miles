"""Routed experts whose first local slots train in BF16 while the rest stay frozen MXFP4.

Why two grouped linears per projection (TransformerEngine release_v2.17, ``pytorch/module/grouped_linear.py``):

* wgrad is one switch for the whole group. ``_GroupedLinear.forward`` reads
  ``weight_requires_grad = weights[0].requires_grad`` (:253 and :439) and stores it as
  ``ctx.weights_requires_grad`` (:354, :660); backward computes either every wgrad or none (:734-:745).
* fused wgrad accumulation needs ``main_grad`` on every weight: forward keeps
  ``lambda j=i: weights[j].main_grad`` for all ``num_gemms`` (:361-:364, :673-:676) and backward calls
  them all (:742). A frozen uint8 expert has no ``main_grad``.
* ``m_splits`` must be a 1-D tensor of exactly ``num_gemms`` entries (:1711); zero-token experts are
  allowed, the grouped GEMM skips empty splits.

So a GroupedLinear cannot mix trainable and frozen experts: ``PartiallyTrainableGroupedMLP`` holds two
complete ``TEGroupedMLP``: ``trainable`` (``k_local`` BF16 experts on TE's native grouped linears, wgrad
into ``main_grad``) and ``frozen`` (``n_local - k_local`` experts on #4667's ``MXFP4*GroupedLinear``,
uint8 packed bytes dequantized on each weight fetch, ``requires_grad=False``, no wgrad).

Token split: both Megatron dispatchers hand the experts module tokens sorted by local expert, each
expert's rows contiguous (alltoall ``sort_chunks_by_idxs``; flex/DeepEP multihot permute). The slot
permutation (``permute.SlotLayout``) puts the trainable experts at local slots ``[0, k_local)``, so the
first ``sum(tokens_per_expert[:k_local])`` rows go to ``trainable`` and the rest to ``frozen``. Probs
split the same way and are applied inside each half after the activation, as TEGroupedMLP does.

LatentMoE: ``fc1_latent_proj`` runs in ``MoELayer.preprocess`` before dispatch and ``routed_expert_norm``
+ ``fc2_latent_proj`` after combine; both are duplicated TELinear outside the experts module and stay
frozen with the dense side. The experts only ever see latent-width rows; nothing here touches them.

Recompute: under ``--recompute-granularity full`` the whole layer re-runs in backward, including this
module; nothing special is needed. TE keeps the tensors ``_get_weight_tensors()`` returned until
backward, so the frozen half holds one layer's dequantized BF16 experts at a time under full recompute.
"""

from __future__ import annotations

import dataclasses
import functools

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

from miles_plugins.models.kimi_k3.mxfp4_experts import MXFP4ColumnParallelGroupedLinear, MXFP4RowParallelGroupedLinear
from miles_plugins.models.kimi_k3.partial_experts import FROZEN_HALF, TRAINABLE_HALF
from miles_plugins.models.kimi_k3.situ_probs_experts import SituProbsTEGroupedMLP, situ_probs_fused_enabled


class PartiallyTrainableGroupedMLP(MegatronModule):
    """Megatron ``ExpertsInterface`` over two TEGroupedMLP halves: BF16 trainable, MXFP4 frozen."""

    def __init__(
        self,
        num_local_experts: int,
        config,
        submodules: GroupedMLPSubmodules,
        pg_collection=None,
        name: str | None = None,
        *,
        num_local_trainable: int,
    ):
        super().__init__(config=config)
        assert 0 < num_local_trainable < num_local_experts, (num_local_trainable, num_local_experts)
        assert not config.add_bias_linear, "Kimi K3 experts carry no bias"
        # Megatron eb547f33e (the k31 image) has no moe_use_legacy_grouped_gemm; the TE
        # path is the only grouped GEMM there and moe_grouped_gemm must be on
        assert not getattr(  # config-access-exempt: absent on newer Megatron
            config, "moe_use_legacy_grouped_gemm", False
        ), "partial experts need TE grouped GEMM"
        assert config.moe_grouped_gemm, "partial experts need --moe-grouped-gemm"
        assert not getattr(  # config-access-exempt: the op fuser assembles one GroupedLinear itself and cannot split
            config, "use_transformer_engine_op_fuser", False
        ), "partial experts: the TE op fuser path rebuilds a single GroupedLinear"
        assert not config.moe_pad_expert_input_to_capacity, "partial experts split tokens by tokens_per_expert counts"
        self.num_local_experts = num_local_experts
        self.num_local_trainable = num_local_trainable
        half_cls = SituProbsTEGroupedMLP if situ_probs_fused_enabled() else TEGroupedMLP
        self.trainable = half_cls(
            num_local_trainable, config, submodules, pg_collection, name=_sub_name(name, TRAINABLE_HALF)
        )
        frozen_submodules = dataclasses.replace(
            submodules, linear_fc1=MXFP4ColumnParallelGroupedLinear, linear_fc2=MXFP4RowParallelGroupedLinear
        )
        self.frozen = half_cls(
            num_local_experts - num_local_trainable,
            config,
            frozen_submodules,
            pg_collection,
            name=_sub_name(name, FROZEN_HALF),
        )
        # before DDP wraps the model: DDP skips requires_grad=False parameters, so the frozen half gets no
        # grad buffer; model_provider freezes the rest of the model the same way
        self.frozen.requires_grad_(False)

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        k = self.num_local_trainable
        # CPU int64 under alltoall / DeepEP; a device tensor under HybridEP / NCCLEP, where TEGroupedMLP
        # calls .tolist() anyway, so the prefix sum adds no extra synchronization
        counts = torch.as_tensor(tokens_per_expert)
        hot_rows = int(counts[:k].sum())
        cold_rows = permuted_local_hidden_states.shape[0] - hot_rows
        x_hot, x_cold = permuted_local_hidden_states.split([hot_rows, cold_rows])
        p_hot, p_cold = permuted_probs.split([hot_rows, cold_rows])
        # run both halves on every microbatch, even with zero rows: DDP's per-bucket grad-ready hooks
        # need the trainable half's backward each microbatch, and TE handles empty splits
        out_hot, bias_hot = self.trainable(x_hot, counts[:k], p_hot)
        out_cold, bias_cold = self.frozen(x_cold, counts[k:], p_cold)
        assert bias_hot is None and bias_cold is None
        return torch.cat([out_hot, out_cold], dim=0), None

    def backward_dw(self) -> None:
        self.trainable.backward_dw()

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        """Only the trainable half goes into torch_dist checkpoints.

        TEGroupedMLP shards its experts as ``ep_rank * num_local + i`` of ``ep_size * num_local``; for the
        trainable half that is exactly the EP-independent trainable index ``SlotLayout.trainable_index``,
        so a checkpoint reshards across EP sizes. The frozen half reloads from the HF MXFP4 checkpoint on
        every start and has no state to save.
        """
        return self.trainable.sharded_state_dict(f"{prefix}{TRAINABLE_HALF}.", sharded_offsets, metadata)


def _sub_name(name: str | None, part: str) -> str | None:
    return None if name is None else f"{name}.{part}"


def partial_experts_spec(experts_spec: functools.partial, num_local_trainable: int) -> functools.partial:
    """Swap ``partial(TEGroupedMLP, submodules=...)`` for the split module with the same submodules."""
    assert experts_spec.func is TEGroupedMLP, experts_spec.func
    return functools.partial(
        PartiallyTrainableGroupedMLP,
        submodules=experts_spec.keywords["submodules"],
        num_local_trainable=num_local_trainable,
    )
