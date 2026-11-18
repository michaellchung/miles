import torch
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import unwrap_model

from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import DeepseekV3Bridge
from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows, pack_mxfp4_rows
from miles_plugins.mbridge.linear_attn import LinearAttentionBridgeMixin
from miles_plugins.models.kimi_k3.kda_layout import kimi_k3_kda_layers
from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec
from miles_plugins.models.kimi_k3.ops import situ_and_mul
from miles_plugins.models.kimi_k3.partial_experts import FROZEN_HALF, TRAINABLE_HALF, is_split_expert_param_name
from miles_plugins.models.kimi_k3.permute import SlotLayout, get_active_slot_layout


@register_model("kimi_k3")
class KimiK3Bridge(LinearAttentionBridgeMixin, DeepseekV3Bridge):
    TransformerConfigClass = MLATransformerConfig

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": "head_dim",
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "language_model.model.norm.weight",
        "output_layer.weight": "language_model.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "input_layernorm.weight": ["language_model.model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.linear_attn.q_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_proj.weight"
        ],
        "self_attention.linear_attn.k_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.k_proj.weight"
        ],
        "self_attention.linear_attn.v_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.v_proj.weight"
        ],
        "self_attention.linear_attn.q_conv1d.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_conv1d.weight"
        ],
        "self_attention.linear_attn.k_conv1d.weight": [
            "language_model.model.layers.{layer_number}.self_attn.k_conv1d.weight"
        ],
        "self_attention.linear_attn.v_conv1d.weight": [
            "language_model.model.layers.{layer_number}.self_attn.v_conv1d.weight"
        ],
        "self_attention.linear_attn.A_log": ["language_model.model.layers.{layer_number}.self_attn.A_log"],
        "self_attention.linear_attn.dt_bias": ["language_model.model.layers.{layer_number}.self_attn.dt_bias"],
        "self_attention.linear_attn.f_a_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.f_a_proj.weight"
        ],
        "self_attention.linear_attn.f_b_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.f_b_proj.weight"
        ],
        "self_attention.linear_attn.b_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.b_proj.weight"
        ],
        "self_attention.linear_attn.g_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.g_proj.weight"
        ],
        "self_attention.linear_attn.norm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.o_norm.weight"
        ],
        "self_attention.linear_attn.out_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.g_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.g_proj.weight"],
        "self_attention.o_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.q_a_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.q_a_proj.weight"],
        "self_attention.q_a_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_a_layernorm.weight"
        ],
        "self_attention.q_b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.q_b_proj.weight"],
        "self_attention.kv_a_proj_with_mqa.weight": [
            "language_model.model.layers.{layer_number}.self_attn.kv_a_proj_with_mqa.weight"
        ],
        "self_attention.kv_a_layernorm.weight": [
            "language_model.model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"
        ],
        "self_attention.kv_b_proj.weight": ["language_model.model.layers.{layer_number}.self_attn.kv_b_proj.weight"],
        "self_attention_res_norm.weight": [
            "language_model.model.layers.{layer_number}.self_attention_res_norm.weight"
        ],
        "self_attention_res_proj.weight": [
            "language_model.model.layers.{layer_number}.self_attention_res_proj.weight"
        ],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["language_model.model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp_res_norm.weight": ["language_model.model.layers.{layer_number}.mlp_res_norm.weight"],
        "mlp_res_proj.weight": ["language_model.model.layers.{layer_number}.mlp_res_proj.weight"],
        "mlp.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.mlp.gate_proj.weight",
            "language_model.model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["language_model.model.layers.{layer_number}.mlp.down_proj.weight"],
        "mlp.router.weight": ["language_model.model.layers.{layer_number}.block_sparse_moe.gate.weight"],
        "mlp.router.expert_bias": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.gate.e_score_correction_bias"
        ],
        "mlp.fc1_latent_proj.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_down_proj.weight"
        ],
        "mlp.routed_expert_norm.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_norm.weight"
        ],
        "mlp.fc2_latent_proj.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.routed_expert_up_proj.weight"
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.gate_proj.weight",
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.shared_experts.down_proj.weight"
        ],
        "mlp.experts.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight"
        ],
    }

    # a compressed-tensors MXFP4 checkpoint keeps each expert as packed codes plus E8M0 scales; they load as the
    # trainer's frozen MXFP4 experts (miles_plugins.models.kimi_k3.mxfp4_experts), never as BF16
    _MXFP4_EXPERT_MAPPING = {
        "mlp.experts.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight_packed",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight_packed",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight_scale",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight_scale",
        ],
        "mlp.experts.linear_fc2.weight": [
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight_packed",
            "language_model.model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight_scale",
        ],
    }
    _experts_packed = False

    _OTHER_MAPPING = {
        "output_attn_res_norm.weight": ["language_model.model.output_attn_res_norm.weight"],
        "output_attn_res_proj.weight": ["language_model.model.output_attn_res_proj.weight"],
    }

    @property
    def text_config(self):
        return self.hf_config.text_config

    def _build_config(self):
        hf_config = self.text_config
        moe_layer_freq = [0] * hf_config.num_hidden_layers
        for layer_idx in range(hf_config.first_k_dense_replace, hf_config.num_hidden_layers):
            if layer_idx % hf_config.moe_layer_freq == 0:
                moe_layer_freq[layer_idx] = 1

        config = self._build_base_config(
            text_config_key="text_config",
            attention_backend=AttnBackend.auto,
            multi_latent_attention=True,
            qk_layernorm=True,
            q_lora_rank=hf_config.q_lora_rank,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_head_dim=hf_config.qk_nope_head_dim,
            qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            gated_activation_func=situ_and_mul,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            use_te_activation_func=False,
            persist_layer_norm=True,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            moe_latent_size=hf_config.routed_expert_hidden_size,
            moe_latent_use_norm=hf_config.latent_moe_use_norm,
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_token,
            moe_router_score_function=hf_config.moe_router_activation_func,
            moe_router_pre_softmax=True,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_enable_expert_bias=True,
            freeze_e_score_correction_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_shared_expert_intermediate_size=(hf_config.moe_intermediate_size * hf_config.num_shared_experts),
            moe_shared_expert_overlap=False,
            moe_layer_freq=moe_layer_freq,
            disable_bf16_reduced_precision_matmul=True,
        )
        config.kimi_kda_layers = kimi_k3_kda_layers(hf_config, config.num_layers)
        config.kimi_linear_num_heads = hf_config.linear_attn_config["num_heads"]
        config.kimi_linear_head_dim = hf_config.linear_attn_config["head_dim"]
        config.kimi_linear_conv_kernel_size = hf_config.linear_attn_config["short_conv_kernel_size"]
        config.kimi_kda_gate_lower_bound = hf_config.linear_attn_config["gate_lower_bound"]
        config.kimi_attn_res_block_size = hf_config.attn_res_block_size
        return config

    def _get_gptmodel_args(self) -> dict:
        return {
            "vocab_size": self.text_config.vocab_size,
            "max_sequence_length": self.text_config.max_position_embeddings,
            "position_embedding_type": "none",
        }

    def _get_transformer_layer_spec(self, vp_stage=None):
        self.has_vp_stage = True
        return build_kimi_k3_spec(self.config, vp_stage=vp_stage)

    def _get_safetensor_io(self, weights_path: str):
        safetensor_io = SafeTensorIO(self._get_actual_hf_path(weights_path))
        self._experts_packed = any(name.endswith(".weight_packed") for name in safetensor_io.index)
        return safetensor_io

    # ---- partial routed-expert training (--partial-experts-selection) -------------------------------
    #
    # The split experts module names its halves ``mlp.experts.trainable`` / ``mlp.experts.frozen``, which
    # mbridge's ``".mlp.experts.linear_fc" in name`` test does not recognize as EP-sharded: it would load
    # them on TP rank 0 and scatter them over TP. So the base loader skips those names and
    # ``_load_partial_experts`` fills each EP rank's local slots itself from the safetensors shards, in slot
    # order, dequantizing the trainable slots to BF16 and keeping the frozen slots as packed MXFP4 rows.
    # The router weight and expert bias load in HF order through the base loader and are permuted in place.

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep: bool = True) -> dict[str, str]:
        mapping = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=consider_ep)
        return {k: v for k, v in mapping.items() if not is_split_expert_param_name(k)}

    def load_weights(self, models, weights_path: str, memory_efficient: bool = False) -> None:
        super().load_weights(models, weights_path, memory_efficient=memory_efficient)
        layout = _partial_experts_layout(models)
        if layout is not None:
            self._load_partial_experts(models, layout)

    @torch.no_grad()
    def _load_partial_experts(self, models, layout: SlotLayout) -> None:
        assert self._experts_packed, "partial expert training loads the frozen experts from an MXFP4 checkpoint"
        assert self.mpu.etp_size == 1, "partial experts: expert tensor parallelism is not implemented"
        ep_rank = self.mpu.ep_rank
        n_local, k_local = layout.num_local_experts, layout.num_local_trainable
        assert n_local == self.config.num_moe_experts // self.mpu.ep_size, (n_local, self.config.num_moe_experts)
        for model in models:
            for layer in unwrap_model(model).decoder.layers:
                if not layer.is_moe_layer:
                    continue
                hf_layer = layer.layer_number - 1
                assert hf_layer in layout.perm, f"selection has no entry for MoE layer {hf_layer}"
                experts = layer.mlp.experts
                trainable_half, frozen_half = getattr(experts, TRAINABLE_HALF), getattr(
                    experts, FROZEN_HALF
                )  # config-access-exempt: the split module names its halves by these constants
                for local_slot in range(n_local):
                    original_id = layout.local_slot_to_original_id(hf_layer, ep_rank, local_slot)
                    fc1, fc2 = self._packed_expert_rows(hf_layer, original_id)
                    if local_slot < k_local:
                        # the BF16 master starts as the exact dequantization of the shipped MXFP4 expert;
                        # dequantize on the expert's device (the rows load on CPU, where 16 experts x ~31
                        # layers per rank take tens of minutes; the result is bit-identical)
                        device = _expert_device(trainable_half)
                        _copy_expert(
                            trainable_half,
                            local_slot,
                            dequantize_mxfp4_rows(fc1.to(device, non_blocking=True), MXFP4_GROUP_SIZE),
                            dequantize_mxfp4_rows(fc2.to(device, non_blocking=True), MXFP4_GROUP_SIZE),
                        )
                    else:
                        _copy_expert(frozen_half, local_slot - k_local, fc1, fc2)
                router = layer.mlp.router
                router.weight.copy_(layout.permute_router_rows(hf_layer, router.weight))
                if router.expert_bias is not None:
                    router.expert_bias.copy_(layout.permute_router_rows(hf_layer, router.expert_bias))

    def _packed_expert_rows(self, hf_layer: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """fc1 (w1 stacked over w3) and fc2 (w2) of one HF expert as packed MXFP4 rows, as _weight_to_mcore_format builds them."""
        fc1_names = [
            x.format(layer_number=hf_layer, expert_id=expert_id)
            for x in self._MXFP4_EXPERT_MAPPING["mlp.experts.linear_fc1.weight"]
        ]
        fc2_names = [
            x.format(layer_number=hf_layer, expert_id=expert_id)
            for x in self._MXFP4_EXPERT_MAPPING["mlp.experts.linear_fc2.weight"]
        ]
        fc1 = self._weight_to_mcore_format(
            "mlp.experts.linear_fc1.weight", [self.safetensor_io.load_one_hf_weight(x) for x in fc1_names]
        )
        fc2 = self._weight_to_mcore_format(
            "mlp.experts.linear_fc2.weight", [self.safetensor_io.load_one_hf_weight(x) for x in fc2_names]
        )
        return fc1, fc2

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if self._experts_packed:
            for keyword, templates in self._MXFP4_EXPERT_MAPPING.items():
                if keyword in name:
                    layer_number = name.split(".")[2]
                    expert_id = name.split("weight")[-1]
                    return [x.format(layer_number=layer_number, expert_id=expert_id) for x in templates]
        return super()._weight_name_mapping_mlp(name)

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name
        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]
        if "self_attention" in mcore_weights_name or "input_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        return self._weight_name_mapping_other(mcore_weights_name)

    def _weight_to_mcore_format(
        self,
        mcore_weights_name: str,
        hf_weights: list[torch.Tensor],
    ) -> torch.Tensor:
        if mcore_weights_name.endswith(
            (
                "self_attention.linear_attn.q_conv1d.weight",
                "self_attention.linear_attn.k_conv1d.weight",
                "self_attention.linear_attn.v_conv1d.weight",
            )
        ):
            assert len(hf_weights) == 1
            return hf_weights[0].float().contiguous()
        if mcore_weights_name.endswith("self_attention.linear_attn.A_log"):
            assert len(hf_weights) == 1
            return hf_weights[0][: self.config.kimi_linear_num_heads].float().contiguous()
        if mcore_weights_name.endswith("self_attention.linear_attn.dt_bias"):
            assert len(hf_weights) == 1
            return hf_weights[0].float().contiguous()
        if self._experts_packed and "mlp.experts.linear_fc" in mcore_weights_name:
            # the packed codes of w1 and w3 stack as fc1's rows, exactly as their BF16 weights would; the scales follow
            half = len(hf_weights) // 2
            return pack_mxfp4_rows(
                torch.cat(hf_weights[:half], dim=0), torch.cat(hf_weights[half:], dim=0), MXFP4_GROUP_SIZE
            )
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)


def _partial_experts_layout(models) -> SlotLayout | None:
    """The slot layout the model was built with (``get_kimi_k3_spec`` sets it on the config), else the process's."""
    for model in models:
        layout = getattr(  # config-access-exempt: only a partial-experts model spec sets partial_experts_layout
            unwrap_model(model).config, "partial_experts_layout", None
        )
        if layout is not None:
            return layout
    return get_active_slot_layout()


def _expert_device(half) -> torch.device:
    """The device the half's expert weights live on (CPU before they move to the GPU)."""
    return next(half.parameters()).device


def _copy_expert(half, local_index: int, fc1: torch.Tensor, fc2: torch.Tensor) -> None:
    dst1 = getattr(
        half.linear_fc1, f"weight{local_index}"
    )  # config-access-exempt: TE GroupedLinear registers weight0..weightN-1
    dst2 = getattr(
        half.linear_fc2, f"weight{local_index}"
    )  # config-access-exempt: TE GroupedLinear registers weight0..weightN-1
    assert dst1.shape == fc1.shape and dst1.dtype == fc1.dtype, (dst1.shape, dst1.dtype, fc1.shape, fc1.dtype)
    assert dst2.shape == fc2.shape and dst2.dtype == fc2.dtype, (dst2.shape, dst2.dtype, fc2.shape, fc2.dtype)
    dst1.copy_(fc1)
    dst2.copy_(fc2)
