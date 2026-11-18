import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.hf_utils.config import load_hf_config
from miles.utils.lora.utils import is_lora_enabled

from .kda_layout import kimi_k3_kda_layers
from .layers import KimiK3Attention, KimiK3KDAAttention, KimiK3TransformerLayer
from .moe_a2a_overlap import a2a_overlap_enabled, a2a_overlap_moe_spec
from .mxfp4_experts import (
    MXFP4ColumnParallelGroupedLinear,
    MXFP4RowParallelGroupedLinear,
    hf_checkpoint_has_mxfp4_experts,
)
from .ops import situ_and_mul
from .permute import load_slot_layout
from .split_moe import partial_experts_spec
from .stage_recompute import apply_stage_recompute


def configure_kimi_k3(config, kda_layers: tuple[int, ...]) -> None:
    """``kda_layers``: ``kimi_k3_kda_layers`` of the checkpoint's text config."""
    assert config.num_layers <= 93
    assert config.num_attention_heads == 96
    assert config.moe_latent_size == 3584
    config.kimi_kda_layers = tuple(kda_layers)
    config.kimi_linear_num_heads = 96
    config.kimi_linear_head_dim = 128
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    config.kimi_attn_res_block_size = 12
    config.gated_activation_func = situ_and_mul
    config.moe_latent_use_norm = True
    config.bias_activation_fusion = False
    config.use_te_activation_func = False


def build_kimi_k3_spec(
    config,
    vp_stage=None,
    mla_attention=KimiK3Attention,
    *,
    mxfp4_experts: bool = False,
    num_local_trainable_experts: int | None = None,
):
    assert config.virtual_pipeline_model_parallel_size is None, "Kimi K3 does not support VPP yet"

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    # layer_specs holds only this stage's layers; moe_layer_freq is indexed globally
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_number = layer_offset + len(layer_specs) + 1
        attention = KimiK3KDAAttention if layer_number in config.kimi_kda_layers else mla_attention
        layer_spec.submodules.self_attention = ModuleSpec(module=attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm

        if not config.moe_layer_freq[layer_offset + len(layer_specs)]:
            # The dense MLP spec is partial(MLP.as_mlp_submodule, submodules=...).
            layer_spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        elif num_local_trainable_experts is not None:
            # --partial-experts-selection: the first local slots train in BF16, the rest stay MXFP4 bytes
            moe = layer_spec.submodules.mlp.keywords["submodules"]
            moe.experts = partial_experts_spec(moe.experts, num_local_trainable_experts)
        elif mxfp4_experts:
            # partial(MoELayer, submodules=MoESubmodules(experts=partial(TEGroupedMLP, submodules=...)))
            experts = layer_spec.submodules.mlp.keywords["submodules"].experts.keywords["submodules"]
            experts.linear_fc1 = MXFP4ColumnParallelGroupedLinear
            experts.linear_fc2 = MXFP4RowParallelGroupedLinear
        if config.moe_layer_freq[layer_offset + len(layer_specs)] and a2a_overlap_enabled():
            # MILES_K3_MOE_A2A_OVERLAP: the EP all-to-alls overlap the shared experts and the expert dequant
            layer_spec.submodules.mlp = a2a_overlap_moe_spec(layer_spec.submodules.mlp)

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec


def _check_cp_layout(args) -> None:
    # the KDA and DSA layers read zigzag CP shards; --allgather-cp would hand them contiguous ones
    assert args.context_parallel_size == 1 or not args.allgather_cp, "Kimi K3 runs zigzag CP; drop --allgather-cp"


def frozen_experts_are_mxfp4(args) -> bool:
    """The trainer keeps the routed experts as the checkpoint's MXFP4 bytes when it loads a compressed-tensors
    MXFP4 checkpoint. Only a run whose experts stay frozen can: LoRA, or partial expert training, where the
    selected experts get a BF16 master and every other expert stays MXFP4."""
    mxfp4 = hf_checkpoint_has_mxfp4_experts(args.backend.load)
    assert not mxfp4 or is_lora_enabled(args) or args.partial_experts_selection is not None, (
        f"{args.backend.load} keeps its routed experts as MXFP4, which only a LoRA run or --partial-experts-selection "
        "(frozen experts) trains from; convert it with tools/convert_mxfp4_to_bf16.py for full-parameter training"
    )
    return mxfp4


def get_kimi_k3_spec(args, config, vp_stage, mla_attention=KimiK3Attention, text_config=None):
    _check_cp_layout(args)
    if text_config is None:
        text_config = load_hf_config(args.hf_checkpoint).text_config
    configure_kimi_k3(config, kimi_k3_kda_layers(text_config, config.num_layers))
    mxfp4 = frozen_experts_are_mxfp4(args)
    num_local_trainable = None
    if args.partial_experts_selection is not None:
        assert mxfp4, "--partial-experts-selection loads the frozen experts from a compressed-tensors MXFP4 checkpoint"
        layout = load_slot_layout(
            args.partial_experts_selection,
            num_experts=config.num_moe_experts,
            ep_size=args.backend.expert_model_parallel_size,
        )
        # every MoE layer has its own trainable set: the selection must name exactly the model's MoE layers
        moe_layers = [i for i, is_moe in enumerate(config.moe_layer_freq) if is_moe]
        assert (
            layout.moe_layers == moe_layers
        ), f"--partial-experts-selection covers layers {layout.moe_layers}, the model's MoE layers are {moe_layers}"
        # the mbridge loader reads the table from the model config; the converter and R3 read the active layout
        config.partial_experts_layout = layout
        num_local_trainable = layout.num_local_trainable
    block_spec = build_kimi_k3_spec(
        config,
        vp_stage=vp_stage,
        mla_attention=mla_attention,
        mxfp4_experts=mxfp4,
        num_local_trainable_experts=num_local_trainable,
    )
    # MILES_K31_RECOMPUTE_NUM_LAYERS_PER_STAGE: this stage's --recompute-method block layer count
    apply_stage_recompute(config, stage_layers=len(block_spec.layer_specs))
    return block_spec
