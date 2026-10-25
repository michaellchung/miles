import json
from contextlib import contextmanager
from types import MappingProxyType

from miles.utils.replay_base import reject_router_fusion_with_routing_replay

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_dsa_backend_args(provider, args) -> None:
    """Map --dsa-attention-backend onto the provider's dsa_attention_backend (bridge) or dsa_kernel_backend (main)."""
    backend = getattr(args, "dsa_attention_backend", "megatron")
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):
        explicit = getattr(args, "dsa_kernel_backend", None)
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]


# MoE runtime flags that the bridge path copies onto the provider only when they are set
# (model_provider._apply_bridge_runtime_config). Megatron's parser builds each flag from the
# TransformerConfig field of the same name, with the field default as the CLI default; a test checks
# this table against the parser. A bridge provider may hold its own value on purpose
# (Nemotron-H sets moe_permute_fusion and moe_shared_expert_overlap to True), and args hold the CLI
# default when a flag is absent, so a field is copied only when args differ from the CLI default.
# A store-true flag can therefore turn a provider setting on, never off.
BRIDGE_MOE_RUNTIME_ARG_DEFAULTS = MappingProxyType(
    {
        "moe_flex_dispatcher_backend": "deepep",
        "moe_flex_dispatcher_num_sms": None,
        "moe_deepep_num_sms": None,  # deprecated alias; TransformerConfig.__post_init__ routes it
        "moe_hybridep_num_sms": None,  # deprecated alias; same
        "moe_enable_deepep": False,  # deprecated; TransformerConfig.__post_init__ maps it to the deepep backend
        "moe_permute_fusion": False,
        "moe_router_fusion": False,
        "moe_shared_expert_overlap": False,
        "overlap_moe_expert_parallel_comm": False,
        "delay_wgrad_compute": False,
    }
)


def bridge_moe_runtime_overrides(args) -> dict:
    """The BRIDGE_MOE_RUNTIME_ARG_DEFAULTS fields that args set to a value other than the CLI default."""
    return {
        name: getattr(args, name)
        for name, default in BRIDGE_MOE_RUNTIME_ARG_DEFAULTS.items()
        if getattr(args, name, default) != default
    }


def apply_bridge_moe_runtime_args(provider, args) -> None:
    """Copy the MoE runtime flags set in args onto a bridge-built provider, before ``finalize()``.

    Raises instead of dropping a setting: for a set flag that the provider has no field for, for
    ``--cuda-graph-impl``, and for router fusion together with routing replay.
    """
    _reject_bridge_cuda_graphs(cuda_graph_impl=getattr(args, "cuda_graph_impl", "none"))
    apply_bridge_local_cuda_graph_args(provider, args)
    for name, value in bridge_moe_runtime_overrides(args).items():
        if not hasattr(provider, name):
            raise AttributeError(
                f"--{name.replace('_', '-')} is set, but the bridge provider {type(provider).__name__} "
                f"has no field {name!r}"
            )
        setattr(provider, name, value)
    router_fusion = bool(getattr(provider, "moe_router_fusion", False))
    reject_router_fusion_with_routing_replay(router_fusion=router_fusion, args=args)


def describe_bridge_moe_runtime_config(provider, args) -> str:
    """A JSON line: the MoE runtime fields of a bridge provider and the ones that args set."""
    names = ("moe_token_dispatcher_type", *BRIDGE_MOE_RUNTIME_ARG_DEFAULTS, "cuda_graph_impl")
    return json.dumps(
        {
            "effective": {name: getattr(provider, name, "<absent>") for name in names},
            "from_args": sorted(bridge_moe_runtime_overrides(args)),
        },
        default=str,
        sort_keys=True,
    )


# CUDA graph settings copied onto a bridge provider for --cuda-graph-impl local (provider field <- args attr).
BRIDGE_LOCAL_CUDA_GRAPH_FIELDS = MappingProxyType(
    {
        "cuda_graph_impl": "cuda_graph_impl",
        "cuda_graph_modules": "cuda_graph_modules",
        "cuda_graph_warmup_steps": "cuda_graph_warmup_steps",
        "use_te_rng_tracker": "te_rng_tracker",
    }
)


def apply_bridge_local_cuda_graph_args(provider, args) -> None:
    """--cuda-graph-impl local in bridge mode: copy the CUDA graph settings onto the provider and install
    the eager fallback for unmatched graph inputs (megatron_utils/cuda_graph_fallback.py).

    Megatron captures local graphs in its own schedule (``create_cudagraphs()`` at the end of the first
    training forward/backward), so Miles' training loop needs no capture call. The RNG tracker must be
    TE's (Megatron enables ``--te-rng-tracker`` for CUDA graphs with the TE transformer impl)."""
    if getattr(args, "cuda_graph_impl", "none") != "local":
        return
    if not getattr(args, "te_rng_tracker", False):
        raise ValueError("--cuda-graph-impl local needs --te-rng-tracker (the CUDA-graph-safe RNG tracker)")
    _reject_router_graph_without_static_replay(args)
    for field, attr in BRIDGE_LOCAL_CUDA_GRAPH_FIELDS.items():
        if not hasattr(provider, field):
            raise AttributeError(f"bridge provider {type(provider).__name__} has no field {field!r}")
        setattr(provider, field, getattr(args, attr))
    from miles.backends.megatron_utils import cuda_graph_fallback

    cuda_graph_fallback.install()
    cuda_graph_fallback.install_moe_recompute_for_eager_moe()


def _reject_router_graph_without_static_replay(args) -> None:
    """A graph that captures the MoE router (modules moe_router / moe_preprocess / moe, or an empty list =
    whole layers) would fix the first micro batch's replayed routes: the per-layer R3 path copies them from
    host memory inside the router. Routing replay then needs --routing-replay-static-buffer."""
    replay = getattr(args, "use_routing_replay", False) or getattr(args, "use_rollout_routing_replay", False)
    if not replay or getattr(args, "routing_replay_static_buffer", False):
        return
    modules = [getattr(m, "name", str(m)) for m in (getattr(args, "cuda_graph_modules", None) or [])]
    if not modules or any(m in ("moe_router", "moe_preprocess", "moe") for m in modules):
        raise ValueError(
            f"--cuda-graph-modules {modules or '(all)'} captures the MoE router while routing replay is on; "
            "add --routing-replay-static-buffer (the replayed routes are then read from fixed device rows)"
        )


def _reject_bridge_cuda_graphs(*, cuda_graph_impl) -> None:
    # Only the local impl is wired (apply_bridge_local_cuda_graph_args): Megatron captures it inside
    # its schedule. The transformer_engine impl needs TECudaGraphHelper calls in the training loop,
    # which Miles does not make, and full_iteration wraps Megatron's own training loop. Fail instead
    # of training without graphs while args claim otherwise.
    if cuda_graph_impl not in (None, "none", "local"):
        raise ValueError(
            f"--cuda-graph-impl {cuda_graph_impl} is not supported with --megatron-to-hf-mode bridge: Miles' "
            "training loop does not capture transformer_engine / full_iteration graphs. Use "
            "--cuda-graph-impl local, or build the model with --custom-model-provider-path."
        )
