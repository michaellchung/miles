"""Which Kimi-K3 layers are KDA (no torch / Megatron imports, so CPU tests load it)."""

# The released 93-layer model: every fourth layer and the last one are MLA, the rest KDA.
KIMI_K3_KDA_LAYERS = tuple(
    layer_number for layer_number in range(1, 94) if layer_number % 4 != 0 and layer_number != 93
)


def kimi_k3_kda_layers(text_config, num_layers: int) -> tuple[int, ...]:
    """The 1-based KDA layers of a ``num_layers``-layer model, from the checkpoint's
    ``linear_attn_config["kda_layers"]`` (the table the bridge loads the weights by). A checkpoint cut from
    non-contiguous layers renumbers them, so only the checkpoint knows; one without the table is the released
    layout. A checkpoint cut to its first layers may keep the full table: layers past ``num_layers`` drop."""
    linear = getattr(text_config, "linear_attn_config", None)  # config-access-exempt: optional HF field
    table = KIMI_K3_KDA_LAYERS if linear is None or linear.get("kda_layers") is None else linear["kda_layers"]
    return tuple(sorted(layer_number for layer_number in table if layer_number <= num_layers))
