"""How many layers each pipeline stage fully recomputes (``MILES_K31_RECOMPUTE_NUM_LAYERS_PER_STAGE``).

Megatron's ``--recompute-method block --recompute-num-layers N`` recomputes the first N layers of every stage and
keeps the rest's activations for the backward, with one N for all stages. Under 1F1B stage s of PP holds up to
PP - s microbatches at once, so stage 0 can afford the fewest kept layers and the last stage the most: with
``MILES_K31_RECOMPUTE_NUM_LAYERS_PER_STAGE="31,27,24"`` stage 0 recomputes 31 of its layers, stage 1 27 and
stage 2 24. ``TransformerBlock._checkpointed_forward`` reads ``config.recompute_num_layers`` at every forward, and
each pipeline rank builds its own config, so setting it once at model build is enough.

A layer this skips keeps every tensor its backward needs for the PP - s microbatches in flight: with
MILES_K3_SITU_TRITON=1 and MILES_K31_MXFP4_REDEQUANT_BACKWARD=1 about 12 GiB per KDA+MoE layer and 16 GiB per
MLA+DSA+MoE layer per GPU at 262144 tokens, TP8 x CP2 x EP16 (without SiTU Triton about 32 / 36 GiB).

Unset or empty: the config is not touched. Pure Python (Megatron is imported only when the switch is set).
"""

from __future__ import annotations

import os

STAGE_RECOMPUTE_ENV = "MILES_K31_RECOMPUTE_NUM_LAYERS_PER_STAGE"


def stage_recompute_num_layers(
    raw: str,
    *,
    pp_rank: int,
    pp_size: int,
    stage_layers: int,
    granularity: str | None,
    method: str | None,
) -> int | None:
    """This stage's recomputed layer count from ``raw`` (comma- or space-separated, one per stage), else None."""
    tokens = raw.replace(",", " ").split()
    if not tokens:
        return None
    if granularity != "full" or method != "block":
        raise ValueError(
            f"{STAGE_RECOMPUTE_ENV} needs --recompute-granularity full --recompute-method block, "
            f"got granularity {granularity!r}, method {method!r}"
        )
    counts = [int(token) for token in tokens]
    if len(counts) != pp_size:
        raise ValueError(f"{STAGE_RECOMPUTE_ENV}={raw!r}: {len(counts)} counts for {pp_size} pipeline stages")
    count = counts[pp_rank]
    if not 0 <= count <= stage_layers:
        raise ValueError(f"{STAGE_RECOMPUTE_ENV}={raw!r}: stage {pp_rank} has {stage_layers} layers, not {count}")
    return count


def apply_stage_recompute(config, *, stage_layers: int, pp_rank: int | None = None) -> int | None:
    """Set ``config.recompute_num_layers`` for this stage when the switch is set; returns the count it set.
    ``pp_rank`` defaults to Megatron's pipeline rank, read only when the switch is set (converters and tools build
    the spec without model-parallel state)."""
    raw = os.environ.get(STAGE_RECOMPUTE_ENV, "")
    if not raw.strip():
        return None
    if pp_rank is None:
        from megatron.core import parallel_state

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    count = stage_recompute_num_layers(
        raw,
        pp_rank=pp_rank,
        pp_size=config.pipeline_model_parallel_size,
        stage_layers=stage_layers,
        granularity=config.recompute_granularity,
        method=config.recompute_method,
    )
    config.recompute_num_layers = count
    return count
