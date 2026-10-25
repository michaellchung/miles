import functools
import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args

from miles.utils import sync_free
from miles.utils.ft_utils.process_group_utils import GroupInfo

from ..training_utils.parallel import ParallelState, get_parallel_state

logger = logging.getLogger(__name__)


def create_megatron_parallel_state(
    indep_dp: GroupInfo,
) -> ParallelState:
    vpp_size, microbatch_group_size_per_vp_stage = _compute_vpp_fields()
    args = get_args()

    def _create_intra_dp(with_context_parallel: bool):
        return GroupInfo(
            rank=mpu.get_data_parallel_rank(with_context_parallel=with_context_parallel),
            size=mpu.get_data_parallel_world_size(with_context_parallel=with_context_parallel),
            group=mpu.get_data_parallel_group(with_context_parallel=with_context_parallel),
            gloo_group=mpu.get_data_parallel_group_gloo(with_context_parallel=with_context_parallel),
        )

    return ParallelState(
        intra_dp=_create_intra_dp(with_context_parallel=False),
        intra_dp_cp=_create_intra_dp(with_context_parallel=True),
        cp=GroupInfo(
            rank=mpu.get_context_parallel_rank(),
            size=mpu.get_context_parallel_world_size(),
            group=mpu.get_context_parallel_group(),
        ),
        cp_comm_type=getattr(args, "cp_comm_type", None),
        tp=GroupInfo(
            rank=mpu.get_tensor_model_parallel_rank(),
            size=mpu.get_tensor_model_parallel_world_size(),
            group=mpu.get_tensor_model_parallel_group(),
        ),
        pp=GroupInfo(
            rank=mpu.get_pipeline_model_parallel_rank(),
            size=mpu.get_pipeline_model_parallel_world_size(),
            group=mpu.get_pipeline_model_parallel_group(),
        ),
        ep=GroupInfo(
            rank=mpu.get_expert_model_parallel_rank(),
            size=mpu.get_expert_model_parallel_world_size(),
            group=mpu.get_expert_model_parallel_group(),
        ),
        etp=GroupInfo(
            rank=mpu.get_expert_tensor_parallel_rank(),
            size=mpu.get_expert_tensor_parallel_world_size(),
            group=mpu.get_expert_tensor_parallel_group(),
        ),
        indep_dp=indep_dp,
        is_pp_last_stage=mpu.is_pipeline_last_stage(),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
    )


def _compute_vpp_fields() -> tuple[int, int | None]:
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is None or vpp_size_value <= 1:
        return 1, None

    return vpp_size_value, get_args().pipeline_model_parallel_size


def verify_megatron_parallel_state(
    model: torch.nn.Module | Sequence[torch.nn.Module],
) -> None:
    """Verify that ParallelState fields match what the model config produces."""
    parallel_state = get_parallel_state()
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is not None and vpp_size_value > 1:
        model_to_check = model[0] if isinstance(model, Sequence) else model
        config = get_model_config(model_to_check)
        expected = config.microbatch_group_size_per_vp_stage
        actual = parallel_state.microbatch_group_size_per_vp_stage
        assert (
            actual == expected
        ), f"microbatch_group_size_per_vp_stage mismatch: ParallelState has {actual}, model config has {expected}"


# sha256 of the dedented source of PackedSeqParams.__post_init__ at radixark/Megatron-LM
# a84b10547 (megatron/core/packed_seq_params.py:42-80); thd_seq_idx reproduces it.
_KNOWN_POST_INIT_SHA256 = frozenset({"3c6a4723b8174cf87166291112f031d16515bb8ae3ea459f7a11893335177405"})


def get_packed_seq_params(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
    if args.qkv_format != "thd":
        return None
    total_tokens = _packed_total_tokens(batch, args)
    host_cu_seqlens = batch.get("cu_seqlens_cpu")
    seq_idx_on_host = total_tokens is not None and host_cu_seqlens is not None and _host_seq_idx_allowed()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=batch["cu_seqlens"],
        cu_seqlens_kv=batch["cu_seqlens"],
        max_seqlen_q=batch["max_seqlen"],
        max_seqlen_kv=batch["max_seqlen"],
        qkv_format="thd",
        total_tokens=None if seq_idx_on_host else total_tokens,
    )
    if seq_idx_on_host:
        # The fields PackedSeqParams(total_tokens=...) would set, with seq_idx built from the
        # host copy of cu_seqlens: __post_init__ builds it on the GPU and makes the host wait
        # three times (a host-to-device scalar and repeat_interleave with tensor repeats).
        packed_seq_params.total_tokens = total_tokens
        seq_idx = thd_seq_idx(host_cu_seqlens, total_tokens)
        packed_seq_params.seq_idx = sync_free.to_device_async(seq_idx, batch["cu_seqlens"].device)
    batch["packed_seq_params"] = packed_seq_params
    return packed_seq_params


def thd_seq_idx(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """``PackedSeqParams.__post_init__``'s ``seq_idx`` ([1, total_tokens] int32) on ``cu_seqlens``' device.

    Token i gets the index of the packed sample it belongs to; tokens after the last
    boundary (the pad) form one more sample."""

    bounds = torch.cat([cu_seqlens, torch.tensor([total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device)])
    seq_lengths = (bounds[1:] - bounds[:-1]).clamp(min=0)
    sample_ids = torch.arange(seq_lengths.numel(), device=cu_seqlens.device)
    return torch.repeat_interleave(sample_ids, seq_lengths).to(torch.int32).unsqueeze(0)


@functools.cache
def _host_seq_idx_matches_megatron() -> bool:
    digest = sync_free.source_sha256(PackedSeqParams.__post_init__)
    if digest in _KNOWN_POST_INIT_SHA256:
        return True
    logger.warning(
        "PackedSeqParams.__post_init__ does not match the Megatron source thd_seq_idx reproduces (sha256 %s); "
        "letting Megatron build seq_idx on the GPU, which makes the host wait for the GPU in every micro batch",
        digest or "unavailable",
    )
    return False


def _host_seq_idx_allowed() -> bool:
    return sync_free.enabled() and _host_seq_idx_matches_megatron()


def _packed_total_tokens(batch: dict[str, torch.Tensor], args: Namespace) -> int | None:
    """Token count of the packed THD stream, so ``PackedSeqParams.__post_init__`` derives
    ``seq_idx``. Mamba (and GatedDeltaNet) layers read ``seq_idx`` to restart the conv and
    SSM state at each packed sample; without it the state of one sample flows into the
    next one, which the rollout engine (one request per sequence) never does.

    Only without context parallelism: with CP the stream is a per-rank slice and
    ``cu_seqlens`` describes the global stream."""
    if getattr(args, "context_parallel_size", 1) != 1 or getattr(args, "allgather_cp", False):
        return None
    return int(batch["tokens"].size(-1))
