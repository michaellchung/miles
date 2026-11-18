"""Rollout top-k candidates laid out for score centering once per step, not once per microbatch.

``MILES_SC_TOPK_PREP=1`` (default off). Without it, the last pipeline stage builds the candidate ids and
rollout head log-probs of every microbatch on the host inside the training step: a numpy fancy-index gather
of the sample's full ``[R, K]`` arrays at this CP rank's response positions (the ids twice, as int64), then a
pageable host-to-device copy of each, and a device ``.any()`` range check of the ids that makes the host wait.

With it:

- ``get_rollout_data`` (``prepare``) cuts this CP rank's rows of each sample out of the arrays into
  contiguous pinned host tensors (ids as int32), in the order ``slice_log_prob_with_cp`` gives, and
  checks the ids against the vocabulary there, once per step and outside the training step;
- ``forward_step`` (``prefetch``) starts their copy to the device on a side stream before the model
  forward of the microbatch, so the copy overlaps the forward;
- the score-centering loss waits for that copy on its stream and reads the rows as they are, so it no
  longer gathers, copies from pageable memory or range-checks on the device.

The loss computes the same values: the rows are the ones ``PackedSlices.rows_at_positions`` gathers.
Only Megatron runs take this path (``enabled``); the side-stream copy is CUDA only and other devices copy
on the current stream.
"""

import os
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from miles.backends.training_utils.data.context_parallel import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.parallel import get_parallel_state
from miles.platforms import current_platform

ENV = "MILES_SC_TOPK_PREP"
LOCAL_IDS = "rollout_topk_local_token_ids"
LOCAL_LOG_PROBS = "rollout_topk_local_log_probs"
# get_batch hands the microbatch its prefetched rows under this key
PREFETCHED = "rollout_topk_local"
OUT_OF_VOCAB = "Score-centering token ID is outside the model vocabulary"

_copy_stream: torch.cuda.Stream | None = None


def switch_on() -> bool:
    return os.environ.get(ENV, "0") == "1"


def enabled(args: Namespace) -> bool:
    """Whether this run lays the candidates out ahead: the switch, and a run whose loss reads them as rows."""
    if not switch_on():
        return False
    # local import: the loss hub imports the data package, which imports this module
    from miles.backends.training_utils.loss.hub.response_layout import PackedSlices, response_layout_type

    return (
        args.loss_type == "score_centering"
        and args.train_backend == "megatron"
        and not args.use_sampling_support_replay
        and not args.allgather_cp
        and response_layout_type(args) is PackedSlices
    )


def local_response_ranges(
    total_length: int, response_length: int, qkv_format: str, max_seq_len: int | None
) -> list[tuple[int, int]]:
    """This CP rank's response positions, as the ranges ``slice_log_prob_with_cp`` concatenates."""
    cp_size = get_parallel_state().cp.size
    if cp_size == 1:
        return [(0, response_length)]
    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )
    ranges = []
    for start, end in logits_offset:
        # python slicing of [R] with these bounds, as slice_log_prob_with_cp does
        start, end, _ = slice(start - (prompt_length - 1), end - (prompt_length - 1)).indices(response_length)
        ranges.append((start, max(start, end)))
    return ranges


def _local_rows(
    values: torch.Tensor, ranges: Sequence[tuple[int, int]], dtype: torch.dtype, pin: bool
) -> torch.Tensor:
    rows = sum(end - start for start, end in ranges)
    out = torch.empty((rows, *values.shape[1:]), dtype=dtype, pin_memory=pin)
    offset = 0
    for start, end in ranges:
        out[offset : offset + end - start].copy_(values[start:end])
        offset += end - start
    return out


def _check_vocab(values: torch.Tensor, vocab_size: int) -> None:
    if values.numel() == 0:
        return
    low, high = torch.aminmax(values)
    if int(low) < -1 or int(high) >= vocab_size:
        raise ValueError(OUT_OF_VOCAB)


def prepare(args: Namespace, rollout_data: dict) -> None:
    """Add this rank's contiguous top-k rows of every sample (``LOCAL_IDS`` / ``LOCAL_LOG_PROBS``).

    Only the last pipeline stage scores the loss, so only it builds them. The sampled tokens the loss
    scores next to the candidates are range-checked here as well, in one wait for the whole step.
    """
    ids_batch = rollout_data.get("rollout_topk_token_ids")
    log_probs_batch = rollout_data.get("rollout_topk_log_probs")
    if ids_batch is None or log_probs_batch is None or not get_parallel_state().is_pp_last_stage:
        return
    vocab_size = args.backend.vocab_size
    pin = current_platform.device_type == "cuda" and torch.accelerator.is_available()
    max_seq_lens = rollout_data.get("max_seq_lens")
    local_ids, local_log_probs = [], []
    for i, (ids, log_probs, total_length, response_length) in enumerate(
        zip(
            ids_batch,
            log_probs_batch,
            rollout_data["total_lengths"],
            rollout_data["response_lengths"],
            strict=True,
        )
    ):
        ids, log_probs = torch.as_tensor(ids), torch.as_tensor(log_probs)
        if ids.ndim != 2 or ids.shape != log_probs.shape or ids.size(0) != response_length:
            raise ValueError(
                f"sample {i}: rollout top-k ids {tuple(ids.shape)} / log-probs {tuple(log_probs.shape)} "
                f"do not cover response_length {response_length} as [R, K]"
            )
        ranges = local_response_ranges(
            total_length,
            response_length,
            args.qkv_format,
            None if max_seq_lens is None else max_seq_lens[i],
        )
        for start, end in ranges:
            _check_vocab(ids[start:end], vocab_size)
        local_ids.append(_local_rows(ids, ranges, torch.int32, pin))
        local_log_probs.append(_local_rows(log_probs, ranges, log_probs.dtype, pin))
    _check_sampled_tokens(rollout_data["tokens"], rollout_data["response_lengths"], vocab_size)
    rollout_data[LOCAL_IDS] = local_ids
    rollout_data[LOCAL_LOG_PROBS] = local_log_probs


def _check_sampled_tokens(tokens: Sequence[torch.Tensor], response_lengths: Sequence[int], vocab_size: int) -> None:
    responses = [t[t.numel() - length :] for t, length in zip(tokens, response_lengths, strict=True) if length]
    if not responses:
        return
    flags = torch.stack([((r < -1) | (r >= vocab_size)).any() for r in responses])
    if bool(flags.any()):
        raise ValueError(OUT_OF_VOCAB)


@dataclass
class PrefetchedRows:
    """One microbatch's top-k rows on their way to the device; ``rows`` waits for them on the caller's stream."""

    ids: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    done: torch.cuda.Event | None = None
    _waited: set = field(default_factory=set)

    def rows(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.done is not None:
            stream = torch.cuda.current_stream()
            if stream not in self._waited:
                stream.wait_event(self.done)
                self._waited.add(stream)
        return self.ids, self.log_probs


def prefetch(batch: dict, device: torch.device | None = None) -> None:
    """Start copying the microbatch's rows to ``device`` (last stage only); the loss reads ``batch[PREFETCHED]``."""
    ids, log_probs = batch.pop(LOCAL_IDS, None), batch.pop(LOCAL_LOG_PROBS, None)
    if ids is None or log_probs is None or not switch_on() or not get_parallel_state().is_pp_last_stage:
        return
    device = current_platform.current_device() if device is None else device
    if device.type != "cuda":
        batch[PREFETCHED] = PrefetchedRows([t.to(device) for t in ids], [t.to(device) for t in log_probs])
        return
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.cuda.Stream(device=device)
    main = torch.cuda.current_stream(device)
    # allocated on the main stream; the copy may only start once the main stream is done with the memory
    dst_ids = [torch.empty(t.shape, dtype=t.dtype, device=device) for t in ids]
    dst_log_probs = [torch.empty(t.shape, dtype=t.dtype, device=device) for t in log_probs]
    _copy_stream.wait_stream(main)
    with torch.cuda.stream(_copy_stream):
        for src, dst in zip([*ids, *log_probs], [*dst_ids, *dst_log_probs], strict=True):
            dst.copy_(src, non_blocking=src.is_pinned())
            dst.record_stream(_copy_stream)
        done = torch.cuda.Event()
        done.record(_copy_stream)
    batch[PREFETCHED] = PrefetchedRows(dst_ids, dst_log_probs, done)
