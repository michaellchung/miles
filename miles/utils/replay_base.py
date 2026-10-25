import logging
import os

import torch
import torch.distributed as dist

from miles.utils import sync_free

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _to_current_device(top_indices: torch.Tensor) -> torch.Tensor:
    # record() keeps each replay buffer in pinned host memory until clear(), so the copy
    # need not make the host wait for the GPU queue: kernels queued after it on the
    # current stream see the data, and PyTorch's pinned allocator does not reuse a
    # buffer before its pending copies finish.  (A pageable buffer, e.g. one set by the
    # debug replay tools, is staged by the driver; non_blocking stays safe for it.)
    return top_indices.to(torch.cuda.current_device(), non_blocking=sync_free.enabled())


def fill_all_invalid_rows(top_indices: torch.Tensor, all_invalid: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Rows whose picks are all -1 (padding tokens) get picks 0..topk-1 (mod num_experts).

    Other rows are returned unchanged, so applying it when no row is all -1 is a copy.
    """
    ar = torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype) % num_experts
    return torch.where(all_invalid.unsqueeze(-1), ar, top_indices)


class Replay:
    def __init__(self, stream_idx: int | None = None):
        self.stream_idx = stream_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []
        # True when micro batches are padded to THD buckets (set by fill_replay_data).
        self.spread_padding_routes = False

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return _to_current_device(top_indices)

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return _to_current_device(top_indices)

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


def _spread_padding_routes(top_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Routes for padding rows that spread them evenly over all experts: row r takes experts
    (r * topk + j) % num_experts, j < topk (distinct within a row when topk <= num_experts).
    Computed for every row without a data-dependent branch (no host sync, CUDA-graph safe)."""
    rows, topk = top_indices.shape
    row = torch.arange(rows, device=top_indices.device, dtype=top_indices.dtype).unsqueeze(-1)
    col = torch.arange(topk, device=top_indices.device, dtype=top_indices.dtype).unsqueeze(0)
    return (row * topk + col) % num_experts


class StaticReplayBuffer:
    """Resident device rows that the top-k hook reads instead of popping a host copy.

    Path (d) of SESSION27/CUDA-GRAPH.md: a CUDA graph that captures the MoE router fixes the
    address it reads the replayed routes from, and it may not copy from host memory or read a
    value back on the host.  So every micro batch is staged once, before its forward and outside
    any graph: one non-blocking host-to-device copy of all layers' routes (a pinned
    ``[slots, n, topk]`` stack per micro batch, built once per rollout from the per-layer lists)
    into ``device_rows[:, :n]``.  Layer ``slot`` then reads ``device_rows[slot, :n]``: the same
    address for every micro batch with the same ``n`` (one THD bucket).

    The forward and the recompute in the backward of a micro batch read the same rows.  That
    holds without pipeline parallelism: the backward of micro batch i runs before micro batch
    i + 1 is staged, and the copy is queued on the stream that runs the router.  The device rows
    are allocated once (``capacity`` rows) and never reallocated, so a captured address stays valid.
    """

    def __init__(self, capacity: int, device=None, pin_memory: bool = True):
        if capacity <= 0:
            raise ValueError(f"static replay buffer capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.device = device
        self.pin_memory = pin_memory
        self.device_rows: torch.Tensor | None = None
        self.staged_tokens: int | None = None
        self.host_stacks: list[torch.Tensor] | None = None
        self.next_index = 0

    def reset_rollout(self) -> None:
        self.host_stacks = None
        self.next_index = 0
        self.staged_tokens = None

    def rows(self, slot: int, num_tokens: int) -> torch.Tensor:
        if self.staged_tokens is None:
            raise RuntimeError("static replay buffer read before any micro batch was staged")
        if num_tokens != self.staged_tokens:
            raise AssertionError(
                f"rank {_get_rank()}: router sees {num_tokens} tokens but the staged micro batch has "
                f"{self.staged_tokens} rows"
            )
        return self.device_rows[slot, :num_tokens]


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None
        # Set by enable_static_buffer (--routing-replay-static-buffer); None = the per-layer pop path.
        self.static_buffer: StaticReplayBuffer | None = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        replay = Replay(stream_idx=stream_idx)
        replay.static_slot = len(self.replays)
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()
        if self.static_buffer is not None:
            self.static_buffer.reset_rollout()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()
        if self.static_buffer is not None:
            self.static_buffer.next_index = 0
            self.static_buffer.staged_tokens = None

    def enable_static_buffer(self, capacity: int, device=None, pin_memory: bool = True) -> None:
        """Read replayed routes from resident device rows (see StaticReplayBuffer)."""
        self.static_buffer = StaticReplayBuffer(capacity, device=device, pin_memory=pin_memory)

    def _build_host_stacks(self) -> list[torch.Tensor]:
        lists = [replay.top_indices_list for replay in self.replays]
        num_micro_batches = len(lists[0]) if lists else 0
        if any(len(lst) != num_micro_batches for lst in lists):
            raise AssertionError(
                f"rank {_get_rank()}: replay layers recorded different micro batch counts "
                f"{[len(lst) for lst in lists]}"
            )
        stacks = []
        for i in range(num_micro_batches):
            first = lists[0][i]
            stack = torch.empty(
                (len(lists), *first.shape), dtype=first.dtype, pin_memory=self.static_buffer.pin_memory
            )
            for slot, lst in enumerate(lists):
                stack[slot].copy_(lst[i])
            stacks.append(stack)
        return stacks

    def stage_static_micro_batch(self) -> None:
        """Copy the next micro batch's routes of every layer into the device rows (one async copy)."""
        buf = self.static_buffer
        if buf.host_stacks is None:
            buf.host_stacks = self._build_host_stacks()
        if buf.next_index >= len(buf.host_stacks):
            raise IndexError(
                f"rank {_get_rank()}: static replay staging index {buf.next_index} out of range "
                f"({len(buf.host_stacks)} recorded micro batches)"
            )
        host = buf.host_stacks[buf.next_index]
        buf.next_index += 1
        slots, num_tokens, topk = host.shape
        if num_tokens > buf.capacity:
            raise RuntimeError(
                f"rank {_get_rank()}: micro batch has {num_tokens} routed rows, more than the static replay "
                f"buffer capacity {buf.capacity}; raise the capacity (it is never reallocated, a captured "
                f"CUDA graph keeps its address)"
            )
        if buf.device_rows is None:
            device = torch.cuda.current_device() if buf.device is None else buf.device
            buf.device_rows = torch.full((slots, buf.capacity, topk), -1, dtype=host.dtype, device=device)
        elif buf.device_rows.shape[0] != slots or buf.device_rows.shape[2] != topk or buf.device_rows.dtype != host.dtype:
            raise AssertionError(
                f"rank {_get_rank()}: staged routes {tuple(host.shape)} {host.dtype} do not fit the static rows "
                f"{tuple(buf.device_rows.shape)} {buf.device_rows.dtype}"
            )
        buf.device_rows[:, :num_tokens].copy_(host, non_blocking=True)
        buf.staged_tokens = num_tokens

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, spread_padding_routes=False, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # fill padding tokens with arange to avoid invalid reading; the sync-free path
            # fills unconditionally (a no-op for rows with a valid pick) instead of reading
            # all_invalid.any() on the host.  With THD buckets the (possibly many) padding
            # rows are spread over all experts instead (also without a host read).
            all_invalid = (top_indices == -1).all(dim=-1)
            if spread_padding_routes:
                top_indices = torch.where(
                    all_invalid.unsqueeze(-1), _spread_padding_routes(top_indices, scores.shape[1]), top_indices
                )
            elif sync_free.enabled() or all_invalid.any():
                top_indices = fill_all_invalid_rows(top_indices, all_invalid, scores.shape[1])

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage in ("replay_forward", "replay_backward") and manager.static_buffer is not None:
                # Forward and backward recompute of one micro batch read the same staged rows.
                return _get_replay_result(
                    manager.static_buffer.rows(replay.static_slot, scores.shape[0]), scores, topk, *args,
                    spread_padding_routes=getattr(replay, "spread_padding_routes", False), **kwargs,
                )

            elif stage == "replay_forward":
                return _get_replay_result(
                    replay.pop_forward(), scores, topk, *args,
                    spread_padding_routes=getattr(replay, "spread_padding_routes", False), **kwargs,
                )

            elif stage == "replay_backward":
                return _get_replay_result(
                    replay.pop_backward(), scores, topk, *args,
                    spread_padding_routes=getattr(replay, "spread_padding_routes", False), **kwargs,
                )

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None):
        if not self.enabled:
            return
        replay = self.create_replay(stream_idx=stream_idx)
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        logger.info(
            f"Replay check (rank {_get_rank()}, stage {self.stage}): "
            f"mismatch {mismatch_count}/{orig_flat.shape[0]} tokens, threshold {mismatch_threshold:.0f}"
        )
        if mismatch_count == 0:
            return

        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]


def reject_router_fusion_with_routing_replay(*, router_fusion: bool, args) -> None:
    """Raise if the MoE router is fused while routing replay is on.

    Megatron's fused top-k (``topk_routing_with_score_function(fused=True)``, from
    ``moe_router_fusion``) returns before ``routing_replay_manager``'s top-k hook. A replayed run
    (``--use-routing-replay``, or R3 ``--use-rollout-routing-replay``) would then route by the
    training logits and not replay anything, without an error.
    """
    replay = getattr(args, "use_routing_replay", False) or getattr(args, "use_rollout_routing_replay", False)
    if router_fusion and replay:
        raise ValueError(
            "--moe-router-fusion cannot be used with routing replay (--use-routing-replay / "
            "--use-rollout-routing-replay): Megatron's fused router top-k returns before the replay hook, "
            "so the replayed routes would be skipped silently. Remove --moe-router-fusion or turn routing "
            "replay off."
        )


def stage_static_replays() -> None:
    """Before a forward: stage the next micro batch for every manager that replays from static rows.

    Called by the Megatron forward steps (log-prob and train) right before the model call; a no-op
    unless --routing-replay-static-buffer is on and the manager is in a replay stage."""
    for m in all_replay_managers:
        if m.enabled and m.static_buffer is not None and m.stage in ("replay_forward", "replay_backward"):
            m.stage_static_micro_batch()


def static_replay_capacity(args) -> int:
    """Rows of the static replay buffer: the largest packed micro batch the train or log-prob pass can
    build (max tokens per GPU and the largest THD bucket, rounded up to the pad size), or
    --routing-replay-static-capacity when set."""
    explicit = getattr(args, "routing_replay_static_capacity", None)
    if explicit:
        return int(explicit)
    pad = getattr(args, "tensor_model_parallel_size", 1) * getattr(args, "data_pad_size_multiplier", 128)
    biggest = max(
        getattr(args, "max_tokens_per_gpu", None) or 0,
        getattr(args, "log_probs_max_tokens_per_gpu", None) or 0,
        max(getattr(args, "thd_bucket_sizes", None) or [0]),
    )
    if biggest <= 0:
        raise ValueError("--routing-replay-static-buffer needs --max-tokens-per-gpu or --routing-replay-static-capacity")
    return -(-biggest // pad) * pad


def validate_static_replay_args(args) -> None:
    """--routing-replay-static-buffer: one micro batch in flight between its forward and backward."""
    if not getattr(args, "routing_replay_static_buffer", False):
        return
    if not (getattr(args, "use_routing_replay", False) or getattr(args, "use_rollout_routing_replay", False)):
        raise ValueError("--routing-replay-static-buffer needs routing replay (--use-rollout-routing-replay)")
    if getattr(args, "pipeline_model_parallel_size", 1) != 1 or getattr(args, "virtual_pipeline_model_parallel_size", None):
        raise ValueError(
            "--routing-replay-static-buffer needs pipeline parallel size 1: with pipelining several micro batches "
            "are between their forward and backward, and they would share one set of static rows"
        )
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        raise ValueError("--routing-replay-static-buffer does not support --overlap-moe-expert-parallel-comm")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("--routing-replay-static-buffer is only implemented for --qkv-format thd")
