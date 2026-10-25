"""CPU tests: THD bucket padding (--thd-bucket-sizes) gives every micro batch of a bucket one CUDA graph
signature, keeps the default padding unchanged, pads routing-replay rows to the same length, and spreads
the dummy routes of padding rows over all experts."""

import argparse

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import replay_data as replay_data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.backends.training_utils.thd_buckets import (
    bucket_target,
    pad_cu_seqlens,
    parse_bucket_sizes,
    thd_bucket_kwargs,
)
from miles.utils.replay_base import BaseReplayManager, Replay, _spread_padding_routes

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

PAD_MULT, TP = 7, 2  # pad size 14
BUCKETS = [56, 112]


def _state() -> ParallelState:
    group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=group,
        intra_dp_cp=group,
        cp=group,
        tp=GroupInfo(rank=0, size=TP, group=None),
        pp=group,
        ep=group,
        etp=group,
        indep_dp=group,
    )


@pytest.fixture
def cpu_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    state = _state()
    for module in (data_utils, cp_utils, replay_data_utils):
        monkeypatch.setattr(module, "get_parallel_state", lambda state=state: state)


def _rollout(lengths: list[int]) -> dict:
    tokens = [torch.arange(1000 * i + 1, 1000 * i + n + 1) for i, n in enumerate(lengths)]
    masks = [torch.ones(max(1, n // 2), dtype=torch.int) for n in lengths]
    return {
        "tokens": tokens,
        "loss_masks": masks,
        "total_lengths": list(lengths),
        "response_lengths": [len(m) for m in masks],
        "max_seq_lens": [max(lengths)] * len(lengths),
    }


def _batch(rollout: dict, **kwargs) -> dict:
    return data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=len(rollout["tokens"])),
        list(rollout),
        pad_multiplier=PAD_MULT,
        qkv_format="thd",
        **kwargs,
    )


def test_bucket_helpers() -> None:
    assert parse_bucket_sizes([112, 56, 56], 14) == [56, 112]
    assert parse_bucket_sizes(None, 14) is None
    with pytest.raises(ValueError, match="multiple of the pad size"):
        parse_bucket_sizes([50], 14)
    assert bucket_target(46, [56, 112], 14) == (56, True)
    assert bucket_target(56, [56, 112], 14) == (56, True)
    assert bucket_target(113, [56, 112], 14) == (126, False)  # above the largest bucket: old rule
    assert bucket_target(46, None, 14) == (56, False)  # no buckets: next multiple of 14
    assert pad_cu_seqlens([0, 5, 9], 12, None) == ([0, 5, 9, 12], False)
    assert pad_cu_seqlens([0, 5, 12], 12, 6) == ([0, 5, 12, 12, 12, 12], True)
    assert pad_cu_seqlens([0, 5, 9], 12, 3) == ([0, 5, 9, 12], False)  # too many sequences for 3 entries
    args = argparse.Namespace(thd_bucket_sizes=[56], thd_max_packed_sequences=8)
    assert thd_bucket_kwargs(args) == {"thd_bucket_sizes": [56], "thd_max_cu_entries": 9}
    assert thd_bucket_kwargs(argparse.Namespace()) == {}


def test_default_padding_is_unchanged(cpu_state) -> None:
    rollout = _rollout([13, 11, 9])  # 33 tokens -> 42
    old = _batch(rollout)
    new = _batch(rollout, thd_bucket_sizes=None, thd_max_cu_entries=None)
    assert torch.equal(old["tokens"], new["tokens"])
    assert old["tokens"].shape == (1, 42)
    assert old["cu_seqlens"].tolist() == [0, 13, 24, 33, 42] == new["cu_seqlens"].tolist()
    assert old["max_seqlen"] == new["max_seqlen"] == 13


def test_micro_batches_of_one_bucket_share_one_signature(cpu_state) -> None:
    sigs = []
    for lengths in ([13, 11, 9], [20, 20, 3, 2], [7]):
        batch = _batch(_rollout(lengths), thd_bucket_sizes=BUCKETS, thd_max_cu_entries=8)
        real = sum(lengths)
        tokens = batch["tokens"].flatten()
        assert tokens.shape == (56,)
        assert torch.all(tokens[real:] == 0)
        cu = batch["cu_seqlens"].tolist()
        assert len(cu) == 8 and cu[-1] == 56 and cu[len(lengths)] == real
        assert batch["max_seqlen"] == 56
        assert torch.all(batch["input_loss_masks"].flatten()[real:] == 0)
        sigs.append((tuple(batch["tokens"].shape), tuple(batch["cu_seqlens"].shape), batch["max_seqlen"]))
    assert len(set(sigs)) == 1


def test_overflow_keeps_the_old_shape(cpu_state) -> None:
    # 120 tokens > largest bucket 112: next multiple of 14, real max_seqlen, own entry count
    batch = _batch(_rollout([60, 60]), thd_bucket_sizes=BUCKETS, thd_max_cu_entries=8)
    assert batch["tokens"].shape == (1, 126)
    assert batch["cu_seqlens"].tolist() == [0, 60, 120, 126]
    assert batch["max_seqlen"] == 60
    # 9 sequences do not fit 8 cu_seqlens entries: bucket length, but real entries and max_seqlen
    batch = _batch(_rollout([3] * 9), thd_bucket_sizes=BUCKETS, thd_max_cu_entries=8)
    assert batch["tokens"].shape == (1, 56)
    assert len(batch["cu_seqlens"]) == 11
    assert batch["max_seqlen"] == 29


class _Recorder:
    def __init__(self):
        self.shapes = []

    def __call__(self, replay_list, replay_data, **kwargs):
        self.shapes.append(tuple(replay_data.shape))


def test_replay_rows_are_padded_like_the_tokens(cpu_state) -> None:
    lengths = [13, 11, 9]
    rollout = _rollout(lengths)
    rollout["rollout_routed_experts"] = [torch.zeros(n - 1, 3, 2, dtype=torch.int32) for n in lengths]
    args = argparse.Namespace(
        qkv_format="thd", data_pad_size_multiplier=PAD_MULT, allgather_cp=False, sequence_parallel=False,
        thd_bucket_sizes=BUCKETS,
    )
    replay_list = [Replay(), Replay()]
    recorder = _Recorder()
    replay_data_utils.fill_replay_data(
        args=args, models=None,
        data_iterator=[data_utils.DataIterator(rollout, micro_batch_size=len(lengths))],
        num_microbatches=[1], rollout_data=rollout, data_key="rollout_routed_experts",
        replay_list=replay_list, register_replay_list_func=recorder,
    )
    assert recorder.shapes == [(56, 3, 2)]
    assert all(r.spread_padding_routes for r in replay_list)


def test_spread_padding_routes_cover_all_experts() -> None:
    routes = _spread_padding_routes(torch.full((64, 6), -1, dtype=torch.int64), 128)
    assert all(len(set(row.tolist())) == 6 for row in routes)
    counts = torch.bincount(routes.flatten(), minlength=128)
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("spread", [False, True])
def test_replay_keeps_real_rows_and_fills_padding_rows(monkeypatch: pytest.MonkeyPatch, spread: bool) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    manager = BaseReplayManager()
    manager.enabled, manager.stage = True, "replay_forward"
    manager.enable_check_replay_result = False
    replay = Replay()
    replay.spread_padding_routes = spread
    real = torch.tensor([[5, 9], [1, 2]])
    # Replay.record keeps a pinned host copy (needs CUDA); append the host tensor directly
    replay.top_indices_list.append(torch.cat([real, torch.full((3, 2), -1, dtype=real.dtype)]))
    manager.set_current(replay)
    topk_fn = manager.get_topk_fn(old_topk_fn=None, return_probs=False)
    out = topk_fn(torch.rand(5, 16), 2)
    assert torch.equal(out[:2], real)
    pad = out[2:]
    if spread:
        assert pad.tolist() == [[4, 5], [6, 7], [8, 9]]  # row r: (r * 2 + j) % 16
    else:
        assert pad.tolist() == [[0, 1]] * 3
