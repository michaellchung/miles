"""R3 static device rows (--routing-replay-static-buffer): same routes as the per-layer pop path (CPU)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.utils import replay_base
from miles.utils.replay_base import BaseReplayManager, static_replay_capacity, validate_static_replay_args


@pytest.fixture(autouse=True)
def _host_only(monkeypatch):
    # the pop path copies to the current CUDA device; these tests run on the host
    monkeypatch.setattr(replay_base, "_to_current_device", lambda t: t)


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices


def _manager(num_layers, *, static_capacity=None):
    m = BaseReplayManager()
    m.enabled = True
    m.enable_check_replay_result = False
    for _ in range(num_layers):
        m.create_replay()
    if static_capacity is not None:
        m.enable_static_buffer(static_capacity, device="cpu", pin_memory=False)
    return m


def _micro_batches(num_layers, sizes, topk, num_experts, seed):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for n in sizes:
        layers = []
        for _ in range(num_layers):
            top = torch.rand(n, num_experts, generator=gen).argsort(dim=1)[:, :topk]
            top[n - max(1, n // 5) :] = -1  # trailing padding rows
            layers.append(top)
        out.append(layers)
    return out


def _record(m, micro_batches):
    for layers in micro_batches:
        for replay, top in zip(m.replays, layers, strict=True):
            replay.top_indices_list.append(top.clone())  # what Replay.record stores (host copy)


def _run_pass(m, micro_batches, num_experts, *, stage_static, train):
    """Forward of every layer, then (train) the recompute in the backward, per micro batch."""
    topk_fn = m.get_topk_fn(_topk, return_probs=True)
    outs = []
    for layers in micro_batches:
        n, topk = layers[0].shape
        scores = torch.rand(n, num_experts, generator=torch.Generator().manual_seed(n))
        m.stage = "replay_forward"
        if stage_static:
            m.stage_static_micro_batch()
        for replay in m.replays:
            m.set_current(replay)
            outs.append(topk_fn(scores, topk))
        if train:
            m.stage = "replay_backward"
            for replay in reversed(m.replays):
                m.set_current(replay)
                outs.append(topk_fn(scores, topk))
    return outs


@pytest.mark.parametrize("spread", [False, True])
def test_static_rows_match_the_pop_path(spread):
    layers, experts = 3, 16
    mbs = _micro_batches(layers, [64, 48, 64], topk=4, num_experts=experts, seed=1)
    ref, new = _manager(layers), _manager(layers, static_capacity=64)
    for m in (ref, new):
        _record(m, mbs)
        for r in m.replays:
            r.spread_padding_routes = spread
    # log-prob pass (forward only), then the train pass (forward + backward recompute)
    for train in (False, True):
        a = _run_pass(ref, mbs, experts, stage_static=False, train=train)
        b = _run_pass(new, mbs, experts, stage_static=True, train=train)
        for (pa, ia), (pb, ib) in zip(a, b, strict=True):
            assert torch.equal(ia, ib) and torch.equal(pa, pb)
        ref.clear_all_forward()
        new.clear_all_forward()


def test_one_bucket_reads_one_address_and_restages_after_clear():
    m = _manager(2, static_capacity=32)
    mbs = _micro_batches(2, [32, 32], topk=2, num_experts=8, seed=2)
    _record(m, mbs)
    ptrs = []
    for layers in mbs:
        m.stage = "replay_forward"
        m.stage_static_micro_batch()
        rows = m.static_buffer.rows(1, 32)
        ptrs.append(rows.data_ptr())
        assert torch.equal(rows, layers[1])
    assert ptrs[0] == ptrs[1]
    with pytest.raises(IndexError):
        m.stage_static_micro_batch()
    m.clear_all_forward()
    m.stage_static_micro_batch()
    assert torch.equal(m.static_buffer.rows(0, 32), mbs[0][0])
    m.clear_all()
    assert m.static_buffer.host_stacks is None and m.static_buffer.next_index == 0
    assert all(r.top_indices_list == [] for r in m.replays)


def test_capacity_and_token_count_are_checked():
    m = _manager(1, static_capacity=16)
    _record(m, _micro_batches(1, [32], topk=2, num_experts=8, seed=3))
    m.stage = "replay_forward"
    with pytest.raises(RuntimeError, match="capacity"):
        m.stage_static_micro_batch()
    m = _manager(1, static_capacity=64)
    _record(m, _micro_batches(1, [32], topk=2, num_experts=8, seed=3))
    m.stage_static_micro_batch()
    with pytest.raises(AssertionError, match="staged micro batch"):
        m.static_buffer.rows(0, 31)


def test_stage_helper_only_stages_replay_stages(monkeypatch):
    m = _manager(1, static_capacity=32)
    _record(m, _micro_batches(1, [32, 32], topk=2, num_experts=8, seed=4))
    monkeypatch.setattr(replay_base, "all_replay_managers", [m])
    m.stage = "record"
    replay_base.stage_static_replays()
    assert m.static_buffer.next_index == 0
    m.stage = "replay_forward"
    replay_base.stage_static_replays()
    assert m.static_buffer.next_index == 1
    m.enabled = False
    replay_base.stage_static_replays()
    assert m.static_buffer.next_index == 1


def _args(**kw):
    base = dict(
        routing_replay_static_buffer=True, use_rollout_routing_replay=True, use_routing_replay=True,
        pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
        overlap_moe_expert_parallel_comm=False, qkv_format="thd", tensor_model_parallel_size=1,
        data_pad_size_multiplier=128, max_tokens_per_gpu=8000, log_probs_max_tokens_per_gpu=None,
        thd_bucket_sizes=None, routing_replay_static_capacity=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_capacity_from_args():
    assert static_replay_capacity(_args()) == 8064
    assert static_replay_capacity(_args(thd_bucket_sizes=[4096, 8192])) == 8192
    assert static_replay_capacity(_args(log_probs_max_tokens_per_gpu=16384)) == 16384
    assert static_replay_capacity(_args(routing_replay_static_capacity=100)) == 100


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(use_rollout_routing_replay=False, use_routing_replay=False), "needs routing replay"),
        (dict(pipeline_model_parallel_size=2), "pipeline parallel size 1"),
        (dict(virtual_pipeline_model_parallel_size=2), "pipeline parallel size 1"),
        (dict(overlap_moe_expert_parallel_comm=True), "overlap-moe"),
        (dict(qkv_format="bshd"), "thd"),
    ],
)
def test_validation_rejects_unsupported_layouts(kw, msg):
    with pytest.raises(ValueError, match=msg):
        validate_static_replay_args(_args(**kw))
    validate_static_replay_args(_args())
    validate_static_replay_args(_args(routing_replay_static_buffer=False, pipeline_model_parallel_size=4))
