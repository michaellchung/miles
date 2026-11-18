"""MILES_SC_TOPK_PREP / MILES_SC_LOGPROB_LEAN / MILES_CP_RESPONSE_VIEW reproduce the score-centering loss bit for bit.

TP=2, CP=2 on CPU/Gloo: the loss, every metric and the logits gradient with each switch (and all of them)
against the default path, compared as bits.
"""

import importlib
import itertools
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.data import rollout_topk_prep
from miles.backends.training_utils.data.context_parallel import (
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
    slice_with_cp,
)
from miles.backends.training_utils.loss.hub.logit_processors import _merge_adjacent_ranges
from miles.backends.training_utils.loss.hub.score_centering import selected_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.score_centering.selected_log_probs import LEAN_ENV, lean_mode
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=["precision"])

SWITCHES = (rollout_topk_prep.ENV, LEAN_ENV, "MILES_CP_RESPONSE_VIEW")
# (prep, lean, view) settings compared with the default path
VARIANTS = (
    ("1", "0", "0"),
    ("0", "1", "0"),
    ("0", "recompute", "0"),
    ("0", "0", "1"),
    ("1", "1", "1"),
    ("1", "recompute", "1"),
)
TOTALS, RESPONSES, K, MAX_SEQ_LEN = (23, 41, 9, 6), (17, 30, 0, 4), 4, 44


def _bits(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if value.dtype in (torch.float32, torch.int32):
        return value.contiguous().view(torch.int32)
    if value.dtype in (torch.bfloat16, torch.float16):
        return value.contiguous().view(torch.int16)
    return value


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    assert torch.equal(_bits(actual), _bits(expected)), f"{what}: max diff {(actual - expected).abs().max()}"


def _case(tp: GroupInfo, cp: GroupInfo, *, qkv_format, mode, dtype, vocab_size, temperature, entropy, seed):
    generator = torch.Generator().manual_seed(seed)
    args = Namespace(
        log_probs_backend="torch",
        loss_type="score_centering",
        qkv_format=qkv_format,
        allgather_cp=False,
        true_on_policy_mode=False,
        rollout_temperature=temperature,
        log_probs_chunk_size=3,
        train_backend="megatron",
        backend=Namespace(vocab_size=vocab_size, calculate_per_token_loss=False),
        score_centering_is=mode,
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        entropy_coef=0.03 if entropy else 0,
        observe_training_entropy=entropy,
        use_kl_loss=False,
        use_sampling_support_replay=False,
        use_unbiased_kl=False,
    )
    max_seq_lens = [MAX_SEQ_LEN] * len(TOTALS) if qkv_format == "bshd" else None
    parts = [torch.randn(total, 8, generator=generator) * 3 for total in TOTALS]
    tokens = [torch.randint(0, vocab_size, (total,), generator=generator) for total in TOTALS]
    masks = [(torch.rand(response, generator=generator) > 0.2).int() for response in RESPONSES]
    advantages = [torch.randn(response, generator=generator) for response in RESPONSES]
    advantages[1][3] = 0  # an inactive token
    candidates = [
        (
            torch.stack([torch.randperm(vocab_size, generator=generator)[:K] for _ in range(response)])
            if response
            else torch.zeros(0, K, dtype=torch.long)
        )
        for response in RESPONSES
    ]
    if RESPONSES[0]:
        candidates[0][1, 2:] = -1  # padded slots
    head_logps = [
        torch.randn(response, K, generator=generator).log_softmax(-1).masked_fill(c < 0, -torch.inf)
        for response, c in zip(RESPONSES, candidates, strict=True)
    ]
    rollout_logps = [torch.randn(response, generator=generator) - 2 for response in RESPONSES]

    def sliced(values):
        return [
            slice_log_prob_with_cp(v, t, r, qkv_format, MAX_SEQ_LEN)
            for v, t, r in zip(values, TOTALS, RESPONSES, strict=True)
        ]

    pieces = [slice_with_cp(part, 0, qkv_format, MAX_SEQ_LEN) for part in parts]
    full_local = torch.stack(pieces) if qkv_format == "bshd" else torch.cat(pieces).unsqueeze(0)
    base_batch = dict(
        total_lengths=list(TOTALS),
        response_lengths=list(RESPONSES),
        unconcat_tokens=tokens,
        loss_masks=masks,
        max_seq_lens=max_seq_lens,
        rollout_topk_token_ids=[c.numpy().astype(np.int32) for c in candidates],
        rollout_topk_log_probs=[h.numpy().astype(np.float32) for h in head_logps],
        rollout_log_probs=sliced(rollout_logps),
        advantages=sliced(advantages),
        ref_log_probs=None,
    )
    local_logits = full_local[..., tp.rank * 4 : (tp.rank + 1) * 4].to(dtype)
    return args, base_batch, local_logits


def _run(args, base_batch, local_logits, switches: tuple[str, str, str]):
    import os

    for name, value in zip(SWITCHES, switches, strict=True):
        os.environ[name] = value
    batch = dict(base_batch)
    if switches[0] == "1":
        assert rollout_topk_prep.enabled(args)
        rollout_data = dict(base_batch, tokens=base_batch["unconcat_tokens"])
        rollout_topk_prep.prepare(args, rollout_data)
        batch[rollout_topk_prep.LOCAL_IDS] = rollout_data[rollout_topk_prep.LOCAL_IDS]
        batch[rollout_topk_prep.LOCAL_LOG_PROBS] = rollout_data[rollout_topk_prep.LOCAL_LOG_PROBS]
        rollout_topk_prep.prefetch(batch, device=local_logits.device)
        assert rollout_topk_prep.PREFETCHED in batch
    logits = local_logits.clone().requires_grad_()
    reduce = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        qkv_format=args.qkv_format,
        max_seq_lens=batch["max_seq_lens"],
    )
    loss, metrics = score_centering_loss_function(args, batch, logits, reduce)
    loss.backward()
    for name in SWITCHES:
        os.environ[name] = "0"
    return loss.detach(), {k: v.detach() for k, v in metrics.items()}, logits.grad


def _check_all(tp: GroupInfo, cp: GroupInfo) -> None:
    grid = itertools.product(
        ("thd", "bshd"), ("tis", "none"), (torch.bfloat16, torch.float32), (8, 7), (1.0, 0.7), (False, True)
    )
    for seed, (qkv_format, mode, dtype, vocab_size, temperature, entropy) in enumerate(grid):
        case = _case(
            tp,
            cp,
            qkv_format=qkv_format,
            mode=mode,
            dtype=dtype,
            vocab_size=vocab_size,
            temperature=temperature,
            entropy=entropy,
            seed=seed,
        )
        ref_loss, ref_metrics, ref_grad = _run(*case, ("0", "0", "0"))
        assert torch.isfinite(ref_loss) and torch.isfinite(ref_grad.float()).all()
        assert ref_grad.abs().sum() > 0
        for switches in VARIANTS:
            what = f"{qkv_format} {mode} {dtype} V={vocab_size} T={temperature} H={entropy} {switches}"
            loss, metrics, grad = _run(*case, switches)
            _assert_bitwise(loss, ref_loss, f"loss {what}")
            assert metrics.keys() == ref_metrics.keys(), what
            for key in ref_metrics:
                _assert_bitwise(metrics[key], ref_metrics[key], f"{key} {what}")
            _assert_bitwise(grad, ref_grad, f"grad {what}")


def _check_lean_selected(tp: GroupInfo) -> None:
    """The lean op on its own, entropy and padding included, against the reference op, as bits."""
    import os

    generator = torch.Generator().manual_seed(5)
    for dtype, vocab_size, with_entropy, mode in itertools.product(
        (torch.bfloat16, torch.float32), (8, 7), (False, True), ("1", "recompute")
    ):
        logits = (torch.randn(11, 4, generator=generator) * 4).to(dtype)
        ids = torch.randint(-1, vocab_size, (11, 6), generator=generator)
        weights = torch.randn(11, 6, generator=generator)
        results = []
        for setting in ("0", mode):
            os.environ[LEAN_ENV] = setting
            leaf = logits.clone().requires_grad_()
            selected, entropy = selected_log_probs_and_entropy(
                leaf, ids, group=tp.group, vocab_size=vocab_size, chunk_size=4, with_entropy=with_entropy
            )
            ((selected * weights).sum() + 0.3 * entropy.sum()).backward()
            results.append((selected.detach(), entropy.detach(), leaf.grad))
        os.environ[LEAN_ENV] = "0"
        for actual, expected, name in zip(results[1], results[0], ("selected", "entropy", "grad"), strict=True):
            _assert_bitwise(actual, expected, f"{name} {dtype} V={vocab_size} H={with_entropy} {mode}")


def _count_lean_calls() -> list[int]:
    # the package re-exports a function of the module's name
    module = importlib.import_module("miles.backends.training_utils.loss.hub.score_centering.selected_log_probs")

    calls = [0]
    apply = module._LeanSelectedLogProbs.apply

    def counted(*args):
        calls[0] += 1
        return apply(*args)

    module._LeanSelectedLogProbs.apply = counted
    return calls


def _worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        tp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        cp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        tp = GroupInfo(rank=rank % 2, size=2, group=tp_groups[rank // 2])
        cp = GroupInfo(rank=rank // 2, size=2, group=cp_groups[rank % 2])
        singleton = GroupInfo(rank=0, size=1, group=None)
        state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "pp", "ep", "etp", "indep_dp")}
        set_parallel_state(ParallelState(**state, tp=tp, cp=cp))
        calls = _count_lean_calls()
        _check_lean_selected(tp)
        _check_all(tp, cp)
        assert calls[0] > 0, "the lean op never ran"
    finally:
        dist.destroy_process_group()


def test_switches_reproduce_the_loss_bitwise(tmp_path: Path) -> None:
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_local_response_ranges_match_slice_log_prob_with_cp(cp_size: int) -> None:
    singleton = GroupInfo(rank=0, size=1, group=None)
    state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "pp", "ep", "etp", "indep_dp", "tp")}
    for cp_rank, total, qkv_format in itertools.product(range(cp_size), range(1, 40), ("thd", "bshd")):
        set_parallel_state(ParallelState(**state, cp=GroupInfo(rank=cp_rank, size=cp_size, group=None)))
        for response in range(0, total):
            values = torch.arange(response * 2).view(response, 2)
            expected = slice_log_prob_with_cp(values, total, response, qkv_format, 48)
            ranges = rollout_topk_prep.local_response_ranges(total, response, qkv_format, 48)
            actual = torch.cat([values[start:end] for start, end in ranges]) if ranges else values[:0]
            assert torch.equal(actual, expected), (cp_rank, total, response, qkv_format, ranges)


def test_prepare_refuses_out_of_vocab_ids() -> None:
    singleton = GroupInfo(rank=0, size=1, group=None)
    state = {name: singleton for name in ("intra_dp", "intra_dp_cp", "pp", "ep", "etp", "indep_dp", "tp", "cp")}
    set_parallel_state(ParallelState(**state))
    args = Namespace(backend=Namespace(vocab_size=8), qkv_format="thd")
    good = dict(
        tokens=[torch.tensor([1, 2, 3, 4])],
        total_lengths=[4],
        response_lengths=[2],
        rollout_topk_token_ids=[np.array([[1, -1], [7, 0]], dtype=np.int32)],
        rollout_topk_log_probs=[np.zeros((2, 2), dtype=np.float32)],
    )
    rollout_topk_prep.prepare(args, dict(good))
    for key, bad in (
        ("rollout_topk_token_ids", [np.array([[1, 8], [7, 0]], dtype=np.int32)]),
        ("rollout_topk_token_ids", [np.array([[1, -2], [7, 0]], dtype=np.int32)]),
        ("tokens", [torch.tensor([1, 2, 9, 4])]),
    ):
        with pytest.raises(ValueError, match="outside the model vocabulary"):
            rollout_topk_prep.prepare(args, dict(good, **{key: bad}))
    # a prompt token is not scored, as in the device check
    rollout_topk_prep.prepare(args, dict(good, tokens=[torch.tensor([9, 2, 3, 4])]))


def test_merge_adjacent_ranges() -> None:
    assert _merge_adjacent_ranges(((3, 8), (8, 12))) == [(3, 12)]
    assert _merge_adjacent_ranges(((3, 8), (9, 12))) == [(3, 8), (9, 12)]
    assert _merge_adjacent_ranges(((3, 3), (5, 9))) == [(5, 9)]
    assert _merge_adjacent_ranges(((4, 4), (4, 4))) == [(4, 4)]


def test_lean_mode_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value, expected in (("0", None), ("1", "1"), ("recompute", "recompute")):
        monkeypatch.setenv(LEAN_ENV, value)
        assert lean_mode() == expected
    monkeypatch.setenv(LEAN_ENV, "yes")
    with pytest.raises(ValueError):
        lean_mode()
