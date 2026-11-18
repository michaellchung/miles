"""The DSA top-k recompute cache: which pass stores and which reuses, FIFO pairing by key identity, bounded size."""

import importlib.util
import logging
from pathlib import Path

import pytest
import torch

# Load the module file directly: the package's model module imports Megatron, which CPU CI does not install.
_ROOT = Path(__file__).resolve().parents[5]
_SPEC = importlib.util.spec_from_file_location(
    "_indexer_recompute_cache_under_test",
    _ROOT / "miles_plugins" / "models" / "kimi_k31" / "indexer_recompute_cache.py",
)
irc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(irc)


@pytest.mark.parametrize(
    "checkpointing,grad,phase",
    [
        (False, False, irc.Phase.PLAIN),
        (False, True, irc.Phase.PLAIN),
        (True, False, irc.Phase.STORE),
        (True, True, irc.Phase.REUSE),
    ],
)
def test_phase(checkpointing, grad, phase):
    # forward-only passes (no grad, Megatron skips the checkpoint) and layers that are not recomputed are plain
    assert irc.recompute_phase(checkpointing, grad) is phase


def _key(n):
    return torch.tensor([0, n], dtype=torch.int32)


def test_take_pairs_by_identity_in_fifo_order():
    cache = irc.TopkRecomputeCache(max_pending=3)
    a, b = _key(4), _key(8)
    cache.put(a, torch.zeros(1))
    cache.put(b, torch.ones(1))
    assert cache.take(_key(4)) is None  # equal values, another microbatch's tensor: no match
    assert len(cache) == 2
    assert torch.equal(cache.take(a), torch.zeros(1)) and len(cache) == 1
    assert torch.equal(cache.take(b), torch.ones(1)) and len(cache) == 0
    assert cache.take(a) is None


def test_same_key_twice_is_taken_oldest_first():
    cache = irc.TopkRecomputeCache(max_pending=3)
    key = _key(4)
    cache.put(key, torch.zeros(1))
    cache.put(key, torch.ones(1))
    assert torch.equal(cache.take(key), torch.zeros(1))
    assert torch.equal(cache.take(key), torch.ones(1))


def test_stale_entries_before_the_match_are_dropped(caplog):
    cache = irc.TopkRecomputeCache(max_pending=4)
    stale, live, later = _key(1), _key(2), _key(3)
    for key in (stale, live, later):
        cache.put(key, torch.zeros(1))
    with caplog.at_level(logging.WARNING, logger=irc.logger.name):
        assert cache.take(live) is not None
    assert len(cache) == 1 and "never ran" in caplog.text
    assert cache.take(stale) is None and cache.take(later) is not None


def test_the_cache_never_holds_more_than_its_bound(caplog):
    cache = irc.TopkRecomputeCache(max_pending=2)
    keys = [_key(i) for i in range(3)]
    with caplog.at_level(logging.WARNING, logger=irc.logger.name):
        for key in keys:
            cache.put(key, torch.zeros(1))
    assert len(cache) == 2 and "dropped the oldest" in caplog.text
    assert cache.take(keys[0]) is None  # its recompute scores the indexer again
    assert cache.take(keys[1]) is not None and cache.take(keys[2]) is not None
    cache.put(keys[0], torch.zeros(1))
    cache.clear()
    assert len(cache) == 0


def test_switch(monkeypatch):
    monkeypatch.delenv(irc.CACHE_ENV, raising=False)
    assert not irc.recompute_cache_enabled()
    monkeypatch.setenv(irc.CACHE_ENV, "1")
    assert irc.recompute_cache_enabled()
