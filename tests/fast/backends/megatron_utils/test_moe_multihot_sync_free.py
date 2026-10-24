"""The host-sync-free ``_indices_to_multihot`` gives Megatron's results bit for bit (CPU)."""

import importlib.util
import logging

import pytest

torch = pytest.importorskip("torch")

from miles.backends.megatron_utils import moe_multihot_sync_free as multihot  # noqa: E402
from miles.utils import sync_free  # noqa: E402

# _DeepepManager._indices_to_multihot verbatim from radixark/Megatron-LM a84b10547
# (megatron/core/transformer/moe/token_dispatcher.py:1401-1429), class indentation kept.
A84B105_METHOD_SOURCE = '''    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            A tuple of (routing_map, probs), where routing_map is the multihot vector
            and probs is the multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs
'''


def _manager_module(tmp_path, method_source, name):
    """Import a module with ``class _DeepepManager`` whose method has ``method_source``."""

    path = tmp_path / f"{name}.py"
    path.write_text("import torch\n\n\nclass _DeepepManager:\n    num_local_experts = 0\n\n" + method_source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def upstream(tmp_path):
    return _manager_module(tmp_path, A84B105_METHOD_SOURCE, "a84b105_dispatcher")._DeepepManager


def _routing(num_tokens, num_local_experts, topk, *, seed, all_invalid_rows=0):
    """DeepEP-shaped output: per row distinct local experts or -1 (routed to another rank)."""

    gen = torch.Generator().manual_seed(seed)
    picks = torch.rand(num_tokens, num_local_experts * 2, generator=gen).argsort(dim=1)[:, :topk]
    indices = torch.where(picks < num_local_experts, picks, torch.full_like(picks, -1))
    if all_invalid_rows:
        indices[torch.randperm(num_tokens, generator=gen)[:all_invalid_rows]] = -1
    probs = torch.rand(num_tokens, topk, generator=gen, dtype=torch.float32)
    probs[indices == -1] = torch.rand(int((indices == -1).sum()), generator=gen) * 7  # junk in masked slots
    return indices.long(), probs


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


CASES = [(0, 4, 2), (1, 1, 1), (7, 4, 4), (300, 16, 6), (1024, 16, 8), (513, 32, 6)]


def test_embedded_source_is_the_known_a84b105_source():
    assert sync_free.source_text_sha256(A84B105_METHOD_SOURCE) in multihot.KNOWN_SOURCE_SHA256


@pytest.mark.parametrize("num_tokens,num_local_experts,topk", CASES)
@pytest.mark.parametrize("all_invalid_rows", [0, 3])
def test_forward_and_backward_bitwise(upstream, num_tokens, num_local_experts, topk, all_invalid_rows):
    all_invalid_rows = min(all_invalid_rows, num_tokens)
    indices, probs = _routing(
        num_tokens, num_local_experts, topk, seed=num_tokens + topk, all_invalid_rows=all_invalid_rows
    )
    manager = upstream()
    manager.num_local_experts = num_local_experts

    ref_probs = probs.clone().requires_grad_(True)
    ref_map, ref_multihot = manager._indices_to_multihot(indices, ref_probs)
    new_probs = probs.clone().requires_grad_(True)
    new_map, new_multihot = multihot.indices_to_multihot(indices, new_probs, num_local_experts=num_local_experts)

    assert new_map.dtype == ref_map.dtype == torch.bool and new_map.shape == ref_map.shape
    assert new_multihot.dtype == ref_multihot.dtype == torch.float32 and new_multihot.shape == ref_multihot.shape
    assert new_map.is_contiguous() and new_multihot.is_contiguous()
    assert torch.equal(new_map, ref_map)
    assert torch.equal(_bits(new_multihot), _bits(ref_multihot))

    grad = torch.randn(ref_multihot.shape, generator=torch.Generator().manual_seed(1))
    ref_multihot.backward(grad)
    new_multihot.backward(grad)
    assert torch.equal(_bits(new_probs.grad), _bits(ref_probs.grad))


def test_bitwise_under_deterministic_algorithms(upstream):
    indices, probs = _routing(257, 16, 6, seed=3, all_invalid_rows=5)
    manager = upstream()
    manager.num_local_experts = 16
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        ref = manager._indices_to_multihot(indices, probs)
        new = multihot.indices_to_multihot(indices, probs, num_local_experts=16)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.equal(new[0], ref[0]) and torch.equal(_bits(new[1]), _bits(ref[1]))


def test_patch_replaces_the_known_method_once(upstream):
    original = upstream._indices_to_multihot
    assert multihot.patch_manager_class(upstream)
    patched = upstream._indices_to_multihot
    assert patched is not original and patched.__wrapped__ is original
    assert multihot.patch_manager_class(upstream)  # second call keeps the patched method
    assert upstream._indices_to_multihot is patched

    manager = upstream()
    manager.num_local_experts = 16
    indices, probs = _routing(64, 16, 6, seed=9, all_invalid_rows=2)
    ref = original(manager, indices, probs)
    new = manager._indices_to_multihot(indices, probs)
    assert torch.equal(new[0], ref[0]) and torch.equal(_bits(new[1]), _bits(ref[1]))


def test_patched_method_calls_the_original_for_other_inputs(upstream, monkeypatch):
    assert multihot.patch_manager_class(upstream)

    def fail(*_args, **_kwargs):
        raise AssertionError("sync-free path used")

    monkeypatch.setattr(multihot, "indices_to_multihot", fail)
    manager = upstream()
    manager.num_local_experts = 4
    indices, probs = _routing(5, 4, 2, seed=1)
    routing_map, _ = manager._indices_to_multihot(indices.int(), probs)  # int32: not DeepEP's dtype
    assert routing_map.shape == (5, 4)
    with pytest.raises(AssertionError, match="sync-free path used"):
        manager._indices_to_multihot(indices, probs)


def test_unknown_source_keeps_megatron_and_warns(tmp_path, caplog):
    changed = A84B105_METHOD_SOURCE.replace("mask = indices != -1", "mask = indices >= 0")
    cls = _manager_module(tmp_path, changed, "changed_dispatcher")._DeepepManager
    original = cls._indices_to_multihot
    with caplog.at_level(logging.WARNING, logger=multihot.__name__):
        assert not multihot.patch_manager_class(cls)
    assert cls._indices_to_multihot is original
    assert "does not match the Megatron source" in caplog.text


def test_missing_method_warns(caplog):
    class Other:
        pass

    with caplog.at_level(logging.WARNING, logger=multihot.__name__):
        assert not multihot.patch_manager_class(Other)
    assert "has no _indices_to_multihot" in caplog.text


def test_install_respects_the_switch(monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, "0")
    assert multihot.install() is False


def test_real_megatron_class_is_recognised_and_bitwise():
    """Runs where Megatron (a84b105) is importable, e.g. in the training image."""

    token_dispatcher = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")
    cls = token_dispatcher._DeepepManager
    original = getattr(cls._indices_to_multihot, "__wrapped__", cls._indices_to_multihot)
    assert sync_free.source_sha256(original) in multihot.KNOWN_SOURCE_SHA256
    assert multihot.patch_manager_class(cls)
    manager = object.__new__(cls)
    manager.num_local_experts = 16
    indices, probs = _routing(300, 16, 6, seed=5, all_invalid_rows=4)
    ref = original(manager, indices, probs)
    new = cls._indices_to_multihot(manager, indices, probs)
    assert torch.equal(new[0], ref[0]) and torch.equal(_bits(new[1]), _bits(ref[1]))
