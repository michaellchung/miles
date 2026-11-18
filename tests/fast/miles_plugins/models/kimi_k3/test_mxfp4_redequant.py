"""Frozen MXFP4 experts re-dequantized in the backward give bitwise the gradients of keeping the BF16 copies.

TE needs CUDA, so the grouped linear here is a CPU stand-in with TE's saving contract (release_v2.17
``_GroupedLinear.forward``): the forward fetches ``_get_weight_tensors()``, runs one GEMM per expert, and
``save_for_backward``-s the very weight tensors it used (no input rows: the weights take no gradient); the backward
reads them for the dgrad GEMM. ``RedequantFrozenExperts`` is the production mixin, unchanged.
"""

import gc
import importlib.util
import logging
import weakref
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, mxfp4_row_width, pack_mxfp4_rows, quantize_mxfp4

# Load the module file directly: the package's model module imports Megatron, which CPU CI does not install.
_PATH = Path(__file__).resolve().parents[5] / "miles_plugins" / "models" / "kimi_k3" / "mxfp4_redequant.py"
_SPEC = importlib.util.spec_from_file_location("_mxfp4_redequant_under_test", _PATH)
redequant = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(redequant)


class _GroupedGemm(torch.autograd.Function):
    """TE's legacy grouped-linear path on CPU: save [inputs (None without wgrad), weights, None, biases]."""

    @staticmethod
    def forward(ctx, inp, m_splits, save_input, *weights):
        rows = torch.split(inp, m_splits)
        out = torch.cat([x @ w.t() for x, w in zip(rows, weights, strict=True)])
        n = len(weights)
        inputs = list(rows) if save_input else [None] * n
        ctx.save_for_backward(*inputs, *weights, *([None] * n))
        ctx.m_splits, ctx.n = m_splits, n
        return out

    @staticmethod
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        weights = saved[ctx.n : 2 * ctx.n]
        grads = torch.split(grad, ctx.m_splits)
        dgrad = torch.cat([g @ w for g, w in zip(grads, weights, strict=True)])
        return (dgrad, None, None, *([None] * ctx.n))


class _FakeTEGroupedLinear(torch.nn.Module):
    """``num_gemms`` uint8 ``weight{i}`` parameters and a forward that saves what ``_get_weight_tensors`` returns."""

    def __init__(self, rows, save_input=False, copy_weights=False):
        super().__init__()
        self.num_gemms = len(rows)
        self.save_input = save_input
        self.copy_weights = copy_weights  # a TE that saves its own copies, not the fetched tensors
        for i, packed in enumerate(rows):
            self.register_parameter(f"weight{i}", torch.nn.Parameter(packed.clone(), requires_grad=False))
        self.fetched = []

    def forward(self, inp, m_splits):
        weights = self._get_weight_tensors()
        self.fetched = [weakref.ref(w) for w in weights]
        if self.copy_weights:
            weights = [w.clone() for w in weights]
        return _GroupedGemm.apply(inp, m_splits, self.save_input, *weights)


class _Frozen(redequant.RedequantFrozenExperts, _FakeTEGroupedLinear):
    pass


def _rows(num, out_features, in_features, seed):
    g = torch.Generator().manual_seed(seed)
    encoded = [
        quantize_mxfp4(torch.randn(out_features, in_features, generator=g) * 0.05, MXFP4_GROUP_SIZE)
        for _ in range(num)
    ]
    return [pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE) for packed, scale in encoded]


M_SPLITS = [5, 0, 7, 3]  # an empty expert, like a cold one in a real microbatch
HIDDEN, WIDTH = 64, 96


def _experts(**kwargs):
    fc1 = _Frozen(_rows(len(M_SPLITS), 2 * WIDTH, HIDDEN, seed=1), **kwargs)
    fc2 = _Frozen(_rows(len(M_SPLITS), HIDDEN, WIDTH, seed=2), **kwargs)
    return fc1, fc2


def _mlp(fc1, fc2, x):
    h = fc1(x, M_SPLITS)
    gate, up = h.chunk(2, dim=-1)
    return fc2(torch.nn.functional.silu(gate) * up, M_SPLITS)


def _run(monkeypatch, enabled, wrap=None, **kwargs):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1" if enabled else "0")
    torch.manual_seed(0)
    proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False).bfloat16()  # a trainable layer before the experts
    fc1, fc2 = _experts(**kwargs)
    tokens = torch.randn(sum(M_SPLITS), HIDDEN).bfloat16().requires_grad_(True)

    def body(t):
        return _mlp(fc1, fc2, proj(t))

    out = wrap(body, tokens) if wrap is not None else body(tokens)
    gc.collect()
    alive = sum(ref() is not None for layer in (fc1, fc2) for ref in layer.fetched)
    out.backward(torch.randn(out.shape, generator=torch.Generator().manual_seed(3)).bfloat16())
    return out.detach(), tokens.grad, proj.weight.grad, alive


def _assert_bitwise(a, b):
    for x, y in zip(a[:3], b[:3], strict=True):
        assert x.dtype == y.dtype and torch.equal(x, y)


@pytest.mark.parametrize("lut", ["0", "1"])
def test_redequant_gradients_are_bitwise_equal_and_no_bf16_copy_survives(monkeypatch, lut):
    plain = _run(monkeypatch, enabled=False)
    monkeypatch.setenv(redequant.LUT_DEQUANT_ENV, lut)  # MILES_K31_MXFP4_LUT_DEQUANT: same bits, either way
    redone = _run(monkeypatch, enabled=True)
    _assert_bitwise(plain, redone)
    # off: TE's graph holds every dequantized expert of both projections until the backward; on: none
    assert plain[3] == 2 * len(M_SPLITS)
    assert redone[3] == 0


def test_other_saved_tensors_pass_through(monkeypatch):
    # a grouped linear that also saves its input rows (the trainable half's wgrad contract) still matches
    plain = _run(monkeypatch, enabled=False, save_input=True)
    _assert_bitwise(plain, _run(monkeypatch, enabled=True, save_input=True))


@pytest.mark.parametrize("reentrant", [True, False])
def test_under_activation_checkpointing(monkeypatch, reentrant):
    # full recompute (Megatron's reentrant CheckpointFunction) and torch's non-reentrant checkpoint around the experts
    def wrap(fn, x):
        return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=reentrant)

    plain = _run(monkeypatch, enabled=False)
    _assert_bitwise(plain, _run(monkeypatch, enabled=False, wrap=wrap))
    _assert_bitwise(plain, _run(monkeypatch, enabled=True, wrap=wrap))


def test_no_grad_forward_installs_no_hooks(monkeypatch):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1")
    fc1, _ = _experts()
    seen = []
    original = redequant.redequant_saved_weights
    monkeypatch.setattr(redequant, "redequant_saved_weights", lambda *a: seen.append(1) or original(*a))
    with torch.no_grad():
        fc1(torch.randn(sum(M_SPLITS), HIDDEN).bfloat16(), M_SPLITS)
    assert not seen and fc1._redequant_scope is None


def test_changed_bytes_between_forward_and_backward_are_refused(monkeypatch):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1")
    fc1, _ = _experts()
    x = torch.randn(sum(M_SPLITS), HIDDEN).bfloat16().requires_grad_(True)
    out = fc1(x, M_SPLITS)
    with torch.no_grad():
        fc1.weight0.add_(1)
    with pytest.raises(AssertionError, match="changed between its forward and its backward"):
        out.sum().backward()


def test_a_te_that_saves_other_tensors_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(redequant, "_reported", False)
    with caplog.at_level(logging.WARNING, logger=redequant.logger.name):
        plain = _run(monkeypatch, enabled=False, copy_weights=True)
        redone = _run(monkeypatch, enabled=True, copy_weights=True)
    _assert_bitwise(plain, redone)
    assert "saved none of the" in caplog.text


def test_packed_ref_is_what_the_graph_keeps(monkeypatch):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1")
    fc1, _ = _experts()
    x = torch.randn(sum(M_SPLITS), HIDDEN).bfloat16().requires_grad_(True)
    packed = []
    original = redequant.RedequantScope.pack

    def spy(self, tensor):
        result = original(self, tensor)
        packed.append(result)
        return result

    monkeypatch.setattr(redequant.RedequantScope, "pack", spy)
    fc1(x, M_SPLITS)
    refs = [p for p in packed if isinstance(p, redequant.PackedWeightRef)]
    assert len(refs) == len(M_SPLITS)
    width = mxfp4_row_width(HIDDEN, MXFP4_GROUP_SIZE)
    assert all(r.rows.dtype == torch.uint8 and r.rows.shape == (2 * WIDTH, width) for r in refs)
    assert all(r.shape == (2 * WIDTH, HIDDEN) and r.dtype == torch.bfloat16 for r in refs)


# ----------------------------------------------------------- MILES_K31_MXFP4_BATCHED_DEQUANT (all experts of a fetch)


@pytest.mark.parametrize("lut", ["0", "1"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("chunk", [None, "1", str(2 * WIDTH * HIDDEN * 2)])
def test_batched_dequant_gradients_are_bitwise_equal(monkeypatch, lut, enabled, chunk):
    monkeypatch.setenv(redequant.LUT_DEQUANT_ENV, lut)
    monkeypatch.setenv(redequant.BATCHED_DEQUANT_ENV, "0")
    plain = _run(monkeypatch, enabled=False)
    monkeypatch.setenv(redequant.BATCHED_DEQUANT_ENV, "1")
    if chunk is not None:
        monkeypatch.setenv(redequant.BATCHED_DEQUANT_CHUNK_ENV, chunk)
    batched = _run(monkeypatch, enabled=enabled)
    _assert_bitwise(plain, batched)
    assert batched[3] == (0 if enabled else 2 * len(M_SPLITS))


@pytest.mark.parametrize("reentrant", [True, False])
def test_batched_dequant_under_activation_checkpointing(monkeypatch, reentrant):
    def wrap(fn, x):
        return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=reentrant)

    plain = _run(monkeypatch, enabled=False)
    monkeypatch.setenv(redequant.BATCHED_DEQUANT_ENV, "1")
    _assert_bitwise(plain, _run(monkeypatch, enabled=False, wrap=wrap))
    _assert_bitwise(plain, _run(monkeypatch, enabled=True, wrap=wrap))


def test_batched_unpack_dequantizes_the_group_once(monkeypatch):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1")
    monkeypatch.setenv(redequant.BATCHED_DEQUANT_ENV, "1")
    fc1, _ = _experts()
    calls = []
    real = redequant.dequantize_expert_rows
    monkeypatch.setattr(
        redequant, "dequantize_expert_rows", lambda rows, *a: calls.append(len(rows)) or real(rows, *a)
    )
    x = torch.randn(sum(M_SPLITS), HIDDEN).bfloat16().requires_grad_(True)
    out = fc1(x, M_SPLITS)
    assert calls == [len(M_SPLITS)]  # the forward fetch: one batched call
    out.sum().backward()
    assert calls == [len(M_SPLITS), len(M_SPLITS)]  # the backward: one more for all four experts, not four


def test_batched_group_survives_a_second_read():
    rows = _rows(3, 2 * WIDTH, HIDDEN, seed=9)
    group = redequant._PackedGroup(rows, MXFP4_GROUP_SIZE)
    first = [group.take(i) for i in range(3)]
    assert group._weights is None  # every slice handed out: nothing cached
    again = group.take(1)  # a second unpack of the saved tensors re-dequantizes
    assert torch.equal(again.view(torch.int16), first[1].view(torch.int16))


def test_batched_changed_bytes_are_still_refused(monkeypatch):
    monkeypatch.setenv(redequant.REDEQUANT_ENV, "1")
    monkeypatch.setenv(redequant.BATCHED_DEQUANT_ENV, "1")
    fc1, _ = _experts()
    x = torch.randn(sum(M_SPLITS), HIDDEN).bfloat16().requires_grad_(True)
    out = fc1(x, M_SPLITS)
    with torch.no_grad():
        fc1.weight2.add_(1)
    with pytest.raises(AssertionError, match="changed between its forward and its backward"):
        out.sum().backward()
