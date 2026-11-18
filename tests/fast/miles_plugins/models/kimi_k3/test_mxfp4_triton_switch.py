"""MILES_K31_MXFP4_TRITON_DEQUANT: default off (the per-expert path, untouched); on, the frozen experts' fetches and the
redequant backward give bitwise the outputs and gradients of the fp32 chain (CPU: TRITON_INTERPRET=1)."""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported: the kernels must be interpreted on CPU

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_mxfp4_redequant_tests", _HERE / "test_mxfp4_redequant.py")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
redequant = base.redequant
mxfp4_triton = pytest.importorskip("miles.utils.mxfp4_triton")
if mxfp4_triton.triton is None:
    pytest.skip("triton is not installed", allow_module_level=True)
if not torch.cuda.is_available() and type(mxfp4_triton._dequant_group_kernel).__name__ != "InterpretedFunction":
    pytest.skip(
        "triton was imported before TRITON_INTERPRET=1 was set: run this file on its own", allow_module_level=True
    )


def _count_triton_calls(monkeypatch):
    calls = []
    original = redequant.dequantize_mxfp4_rows_triton
    monkeypatch.setattr(
        redequant, "dequantize_mxfp4_rows_triton", lambda rows, g: calls.append(len(rows)) or original(rows, g)
    )
    return calls


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(redequant.TRITON_DEQUANT_ENV, raising=False)
    calls = _count_triton_calls(monkeypatch)
    base._run(monkeypatch, enabled=True)
    assert not redequant.triton_dequant_enabled() and calls == []


@pytest.mark.parametrize("redo", [False, True])
def test_triton_fetch_is_bitwise_the_fp32_chain(monkeypatch, redo):
    monkeypatch.delenv(redequant.TRITON_DEQUANT_ENV, raising=False)
    plain = base._run(monkeypatch, enabled=False)
    monkeypatch.setenv(redequant.TRITON_DEQUANT_ENV, "1")
    calls = _count_triton_calls(monkeypatch)
    fast = base._run(monkeypatch, enabled=redo)
    base._assert_bitwise(plain, fast)
    n = len(base.M_SPLITS)
    # one call per projection and fetch: fc1 + fc2 forward; with redequant one more each in the backward (not per expert)
    assert calls == [n, n] + ([n, n] if redo else [])
    assert fast[3] == (0 if redo else 2 * n)


@pytest.mark.parametrize("reentrant", [True, False])
def test_triton_under_activation_checkpointing(monkeypatch, reentrant):
    def wrap(fn, x):
        return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=reentrant)

    monkeypatch.delenv(redequant.TRITON_DEQUANT_ENV, raising=False)
    plain = base._run(monkeypatch, enabled=False)
    monkeypatch.setenv(redequant.TRITON_DEQUANT_ENV, "1")
    base._assert_bitwise(plain, base._run(monkeypatch, enabled=True, wrap=wrap))


def test_a_second_read_of_the_saved_weights_dequantizes_again(monkeypatch):
    monkeypatch.setenv(redequant.TRITON_DEQUANT_ENV, "1")
    rows = base._rows(3, 8, 64, seed=5)
    group = redequant._FetchGroup(rows, 32)
    want = [redequant.dequantize_mxfp4_rows(r, 32) for r in rows]
    for i in (0, 1, 2, 1):  # expert 1 read twice (retain_graph-style): re-dequantized, same bits
        assert torch.equal(group.take(i).view(torch.int16), want[i].view(torch.int16))


def test_a2a_overlap_prefetch_goes_through_triton(monkeypatch):
    """moe_a2a_overlap's ``prefetch_weights`` uses the same entry: one Triton call per projection, the prefetched
    weights are the ones the next fetch takes, and they are bitwise the fp32 chain's."""
    monkeypatch.setenv(redequant.TRITON_DEQUANT_ENV, "1")
    calls = _count_triton_calls(monkeypatch)
    fc1, _ = base._experts()
    fc1.prefetch_weights()
    assert calls == [fc1.num_gemms]
    prefetched = fc1._prefetched[0]
    taken = fc1._get_weight_tensors()
    assert calls == [fc1.num_gemms] and all(a is b for a, b in zip(taken, prefetched, strict=True))
    for w, rows in zip(taken, fc1._packed_rows(), strict=True):
        want = redequant.dequantize_mxfp4_rows(rows, redequant.MXFP4_GROUP_SIZE)
        assert torch.equal(w.view(torch.int16), want.view(torch.int16))
    # a switch flip between the prefetch and the fetch drops the prefetch (it keys on every dequant switch)
    fc1.prefetch_weights()
    stale = fc1._prefetched[0][0]
    monkeypatch.setenv(redequant.TRITON_DEQUANT_ENV, "0")
    assert fc1._get_weight_tensors()[0] is not stale and fc1._prefetched is None
