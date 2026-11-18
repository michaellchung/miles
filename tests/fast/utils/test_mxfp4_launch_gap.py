"""Host-launch fixes for the MXFP4 weight fetches give the same bits as the paths they replace.

* ``dequantize_mxfp4_rows_batched`` (``MILES_K31_MXFP4_BATCHED_DEQUANT``) against one ``dequantize_mxfp4_rows`` per
  tensor, through both the fp32 chain and the byte table, for every chunking;
* ``round_in_place_without_sync`` (``MILES_K31_MXFP4_QAT_NO_SYNC``) against ``round_in_place_if_off_grid``: the same
  parameter bits (signed zeros included), no host sync, no autograd version bump.
"""

import pytest
import torch

from miles.utils import mxfp4_fake_qat
from miles.utils.mxfp4 import (
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4_rows,
    dequantize_mxfp4_rows_batched,
    mxfp4_row_width,
    mxfp4_row_width_inverse,
    pack_mxfp4_rows,
    quantize_mxfp4,
)
from miles.utils.mxfp4_fake_qat import (
    fake_quantize_mxfp4,
    fake_quantize_mxfp4_weight_tensors,
    round_in_place_if_off_grid,
    round_in_place_without_sync,
)


def _bits(t):
    return t.contiguous().view(torch.int16)  # tells -0.0 from +0.0


def _rows(num, out_features, in_features, seed):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(num):
        w = torch.randn(out_features, in_features, generator=g) * torch.logspace(-20, 20, out_features)[:, None]
        rows.append(pack_mxfp4_rows(*quantize_mxfp4(w, MXFP4_GROUP_SIZE), MXFP4_GROUP_SIZE))
    return rows


def _random_bytes(num, out_features, in_features, seed):
    # every byte value, scales over the whole finite E8M0 range (255 = NaN excluded)
    g = torch.Generator().manual_seed(seed)
    width = mxfp4_row_width(in_features, MXFP4_GROUP_SIZE)
    out = []
    for _ in range(num):
        r = torch.randint(0, 256, (out_features, width), generator=g, dtype=torch.uint8)
        r[:, in_features // 2 :] = torch.randint(0, 255, (out_features, width - in_features // 2), generator=g).to(
            torch.uint8
        )
        out.append(r)
    return out


@pytest.mark.parametrize("lut", [False, True])
@pytest.mark.parametrize("max_elements", [1, 64 * 96, 3 * 64 * 96, 1 << 40])
def test_batched_equals_one_call_per_tensor(lut, max_elements):
    rows = _rows(5, 64, 96, seed=1) + _random_bytes(3, 64, 96, seed=2) + _rows(2, 96, 64, seed=3)
    rows.insert(4, _rows(1, 0, 96, seed=4)[0])  # an empty expert in the middle splits the chunk
    ref = [dequantize_mxfp4_rows(r, MXFP4_GROUP_SIZE, lut=lut) for r in rows]
    got = dequantize_mxfp4_rows_batched(rows, MXFP4_GROUP_SIZE, lut=lut, max_elements=max_elements)
    assert len(got) == len(ref)
    for a, b in zip(ref, got, strict=True):
        assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape and b.is_contiguous()
        assert torch.equal(_bits(a), _bits(b))


def test_batched_lut_equals_batched_fp32_chain():
    rows = _random_bytes(6, 32, 128, seed=5)
    a = dequantize_mxfp4_rows_batched(rows, MXFP4_GROUP_SIZE, lut=False)
    b = dequantize_mxfp4_rows_batched(rows, MXFP4_GROUP_SIZE, lut=True)
    assert all(torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b, strict=True))


def test_chunking_launches_one_dequantization_per_chunk(monkeypatch):
    import miles.utils.mxfp4 as m

    calls = []
    real = m.dequantize_mxfp4_rows
    monkeypatch.setattr(m, "dequantize_mxfp4_rows", lambda r, *a, **k: calls.append(r.shape) or real(r, *a, **k))
    rows = _rows(7, 64, 96, seed=6)
    m.dequantize_mxfp4_rows_batched(rows, MXFP4_GROUP_SIZE, lut=True, max_elements=3 * 64 * 96)
    assert calls == [(192, 51), (192, 51), (64, 51)]
    calls.clear()
    m.dequantize_mxfp4_rows_batched(rows, MXFP4_GROUP_SIZE, lut=True)
    assert calls == [(7 * 64, 51)]


def test_row_width_inverse():
    for k in (32, 64, 3072, 3584):
        assert mxfp4_row_width_inverse(mxfp4_row_width(k, MXFP4_GROUP_SIZE), MXFP4_GROUP_SIZE) == k


# ------------------------------------------------------------------------------------- QAT without the host sync


def _qat_weights():
    g = torch.Generator().manual_seed(7)
    off = (torch.randn(16, 4 * MXFP4_GROUP_SIZE, generator=g) * 0.02).bfloat16()
    on = fake_quantize_mxfp4(off.clone())
    on_with_neg_zero = on.clone()
    on_with_neg_zero.view(-1)[::7] = -0.0  # on the grid, but -0 re-encodes as +0
    off_with_neg_zero = off.clone()
    off_with_neg_zero.view(-1)[::5] = -0.0
    return {"off": off, "on": on, "on_neg_zero": on_with_neg_zero, "off_neg_zero": off_with_neg_zero}


@pytest.mark.parametrize("case", ["off", "on", "on_neg_zero", "off_neg_zero"])
def test_no_sync_rounding_leaves_the_same_bits(case):
    w = _qat_weights()[case]
    a = torch.nn.Parameter(w.clone())
    b = torch.nn.Parameter(w.clone())
    round_in_place_if_off_grid(a)
    round_in_place_without_sync(b)
    assert torch.equal(_bits(a.detach()), _bits(b.detach()))
    if case == "on_neg_zero":  # the synced path leaves an on-grid weight untouched, -0 included
        assert torch.equal(_bits(b.detach()), _bits(w))


def test_no_sync_rounding_does_not_sync_or_bump_the_version(monkeypatch):
    w = torch.nn.Parameter(_qat_weights()["on"].clone())
    version = w._version
    monkeypatch.setattr(torch, "equal", lambda *a: pytest.fail("torch.equal is a host sync"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        round_in_place_without_sync(w)
    names = {e.name for e in prof.events()}
    assert not names & {"aten::_local_scalar_dense", "aten::item", "aten::is_nonzero", "aten::equal"}, names
    assert w._version == version  # an unchanged write must not invalidate tensors autograd saved


def test_weight_tensors_hook_honours_the_flag(monkeypatch):
    weights = _qat_weights()
    plain = [torch.nn.Parameter(w.clone()) for w in weights.values()]
    nosync = [torch.nn.Parameter(w.clone()) for w in weights.values()]
    monkeypatch.setenv(mxfp4_fake_qat.NO_SYNC_FLAG, "0")
    fake_quantize_mxfp4_weight_tensors(plain)
    monkeypatch.setenv(mxfp4_fake_qat.NO_SYNC_FLAG, "1")
    seen = []
    monkeypatch.setattr(mxfp4_fake_qat, "round_in_place_if_off_grid", lambda w: seen.append(w))
    fake_quantize_mxfp4_weight_tensors(nosync)
    assert not seen
    for a, b in zip(plain, nosync, strict=True):
        assert torch.equal(_bits(a.detach()), _bits(b.detach()))


def test_no_sync_rounding_keeps_a_saved_weight_valid_in_the_backward():
    # a weight saved by one microbatch's forward and re-fetched (on grid) by the next before the first's backward:
    # the synced path never writes it; the device path writes the same bits without bumping the version
    w = torch.nn.Parameter(_qat_weights()["off"].clone())
    round_in_place_without_sync(w)
    x = torch.randn(3, w.shape[1]).bfloat16().requires_grad_(True)
    y = x @ w.t()
    round_in_place_without_sync(w)
    y.sum().backward()  # raises on a version mismatch
    assert w.grad is not None
