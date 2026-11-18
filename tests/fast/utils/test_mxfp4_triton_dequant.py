"""The one-pass Triton MXFP4 dequantization equals the fp32 chain bit for bit (CPU: under TRITON_INTERPRET=1)."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported: the kernels must be interpreted on CPU

import pytest
import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows, pack_mxfp4_rows, quantize_mxfp4

mxfp4_triton = pytest.importorskip("miles.utils.mxfp4_triton")
if mxfp4_triton.triton is None:
    pytest.skip("triton is not installed", allow_module_level=True)
if not torch.cuda.is_available() and type(mxfp4_triton._dequant_group_kernel).__name__ != "InterpretedFunction":
    pytest.skip(
        "triton was imported before TRITON_INTERPRET=1 was set: run this file on its own", allow_module_level=True
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _assert_same_bits(got, want):
    """BF16 bits equal, also -0.0 against +0.0; NaN only needs to be NaN on both sides (payloads differ by backend)."""
    assert got.dtype == want.dtype == torch.bfloat16 and got.shape == want.shape
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got.view(torch.int16)[~nan], want.view(torch.int16)[~nan])


def test_every_byte_under_every_scale():
    # row s: every packed byte (all nibble pairs, so every E2M1 code in both halves) under E8M0 scale s = 0..255:
    # BF16 subnormals (s <= 1), normals, overflow to inf (s = 254 with |value| >= 2) and s = 255 (inf / NaN)
    codes = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 512 // MXFP4_GROUP_SIZE)
    rows = pack_mxfp4_rows(codes, scales.contiguous(), MXFP4_GROUP_SIZE)
    want = dequantize_mxfp4_rows(rows, MXFP4_GROUP_SIZE)
    (got,) = mxfp4_triton.dequantize_mxfp4_rows_triton([rows.to(DEVICE)], MXFP4_GROUP_SIZE)
    _assert_same_bits(got.cpu(), want)
    bits = want.view(torch.int16)
    assert (bits == -32768).any() and ((bits & 0x7F80) == 0).logical_and(bits & 0x7F != 0).any()  # -0.0, subnormals
    assert torch.isinf(want).any() and torch.isnan(want).any()


def _expert_rows(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    packed, scale = quantize_mxfp4(torch.randn(n, k, generator=g) * 0.05, MXFP4_GROUP_SIZE)
    return pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE)


def test_a_fetch_of_separate_experts_and_shapes():
    # separately allocated experts (the kernel addresses them as offsets from the lowest one, in any order), two runs
    # of one shape split by another shape, and an expert with zero rows
    shapes = [(6, 96), (6, 96), (6, 96), (5, 64), (0, 64), (5, 64), (6, 96)]
    rows = [_expert_rows(n, k, seed).to(DEVICE) for seed, (n, k) in enumerate(shapes)]
    rows = rows[::-1]
    got = mxfp4_triton.dequantize_mxfp4_rows_triton(rows, MXFP4_GROUP_SIZE)
    assert len(got) == len(rows)
    for g, r in zip(got, rows, strict=True):
        assert g.is_contiguous()
        _assert_same_bits(g.cpu(), dequantize_mxfp4_rows(r.cpu(), MXFP4_GROUP_SIZE))
    # reversed: (6, 96), (5, 64), (0, 64), (5, 64), then a run of three (6, 96) views of one [3, N, K] allocation
    ptr = [g.untyped_storage().data_ptr() for g in got]
    assert ptr[4] == ptr[5] == ptr[6] and len({ptr[0], ptr[1], ptr[3], ptr[4]}) == 4


def test_more_rows_and_bytes_than_one_program():
    # rows not a multiple of the program's row block and K/2 past one byte block, scales that vary along a row
    rows = [_expert_rows(13, 32 * 40, seed=s).to(DEVICE) for s in range(3)]
    got = mxfp4_triton.dequantize_mxfp4_rows_triton(rows, MXFP4_GROUP_SIZE)
    for g, r in zip(got, rows, strict=True):
        _assert_same_bits(g.cpu(), dequantize_mxfp4_rows(r.cpu(), MXFP4_GROUP_SIZE))


def test_offsets_are_cached_per_fetch():
    rows = [_expert_rows(4, 64, seed=s).to(DEVICE) for s in range(4)]
    mxfp4_triton._OFFSETS.clear()
    mxfp4_triton.dequantize_mxfp4_rows_triton(rows, MXFP4_GROUP_SIZE)
    first = dict(mxfp4_triton._OFFSETS)
    mxfp4_triton.dequantize_mxfp4_rows_triton(rows, MXFP4_GROUP_SIZE)
    assert len(first) == 1 and all(mxfp4_triton._OFFSETS[k] is v for k, v in first.items())
    assert (next(iter(first.values())) >= 0).all()
