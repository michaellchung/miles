"""The one-pass Triton MXFP4 dequant on the GPU: bit for bit the fp32 chain (every byte x every scale, and a K3.1
EP16 fetch of 48 frozen experts), and one launch per same-shape run. Timings print with -s."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4_rows, mxfp4_row_width, pack_mxfp4_rows  # noqa: E402
from miles.utils.mxfp4_triton import dequantize_mxfp4_rows_triton  # noqa: E402


def _assert_same_bits(got, want):
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got.view(torch.int16)[~nan], want.view(torch.int16)[~nan])


def test_every_byte_under_every_scale_on_gpu():
    codes = torch.arange(256, dtype=torch.uint8, device="cuda").repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8, device="cuda")[:, None].expand(256, 16).contiguous()
    rows = pack_mxfp4_rows(codes, scales, MXFP4_GROUP_SIZE)
    (got,) = dequantize_mxfp4_rows_triton([rows])
    _assert_same_bits(got, dequantize_mxfp4_rows(rows, MXFP4_GROUP_SIZE))  # the chain on CUDA: subnormals not flushed
    _assert_same_bits(got.cpu(), dequantize_mxfp4_rows(rows.cpu(), MXFP4_GROUP_SIZE))


def _random_experts(n_experts, out_features, in_features, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = mxfp4_row_width(in_features, MXFP4_GROUP_SIZE)
    rows = []
    for _ in range(n_experts):
        r = torch.randint(0, 256, (out_features, width), dtype=torch.uint8, device="cuda", generator=g)
        # realistic E8M0 scales (weights ~1e-3..1e-1): 2 ** (e - 127), e in [110, 130]
        r[:, in_features // 2 :] = torch.randint(
            110, 131, (out_features, width - in_features // 2), dtype=torch.uint8, device="cuda", generator=g
        )
        rows.append(r)
    return rows


def _ms(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.parametrize("shape", [(6144, 3584), (3584, 3072)])  # K3.1 fc1 (gate+up) / fc2, 48 frozen experts at EP16
def test_k31_fetch_bitwise_and_timing(shape):
    rows = _random_experts(48, *shape, seed=shape[0])
    got = dequantize_mxfp4_rows_triton(rows)
    for g, r in zip(got, rows, strict=True):
        _assert_same_bits(g, dequantize_mxfp4_rows(r, MXFP4_GROUP_SIZE))
    t_tri = _ms(lambda: dequantize_mxfp4_rows_triton(rows))
    t_lut = _ms(lambda: [dequantize_mxfp4_rows(r, MXFP4_GROUP_SIZE, lut=True) for r in rows], iters=5)
    gb = sum(r.numel() for r in rows) / 1e9 + sum(g.numel() * 2 for g in got) / 1e9
    print(f"\n[mxfp4-triton] {shape} x48: triton {t_tri:.3f} ms ({gb / t_tri:.2f} TB/s), lut {t_lut:.3f} ms")
