"""The byte-table MXFP4 dequantization equals the fp32-chain one bit for bit, over every byte and every scale."""

import pytest
import torch

from miles.utils.mxfp4 import (
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4,
    dequantize_mxfp4_lut,
    dequantize_mxfp4_rows,
    pack_mxfp4_rows,
    quantize_mxfp4,
)


def _bits(t):
    return t.view(torch.int16)  # also tells -0.0 (E2M1 code 8) from +0.0


def test_every_byte_under_every_finite_scale():
    # rows: scale e = 0..254 (255 is E8M0 NaN); columns: all 256 packed bytes, i.e. every nibble pair
    packed = torch.arange(256, dtype=torch.uint8).repeat(255, 1)
    scale = torch.arange(255, dtype=torch.uint8)[:, None].expand(255, 512 // MXFP4_GROUP_SIZE).contiguous()
    chain = dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE)
    lut = dequantize_mxfp4_lut(packed, scale, MXFP4_GROUP_SIZE)
    assert lut.dtype == chain.dtype == torch.bfloat16 and lut.shape == chain.shape == (255, 512)
    assert torch.equal(_bits(lut), _bits(chain))
    assert (_bits(lut) == _bits(torch.tensor(-0.0, dtype=torch.bfloat16))).any()


@pytest.mark.parametrize("shape", [(7, 64), (3, 5, 96), (0, 32)])
def test_quantized_weights_and_packed_rows(shape):
    weight = torch.randn(*shape, generator=torch.Generator().manual_seed(len(shape))) * 0.02
    packed, scale = quantize_mxfp4(weight, MXFP4_GROUP_SIZE)
    lut = dequantize_mxfp4_lut(packed, scale, MXFP4_GROUP_SIZE)
    assert torch.equal(_bits(lut), _bits(dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE)))
    if len(shape) == 2:
        rows = pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE)
        assert torch.equal(
            _bits(dequantize_mxfp4_rows(rows, MXFP4_GROUP_SIZE, lut=True)),
            _bits(dequantize_mxfp4_rows(rows, MXFP4_GROUP_SIZE)),
        )
