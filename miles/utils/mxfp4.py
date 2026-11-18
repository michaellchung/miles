"""MXFP4 (E2M1 elements + E8M0 block scales) pack/unpack.

Torch-only, so checkpoint tooling can use it without importing Megatron.
"""

import torch

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = _E2M1_VALUES
E2M1_MAX = 6.0
MXFP4_GROUP_SIZE = 32  # OCP MXFP4: one E8M0 scale per 32 elements
# Block amaxes below this floor share the smallest normal power-of-two scale, 2**-126.
MXFP4_AMAX_FLOOR = E2M1_MAX * 2.0**-126


def _biased_exponent_of_ceil_pow2(value: torch.Tensor) -> torch.Tensor:
    """Biased fp32 exponent of 2 ** ceil(log2(value)) for positive normal fp32 values, exact at powers of two."""
    bits = value.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def quantize_mxfp4(weight, group_size):
    """Quantize to MXFP4: E2M1 elements rounded to nearest with ties to even, and E8M0 block scales.

    Each block's scale is the smallest power of two that keeps its amax within E2M1_MAX, computed as
    ``2 ** ceil(log2(amax * (1 / 6)))`` in fp32 with the amax floored at ``MXFP4_AMAX_FLOOR``, so no
    element saturates. Every step is exact fp32 arithmetic, so the result does not depend on the device.
    The rules are those of ``miles.kernels.quant.fake_quant.fake_quant_fp4_forward`` (amax floor,
    ``_ceil_pow2`` scale, ``rint`` on the 0.5/1/2 steps), so routed-expert QAT
    (``miles.utils.mxfp4_fake_qat``) rounds the trained weights onto exactly the values a rollout
    decodes from this encoding.
    """
    assert weight.shape[-1] % group_size == 0
    assert weight.shape[-1] % 2 == 0

    blocks = weight.reshape(-1, group_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(MXFP4_AMAX_FLOOR)
    biased_exponent = _biased_exponent_of_ceil_pow2(amax * (1.0 / E2M1_MAX))
    normalized = blocks / (biased_exponent << 23).view(torch.float32)

    magnitude = normalized.abs().clamp_max(E2M1_MAX)
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    rounded = torch.round(magnitude / step) * step  # torch.round: half to even, like libdevice.rint
    code = torch.where(rounded < 2.0, rounded * 2.0, torch.where(rounded < 4.0, rounded + 2.0, rounded * 0.5 + 4.0))
    encoded = code.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)

    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2).contiguous()
    scale = biased_exponent.to(torch.uint8)
    scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // group_size).contiguous()
    return packed, scale


def dequantize_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    assert weight_packed.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8
    assert group_size > 0

    unpacked = torch.empty(
        *weight_packed.shape[:-1],
        weight_packed.shape[-1] * 2,
        dtype=torch.uint8,
        device=weight_packed.device,
    )
    unpacked[..., 0::2] = weight_packed & 0x0F
    unpacked[..., 1::2] = (weight_packed >> 4) & 0x0F

    signs = 1.0 - 2.0 * ((unpacked & 0b1000) >> 3).float()
    magnitudes = unpacked & 0b0111
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=weight_packed.device)
    dequantized = signs * values[magnitudes.long()]

    assert dequantized.numel() % group_size == 0
    assert weight_scale.numel() == dequantized.numel() // group_size
    dequantized = dequantized.reshape(-1, group_size)
    scales = torch.exp2(weight_scale.float().reshape(-1, 1) - 127.0)
    return (dequantized * scales).reshape(unpacked.shape).to(torch.bfloat16).contiguous()


def _byte_lut(device: torch.device) -> torch.Tensor:
    """[256, 2] BF16: the low-nibble and high-nibble E2M1 values of every packed byte (sign bit 3 of a nibble)."""
    magnitudes = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
    nibble = torch.arange(16)
    values = torch.where(nibble >= 8, -magnitudes[nibble % 8], magnitudes[nibble % 8])  # code 8 is -0.0
    byte = torch.arange(256)
    return torch.stack([values[byte & 0x0F], values[byte >> 4]], dim=-1).to(torch.bfloat16).to(device)


_BYTE_LUTS: dict[torch.device, torch.Tensor] = {}


def dequantize_mxfp4_lut(weight_packed: torch.Tensor, weight_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """``dequantize_mxfp4`` bit for bit, in BF16 through a 256-entry byte table instead of an fp32 chain.

    Every E2M1 value and every E8M0 scale 2 ** (e - 127), e <= 254, is exact in BF16 and so is their product (at
    most 3 significant bits, exponent within BF16's range), so the BF16 multiply rounds nothing and the result equals
    the fp32 product cast to BF16. About 9 B of memory traffic per weight instead of about 77.
    """
    assert weight_packed.dtype == torch.uint8 and weight_scale.dtype == torch.uint8 and group_size > 0
    lut = _BYTE_LUTS.get(weight_packed.device)
    if lut is None:
        lut = _BYTE_LUTS[weight_packed.device] = _byte_lut(weight_packed.device)
    *lead, half = weight_packed.shape
    values = lut.index_select(0, weight_packed.reshape(-1).to(torch.int32)).view(*lead, half * 2)
    assert values.numel() % group_size == 0 and weight_scale.numel() == values.numel() // group_size
    scales = torch.exp2(weight_scale.float() - 127.0).to(torch.bfloat16)
    blocks = values.view(-1, group_size) * scales.reshape(-1, 1)
    return blocks.view(*lead, half * 2)


def mxfp4_row_width(in_features: int, group_size: int) -> int:
    """Bytes per output row of a packed row tensor: K/2 codes, then K/group_size scales."""
    assert in_features % group_size == 0 and in_features % 2 == 0, (in_features, group_size)
    return in_features // 2 + in_features // group_size


def pack_mxfp4_rows(weight_packed: torch.Tensor, weight_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """[N, K/2] codes and [N, K/group_size] scales -> one [N, K/2 + K/group_size] uint8 tensor."""
    assert weight_packed.dtype == torch.uint8 and weight_scale.dtype == torch.uint8
    assert weight_packed.shape[0] == weight_scale.shape[0]
    assert weight_packed.shape[1] * 2 == weight_scale.shape[1] * group_size
    return torch.cat([weight_packed, weight_scale], dim=1)


def split_mxfp4_rows(rows: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes and scales a packed row tensor holds; the inverse of ``pack_mxfp4_rows``."""
    assert rows.dtype == torch.uint8
    width = rows.shape[-1]
    # width = K/2 + K/group_size = K (group_size + 2) / (2 group_size)
    assert (width * 2 * group_size) % (group_size + 2) == 0, (width, group_size)
    in_features = width * 2 * group_size // (group_size + 2)
    return rows[:, : in_features // 2], rows[:, in_features // 2 :]


def dequantize_mxfp4_rows(rows: torch.Tensor, group_size: int, lut: bool = False) -> torch.Tensor:
    """The BF16 [N, K] weight a packed row tensor encodes; exact. ``lut``: through ``dequantize_mxfp4_lut``."""
    dequantize = dequantize_mxfp4_lut if lut else dequantize_mxfp4
    return dequantize(*split_mxfp4_rows(rows, group_size), group_size)


# elements per batched dequantization chunk: about 0.13 G weights, ~1 GB of transient LUT intermediates
BATCHED_DEQUANT_CHUNK_ELEMENTS = 1 << 27


def dequantize_mxfp4_rows_batched(
    rows: list[torch.Tensor] | tuple[torch.Tensor, ...],
    group_size: int,
    lut: bool = False,
    max_elements: int = BATCHED_DEQUANT_CHUNK_ELEMENTS,
) -> list[torch.Tensor]:
    """``[dequantize_mxfp4_rows(r, group_size, lut) for r in rows]`` bit for bit, in a few kernel sequences.

    Consecutive row tensors of one shape are stacked into chunks of at most ``max_elements`` weights (at least one
    tensor each) and each chunk is dequantized as one [c * N, width] row tensor; both dequantizations are
    row-wise elementwise, so every output row is the same bits. The outputs are contiguous views of their chunk.
    One call per chunk instead of per tensor: a grouped linear's 48 frozen experts take a handful of kernel
    sequences instead of 48, which is what keeps the host from becoming the bottleneck of the weight fetch.
    """
    out: list[torch.Tensor] = []
    i, n = 0, len(rows)
    while i < n:
        shape = rows[i].shape
        assert rows[i].dim() == 2, shape
        width = mxfp4_row_width_inverse(shape[-1], group_size)
        per = max(1, shape[0] * width)
        j = i + 1
        while j < n and rows[j].shape == shape and (j - i + 1) * per <= max_elements:
            j += 1
        if j - i == 1:
            out.append(dequantize_mxfp4_rows(rows[i], group_size, lut=lut))
        else:
            stacked = torch.stack(list(rows[i:j])).view(-1, shape[-1])
            weights = dequantize_mxfp4_rows(stacked, group_size, lut=lut)
            out.extend(weights.view(j - i, shape[0], width).unbind(0))
        i = j
    return out


def mxfp4_row_width_inverse(row_width: int, group_size: int) -> int:
    """``in_features`` of a packed row tensor ``row_width`` bytes wide; the inverse of ``mxfp4_row_width``."""
    assert (row_width * 2 * group_size) % (group_size + 2) == 0, (row_width, group_size)
    return row_width * 2 * group_size // (group_size + 2)
