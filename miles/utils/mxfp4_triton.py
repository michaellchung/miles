"""One-pass Triton dequantization of a grouped linear's packed MXFP4 experts to BF16.

``dequantize_mxfp4_rows`` (fp32 chain, ~77 B of memory traffic per weight) and its byte-table variant
(``dequantize_mxfp4_lut``, measured 9.2 ms per MoE layer and fetch for the 48 frozen experts of a K3.1 EP16 GPU in
the n1 trace: seven kernels per expert, an int32 index tensor and a gather in between) both run once per expert.
``dequantize_mxfp4_rows_triton`` runs one kernel per group of same-shape experts: it reads the 0.53 B per weight of
packed codes and scales and writes the 2 B of BF16, into one contiguous [E, N, K] buffer whose per-expert views it
returns (no ``torch.stack`` of the inputs, no per-expert launch).

Bit for bit ``dequantize_mxfp4_rows(rows, 32)``: the kernel builds each BF16 bit pattern from the code and the
exponent with integer arithmetic, i.e. the exactly rounded product sign * E2M1 * 2 ** (e - 127), which the fp32
chain also yields (every factor and the product are exact in fp32, the BF16 cast is exact): normal results, the
BF16 subnormals of small scales (e <= 1), -0.0 for code 8, +-inf when the exponent overflows (e = 254 with
magnitude >= 2) and for e = 255 (the chain's 2 ** 128 = inf). The only difference is the bit pattern of a NaN
(code 0 or 8 under e = 255, the chain's 0 * inf): this kernel writes 0x7FC0; torch's CPU cast also gives 0x7FC0,
CUDA's may give 0x7FFF. Real checkpoints carry no e = 255 scale.

CUDA-only module (plus ``TRITON_INTERPRET=1`` for the CPU tests): only the Megatron MXFP4 expert path imports it,
and ``triton_dequant_available`` gates every use.

Integer-only on purpose: no float multiply means no FTZ/denormal mode question on any backend, including
``TRITON_INTERPRET=1`` (the CPU tests). Packed rows are Miles' layout (``pack_mxfp4_rows``): [N, K/2 codes |
K/32 E8M0 scales] uint8, low nibble = even element.

The experts of a group live in separate parameter storages; the kernel addresses them as byte offsets from the
lowest-addressed one (an int64 device tensor, cached per tuple of data pointers: frozen parameters do not move, so after the
first fetch no host-to-device copy happens on the fetch path).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments without triton: the switch then falls back to the torch path
    triton = None
    tl = None

GROUP_SIZE = 32
_ROWS_PER_PROGRAM = 4
_BYTES_PER_PROGRAM = 512

if triton is not None:

    @triton.jit
    def _nibble_bits(code, scale):
        """BF16 bits (int32, low 16) of sign(code) * E2M1(code & 7) * 2 ** (scale - 127), exactly rounded."""
        sign = (code & 8) << 12
        m = code & 7
        e_m = (m >> 1) - 1  # 0.5 -> -1, 1 / 1.5 -> 0, 2 / 3 -> 1, 4 / 6 -> 2
        frac = (m & 1) & (m > 1).to(tl.int32)  # 1.5, 3, 6 carry the half mantissa bit
        biased = scale + e_m
        normal = sign | (biased << 7) | (frac << 6)
        sub_shift = tl.maximum(biased + 5, 0)
        subnormal = sign | ((2 + frac) << sub_shift)  # (1 + f/2) * 2 ** (biased - 127) = (2 + f) << (biased + 5) ulps
        bits = tl.where(biased >= 1, normal, subnormal)
        bits = tl.where(biased >= 255, sign | 0x7F80, bits)  # overflow -> +-inf
        bits = tl.where(m == 0, sign, bits)  # +-0
        inf_scale = scale == 255  # E8M0 255: the chain's 2 ** 128 = inf; 0 * inf = NaN
        bits = tl.where(inf_scale, tl.where(m == 0, 0x7FC0, sign | 0x7F80), bits)
        return bits

    @triton.jit
    def _dequant_group_kernel(
        base_ptr,  # uint8: the lowest-addressed expert's packed rows
        offsets_ptr,  # int64 [E]: byte offset of expert e's rows from base_ptr
        out_ptr,  # int32 view of the [E, N, K] BF16 output: one word = two BF16 (low half = even element)
        total_rows,
        n_rows: tl.constexpr,
        half_k: tl.constexpr,  # K / 2: code bytes per row = output words per row
        row_bytes: tl.constexpr,  # K / 2 + K / 32
        ROWS: tl.constexpr,
        BYTES: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_b = tl.program_id(1)
        rows = pid_r * ROWS + tl.arange(0, ROWS)
        row_ok = rows < total_rows
        rows64 = rows.to(tl.int64)
        expert = rows64 // n_rows
        in_expert = rows64 - expert * n_rows
        base = tl.load(offsets_ptr + expert, mask=row_ok, other=0) + in_expert * row_bytes  # [ROWS] int64
        cols = pid_b * BYTES + tl.arange(0, BYTES)
        col_ok = cols < half_k
        mask = row_ok[:, None] & col_ok[None, :]
        src = base[:, None] + cols[None, :]
        byte = tl.load(base_ptr + src, mask=mask, other=0).to(tl.int32)
        scale = tl.load(base_ptr + base[:, None] + half_k + cols[None, :] // 16, mask=mask, other=127).to(tl.int32)
        lo = _nibble_bits(byte & 15, scale)
        hi = _nibble_bits(byte >> 4, scale)
        word = lo | (hi << 16)
        dst = rows64[:, None] * half_k + cols[None, :]
        tl.store(out_ptr + dst, word, mask=mask)


_OFFSETS: dict[tuple, torch.Tensor] = {}
_OFFSETS_MAX = 4096


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def triton_dequant_available(device: torch.device) -> bool:
    return triton is not None and (device.type == "cuda" or _interpreting())


def _base(rows: list[torch.Tensor]) -> torch.Tensor:
    """The expert whose storage starts lowest: every offset from it is non-negative."""
    return min(rows, key=lambda r: r.data_ptr())


def _offsets(rows: list[torch.Tensor]) -> torch.Tensor:
    base = _base(rows).data_ptr()
    key = (rows[0].device, *(r.data_ptr() for r in rows))
    off = _OFFSETS.get(key)
    if off is None:
        if len(_OFFSETS) >= _OFFSETS_MAX:
            _OFFSETS.clear()
        host = torch.tensor([r.data_ptr() - base for r in rows], dtype=torch.int64)
        if rows[0].device.type == "cuda":
            host = host.pin_memory()
        off = _OFFSETS[key] = host.to(rows[0].device, non_blocking=True)
    return off


def _dequant_group(rows: list[torch.Tensor], group_size: int) -> list[torch.Tensor]:
    n, width = rows[0].shape
    assert (width * 2 * group_size) % (group_size + 2) == 0, (width, group_size)
    k = width * 2 * group_size // (group_size + 2)
    half_k = k // 2
    e = len(rows)
    out = torch.empty((e, n, k), dtype=torch.bfloat16, device=rows[0].device)
    if out.numel():
        total = e * n
        grid = (triton.cdiv(total, _ROWS_PER_PROGRAM), triton.cdiv(half_k, _BYTES_PER_PROGRAM))
        _dequant_group_kernel[grid](
            _base(rows),
            _offsets(rows),
            out.view(torch.int32),
            total,
            n,
            half_k,
            width,
            ROWS=_ROWS_PER_PROGRAM,
            BYTES=_BYTES_PER_PROGRAM,
            num_warps=4,
        )
    return list(out.unbind(0))


def dequantize_mxfp4_rows_triton(rows, group_size: int = GROUP_SIZE) -> list[torch.Tensor]:
    """``[dequantize_mxfp4_rows(r, group_size) for r in rows]`` bit for bit (NaN payloads aside, see the module
    docstring): one kernel per run of consecutive same-shape experts; each output is a contiguous view of its
    run's [E, N, K] buffer."""
    assert group_size == GROUP_SIZE, f"the Triton MXFP4 dequant is built for group size {GROUP_SIZE}, got {group_size}"
    rows = list(rows)
    out: list[torch.Tensor] = []
    i = 0
    while i < len(rows):
        r0 = rows[i]
        assert r0.dtype == torch.uint8 and r0.dim() == 2 and r0.is_contiguous(), (r0.dtype, r0.shape)
        j = i + 1
        while (
            j < len(rows)
            and rows[j].shape == r0.shape
            and rows[j].device == r0.device
            and rows[j].dtype == torch.uint8
            and rows[j].is_contiguous()
        ):
            j += 1
        out.extend(_dequant_group(rows[i:j], group_size))
        i = j
    return out
