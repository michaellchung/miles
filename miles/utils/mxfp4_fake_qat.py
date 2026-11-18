"""Routed-expert MXFP4 fake QAT for BF16 training (``--routed-expert-qat mxfp4``).

The trainable routed-expert weights are rounded onto the MXFP4 grid in place before the grouped GEMMs
read them, so the trainer's forward uses exactly the weights the rollout decodes from the MXFP4 release
(``miles.utils.mxfp4.quantize_mxfp4`` + ``dequantize_mxfp4``).

Straight-through: Megatron's mixed-precision optimizer keeps fp32 master weights and copies them into the
BF16 parameters after every step. Rounding the BF16 parameters changes what the GEMMs see, not what the
optimizer updates; the weight gradient (taken w.r.t. the rounded weight) reaches ``main_grad`` and the
master unchanged. Transformer Engine receives the parameters themselves, so fused weight-gradient
accumulation into ``main_grad`` works as for any other weight (an out-of-place STE copy, as the NVFP4
hook returns, has no ``main_grad``).

Where it hooks: ``TEGroupedLinear._get_weight_tensors``, the method Megatron's INT4 / NVFP4 fake-QAT hooks
already wrap. Megatron-LM at the k31 pin has no ``moe_expert_fake_quant`` field, so the hook is installed
per module instance on the trainable expert grouped linears only. The frozen MXFP4 experts
(``miles_plugins.models.kimi_k3.mxfp4_experts``) override ``_get_weight_tensors`` with an exact
dequantization and are already on the grid; they are skipped.

When it rounds: the parameters change only when the optimizer (or a checkpoint load) writes them, never
inside a forward-backward pass. Megatron marks the first microbatch of a pass only under an FP8/FP4
recipe, so a BF16 run cannot rely on ``is_first_microbatch``. Instead every fetch rounds into a scratch
tensor and writes back only when the result differs from the parameter. Within a pass the parameter is
already on the grid and is never written again, so autograd's saved-tensor version checks (recompute,
pipeline interleaving) stay valid.

Rounding: by default the release codec itself (``quantize_mxfp4`` then ``dequantize_mxfp4``; exact fp32
arithmetic, so CPU and GPU agree), which makes the trainer weights equal the release by construction.
``MILES_MXFP4_QAT_TRITON=1`` switches to the existing Triton ``fake_quant_fp4_forward`` (same rules:
block 32, scale ``ceil_pow2(max(amax, 6 * 2**-126) / 6)``, ``rint`` ties to even), which is cheaper but
whose bit equality with the codec is only checked by the CUDA tests.
Signed zeros: a -0 parameter element re-encodes as +0 on both paths; values (and GEMM results) agree.

``MILES_K31_MXFP4_QAT_NO_SYNC=1``: the on-grid test stays on the device. ``torch.equal`` is a host sync (16 per
MoE layer and pass at EP16 with 128 trainable experts; in the a1d trace each one drains the stream and the next
kernels start late). Instead the fetch writes ``where(all(rounded == data), data, rounded)``: the parameter's own
bits when it is on the grid, the rounded tensor otherwise, exactly what the synced path leaves behind. The write
goes through ``weight.data``, which does not bump the autograd version counter, so the unchanged on-grid write
inside a pass does not invalidate saved tensors (the synced path's off-grid write, which only happens before the
first forward after an optimizer step, did bump it). Costs one extra read and write of the trainable experts per
fetch (~2 GB per MoE layer at EP16, ~0.3 ms) instead of 16 syncs.
"""

from __future__ import annotations

import os
import types
from collections.abc import Sequence

import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, dequantize_mxfp4, quantize_mxfp4

MXFP4 = "mxfp4"
MOE_EXPERT_FAKE_QUANT_FIELD = "moe_expert_fake_quant"
_OTHER_FAKE_QAT_FLAGS = ("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "OPEN_TRAINING_INT4_FAKE_QAT_FLAG")
# Opt-in: round with the Triton fake_quant_fp4_forward instead of the release codec. Its bit equality
# with the codec is checked only by the CUDA tests in tests/fast/utils/test_mxfp4_fake_qat.py.
TRITON_KERNEL_FLAG = "MILES_MXFP4_QAT_TRITON"
_INSTALLED = "_miles_mxfp4_qat_installed"
# Opt-in: decide on the device whether to write the rounded weight back (no torch.equal host sync); same bits
NO_SYNC_FLAG = "MILES_K31_MXFP4_QAT_NO_SYNC"


def mxfp4_round_trip(weight: torch.Tensor) -> torch.Tensor:
    """The weight a rollout decodes from the MXFP4 release encoding of ``weight`` (same dtype and shape)."""
    rows = weight.reshape(-1, weight.shape[-1])
    packed, scale = quantize_mxfp4(rows, MXFP4_GROUP_SIZE)
    return dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE).to(weight.dtype).reshape(weight.shape)


def fake_quantize_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Round ``weight`` onto the MXFP4 grid (out of place, no autograd)."""
    if weight.is_cuda and os.getenv(TRITON_KERNEL_FLAG, "0") == "1":
        from miles.kernels.quant.fake_quant import fake_quant_fp4_forward

        return fake_quant_fp4_forward(weight)
    return mxfp4_round_trip(weight)


def round_in_place_if_off_grid(weight: torch.Tensor) -> bool:
    """Round ``weight`` onto the MXFP4 grid in place unless it is already there; returns whether it wrote."""
    with torch.no_grad():
        data = weight.detach()
        rounded = fake_quantize_mxfp4(data)
        if torch.equal(rounded, data):
            return False
        data.copy_(rounded)
        return True


def round_in_place_without_sync(weight: torch.Tensor) -> None:
    """``round_in_place_if_off_grid`` without the host sync: the weight ends up with the same bits."""
    with torch.no_grad():
        data = weight.detach()
        rounded = fake_quantize_mxfp4(data)
        on_grid = (rounded == data).all()  # torch.equal for equal shapes and dtypes (NaN never equal, -0 == +0)
        weight.data.copy_(torch.where(on_grid, data, rounded))


def fake_quantize_mxfp4_weight_tensors(weight_tensors: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """The grouped linear's expert weights, rounded onto the MXFP4 grid in place; same tensors returned."""
    no_sync = os.getenv(NO_SYNC_FLAG, "0") == "1"
    for weight in weight_tensors:
        if hasattr(weight, "__fsdp_param__"):  # config-access-exempt: Megatron FSDP marks its weights so
            raise RuntimeError("MXFP4 fake QAT does not support Megatron FSDP weights.")
        assert weight.shape[-1] % MXFP4_GROUP_SIZE == 0, tuple(weight.shape)
        if no_sync:
            round_in_place_without_sync(weight)
        else:
            round_in_place_if_off_grid(weight)
    return weight_tensors


def validate_mxfp4_qat_config(config) -> None:
    """Reject model configurations under which rounding the grouped-expert weights would be skipped or wrong."""
    if not getattr(config, "moe_grouped_gemm", False):  # config-access-exempt: Megatron versions differ
        raise ValueError("MXFP4 fake QAT requires grouped-GEMM routed experts (--moe-grouped-gemm).")
    if getattr(config, "use_transformer_engine_op_fuser", False):  # config-access-exempt: Megatron versions differ
        raise ValueError("MXFP4 fake QAT does not support the Transformer Engine op fuser.")
    if getattr(config, "delay_wgrad_compute", False):  # config-access-exempt: Megatron versions differ
        raise ValueError("MXFP4 fake QAT does not support delayed weight-gradient computation.")
    if getattr(config, MOE_EXPERT_FAKE_QUANT_FIELD, None):  # config-access-exempt: absent at the k31 pin
        raise ValueError(f"{MOE_EXPERT_FAKE_QUANT_FIELD} is set in Megatron as well; enable only one of them.")
    enabled = [flag for flag in _OTHER_FAKE_QAT_FLAGS if os.getenv(flag, "0") == "1"]
    if enabled:
        raise ValueError(f"MXFP4 fake QAT cannot be combined with {', '.join(enabled)}.")


def _is_trainable_grouped_linear(module: torch.nn.Module) -> bool:
    num_gemms = getattr(module, "num_gemms", None)  # config-access-exempt: TE GroupedLinear attribute
    if not isinstance(num_gemms, int) or not callable(getattr(module, "_get_weight_tensors", None)):
        return False
    weights = [getattr(module, f"weight{i}", None) for i in range(num_gemms)]  # config-access-exempt: TE names
    return all(
        isinstance(w, torch.nn.Parameter) and w.requires_grad and w.dtype in (torch.bfloat16, torch.float16)
        for w in weights
    )


def install_routed_expert_mxfp4_qat(model: torch.nn.Module) -> int:
    """Wrap ``_get_weight_tensors`` of every trainable routed-expert grouped linear; returns how many.

    Run after the freeze of ``--partial-experts-selection`` so the frozen half (``requires_grad=False``,
    uint8 packed bytes) is left alone. Only modules under ``.mlp.experts.`` are touched.
    """
    count = 0
    for name, module in model.named_modules():
        if ".mlp.experts." not in f".{name}." or not _is_trainable_grouped_linear(module):
            continue
        if getattr(module, _INSTALLED, False):  # config-access-exempt: our own marker
            continue
        validate_mxfp4_qat_config(module.config)
        inner = module._get_weight_tensors

        def _get_weight_tensors(self, _inner=inner):
            return fake_quantize_mxfp4_weight_tensors(_inner())

        module._get_weight_tensors = types.MethodType(_get_weight_tensors, module)
        setattr(module, _INSTALLED, True)
        count += 1
    return count


__all__ = [
    "MXFP4",
    "fake_quantize_mxfp4",
    "fake_quantize_mxfp4_weight_tensors",
    "install_routed_expert_mxfp4_qat",
    "mxfp4_round_trip",
    "round_in_place_if_off_grid",
    "round_in_place_without_sync",
    "validate_mxfp4_qat_config",
]
