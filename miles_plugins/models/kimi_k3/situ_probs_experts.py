"""Routed experts whose SiTU activation and router-prob scale run as one fused Triton kernel
(``MILES_K3_SITU_PROBS_FUSED=1``, default off).

Megatron's ``TEGroupedMLP.forward`` computes ``fc1 -> config.gated_activation_func -> (* permuted_probs).to(bf16)
-> fc2`` with the activation and the scale as separate eager kernels (``bias_act_func``, a closure that cannot be
overridden on its own). ``SituProbsTEGroupedMLP`` replaces only that middle step with
``situ_triton.fused_situ_and_mul(fc1_output, permuted_probs)``, which replays the eager chain and Megatron's
dtype casts op for op (forward, input gradient and prob gradient bit for bit; checked on CUDA in
tests/fast-gpu/test_kimi_k3_situ_probs_experts.py). It applies only on the path K3.1 runs: BF16 experts without
FP8/FP4, the TE op fuser, activation offload/recompute, GLU interleaving, fp32 activation, probs-on-input or
fp32 combine; with
the switch on, any of those is an error at construction, never a silent fallback.
"""

from __future__ import annotations

import os

import torch
from megatron.core.transformer.moe.experts import TEGroupedMLP

from miles_plugins.models.kimi_k3.ops import situ_and_mul

SITU_PROBS_FUSED_ENV = "MILES_K3_SITU_PROBS_FUSED"

# config fields (or module attributes) that route TEGroupedMLP.forward around the eager bias_act_func branch
_UNSUPPORTED_CONFIG = (
    "fp8",
    "fp4",
    "moe_apply_probs_on_input",
    "moe_combine_in_fp32",
    "moe_activation_in_fp32",
    "add_bias_linear",
    "bias_activation_fusion",
    "use_te_activation_func",
    "moe_mlp_glu_interleave_size",
)
_UNSUPPORTED_MODULE = ("_with_fused_impl", "activation_recompute", "offload_expert_fc1", "offload_moe_act")


def situ_probs_fused_enabled() -> bool:
    return os.environ.get(SITU_PROBS_FUSED_ENV, "0") == "1"


class SituProbsTEGroupedMLP(TEGroupedMLP):
    """``TEGroupedMLP`` with the SiTU activation and the router-prob scale fused into one Triton kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_supported()

    def _check_supported(self) -> None:
        reasons = [
            f"config.{name}" for name in _UNSUPPORTED_CONFIG if getattr(self.config, name, None)
        ]  # config-access-exempt: fields differ across Megatron versions
        reasons += [
            name for name in _UNSUPPORTED_MODULE if getattr(self, name, False)
        ]  # config-access-exempt: TEGroupedMLP attributes differ by version
        if self.config.gated_activation_func is not situ_and_mul:
            reasons.append(f"gated_activation_func={self.config.gated_activation_func!r}")
        if reasons:
            raise ValueError(f"{SITU_PROBS_FUSED_ENV}=1 does not apply with {', '.join(reasons)}")

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert = torch.as_tensor(tokens_per_expert).tolist()
        fc1_output, fc1_bias = self.linear_fc1(permuted_local_hidden_states, tokens_per_expert)
        assert fc1_bias is None
        intermediate = _fused_situ_and_mul(fc1_output.contiguous(), permuted_probs)
        output, fc2_bias = self.linear_fc2(intermediate, tokens_per_expert)
        assert fc2_bias is None
        return output, None


def _fused_situ_and_mul(x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    from miles_plugins.models.kimi_k3.situ_triton import fused_situ_and_mul  # Triton only once the switch is used

    return fused_situ_and_mul(x, probs)
