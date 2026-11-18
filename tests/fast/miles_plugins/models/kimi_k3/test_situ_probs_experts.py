"""MILES_K3_SITU_PROBS_FUSED (default off) swaps both PartiallyTrainableGroupedMLP halves for
SituProbsTEGroupedMLP. On CPU, with the Triton kernel replaced by Megatron's own eager formula, its forward
equals TEGroupedMLP.forward bit for bit (output and every gradient), so the module drops nothing around the
activation; the kernel's own bit equality with that formula is checked on CUDA in
tests/fast-gpu/test_kimi_k3_situ_triton.py and, end to end with TE grouped linears, in
tests/fast-gpu/test_kimi_k3_situ_probs_experts.py. Unsupported configs fail at construction."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.kimi_k3 import situ_probs_experts, split_moe
from miles_plugins.models.kimi_k3.ops import situ_and_mul
from miles_plugins.models.kimi_k3.situ_probs_experts import SituProbsTEGroupedMLP

HIDDEN, FFN, EXPERTS = 64, 48, 3


def _config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        gated_activation_func=situ_and_mul,
        bias_activation_fusion=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _GroupedLinear(torch.nn.Module):
    """Per-expert BF16 matmul with the TE grouped-linear call signature ``(x, m_splits) -> (out, bias)``."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weights.clone())

    def forward(self, x, m_splits):
        assert isinstance(m_splits, list)
        chunks = x.split(m_splits)
        return torch.cat([c @ w.t() for c, w in zip(chunks, self.weight, strict=True)]), None


def _bare(cls, config, w1, w2):
    """A ``cls`` instance with only the attributes TEGroupedMLP.forward reads (TE linears need CUDA)."""
    mod = cls.__new__(cls)
    torch.nn.Module.__init__(mod)
    mod.config = config
    mod.activation_func = F.silu
    mod._with_fused_impl = False
    mod.activation_recompute = False
    mod.offload_expert_fc1 = False
    mod.offload_moe_act = False
    mod.linear_fc1 = _GroupedLinear(w1)
    mod.linear_fc2 = _GroupedLinear(w2)
    return mod


def _megatron_formula(x, probs):
    y = situ_and_mul(x)
    return (y * probs.unsqueeze(-1)).to(y.dtype)  # TEGroupedMLP.bias_act_func


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.int32)


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(situ_probs_experts.SITU_PROBS_FUSED_ENV, raising=False)
    assert situ_probs_experts.situ_probs_fused_enabled() is False
    monkeypatch.setenv(situ_probs_experts.SITU_PROBS_FUSED_ENV, "1")
    assert situ_probs_experts.situ_probs_fused_enabled() is True


@pytest.mark.parametrize("counts", [[5, 0, 11], [0, 0, 0], [7, 3, 2]])
def test_forward_matches_te_grouped_mlp_bitwise(monkeypatch, counts):
    monkeypatch.setattr(situ_probs_experts, "_fused_situ_and_mul", _megatron_formula)
    torch.manual_seed(0)
    config = _config()
    w1 = (torch.randn(EXPERTS, 2 * FFN, HIDDEN) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(EXPERTS, HIDDEN, FFN) * 0.3).to(torch.bfloat16)
    rows = sum(counts)
    x = (torch.randn(rows, HIDDEN) * 2).to(torch.bfloat16)
    probs = torch.rand(rows)
    dy = torch.randn(rows, HIDDEN).to(torch.bfloat16)
    tokens = torch.tensor(counts)  # TEGroupedMLP.forward calls .tolist(); split_moe passes tensor slices

    results = []
    for cls in (TEGroupedMLP, SituProbsTEGroupedMLP):
        mod = _bare(cls, config, w1, w2)
        xi, pi = x.clone().requires_grad_(), probs.clone().requires_grad_()
        out, bias = mod(xi, tokens, pi)
        assert bias is None
        out.backward(dy)
        results.append((out, xi.grad, pi.grad, mod.linear_fc1.weight.grad, mod.linear_fc2.weight.grad))
    for ref, got in zip(*results, strict=True):
        assert got.dtype == ref.dtype and got.shape == ref.shape
        assert torch.equal(_bits(got), _bits(ref))


@pytest.mark.parametrize(
    "config_overrides,module_overrides,reason",
    [
        ({"moe_apply_probs_on_input": True, "moe_router_topk": 1}, {}, "config.moe_apply_probs_on_input"),
        ({"moe_combine_in_fp32": True}, {}, "config.moe_combine_in_fp32"),
        ({"gated_activation_func": None}, {}, "gated_activation_func=None"),
        ({}, {"activation_recompute": True}, "activation_recompute"),
        ({}, {"offload_moe_act": True}, "offload_moe_act"),
        ({}, {"_with_fused_impl": True}, "_with_fused_impl"),
    ],
)
def test_unsupported_paths_fail_loud(config_overrides, module_overrides, reason):
    config = _config()
    for name, value in config_overrides.items():
        setattr(config, name, value)
    mod = _bare(
        SituProbsTEGroupedMLP, config, torch.zeros(EXPERTS, 2 * FFN, HIDDEN), torch.zeros(EXPERTS, HIDDEN, FFN)
    )
    for name, value in module_overrides.items():
        setattr(mod, name, value)
    with pytest.raises(ValueError, match=reason):
        mod._check_supported()


def test_supported_config_passes():
    mod = _bare(
        SituProbsTEGroupedMLP, _config(), torch.zeros(EXPERTS, 2 * FFN, HIDDEN), torch.zeros(EXPERTS, HIDDEN, FFN)
    )
    mod._check_supported()


@pytest.mark.parametrize("enabled", [False, True])
def test_split_moe_halves_follow_switch(monkeypatch, enabled):
    built = []

    def _stub(tag):
        class _Half(torch.nn.Module):
            def __init__(self, num_local_experts, config, submodules, pg_collection, name=None):
                super().__init__()
                built.append((tag, num_local_experts, submodules.linear_fc1))

        return _Half

    monkeypatch.setattr(split_moe, "TEGroupedMLP", _stub("te"))
    monkeypatch.setattr(split_moe, "SituProbsTEGroupedMLP", _stub("fused"))
    if enabled:
        monkeypatch.setenv(situ_probs_experts.SITU_PROBS_FUSED_ENV, "1")
    else:
        monkeypatch.delenv(situ_probs_experts.SITU_PROBS_FUSED_ENV, raising=False)
    submodules = GroupedMLPSubmodules(linear_fc1="fc1", linear_fc2="fc2")
    split_moe.PartiallyTrainableGroupedMLP(5, _config(), submodules, None, num_local_trainable=2)
    tag = "fused" if enabled else "te"
    assert [(t, n) for t, n, _ in built] == [(tag, 2), (tag, 3)]
    assert built[0][2] == "fc1" and built[1][2] is split_moe.MXFP4ColumnParallelGroupedLinear
