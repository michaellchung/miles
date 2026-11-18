"""The trainer's frozen MXFP4 experts compute exactly what BF16 experts holding the dequantized weights
compute: a packed row tensor round-trips the checkpoint's bytes, and a TE grouped linear built on packed rows
matches a BF16 ``TEColumnParallelGroupedLinear`` forward and backward bit for bit, owning no BF16 copy and
taking no weight gradient."""

import os
import sys

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles.utils.mxfp4 import (  # noqa: E402
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4,
    dequantize_mxfp4_rows,
    mxfp4_row_width,
    pack_mxfp4_rows,
    quantize_mxfp4,
    split_mxfp4_rows,
)
from miles_plugins.models.kimi_k3.mxfp4_experts import (  # noqa: E402
    MXFP4ColumnParallelGroupedLinear,
    MXFP4RowParallelGroupedLinear,
)

_LINEARS = {
    "fc1": (MXFP4ColumnParallelGroupedLinear, TEColumnParallelGroupedLinear),
    "fc2": (MXFP4RowParallelGroupedLinear, TERowParallelGroupedLinear),
}


@pytest.fixture(scope="module")
def config():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=3,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
    )
    yield config
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _packed_experts(num: int, out_features: int, in_features: int):
    torch.manual_seed(1)
    encoded = [
        quantize_mxfp4(torch.randn(out_features, in_features, device="cuda"), MXFP4_GROUP_SIZE) for _ in range(num)
    ]
    rows = [pack_mxfp4_rows(packed, scale, MXFP4_GROUP_SIZE) for packed, scale in encoded]
    dequantized = [dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE) for packed, scale in encoded]
    return rows, dequantized


def test_packed_rows_round_trip(config):
    rows, dequantized = _packed_experts(2, 64, 128)
    for row, reference in zip(rows, dequantized, strict=True):
        assert row.dtype == torch.uint8 and row.shape == (64, mxfp4_row_width(128, MXFP4_GROUP_SIZE))
        packed, scale = split_mxfp4_rows(row, MXFP4_GROUP_SIZE)
        assert packed.shape == (64, 64) and scale.shape == (64, 4)
        assert torch.equal(dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE), reference)
        assert torch.equal(dequantize_mxfp4_rows(row, MXFP4_GROUP_SIZE), reference)


@pytest.mark.parametrize("linear", ["fc1", "fc2"])
def test_grouped_linear_matches_bf16_host(config, linear):
    mxfp4_cls, bf16_cls = _LINEARS[linear]
    num, in_features, out_features = 3, 256, 384
    kwargs = dict(config=config, init_method=torch.nn.init.zeros_, bias=False, skip_bias_add=False, is_expert=True)
    frozen = mxfp4_cls(num, in_features, out_features, **kwargs)
    plain = bf16_cls(num, in_features, out_features, **kwargs)
    rows, dequantized = _packed_experts(num, out_features, in_features)
    for i in range(num):
        frozen_weight = getattr(frozen, f"weight{i}")
        assert frozen_weight.dtype == torch.uint8 and not frozen_weight.requires_grad
        assert frozen_weight.shape == (out_features, mxfp4_row_width(in_features, MXFP4_GROUP_SIZE))
        with torch.no_grad():
            frozen_weight.copy_(rows[i])
            getattr(plain, f"weight{i}").copy_(dequantized[i])

    m_splits = torch.tensor([5, 3, 7])
    x = torch.randn(int(m_splits.sum()), in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    out, _ = frozen(x, m_splits)
    reference, _ = plain(x_reference, m_splits)
    assert torch.equal(out, reference)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    reference.backward(grad_out)
    assert torch.equal(x.grad, x_reference.grad)
    assert all(getattr(frozen, f"weight{i}").grad is None for i in range(num))


@pytest.mark.parametrize("linear", ["fc1", "fc2"])
def test_redequant_in_backward_matches_and_frees_the_bf16_copies(config, linear, monkeypatch):
    """MILES_K31_MXFP4_REDEQUANT_BACKWARD=1 on the real TE grouped linear: the forward output and the input
    gradient are bitwise those of keeping the dequantized weights, and the graph holds no BF16 copy of them."""
    from miles_plugins.models.kimi_k3.mxfp4_redequant import REDEQUANT_ENV

    mxfp4_cls, _ = _LINEARS[linear]
    num, in_features, out_features = 3, 256, 384
    kwargs = dict(config=config, init_method=torch.nn.init.zeros_, bias=False, skip_bias_add=False, is_expert=True)
    frozen = mxfp4_cls(num, in_features, out_features, **kwargs)
    rows, _ = _packed_experts(num, out_features, in_features)
    with torch.no_grad():
        for i in range(num):
            getattr(frozen, f"weight{i}").copy_(rows[i])
    m_splits = torch.tensor([5, 0, 7])
    x0 = torch.randn(int(m_splits.sum()), in_features, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(int(m_splits.sum()), out_features, device="cuda", dtype=torch.bfloat16)

    def run(enabled):
        monkeypatch.setenv(REDEQUANT_ENV, "1" if enabled else "0")
        x = x0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out, _ = frozen(x, m_splits)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before - out.numel() * out.element_size()
        out.backward(grad_out)
        return out.detach(), x.grad, held

    out_plain, grad_plain, held_plain = run(False)
    out_redo, grad_redo, held_redo = run(True)
    assert torch.equal(out_plain, out_redo) and torch.equal(grad_plain, grad_redo)
    bf16_copies = num * out_features * in_features * 2
    assert held_plain >= bf16_copies
    assert held_redo <= held_plain - bf16_copies


def test_lut_dequant_matches_the_fp32_chain_on_cuda(config):
    """MILES_K31_MXFP4_LUT_DEQUANT=1's byte table against the fp32 chain, every byte under every finite scale."""
    from miles.utils.mxfp4 import dequantize_mxfp4_lut

    packed = torch.arange(256, dtype=torch.uint8, device="cuda").repeat(255, 1)
    scale = torch.arange(255, dtype=torch.uint8, device="cuda")[:, None].expand(255, 16).contiguous()
    chain = dequantize_mxfp4(packed, scale, MXFP4_GROUP_SIZE)
    lut = dequantize_mxfp4_lut(packed, scale, MXFP4_GROUP_SIZE)
    assert torch.equal(chain.view(torch.int16), lut.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
