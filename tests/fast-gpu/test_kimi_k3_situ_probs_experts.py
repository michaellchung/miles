"""SituProbsTEGroupedMLP (MILES_K3_SITU_PROBS_FUSED) equals Megatron's TEGroupedMLP bit for bit on TE grouped
linears: output, input gradient, router-prob gradient and both expert weight gradients, including empty
experts and an all-empty microbatch."""

import os

# cuBLAS on several streams (TE grouped GEMM) is only run-to-run deterministic with a fixed workspace config
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from tests.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from miles_plugins.models.kimi_k3.ops import situ_and_mul  # noqa: E402
from miles_plugins.models.kimi_k3.situ_probs_experts import SituProbsTEGroupedMLP  # noqa: E402

HIDDEN, FFN, EXPERTS = 256, 3072, 4


@pytest.fixture(scope="module")
def setup():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=FFN,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        gated_linear_unit=True,
        gated_activation_func=situ_and_mul,
        bias_activation_fusion=False,
    )
    yield config, ProcessGroupCollection.use_mpu_process_groups()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.int32)


def _mismatch(name, got, ref) -> str:
    if got.dtype != ref.dtype or got.shape != ref.shape:
        return f"{name}: {got.dtype}{tuple(got.shape)} vs {ref.dtype}{tuple(ref.shape)}"
    differ = _bits(got) != _bits(ref)
    if not differ.any():
        return ""
    delta = (got.float() - ref.float()).abs()
    return f"{name}: {int(differ.sum())}/{differ.numel()} differ, max |diff| {delta.max().item():.3e}"


def _run(mod, x, tokens, probs, dy):
    """Output, input / prob / parameter grads and the fc2 input (the SiTU*probs activation), by name."""
    seen = {}
    hook = mod.linear_fc2.register_forward_pre_hook(lambda _m, args: seen.update(act=args[0].detach().clone()))
    xi, pi = x.clone().requires_grad_(), probs.clone().requires_grad_()
    out, bias = mod(xi, tokens, pi)
    hook.remove()
    assert bias is None
    out.backward(dy)
    named = {"act": seen["act"], "out": out.detach(), "dx": xi.grad, "dprobs": pi.grad}
    # TE skips the wgrad GEMM of an empty expert, so without fused accumulation its .grad holds whatever the
    # allocator returned (production accumulates into main_grad); compare the experts that received tokens
    named.update({f"grad.{n}": p.grad for n, p in mod.named_parameters() if tokens[int(n.rsplit("weight", 1)[1])] > 0})
    return named


@pytest.mark.parametrize("counts", [[700, 300, 1301, 47], [700, 0, 1301, 47], [0, 0, 0, 0]])
def test_matches_te_grouped_mlp_bitwise(setup, counts):
    config, pg_collection = setup
    submodules = GroupedMLPSubmodules(linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear)
    ref = TEGroupedMLP(EXPERTS, config, submodules, pg_collection).cuda()
    fused = SituProbsTEGroupedMLP(EXPERTS, config, submodules, pg_collection).cuda()
    torch.manual_seed(2)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    fused.load_state_dict(ref.state_dict())

    rows = sum(counts)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 2).to(torch.bfloat16)
    probs = torch.rand(rows, device="cuda")
    dy = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
    tokens = torch.tensor(counts)

    want = _run(ref, x, tokens, probs, dy)
    for p in ref.parameters():
        p.grad = None
    again = _run(ref, x, tokens, probs, dy)  # control: TE / cuBLAS run-to-run determinism on this host
    got = _run(fused, x, tokens, probs, dy)
    assert want.keys() == got.keys()
    assert rows == 0 or all(t is not None for t in want.values())
    unstable = [m for n in want if want[n] is not None and (m := _mismatch(n, again[n], want[n]))]
    assert not unstable, f"reference TEGroupedMLP is not deterministic: {unstable}"
    problems = []
    for name, ref_t in want.items():
        if ref_t is None or got[name] is None:
            if ref_t is not got[name]:
                problems.append(f"{name}: grad None mismatch")
            continue
        if msg := _mismatch(name, got[name], ref_t):
            problems.append(msg)
    assert not problems, problems
