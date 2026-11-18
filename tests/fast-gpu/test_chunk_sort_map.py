"""MILES_K31_CHUNK_SORT_MAP_TORCH on CUDA: the torch row map equals TE's Triton kernel element for element at the
K3.1 EP16 shape (16 ranks x 56 local experts, ~330k routed rows), and the sorted tensor is bitwise the same."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", num_gpus=1, labels=["megatron"], hardware=["hopper", "blackwell"]
)

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
te_perm = pytest.importorskip("transformer_engine.pytorch.triton.permutation")

from miles.utils.chunk_sort_map import make_chunk_sort_map_torch  # noqa: E402


@pytest.mark.parametrize("ep,n_local,mean", [(16, 56, 370), (8, 3, 5), (2, 1, 0)])
def test_torch_map_is_the_triton_map(ep, n_local, mean):
    g = torch.Generator(device="cuda").manual_seed(0)
    n = ep * n_local
    sizes = torch.poisson(torch.full((n,), float(mean), device="cuda"), generator=g).to(torch.int64)
    order = torch.arange(n, device="cuda").view(ep, n_local).t().reshape(-1)  # [rank, expert] -> [expert, rank]
    tokens = int(sizes.sum()) + 5  # a few padding rows: identity
    want = te_perm.make_chunk_sort_map(sizes, order, tokens, n)
    got = make_chunk_sort_map_torch(sizes, order, tokens, n)
    assert torch.equal(got, want)
    x = torch.randn(tokens, 64, device="cuda").bfloat16()
    a, _ = te_perm.sort_chunks_by_map(x, want, None, tokens, 64, is_forward=True)
    b, _ = te_perm.sort_chunks_by_map(x, got, None, tokens, 64, is_forward=True)
    assert torch.equal(a, b)
    impls = (te_perm.make_chunk_sort_map, make_chunk_sort_map_torch)
    for impl in impls:
        impl(sizes, order, tokens, n)
    torch.cuda.synchronize()
    times = []
    for impl in impls:
        fn = lambda impl=impl: impl(sizes, order, tokens, n)  # noqa: E731
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 20)
    print(
        f"BENCH chunk_sort_map ep={ep} n_local={n_local} rows={tokens} "
        f"triton_ms={times[0]:.3f} torch_ms={times[1]:.3f}"
    )
