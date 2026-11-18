"""MILES_K31_CHUNK_SORT_MAP_TORCH: the torch chunk-sort row map is element for element TE's Triton kernel's."""

import types

import pytest
import torch

from miles.utils import chunk_sort_map as csm


def _triton_reference(split_sizes, sorted_indices, num_tokens, num_splits):
    """``_make_chunk_sort_map_kernel`` (TE 2.17 common/triton/permutation.py), one program per row, in Python."""
    sizes = [int(x) for x in split_sizes[:num_splits]]
    order = [int(x) for x in sorted_indices[:num_splits]]
    cumsum, acc = [], 0
    for s in sizes:
        acc += s
        cumsum.append(acc)
    total = acc
    out = []
    for pid in range(num_tokens):
        mask = [1 if c <= pid else 0 for c in cumsum]
        chunk = sum(mask)
        presum = sum(s * m for s, m in zip(sizes, mask, strict=True))
        offset = pid - presum
        match = [1 if o == chunk else 0 for o in order]
        out_pos = match.index(1) if 1 in match else 0  # tl.argmax: first max
        out_split = [sizes[o] for o in order]
        dst = sum(s for q, s in enumerate(out_split) if q < out_pos) + offset
        out.append(dst if pid < total else pid)
    return torch.tensor(out, dtype=torch.int32)


@pytest.mark.parametrize("num_splits,extra,seed", [(1, 0, 0), (4, 0, 1), (16, 3, 2), (56, 0, 3), (96, 7, 4)])
def test_matches_the_triton_kernel(num_splits, extra, seed):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(0, 9, (num_splits,), generator=g)
    sizes[torch.rand(num_splits, generator=g) < 0.25] = 0  # cold experts
    # Megatron's sort: [rank, expert] chunks -> [expert, rank]; any permutation works
    order = torch.randperm(num_splits, generator=g)
    num_tokens = int(sizes.sum()) + extra  # phantom rows beyond sum(split) map to themselves
    for dtype in (torch.int64, torch.int32):
        got = csm.make_chunk_sort_map_torch(sizes.to(dtype), order.to(dtype), num_tokens, num_splits)
        want = _triton_reference(sizes, order, num_tokens, num_splits)
        assert got.dtype == torch.int32 and torch.equal(got, want), (got, want)
    if extra == 0 and num_tokens:
        assert torch.equal(torch.sort(got).values, torch.arange(num_tokens, dtype=torch.int32))  # a permutation


def test_megatron_sort_layout():
    # EP 4 x 3 local experts: received chunks [rank, expert]; sorted to [expert, rank]
    ep, n_local = 4, 3
    sizes = torch.tensor([2, 0, 1, 1, 3, 0, 0, 2, 2, 1, 1, 1])
    order = torch.arange(ep * n_local).view(ep, n_local).t().reshape(-1)
    got = csm.make_chunk_sort_map_torch(sizes, order, int(sizes.sum()), ep * n_local)
    assert torch.equal(got, _triton_reference(sizes, order, int(sizes.sum()), ep * n_local))
    x = torch.arange(int(sizes.sum()))
    out = torch.empty_like(x)
    out[got.long()] = x
    chunks = list(torch.split(x, sizes.tolist()))
    assert torch.equal(out, torch.cat([chunks[i] for i in order.tolist()]))


def test_install_is_off_by_default_and_idempotent(monkeypatch):
    monkeypatch.delenv(csm.CHUNK_SORT_MAP_ENV, raising=False)
    assert not csm.chunk_sort_map_enabled()
    monkeypatch.setenv(csm.CHUNK_SORT_MAP_ENV, "1")
    assert csm.chunk_sort_map_enabled()
    fake = types.SimpleNamespace(make_chunk_sort_map=lambda *a: "triton")
    assert csm.install_chunk_sort_map(fake) and csm.install_chunk_sort_map(fake)
    assert fake.make_chunk_sort_map._miles_torch and fake.make_chunk_sort_map._miles_original() == "triton"
    sizes, order = torch.tensor([1, 2]), torch.tensor([1, 0])
    assert torch.equal(fake.make_chunk_sort_map(sizes, order, 3, 2), torch.tensor([2, 0, 1], dtype=torch.int32))
