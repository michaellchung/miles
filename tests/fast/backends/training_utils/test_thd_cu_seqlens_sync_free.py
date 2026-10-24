"""THD cu_seqlens / max_seqlen from the host copy equal the GPU-read values; CUDA is stubbed to run on CPU."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.data as data_mod
from miles.backends.training_utils.data import get_batch, thd_cu_seqlens_on_device
from miles.utils import sync_free


class _Iterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys if key in self._batch}


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)


def _batch(lengths):
    return {
        "tokens": [torch.arange(1, n + 1, dtype=torch.long) for n in lengths],
        "loss_masks": [torch.ones(max(1, n // 2), dtype=torch.int) for n in lengths],
        "total_lengths": list(lengths),
        "response_lengths": [max(1, n // 2) for n in lengths],
    }


def _get_batch(lengths, *, cp_size, cp_rank, allgather_cp, switch, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(size=1))
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
    return get_batch(_Iterator(_batch(lengths)), keys, pad_multiplier=128, qkv_format="thd", allgather_cp=allgather_cp)


def _gpu_read_max_seqlen(cu_list, cp_size):
    # the expression get_batch used before (there on the GPU, with .item())
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int) * cp_size
    return cu_seqlens, (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()


LENGTHS = [[7, 13, 5, 9, 4], [4096], [1, 1, 1], [8191, 1, 300]]


@pytest.mark.parametrize("lengths", LENGTHS, ids=range(len(LENGTHS)))
@pytest.mark.parametrize("cp_size,cp_rank,allgather_cp", [(1, 0, False), (2, 1, False), (2, 0, True)])
def test_get_batch_same_with_and_without_host_syncs(lengths, cp_size, cp_rank, allgather_cp, monkeypatch):
    new = _get_batch(lengths, cp_size=cp_size, cp_rank=cp_rank, allgather_cp=allgather_cp, switch="1",
                     monkeypatch=monkeypatch)
    old = _get_batch(lengths, cp_size=cp_size, cp_rank=cp_rank, allgather_cp=allgather_cp, switch="0",
                     monkeypatch=monkeypatch)
    assert torch.equal(new["tokens"], old["tokens"])
    assert new["cu_seqlens"].dtype == old["cu_seqlens"].dtype == torch.int32
    assert torch.equal(new["cu_seqlens"], old["cu_seqlens"])
    assert type(new["max_seqlen"]) is type(old["max_seqlen"]) is int
    assert new["max_seqlen"] == old["max_seqlen"]
    assert torch.equal(new["cu_seqlens_cpu"], new["cu_seqlens"])


@pytest.mark.parametrize("cu_list", [[0, 7, 20, 25, 34, 38, 128], [0, 4096], [0, 1, 2, 3, 128]])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("switch", ["1", "0"])
def test_thd_cu_seqlens_on_device_matches_the_gpu_read(cu_list, cp_size, switch, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    expected_cu, expected_max = _gpu_read_max_seqlen(cu_list, cp_size)
    cu_seqlens, max_seqlen = thd_cu_seqlens_on_device(torch.tensor(cu_list, dtype=torch.int) * cp_size, "cpu")
    assert torch.equal(cu_seqlens, expected_cu) and cu_seqlens.dtype == torch.int32
    assert max_seqlen == expected_max and type(max_seqlen) is int
