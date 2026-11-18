"""MILES_K3_ATTN_RES_BANK_PAD_ROWS (test only) is 0 unless set, and a padded bank row only takes softmax mass."""

import pytest
import torch

layers = pytest.importorskip("miles_plugins.models.kimi_k3.layers")
from miles_plugins.models.kimi_k3.ops import _attn_res_mix  # noqa: E402


def test_pad_rows_switch(monkeypatch):
    monkeypatch.delenv("MILES_K3_ATTN_RES_BANK_PAD_ROWS", raising=False)
    assert layers.attn_res_bank_pad_rows() == 0
    monkeypatch.setenv("MILES_K3_ATTN_RES_BANK_PAD_ROWS", "5")
    assert layers.attn_res_bank_pad_rows() == 5
    monkeypatch.setenv("MILES_K3_ATTN_RES_BANK_PAD_ROWS", "-1")
    with pytest.raises(AssertionError):
        layers.attn_res_bank_pad_rows()


def test_zero_rows_score_zero_and_add_nothing():
    torch.manual_seed(0)
    prefix, bank, weight = torch.randn(7, 16), torch.randn(7, 2, 16), torch.randn(16)
    padded = torch.cat((torch.zeros(7, 5, 16), bank), dim=-2)
    rows = torch.cat((bank, prefix.unsqueeze(-2)), dim=-2)
    normalized = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + 1e-6)
    scores = (normalized * weight).sum(-1)
    # five extra softmax entries of score 0 with value 0
    want = (torch.softmax(torch.cat((torch.zeros(7, 5), scores), -1), -1)[..., 5:, None] * rows).sum(-2)
    torch.testing.assert_close(_attn_res_mix(prefix, padded, weight, 1e-6), want)
