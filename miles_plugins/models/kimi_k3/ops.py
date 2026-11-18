import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# MILES_K3_SITU_TRITON=1 (default off): run SiTU-and-mul as one fused Triton kernel per direction, the same
# fp32 op chain replayed op for op (situ_triton.py), saving only the BF16 input instead of six fp32
# [rows, N] intermediates (~18 GiB per routed-expert layer at 262k rows).
SITU_TRITON = os.environ.get("MILES_K3_SITU_TRITON", "0") == "1"


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    if SITU_TRITON and x.is_cuda and x.dtype == torch.bfloat16 and beta == 4.0 and linear_beta == 25.0:
        from .situ_triton import fused_situ_and_mul

        lead = x.shape[:-1]
        return fused_situ_and_mul(x.reshape(-1, x.shape[-1]).contiguous()).reshape(*lead, x.shape[-1] // 2)
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


class KimiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


# Tokens per block of the attention residual aggregate; each token mixes only its own rows, so
# blocks are exact. The fp32 rows of one call are [T, blocks + 1, H] (7.9 GiB at 32k tokens and
# 9 rows); scoring them in checkpointed token blocks bounds those temporaries. <= 0 disables.
ATTN_RES_TOKEN_CHUNK = int(os.environ.get("MILES_K3_ATTN_RES_TOKEN_CHUNK", "4096"))
# Fused Triton mix (attn_res_fused.py): reads the bf16 rows directly and saves only its inputs, so the token
# blocks above are not used. CUDA tensors only (or CPU under TRITON_INTERPRET=1); default off.
ATTN_RES_FUSED = os.environ.get("MILES_K3_ATTN_RES_FUSED", "0") == "1"


def _use_fused_mix(prefix_sum: torch.Tensor, block_residual: torch.Tensor, fused: bool | None) -> bool:
    if not (ATTN_RES_FUSED if fused is None else fused):
        return False
    if prefix_sum.dtype != block_residual.dtype:
        return False
    return prefix_sum.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def _attn_res_mix(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + eps)
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
    chunk: int | None = None,
    fused: bool | None = None,
) -> torch.Tensor:
    chunk = ATTN_RES_TOKEN_CHUNK if chunk is None else int(chunk)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    tokens = prefix_sum.shape[0]
    if _use_fused_mix(prefix_sum, block_residual, fused):
        # deferred: triton is needed only on the fused path
        from miles_plugins.models.kimi_k3.attn_res_fused import attn_res_mix_fused

        mixed = attn_res_mix_fused(prefix_sum, block_residual, score_weight, score_norm.eps)
    elif chunk <= 0 or tokens <= chunk:
        mixed = _attn_res_mix(prefix_sum, block_residual, score_weight, score_norm.eps)
    else:
        mixed = torch.cat(
            [
                checkpoint(
                    _attn_res_mix,
                    prefix_sum[start : start + chunk],
                    block_residual[start : start + chunk],
                    score_weight,
                    score_norm.eps,
                    use_reentrant=False,
                )
                for start in range(0, tokens, chunk)
            ]
        )
    return output_norm(mixed)
