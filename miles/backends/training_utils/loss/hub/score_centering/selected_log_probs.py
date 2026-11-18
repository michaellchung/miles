"""Selected-token log-probabilities from vocabulary-sharded logits."""

import os
from typing import Any

import torch
import torch.distributed as dist


class _SelectedLogProbs(torch.autograd.Function):
    """Vocabulary-sharded log-softmax gather with a replicated loss on TP ranks.

    Only selected logits and normalization scalars are communicated. Backward
    computes each rank's vocabulary slice directly; reducing identical output
    gradients across TP ranks would incorrectly multiply the gradient by TP.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        vocab_size: int,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.shape[-1]
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        padded = torch.arange(width, device=logits.device) + rank * width >= vocab_size
        work = logits.masked_fill(padded, -torch.inf)
        maximum = work.amax(-1, keepdim=True)
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        probabilities = (work - maximum).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        selected = torch.where(local, work.gather(-1, local_ids), 0.0)
        if group is not None:
            dist.all_reduce(denominator, group=group)
            dist.all_reduce(selected, group=group)
        probabilities.div_(denominator)
        entropy = probabilities.new_zeros(probabilities.size(0))
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, probabilities.log(), 0.0)).sum(-1)
            if group is not None:
                dist.all_reduce(entropy, group=group)
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid, entropy)
        return torch.where(valid, selected - maximum - denominator.log(), 0.0), entropy

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        grad_logits = probabilities * (-grad.sum(-1, keepdim=True))
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            grad_logits -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        return grad_logits, None, None, None, None


LEAN_ENV = "MILES_SC_LOGPROB_LEAN"
_LEAN_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def lean_mode() -> str | None:
    """``MILES_SC_LOGPROB_LEAN`` (default off): ``1`` keeps the probabilities, ``recompute`` rebuilds them."""
    mode = os.environ.get(LEAN_ENV, "0")
    if mode in ("0", ""):
        return None
    if mode not in ("1", "recompute"):
        raise ValueError(f"{LEAN_ENV} must be 0, 1 or recompute, got {mode!r}")
    return mode


def _unnormalized_probabilities(work: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    """fp32 ``exp(work - maximum)`` without an fp32 copy of native-dtype ``work``.

    The subtraction reads ``work`` as fp32 inside the kernel, so it equals the subtraction on
    ``work.float()``; ``exp_`` runs the kernel ``exp`` runs.
    """
    return torch.sub(work, maximum).exp_()


def _masked(logits: torch.Tensor, padded: torch.Tensor | None) -> torch.Tensor:
    """The reference's fp32 padding-masked logits on a rank with padding columns; the logits as they are otherwise."""
    return logits if padded is None else logits.float().masked_fill(padded, -torch.inf)


class _LeanSelectedLogProbs(torch.autograd.Function):
    """``_SelectedLogProbs`` at temperature 1 on native-dtype logits, bit for bit, with fewer vocabulary passes.

    The reference upcasts a chunk to fp32, divides it by the temperature and masks the vocabulary padding,
    three fp32 copies of the chunk, before the softmax; its backward goes back through the division and the
    upcast. Here the softmax reads the native logits as fp32 inside its first subtraction, padding is masked
    only on a rank that holds padding columns (on the others the reference's mask changes nothing), and the
    backward writes the fp32 gradient into the saved probabilities and casts it once. Dividing by 1 is
    exact, so the results and the gradient are the reference's.

    ``recompute`` saves the logits chunk and the row normalizers instead of the fp32 probabilities and
    recomputes them in backward with the same kernels: one fp32 ``[rows, V/TP]`` buffer per chunk fewer
    held from the forward to the backward, for one more pass in the backward.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        vocab_size: int,
        with_entropy: bool,
        recompute: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.shape[-1]
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        padded = None
        if (rank + 1) * width > vocab_size:
            padded = torch.arange(width, device=logits.device) + rank * width >= vocab_size
        work = _masked(logits, padded)
        maximum = work.amax(-1, keepdim=True).float()
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        probabilities = _unnormalized_probabilities(work, maximum)
        denominator = probabilities.sum(-1, keepdim=True)
        selected = torch.where(local, work.gather(-1, local_ids).float(), 0.0)
        if group is not None:
            dist.all_reduce(denominator, group=group)
            dist.all_reduce(selected, group=group)
        probabilities.div_(denominator)
        entropy = probabilities.new_zeros(probabilities.size(0))
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, probabilities.log(), 0.0)).sum(-1)
            if group is not None:
                dist.all_reduce(entropy, group=group)
        ctx.with_entropy = with_entropy
        ctx.recompute = recompute
        ctx.padded = padded
        ctx.logits_dtype = logits.dtype
        if recompute:
            del probabilities, work
            ctx.save_for_backward(logits, maximum, denominator, local_ids, local, valid, entropy)
        else:
            ctx.save_for_backward(probabilities, local_ids, local, valid, entropy)
        return torch.where(valid, selected - maximum - denominator.log(), 0.0), entropy

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        if ctx.recompute:
            logits, maximum, denominator, local_ids, local, valid, entropy = ctx.saved_tensors
            probabilities = _unnormalized_probabilities(_masked(logits, ctx.padded), maximum).div_(denominator)
        else:
            probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        scale = -grad.sum(-1, keepdim=True)
        if ctx.with_entropy:
            grad_logits = probabilities * scale
        else:
            # the probabilities are not read again: the gradient takes their buffer
            grad_logits = probabilities.mul_(scale)
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            grad_logits -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        return grad_logits.to(ctx.logits_dtype), None, None, None, None, None


class _SupportLogProbs(torch.autograd.Function):
    """Score the sampled token and complete support without saving vocabulary-sized probabilities."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        temperature: float,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.size(-1)
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        selected = torch.where(local, logits.gather(-1, local_ids).to(dtype), 0.0) / temperature
        if group is not None:
            dist.all_reduce(selected, group=group)
        head_valid = valid[:, 1:]
        has_support = head_valid.any(-1, keepdim=True)
        head = selected[:, 1:].masked_fill(~head_valid, -torch.inf)
        head = torch.where(has_support, head, 0.0)
        logp = selected - torch.logsumexp(head, dim=-1, keepdim=True)
        logp = torch.where(valid & has_support, logp, 0.0)
        probabilities = torch.where(head_valid, logp[:, 1:].exp(), 0.0)
        entropy = logits.new_zeros(logits.size(0), dtype=dtype)
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, logp[:, 1:], 0.0)).sum(-1)
        ctx.logits_shape = logits.shape
        ctx.temperature = temperature
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid & has_support, entropy)
        return logp, entropy

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        head_grad = probabilities * (-grad.sum(-1, keepdim=True))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            head_grad -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        grad_logits = grad.new_zeros(ctx.logits_shape)
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        grad_logits.scatter_add_(-1, local_ids[:, 1:], head_grad.masked_fill(~local[:, 1:], 0.0))
        grad_logits.div_(ctx.temperature)
        return grad_logits, None, None, None, None


def selected_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
) -> torch.Tensor:
    """Log-probabilities at global IDs [T, K] from local logits [T, V/TP].

    -1 is a padding ID and produces zero with zero gradient. The true vocabulary
    size excludes Megatron's padded vocabulary entries from normalization.
    """
    return selected_log_probs_and_entropy(
        logits, token_ids, group=group, vocab_size=vocab_size, temperature=temperature, chunk_size=chunk_size
    )[0]


def selected_log_probs_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
    with_entropy: bool = False,
    sampling_support: bool = False,
    validate_ids: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected logprobs and optional entropy of the same unpadded distribution.

    With sampling_support, column zero is the sampled token and remaining
    columns are the complete support. Rows without candidates contribute zero.
    ``validate_ids=False`` skips the range check of ``token_ids``, which makes the
    host wait for the device; the caller has checked them.
    """
    size = dist.get_world_size(group) if group is not None else 1
    vocab_size = vocab_size if vocab_size is not None else logits.size(-1) * size
    if temperature <= 0 or not 0 < vocab_size <= logits.size(-1) * size:
        raise ValueError("Score centering needs a positive temperature and valid vocabulary size")
    if token_ids.ndim != 2 or logits.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected logits [T, V/TP] and selected token IDs [T, K]")
    if validate_ids and ((token_ids < -1) | (token_ids >= vocab_size)).any():
        raise ValueError("Score-centering token ID is outside the model vocabulary")
    if logits.size(0) == 0:
        return logits.sum(-1, keepdim=True).expand_as(token_ids), logits.sum(-1)
    chunk_size = chunk_size if chunk_size > 0 else logits.size(0)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    # the lean path reproduces the reference only where its temperature division is exact
    lean = lean_mode() if temperature == 1.0 and logits.dtype in _LEAN_DTYPES else None
    chunks = []
    for chunk, ids in zip(logits.split(chunk_size), token_ids.split(chunk_size), strict=True):
        if sampling_support:
            chunks.append(_SupportLogProbs.apply(chunk, ids, group, temperature, with_entropy))
        elif lean:
            chunks.append(
                _LeanSelectedLogProbs.apply(chunk, ids, group, vocab_size, with_entropy, lean == "recompute")
            )
        else:
            chunks.append(_SelectedLogProbs.apply(chunk.to(dtype) / temperature, ids, group, vocab_size, with_entropy))
    return torch.cat([chunk[0] for chunk in chunks]), torch.cat([chunk[1] for chunk in chunks])
