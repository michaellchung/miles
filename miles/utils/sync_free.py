"""Switch and helpers for the host-sync-free paths of the Megatron training step.

A training step of a MoE model on Megatron used to make the host wait for the GPU
thousands of times per step: ``.item()``, boolean-mask indexing, ``repeat_interleave``
with tensor repeats, ``torch.tensor`` from GPU scalars and blocking host-to-device
copies.  ``torch.cuda.set_sync_debug_mode("warn")`` counted 2,917 such waits per rank
per step for Nemotron-3 Nano 30B-A3B on 8 x H200 (EP8, DeepEP, routing replay).  After
each wait the GPU idles until the host has launched the next kernels.

The sync-free paths compute the same values bit for bit; they read sizes from host
copies, build tensors on the device and copy host data without waiting.
``MILES_HOST_SYNC_FREE=0`` restores the previous code paths (for A/B measurements and
as an escape hatch); the default is on.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import textwrap
from collections.abc import Callable, Sequence
from numbers import Number

import torch

ENV_VAR = "MILES_HOST_SYNC_FREE"
_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def enabled() -> bool:
    """True unless ``MILES_HOST_SYNC_FREE`` is set to 0/false/no/off."""

    return os.environ.get(ENV_VAR, "1").strip().lower() not in _OFF_VALUES


def source_sha256(function: Callable) -> str | None:
    """SHA-256 of ``function``'s dedented source; None when the source cannot be read.

    The sync-free replacements of Megatron code check it, so they only take over from
    the exact upstream code they reproduce.
    """

    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return None
    return source_text_sha256(source)


def source_text_sha256(source: str) -> str:
    """SHA-256 of ``source`` after ``textwrap.dedent`` (the form :func:`source_sha256` hashes)."""

    return hashlib.sha256(textwrap.dedent(source).encode()).hexdigest()


def to_device_async(cpu_tensor: torch.Tensor, device: torch.device | str | int) -> torch.Tensor:
    """Copy a host tensor to ``device`` without making the host wait for the GPU.

    For a CUDA target the data goes through a pinned staging copy, and the copy runs
    on the current stream with ``non_blocking=True``.  Kernels queued later on that
    stream see the data.  The caller may drop the staging tensor at once: PyTorch's
    pinned allocator does not reuse a block before the copies recorded on it finish.
    """

    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type != "cuda":
        return cpu_tensor.to(device)
    return cpu_tensor.pin_memory().to(device, non_blocking=True)


def _tensor_dtype(value) -> torch.dtype | None:
    """The dtype ``torch.tensor`` infers for one list element; None for other elements."""

    if isinstance(value, torch.Tensor):
        return value.dtype if value.dim() == 0 else None
    if isinstance(value, bool):
        return torch.bool
    if isinstance(value, int):
        return torch.int64
    if isinstance(value, float):
        return torch.get_default_dtype()
    return None


def _number_on_device(value: Number, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # torch.tensor converts the number to dtype on the host (rounding, overflow to inf);
    # torch.full then writes that exact value with a fill kernel, without a host-to-device copy.
    return torch.full((), torch.tensor(value, dtype=dtype).item(), dtype=dtype, device=device)


def _element_on_device(value, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return _number_on_device(value, dtype, device)
    if value.device == device:
        return value.detach().to(dtype)
    if value.device.type == "cpu":
        return _number_on_device(value.item(), dtype, device)
    return None


def scalars_to_tensor(values: Sequence, *, device: torch.device | str | int) -> torch.Tensor:
    """``torch.tensor(values, device=device)`` for a list of 0-d tensors and Python numbers, without host syncs.

    ``torch.tensor`` reads every GPU element on the host (one wait each) and copies the
    result to the GPU (one more wait).  This builds the 1-d result on ``device``: the
    dtype is the one ``torch.tensor`` infers (type promotion over the elements) and the
    values are the same.  Element casts are exact for the element types a loss logs
    (bool, integers up to 2**24, float16/bfloat16/float32/float64).  Other inputs (an
    empty list, nested lists, tensors with dimensions, numpy numbers, tensors on another
    GPU) and ``MILES_HOST_SYNC_FREE=0`` go to ``torch.tensor``.
    """

    dtypes = [_tensor_dtype(v) for v in values]
    if not values or None in dtypes or not enabled():
        return torch.tensor(values, device=device)
    dtype = functools.reduce(torch.promote_types, dtypes)
    target = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    parts = [_element_on_device(v, dtype, target) for v in values]
    if any(part is None for part in parts):
        return torch.tensor(values, device=device)
    return torch.stack(parts)


def scalar_to_tensor(value, *, device: torch.device | str | int) -> torch.Tensor:
    """``torch.tensor(value, device=device)`` for one 0-d tensor or Python number, without host syncs."""

    if not enabled() or _tensor_dtype(value) is None:
        return torch.tensor(value, device=device)
    return scalars_to_tensor([value], device=device).reshape(())
