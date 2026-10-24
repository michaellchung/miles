"""Keep the CPU masters of Megatron's HybridDeviceOptimizer in pinned memory.

With ``--optimizer-cpu-offload``, ``HybridDeviceOptimizer._get_sub_optimizer_param_groups``
copies each offloaded parameter to pinned host memory
(``.clone().cpu().pin_memory()``).  Under the precision-aware optimizer
(``param_update_in_fp32``) it then makes the FP32 master with
``.clone().float()``: a new tensor in pageable memory.  The CPU
sub-optimizers step those masters, and their post-step hook copies each one
back to the GPU with ``non_blocking=True``; from pageable memory that copy is
staged and synchronous.  Measured on 2 x H200 (Nemotron-3 Nano, 15.8B
parameters per GPU): ``Memcpy HtoD (Pageable -> Device)`` took 8.95 s of a
16.3 s optimizer step, and pinning the masters cut the step to 8.5 s.

:func:`install` wraps ``HybridDeviceOptimizer._init_sub_optimizers`` (Megatron
calls it at construction and again after ``load_state_dict``): once Megatron has
built the sub-optimizers, every pageable CPU tensor a CPU sub-optimizer steps
moves to pinned memory in place (``tensor.data = pinned``).  The tensor objects
stay the same, so the sub-optimizer parameter groups and every map Megatron
keeps (``param_to_fp32_param``, ``cpu_copys_map_gpu_param``,
``param_to_inner_param``) stay valid.  The pinned host cache is then emptied:
the BF16 pinned copies Megatron made only to cast from are free by then, and
the FP32 masters cannot reuse their blocks.

The optimizer's own ``pin_cpu_params`` switch (Megatron ``OptimizerConfig``,
default on) also turns this off.  Pinned memory cannot be swapped, and
PyTorch's pinned allocator may round a block up to a power of two (depending on
the PyTorch version and its pinned-allocator settings), so the masters can take
up to twice their size in host memory.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator

import torch

logger = logging.getLogger(__name__)

_WRAPPED_MARK = "_miles_pins_cpu_masters"


def _pin(value: torch.Tensor) -> torch.Tensor:
    return value.pin_memory()


def _is_pinned(value: torch.Tensor) -> bool:
    return value.is_pinned()


def _empty_pinned_cache() -> None:
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None:
        empty()


def _cpu_stepped_tensors(optimizer) -> Iterator[torch.Tensor]:
    """Every CPU tensor the CPU sub-optimizers step, once each."""

    seen: set[int] = set()
    for group in getattr(optimizer, "cpu_param_groups", None) or ():
        for tensor in group["params"]:
            if id(tensor) not in seen and tensor.device.type == "cpu":
                seen.add(id(tensor))
                yield tensor


def pin_cpu_optimizer_params(
    optimizer,
    *,
    pin: Callable[[torch.Tensor], torch.Tensor] | None = None,
    is_pinned: Callable[[torch.Tensor], bool] | None = None,
) -> int:
    """Move every pageable CPU tensor ``optimizer``'s CPU sub-optimizers step into pinned memory.

    In place (``tensor.data``), values unchanged; returns the bytes moved.  No-op
    when the optimizer was built with ``pin_cpu_params=False``."""

    if not getattr(optimizer, "pin_cpu_params", True):
        return 0
    pin = pin or _pin
    is_pinned = is_pinned or _is_pinned
    moved = 0
    for tensor in _cpu_stepped_tensors(optimizer):
        if is_pinned(tensor):
            continue
        pinned = pin(tensor.data)
        if pinned.dtype != tensor.dtype or pinned.shape != tensor.shape:
            raise RuntimeError("pinned copy of a HybridDeviceOptimizer CPU master changed its dtype or shape")
        tensor.data = pinned
        moved += tensor.numel() * tensor.element_size()
    return moved


def wrap_init_sub_optimizers(hybrid_optimizer_cls) -> bool:
    """Wrap ``hybrid_optimizer_cls._init_sub_optimizers`` once; returns False when already wrapped."""

    original = hybrid_optimizer_cls._init_sub_optimizers
    if getattr(original, _WRAPPED_MARK, False):
        return False

    @functools.wraps(original)
    def init_sub_optimizers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        moved = pin_cpu_optimizer_params(self)
        if moved:
            _empty_pinned_cache()
            logger.info("HybridDeviceOptimizer: moved %.2f GB of CPU masters to pinned memory", moved / 1e9)
        return result

    setattr(init_sub_optimizers, _WRAPPED_MARK, True)
    hybrid_optimizer_cls._init_sub_optimizers = init_sub_optimizers
    return True


def install() -> bool:
    """Wrap Megatron's HybridDeviceOptimizer (once per process); False when Megatron has none."""

    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ImportError:
        return False
    wrap_init_sub_optimizers(HybridDeviceOptimizer)
    return True
