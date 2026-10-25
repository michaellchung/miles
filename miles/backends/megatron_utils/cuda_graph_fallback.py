"""Run a Megatron "local" CUDA-graphed layer eagerly when no captured graph matches its inputs.

Megatron (radixark a84b10547) captures the local per-layer graphs at the end of the first training
forward/backward (``schedules.py`` -> ``create_cudagraphs()``). After that,
``CudaGraphManager.get_cudagraph_runner`` raises ``assert False`` for any call whose input signature
(tensor shapes, ``cu_seqlens`` length, ``max_seqlen`` value, ...) no runner was captured with
(``cuda_graphs.py:1493-1499``). With ``--thd-bucket-sizes`` most micro batches fall into captured
buckets, but a micro batch above the largest bucket, with more sequences than
``--thd-max-packed-sequences``, or in a bucket that did not occur in the capture step would stop
training. This patch runs such a call through the module's normal eager forward instead (the same
math, no graph) and counts it in ``STATS``.

Installed once per process by ``install()``, only when ``CudaGraphManager.get_cudagraph_runner`` has
the a84b10547 source (sha256 of the dedented source). Otherwise it logs a WARNING and changes nothing.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import textwrap

import torch

logger = logging.getLogger(__name__)

# sha256 of textwrap.dedent(inspect.getsource(CudaGraphManager.get_cudagraph_runner)) in a84b10547
_EXPECTED_GET_RUNNER_SHA256 = "6bf153e8f395e039367b35607b1a67bdb2fa6ed3b21f5b4c3ec755415ce8f9cf"
STATS = {"eager_fallback_calls": 0}
_INSTALLED = False


class _NoMatchingGraph(Exception):
    """Raised by the patched runner lookup when graphs exist and none fits the call."""


def _source_sha256(fn) -> str | None:
    try:
        return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def _find_runner(manager, args, kwargs, graph_status):
    """The first FWD_READY runner whose captured signature fits (a84b10547 ``is_valid``)."""
    require_grad = manager.need_backward and torch.is_grad_enabled()
    for runner in manager.cudagraph_runners:
        if (
            runner.status == graph_status.FWD_READY
            and not runner.get_mismatch_errors(args, kwargs)
            and (not require_grad or runner.grad_enabled)
        ):
            return runner
    return None


def install() -> bool:
    """Patch ``CudaGraphManager`` once; returns whether the patch is active."""
    global _INSTALLED
    if _INSTALLED:
        return True
    from megatron.core.transformer import cuda_graphs as cg
    from megatron.core.transformer.module import MegatronModule

    manager_cls = cg.CudaGraphManager
    digest = _source_sha256(manager_cls.get_cudagraph_runner)
    if digest != _EXPECTED_GET_RUNNER_SHA256:
        logger.warning(
            "cuda_graph_fallback: CudaGraphManager.get_cudagraph_runner source sha256 %s does not match the "
            "a84b10547 source; keeping Megatron's assert on unmatched CUDA graph inputs",
            digest,
        )
        return False

    orig_get_runner = manager_cls.get_cudagraph_runner
    orig_call = manager_cls.__call__

    def get_cudagraph_runner(self, megatron_module, args, kwargs, reuse_cudagraphs, cache_key=None):
        if reuse_cudagraphs and cache_key is None and cg._CudagraphGlobalRecord.cudagraph_created:
            runner = _find_runner(self, args, kwargs, cg._GraphStatus)
            if runner is None:
                raise _NoMatchingGraph()
            return runner
        return orig_get_runner(self, megatron_module, args, kwargs, reuse_cudagraphs, cache_key=cache_key)

    def call(self, megatron_module, args, kwargs, cache_key=None):
        try:
            return orig_call(self, megatron_module, args, kwargs, cache_key=cache_key)
        except _NoMatchingGraph:
            STATS["eager_fallback_calls"] += 1
            if STATS["eager_fallback_calls"] in (1, 10, 100, 1000):
                logger.info("cuda_graph_fallback: %d eager calls so far", STATS["eager_fallback_calls"])
            if self.func is not None:
                return self.func(*args, **kwargs)
            return super(MegatronModule, megatron_module).__call__(*args, **kwargs)

    manager_cls.get_cudagraph_runner = get_cudagraph_runner
    manager_cls.__call__ = call
    manager_cls._miles_eager_fallback = True
    _INSTALLED = True
    logger.info("cuda_graph_fallback: eager fallback for unmatched local CUDA graph inputs installed")
    return True
