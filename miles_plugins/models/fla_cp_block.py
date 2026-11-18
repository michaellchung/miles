"""MILES_K31_KDA_FLA_CP_BLOCK=32 (default unset: fla's own choice): the column block of fla's CP pre-process kernels.

Under context parallelism fla's chunked delta rule (KDA, GDN) runs a serial scan over the rank-boundary sequence
before the CP state all-gather: ``pre_process_fwd_kernel_merged`` on every rank but the last (forward), and
``pre_process_bwd_kernel_merged`` on every rank but the first (backward). The peer waits in the all-gather the whole
time (S2 trace: 2.9 ms forward / recompute, 4.4 ms backward per KDA layer and microbatch). fla picks
``BLOCK_SIZE = 32 if K <= 64 else 64``, so at K = V = 128 the grid is (2 + 2) x heads CTAs (48 at 12 heads) for a
latency-bound scan. Each CTA owns BLOCK_SIZE columns of the state and scans the same rows, so a smaller block
only splits the columns over more CTAs. On B300 (8-GPU check, T = 131072, 12 heads) BLOCK_SIZE 32 gave bit-equal
``hm`` / ``dhm`` / ``dv`` and 2.86 -> 2.24 ms forward, 4.32 -> 3.68 ms backward; 16 was slower.

fla computes the block inside ``chunk_gated_delta_rule_{fwd_h,bwd_dhu}_pre_process`` (a local, not a parameter), and
those functions look the two kernels up as globals of ``fla.ops.cp.chunk_delta_h`` at call time. So the switch
replaces the two module globals with a launcher that, for fla's K > 64 choice (``BLOCK_SIZE=64``), relaunches with
the requested block and the grid recomputed for it. Nothing else of fla changes. ``install`` checks that structure
and raises if this fla version differs. It applies to every fla CP delta-rule call in the process (KDA and GDN).
Set it the same on every rank (it changes no collective; a mismatch would only make the ranks' numerics differ).
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

FLA_CP_BLOCK_ENV = "MILES_K31_KDA_FLA_CP_BLOCK"
FLA_MODULE = "fla.ops.cp.chunk_delta_h"
# kernel global -> the fla function that launches it
KERNELS = {
    "pre_process_fwd_kernel_merged": "chunk_gated_delta_rule_fwd_h_pre_process",
    "pre_process_bwd_kernel_merged": "chunk_gated_delta_rule_bwd_dhu_pre_process",
}
FLA_BLOCK = 64  # fla's choice for K > 64
ALLOWED = (16, 32)


def fla_cp_block() -> int | None:
    """The requested block, or None (unset / empty / 0 / 64: fla's own choice)."""
    raw = os.environ.get(FLA_CP_BLOCK_ENV, "").strip()
    if raw in ("", "0", str(FLA_BLOCK)):
        return None
    block = int(raw)
    if block not in ALLOWED:
        raise ValueError(f"{FLA_CP_BLOCK_ENV}={raw}: use one of {ALLOWED} (or unset for fla's {FLA_BLOCK})")
    return block


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class BlockOverride:
    """Stands in for a Triton kernel object: ``kernel[grid](**kwargs)`` with ``BLOCK_SIZE=64`` launches with
    ``block`` instead and the column grid ``cdiv(V, block) + cdiv(K, block)``; any other launch passes through."""

    def __init__(self, kernel, block: int):
        self.kernel, self.block = kernel, block

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if (
                kwargs.get("BLOCK_SIZE") == FLA_BLOCK
                and isinstance(grid, tuple)
                and grid[0] == _cdiv(kwargs["V"], FLA_BLOCK) + _cdiv(kwargs["K"], FLA_BLOCK)
            ):
                block = self.block
                kwargs["BLOCK_SIZE"] = block
                return self.kernel[(_cdiv(kwargs["V"], block) + _cdiv(kwargs["K"], block), *grid[1:])](*args, **kwargs)
            return self.kernel[grid](*args, **kwargs)

        return launch

    def __getattr__(self, name):
        return getattr(self.kernel, name)  # config-access-exempt: proxy for the wrapped Triton kernel


def install() -> int | None:
    """Apply MILES_K31_KDA_FLA_CP_BLOCK to fla (idempotent). Returns the block in effect, None for fla's own."""
    block = fla_cp_block()
    if block is None:
        return None
    module = importlib.import_module(FLA_MODULE)
    for kernel_name, launcher_name in KERNELS.items():
        launcher = getattr(module, launcher_name, None)  # config-access-exempt: fla module global by name
        kernel = getattr(module, kernel_name, None)  # config-access-exempt: fla module global by name
        if launcher is None or kernel is None or kernel_name not in launcher.__code__.co_names:
            raise RuntimeError(
                f"{FLA_CP_BLOCK_ENV}: {FLA_MODULE}.{launcher_name} does not launch the global {kernel_name} in "
                "this fla version; unset the switch"
            )
        if isinstance(kernel, BlockOverride):
            kernel.block = block
        else:
            setattr(module, kernel_name, BlockOverride(kernel, block))
    logger.info(f"{FLA_CP_BLOCK_ENV}={block}: fla CP pre-process kernels launch with BLOCK_SIZE {block}")
    print(f"[k31] {FLA_CP_BLOCK_ENV}={block}: fla CP pre-process BLOCK_SIZE {FLA_BLOCK} -> {block}", flush=True)
    return block
