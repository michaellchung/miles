"""Read the stored chunks a full-parameter commit copies, ahead of the copy.

A chunk read back through Ray's zero-copy path is a tensor over the object
store's shared memory.  The first read of those pages in a process maps them
into it page by page; a later read of the same pages does not.  Before
verify-ahead, the prepare on the trainer read every chunk the commit then
copied; with verify-ahead the commit's copy is the first read (S27 E3: the
owner's copy took 0.37 s per fragment after a prepare read and 0.8 s
without).

:class:`CommitChunkReader` takes the commit's reads in copy order.  The
two hashing threads of ``full_parameter_state._hash_pool`` (idle during a
commit) ``ray.get`` each chunk and read one byte of every page the copy will
use, a bounded window ahead of the copy; the copy takes the chunk from the
reader.  The bytes are the same Ray objects (immutable) either way, so the
commit writes the same values; only the page mapping moves off the copying
thread.  A chunk the reader does not hold is read in the caller's thread, as
before.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Executor, Future

import torch

PREFETCH_ENV = "MILES_FULL_PARAMETER_COMMIT_PREFETCH"
# full_parameter_state._ZERO_COPY_ENV (that module imports this one).  With
# zero-copy chunks off, a get deserializes a private copy of the chunk, so
# reading ahead would hold extra host memory instead of mapping shared pages.
_ZERO_COPY_ENV = "MILES_FULL_PARAMETER_ZERO_COPY_CHUNKS"
_PAGE_BYTES = 4096
# Chunks read ahead of the copy (and held as mappings) at most.
DEFAULT_WINDOW = 4

# ``(chunk reference, element spans inside the chunk the copy reads)``;
# ``None`` spans: the whole chunk.
ChunkRead = tuple[object, tuple[tuple[int, int], ...] | None]

# Tests set this to read ahead through an in-process stand-in of Ray, whose
# stored tensors are already in this process's memory.
_PREFETCH_STANDINS = False


def prefetch_enabled(ray_module) -> bool:
    """Whether a commit reads ahead (``MILES_FULL_PARAMETER_COMMIT_PREFETCH``,
    default 1; 0 reads every chunk in the copying thread, as before).  Only
    the real Ray module with zero-copy chunks maps stored chunks lazily.

    The reads run on the caller's pool: page mapping barely scales past two
    threads in a Modal (gVisor) container (S27 G-DZ CPU probe: 1 GiB fresh
    chunks, 1 thread 0.68 s, 2 threads 0.41-0.50 s, 4 and 8 threads
    0.5-0.95 s), which is the size of Miles' hashing pool."""

    raw = os.environ.get(PREFETCH_ENV, "1")
    if raw not in ("0", "1"):
        raise RuntimeError(f"{PREFETCH_ENV} must be 0 or 1, got {raw!r}")
    if raw == "0" or os.environ.get(_ZERO_COPY_ENV, "1") == "0":
        return False
    return getattr(ray_module, "__name__", None) == "ray" or _PREFETCH_STANDINS


def touch_pages(value: object, spans: tuple[tuple[int, int], ...] | None) -> None:
    """Read one byte of every page of ``value`` that ``spans`` (element
    offsets) cover; ``None``: all of it.  Anything but a contiguous CPU tensor
    is left alone (the copy validates it)."""

    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_contiguous():
        return
    flat = value.reshape(-1).view(torch.uint8)
    width = value.element_size()
    for low, high in spans if spans is not None else ((0, value.numel()),):
        first, end = max(low, 0) * width, min(high, value.numel()) * width
        if first < end:
            flat[first:end:_PAGE_BYTES].max()
            flat[end - 1].item()


class CommitChunkReader:
    """``get(reference)`` of the commit's chunks, read ahead in ``reads`` order.

    At most ``window`` chunks are held or being read ahead of the copy.  A
    read error surfaces from ``get`` of that chunk, where the copy reading it
    itself would have raised.  ``close`` drops unstarted reads and waits for
    running ones (they only read)."""

    def __init__(
        self,
        reads: Sequence[ChunkRead],
        *,
        ray_module,
        pool: Executor | None,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        self._ray = ray_module
        self._pool = pool if reads else None
        self._pending: deque[ChunkRead] = deque(reads if self._pool is not None else ())
        self._inflight: deque[tuple[object, Future]] = deque()
        self._window = window
        self.prefetched_chunks = 0
        self.read_seconds = 0.0  # reader threads: get + page reads
        self.wait_seconds = 0.0  # the copying thread waiting for chunks
        self._fill()

    def _load(self, reference: object, spans) -> tuple[object, float]:
        started = time.monotonic()
        value = self._ray.get(reference)
        touch_pages(value, spans)
        return value, time.monotonic() - started

    def _fill(self) -> None:
        while self._pool is not None and self._pending and len(self._inflight) < self._window:
            reference, spans = self._pending.popleft()
            self._inflight.append((reference, self._pool.submit(self._load, reference, spans)))

    def _take(self, reference: object) -> Future | None:
        for index, (queued, future) in enumerate(self._inflight):
            if queued is reference:
                del self._inflight[index]
                return future
        for index, (queued, _spans) in enumerate(self._pending):
            if queued is reference:
                del self._pending[index]
                break
        return None

    def get(self, reference: object) -> object:
        started = time.monotonic()
        try:
            future = self._take(reference)
            if future is None:
                return self._ray.get(reference)
            value, seconds = future.result()
            self.prefetched_chunks += 1
            self.read_seconds += seconds
            return value
        finally:
            self._fill()
            self.wait_seconds += time.monotonic() - started

    def close(self) -> None:
        self._pending.clear()
        inflight, self._inflight = list(self._inflight), deque()
        for _reference, future in inflight:
            future.cancel()
        for _reference, future in inflight:
            if not future.cancelled():
                try:
                    future.result()
                except BaseException:  # noqa: BLE001 - the copy did not need this chunk
                    pass

    def stats(self) -> dict[str, object]:
        return {
            "prefetch": self._pool is not None,
            "prefetched_chunks": self.prefetched_chunks,
            "prefetch_read_seconds": self.read_seconds,
            "fetch_wait_seconds": self.wait_seconds,
        }
