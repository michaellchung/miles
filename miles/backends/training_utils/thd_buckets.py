"""Fixed THD micro-batch shapes ("buckets") for Megatron "local" CUDA graphs.

A captured CUDA graph only replays for the exact input signature it was captured with: the token
count of the packed stream, the number of ``cu_seqlens`` entries and the Python value of
``max_seqlen`` (Megatron ``_CudaGraphRunner.get_mismatch_errors``). Miles packs RL samples of varying
length, so without buckets every micro batch has a new signature and no graph ever replays.

With ``--thd-bucket-sizes`` every THD micro batch (no context parallelism) is padded to the
smallest listed size that holds it; the padding is one trailing dummy sequence (as Miles already
does for the multiple-of-128 padding). With ``--thd-max-packed-sequences N`` the ``cu_seqlens``
list is then padded to ``N + 1`` entries with zero-length sequences (repeating the last offset),
and ``max_seqlen`` is set to the bucket size, so the signature depends on the bucket only.
A micro batch longer than the largest bucket, or with more than N sequences, keeps the old
shape (multiple of the pad size, its own entry count); a graphed layer then runs eagerly
(see ``megatron_utils/cuda_graph_fallback.py``).
"""

from __future__ import annotations

from collections.abc import Sequence


def parse_bucket_sizes(sizes: Sequence[int] | None, pad_size: int) -> list[int] | None:
    """Sorted, de-duplicated bucket sizes; each must be a positive multiple of ``pad_size``."""
    if not sizes:
        return None
    out = sorted({int(s) for s in sizes})
    for s in out:
        if s <= 0 or s % pad_size != 0:
            raise ValueError(f"--thd-bucket-sizes: {s} is not a positive multiple of the pad size {pad_size}")
    return out


def bucket_target(num_tokens: int, bucket_sizes: Sequence[int] | None, pad_size: int) -> tuple[int, bool]:
    """Padded token count of a packed stream of ``num_tokens`` tokens and whether it is a bucket.

    Without buckets (or above the largest bucket) this is the old rule: the next multiple of
    ``pad_size``."""
    if bucket_sizes:
        for size in bucket_sizes:
            if num_tokens <= size:
                return size, True
    return num_tokens + (pad_size - num_tokens % pad_size) % pad_size, False


def pad_cu_seqlens(cu_seqlens: list[int], target_len: int, num_entries: int | None) -> tuple[list[int], bool]:
    """Append the trailing dummy sequence up to ``target_len`` (when there is padding), then repeat the
    last offset until the list has ``num_entries`` entries. Returns (list, entries_fixed)."""
    out = list(cu_seqlens)
    if out[-1] > target_len:
        raise ValueError(f"cu_seqlens end {out[-1]} is past the padded length {target_len}")
    if out[-1] < target_len:
        out.append(target_len)
    if num_entries is None:
        return out, False
    if len(out) > num_entries:
        return out, False
    out.extend([target_len] * (num_entries - len(out)))
    return out, True


def thd_bucket_kwargs(args) -> dict:
    """get_batch keyword arguments for the bucket settings of ``args`` (empty when unset)."""
    sizes = getattr(args, "thd_bucket_sizes", None)
    if not sizes:
        return {}
    n = getattr(args, "thd_max_packed_sequences", None)
    return {"thd_bucket_sizes": sizes, "thd_max_cu_entries": (int(n) + 1) if n else None}
