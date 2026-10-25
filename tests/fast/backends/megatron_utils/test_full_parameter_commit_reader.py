"""The commit reads its chunks ahead of the copy and casts the model on the master's device.

* ``CommitChunkReader``: the hashing pool's threads ``ray.get`` the commit's
  chunks in copy order and read one byte of each page the copy uses, a
  bounded window ahead.
  Every chunk is read once; an error surfaces where the copy reads the chunk.
* ``MasterRange.write``: the values inside a rank's range cross to the master
  once and the model takes the master's cast (Megatron's main-to-model copy);
  a sample of every write is compared bit for bit with the host cast.

Masters and model bits must equal the plain apply (no read-ahead, host cast)
and the whole-master cluster, for values that exercise the BF16 rounding:
ties, subnormals, signed zeros and the top of the BF16 range.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils import full_parameter_chunk_reader as reader_module
from miles.backends.megatron_utils import full_parameter_sharded_masters as sharded
from miles.backends.megatron_utils.full_parameter_chunk_reader import CommitChunkReader, prefetch_enabled, touch_pages
from miles.ray.full_parameter_transport import apply_chunked_cut, export_chunked_cut
from tests.fast.backends.megatron_utils.test_full_parameter_dp_sharded_masters import (  # noqa: F401 - fixture
    _CHUNK_BYTES,
    _assert_same_bits,
    _full_masters,
    _pair,
    _step_and_record,
    _target,
    cluster,
)

pytestmark = pytest.mark.asyncio

# FP32 values whose BF16 rounding is delicate: ties to even both ways, the
# largest finite BF16 neighbourhood, subnormals, signed zeros.
_EDGE = torch.from_numpy(
    np.array(
        [
            0x3F808000,  # 1 + 2**-8: a tie, rounds down to the even 1.0
            0x3F818000,  # a tie that rounds up to the even neighbour
            0xBF808000,  # a negative tie
            0x3F80C000,  # above a tie
            0x7F7F0000,  # the largest finite BF16
            0x7F7F7FFF,  # just below the overflow tie: stays finite
            0x00000001,  # the smallest FP32 subnormal
            0x00008000,  # a subnormal tie, rounds down
            0x00018000,  # a subnormal tie, rounds up
            0x007FFFFF,  # the largest FP32 subnormal: rounds up to the smallest normal
            0x80412345,  # a negative subnormal
            0x80000000,  # -0.0
            0x00000000,  # +0.0
        ],
        dtype=np.uint32,
    ).view(np.float32)
)


def _edge_values(descriptor) -> torch.Tensor:
    base = torch.arange(descriptor.numel, dtype=torch.float32) * -0.37 + 10.3 * (descriptor.fragment_id + 1)
    edge = _EDGE.roll(descriptor.fragment_id)
    count = min(descriptor.numel, edge.numel())
    base[:count] = edge[:count]
    return base


def _host_rne_bf16(values: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even FP32 -> BF16 bits on integers (finite inputs)."""

    bits = values.numpy().view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return torch.from_numpy(rounded.astype(np.uint16).view(np.int16))


def test_host_cast_is_round_to_nearest_even():
    values = torch.cat([_EDGE, torch.randn(4096) * 1e-3, torch.randn(4096) * 1e30])
    assert torch.equal(values.to(torch.bfloat16).view(torch.int16), _host_rne_bf16(values))


def test_a_device_cast_that_differs_from_the_host_cast_stops_the_write():
    values = torch.tensor([1.0 + 2.0**-8, 2.5], dtype=torch.float32)
    model = values.to(torch.bfloat16)
    sharded._require_host_cast_bits(model, values)
    model.view(torch.int16)[0] += 1  # one ulp off
    with pytest.raises(RuntimeError, match="differs from the host cast"):
        sharded._require_host_cast_bits(model, values)


class _Ray:
    """Thread-safe stand-in: counts gets per reference, can fail one."""

    def __init__(self, values, *, fail=None, gate=None):
        self.objects = {object(): value for value in values}
        self.order = list(self.objects)
        self.gets: dict[object, int] = {}
        self.fail = fail
        self.gate = gate
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def get(self, reference):
        with self.lock:
            self.gets[reference] = self.gets.get(reference, 0) + 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.gate is not None:
                self.gate.wait(timeout=5)
            if reference is self.fail:
                raise KeyError("lost chunk")
            return self.objects[reference]
        finally:
            with self.lock:
                self.inflight -= 1


def _values(count, numel=3000):
    return [torch.arange(numel, dtype=torch.float32) + index for index in range(count)]


def test_reader_returns_every_chunk_once_in_copy_order():
    ray = _Ray(_values(9))
    reads = [(reference, None if index % 2 else ((5, 2000),)) for index, reference in enumerate(ray.order)]
    reader = CommitChunkReader(reads, ray_module=ray, pool=ThreadPoolExecutor(3), window=4)
    try:
        for reference in ray.order:
            assert reader.get(reference) is ray.objects[reference]
    finally:
        reader.close()
    assert ray.gets == {reference: 1 for reference in ray.order}
    stats = reader.stats()
    assert stats["prefetch"] and stats["prefetched_chunks"] == 9
    assert stats["fetch_wait_seconds"] >= 0 and stats["prefetch_read_seconds"] >= 0


def test_reader_window_bounds_reads_ahead():
    gate = threading.Event()
    ray = _Ray(_values(12), gate=gate)
    reader = CommitChunkReader([(reference, None) for reference in ray.order], ray_module=ray, pool=ThreadPoolExecutor(8), window=3)
    try:
        assert len(reader._inflight) == 3 and len(reader._pending) == 9
        gate.set()
        for reference in ray.order:
            reader.get(reference)
    finally:
        reader.close()
    assert ray.max_inflight <= 3


def test_reader_falls_back_for_a_chunk_out_of_order():
    ray = _Ray(_values(6))
    reader = CommitChunkReader([(reference, None) for reference in ray.order], ray_module=ray, pool=ThreadPoolExecutor(2), window=2)
    try:
        last = ray.order[-1]  # not read ahead yet: read here and never again
        assert reader.get(last) is ray.objects[last]
        for reference in ray.order[:-1]:
            assert reader.get(reference) is ray.objects[reference]
    finally:
        reader.close()
    assert ray.gets == {reference: 1 for reference in ray.order}
    assert reader.prefetched_chunks == 5


def test_reader_error_surfaces_at_the_failing_chunk():
    ray = _Ray(_values(5))
    ray.fail = ray.order[2]
    reader = CommitChunkReader([(reference, None) for reference in ray.order], ray_module=ray, pool=ThreadPoolExecutor(2))
    try:
        reader.get(ray.order[0])
        reader.get(ray.order[1])
        with pytest.raises(KeyError, match="lost chunk"):
            reader.get(ray.order[2])
    finally:
        reader.close()
    assert not reader._inflight and not reader._pending


def test_reader_off_reads_in_the_calling_thread():
    ray = _Ray(_values(3))
    reader = CommitChunkReader([(reference, None) for reference in ray.order], ray_module=ray, pool=None)
    assert reader._pool is None and not reader._pending
    for reference in ray.order:
        assert reader.get(reference) is ray.objects[reference]
    reader.close()
    assert not reader.stats()["prefetch"] and reader.stats()["prefetched_chunks"] == 0


def test_prefetch_default_env_and_stand_ins(monkeypatch):
    real = type("RayModule", (), {"__name__": "ray"})()
    monkeypatch.delenv(reader_module.PREFETCH_ENV, raising=False)
    monkeypatch.delenv(reader_module._ZERO_COPY_ENV, raising=False)
    assert prefetch_enabled(real)
    assert not prefetch_enabled(object())  # an in-process stand-in maps nothing lazily
    monkeypatch.setenv(reader_module._ZERO_COPY_ENV, "0")  # a get copies: no read-ahead
    assert not prefetch_enabled(real)
    monkeypatch.delenv(reader_module._ZERO_COPY_ENV)
    monkeypatch.setenv(reader_module.PREFETCH_ENV, "0")
    assert not prefetch_enabled(real)
    monkeypatch.setenv(reader_module.PREFETCH_ENV, "2")
    with pytest.raises(RuntimeError, match="0 or 1"):
        prefetch_enabled(real)


def test_touch_pages_reads_without_changing_anything():
    value = torch.arange(5000, dtype=torch.float32)
    before = value.clone()
    for spans in (None, ((0, 1),), ((4999, 5000),), ((10, 10),), ((-3, 99999),), ((1, 1025), (2000, 4500))):
        touch_pages(value, spans)
    touch_pages("not a tensor", None)
    touch_pages(value[::2], None)  # not contiguous: left alone
    read_only = torch.from_numpy(np.frombuffer(value.numpy().tobytes(), dtype=np.float32))
    touch_pages(read_only, None)
    assert torch.equal(value, before)


async def _apply_sequence(cluster, pair):  # noqa: F811 - the cluster fixture
    for side in pair.values():
        cluster.use(side.group)
        exported = await export_chunked_cut(side.group, 0, 0, max_chunk_bytes=_CHUNK_BYTES)
        await apply_chunked_cut(side.group, _target(exported, side.ray, 1, _edge_values), commit_token="apply-1")
    await _step_and_record(cluster, pair, rollout_id=1, base=1, delta=-0.625)
    for side in pair.values():
        cluster.use(side.group)
        deferred = await export_chunked_cut(side.group, 1, 1, max_chunk_bytes=_CHUNK_BYTES, defer_payload=True)
        target = _target(deferred, side.ray, 2, lambda d: -_edge_values(d) if d.fragment_id in (0, 3) else None)
        await apply_chunked_cut(side.group, target, commit_token="apply-2", skip_resident_fragments=True)


def _snapshot(pair):
    return {
        "masters": {name: value.view(torch.int32).clone() for name, value in _full_masters(pair["sharded"].ranks).items()},
        "models": [rank.model_bits() for rank in pair["sharded"].ranks],
    }


@pytest.mark.parametrize(("world", "ep"), [(2, 2), (4, 2), (4, 1)])
async def test_read_ahead_and_device_cast_apply_the_same_bits(cluster, monkeypatch, world, ep):  # noqa: F811 - the cluster fixture
    snapshots = {}
    for prefetch, device_cast in ((0, "0"), (1, "1"), (0, "1"), (1, "0")):
        monkeypatch.setattr(reader_module, "_PREFETCH_STANDINS", prefetch > 0)
        monkeypatch.setenv(reader_module.PREFETCH_ENV, str(prefetch))
        monkeypatch.setenv(sharded._DEVICE_CAST_ENV, device_cast)
        pair = await _pair(cluster, world, ep)
        await _apply_sequence(cluster, pair)
        _assert_same_bits(pair)  # equal to the whole-master cluster
        snapshots[(prefetch, device_cast)] = _snapshot(pair)
        for rank in pair["sharded"].ranks:
            commit = rank._external_full_parameter_apply_stats["commit"]
            if commit["prefetched_chunks"]:
                assert commit["prefetch"]
            assert commit["fetch_wait_seconds"] >= 0 and commit["write_seconds"] >= 0
            if not prefetch:
                assert commit["prefetched_chunks"] == 0
        if prefetch:
            assert sum(rank._external_full_parameter_apply_stats["commit"]["prefetched_chunks"] for rank in pair["sharded"].ranks) > 0
    reference = snapshots[(0, "0")]
    for key, snapshot in snapshots.items():
        assert snapshot["masters"].keys() == reference["masters"].keys()
        for name in reference["masters"]:
            assert torch.equal(snapshot["masters"][name], reference["masters"][name]), (key, name)
        for got, want in zip(snapshot["models"], reference["models"], strict=True):
            for name in want:
                assert torch.equal(got[name], want[name]), (key, name)


async def test_read_ahead_reads_each_chunk_as_often_as_the_plain_commit(cluster, monkeypatch):  # noqa: F811 - the cluster fixture
    totals = {}
    for prefetch in (0, 1):
        monkeypatch.setattr(reader_module, "_PREFETCH_STANDINS", prefetch > 0)
        monkeypatch.setenv(reader_module.PREFETCH_ENV, str(prefetch))
        pair = await _pair(cluster, 4, 2)
        side = pair["sharded"]
        cluster.use(side.group)
        exported = await export_chunked_cut(side.group, 0, 0, max_chunk_bytes=_CHUNK_BYTES)
        target = _target(exported, side.ray, 1, _edge_values)
        order = {id(reference): index for index, reference in enumerate(side.ray.objects)}
        counts: dict[int, int] = {}
        lock = threading.Lock()
        plain_get = side.ray.get

        def counting_get(reference, counts=counts, lock=lock, plain_get=plain_get, order=order):
            with lock:
                counts[order[id(reference)]] = counts.get(order[id(reference)], 0) + 1
            return plain_get(reference)

        monkeypatch.setattr(side.ray, "get", counting_get)
        await apply_chunked_cut(side.group, target, commit_token="apply-1")
        prefetched = sum(rank._external_full_parameter_apply_stats["commit"]["prefetched_chunks"] for rank in side.ranks)
        assert (prefetched > 0) == (prefetch > 0)
        totals[prefetch] = counts
    assert totals[0] == totals[1]


async def test_materialize_reports_its_time_split(cluster):  # noqa: F811 - the cluster fixture
    pair = await _pair(cluster, 4, 2)
    side = pair["sharded"]
    cluster.use(side.group)
    from miles.ray.full_parameter_transport import materialize_chunked_fragments

    deferred = await export_chunked_cut(side.group, 0, 0, max_chunk_bytes=_CHUNK_BYTES, defer_payload=True)
    fragment_ids = [descriptor.fragment_id for shard in deferred.shards for descriptor in shard.fragments]
    await materialize_chunked_fragments(side.group, deferred, fragment_ids)
    owners = [rank._external_full_parameter_apply_stats["materialize"] for rank in side.ranks]
    for stats in owners:
        if stats["fragments"]:
            for key in ("sources_seconds", "copy_out_seconds", "check_seconds", "put_seconds", "model_check_seconds"):
                assert stats[key] >= 0, key
    assert any(stats["fragments"] for stats in owners)
