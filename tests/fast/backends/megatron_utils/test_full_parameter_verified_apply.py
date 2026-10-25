"""A chunked apply skips the prepare checks of fragments the caller verified.

``verify_stored_fragment_chunks`` is the prepare's own per-fragment check
(finite FP32 chunks, chunk hashes, fragment digest).  A caller may run it off
the trainer before the apply and name the fragment in ``verified_fragments``
with its payload hash; the prepare then reads none of that fragment's chunks,
for the owner and for every data-parallel replica.  The committed masters
and model must stay bit-identical to the apply that checks everything.
"""

from __future__ import annotations

import pytest
import torch

from miles.backends.megatron_utils.full_parameter_state import (
    export_full_parameter_chunked_shard,
    prepare_full_parameter_chunked_shard,
    verify_stored_fragment_chunks,
)
from miles.ray.full_parameter_transport import (
    FULL_PARAMETER_TRANSPORT_FEATURES,
    apply_chunked_cut,
    export_chunked_cut,
    record_local_step,
)
from tests.fast.backends.megatron_utils.test_full_parameter_dp_replicas import (  # noqa: F401 - fixture
    _assert_replicas_identical,
    _install,
    _merged,
    cluster,
)
from tests.fast.backends.megatron_utils.test_full_parameter_dp_replicas import _target as _dp_target
from tests.fast.backends.megatron_utils.test_full_parameter_resident_apply import (
    _apply,
    _CountingRay,
    _master_bits,
    _model_bits,
    _split_actor,
    _target,
)


def _verified(target, fragment_ids):
    return {d.fragment_id: d.payload_hash for d in target.fragments if d.fragment_id in fragment_ids}


def _apply_verified(actor, target, ray, token, verified):
    before = ray.gets
    count = prepare_full_parameter_chunked_shard(
        actor, target, commit_token=token, ray_module=ray, skip_resident=True, verified_fragments=verified
    )
    from miles.backends.megatron_utils.full_parameter_state import (
        commit_prepared_full_parameter_shard,
        finalize_full_parameter_commit,
    )

    assert commit_prepared_full_parameter_shard(actor, commit_token=token, ray_module=ray) == count
    assert finalize_full_parameter_commit(actor, commit_token=token) == count
    return count, ray.gets - before


def _two_applies(monkeypatch, *, ahead):
    actor, named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    first = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    second_values = {2: [-0.0, 3.75, 2.5e38]}
    results, stats = [], []
    for token, target in (("apply-1", first), ("apply-2", None)):
        if target is None:
            target = _target(first, ray, 2, second_values)
        replaced = {1} if token == "apply-1" else {2}
        if ahead:
            # Off the trainer: the same check the prepare would run.
            for descriptor, refs in zip(target.fragments, target.chunk_refs, strict=True):
                if descriptor.fragment_id in replaced:
                    verify_stored_fragment_chunks(
                        descriptor, refs, parameter_layout_hash=target.parameter_layout_hash,
                        plan_hash=target.plan_hash, ray_module=ray)
            results.append(_apply_verified(actor, target, ray, token, _verified(target, replaced)))
        else:
            results.append(_apply(actor, target, ray, token, skip=True))
        stats.append(dict(actor._external_full_parameter_apply_stats))
    return named, results, stats


def test_verified_fragments_are_bit_identical_and_read_only_by_the_commit(monkeypatch):
    named_full, full, _ = _two_applies(monkeypatch, ahead=False)
    named, ahead, stats = _two_applies(monkeypatch, ahead=True)
    for got, want in zip(_master_bits(named), _master_bits(named_full), strict=True):
        assert torch.equal(got, want)
    for got, want in zip(_model_bits(named), _model_bits(named_full), strict=True):
        assert torch.equal(got, want)
    assert [count for count, _ in ahead] == [count for count, _ in full]
    # The replaced fragment has 2 chunks: the checking prepare reads them and
    # the commit copies them (4 gets); with the fragment verified ahead only
    # the commit reads them (the verification's own 2 gets ran before).
    assert [gets for _, gets in full] == [4, 4]
    assert [gets for _, gets in ahead] == [2, 2]
    for one in stats:
        assert one["prepare"]["read_fragments"] == 0
        assert one["prepare"]["verified_ahead_fragments"] == 1
        assert one["commit"]["copied_fragments"] == 1


def test_a_verified_entry_with_another_payload_hash_is_checked_as_usual(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    _count, gets = _apply_verified(actor, target, ray, "stale-claim", {1: "0" * 64})
    assert gets == 4
    stats = actor._external_full_parameter_apply_stats["prepare"]
    assert (stats["read_fragments"], stats["verified_ahead_fragments"]) == (1, 0)


def test_the_verification_refuses_changed_or_non_finite_bytes(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    (index,) = [i for i, d in enumerate(target.fragments) if d.fragment_id == 1]
    descriptor, refs = target.fragments[index], target.chunk_refs[index]
    kwargs = dict(parameter_layout_hash=target.parameter_layout_hash, plan_hash=target.plan_hash, ray_module=ray)
    verify_stored_fragment_chunks(descriptor, refs, **kwargs)
    good = ray.objects[refs[0]].clone()
    ray.objects[refs[0]] = good + 1.0
    with pytest.raises(ValueError, match="hash"):
        verify_stored_fragment_chunks(descriptor, refs, **kwargs)
    ray.objects[refs[0]] = torch.full_like(good, float("nan"))
    with pytest.raises(ValueError, match="NaN or Inf"):
        verify_stored_fragment_chunks(descriptor, refs, overlap_checks=False, **kwargs)


def test_a_malformed_verified_mapping_is_refused(monkeypatch):
    actor, _named = _split_actor(monkeypatch)
    ray = _CountingRay()
    exported = export_full_parameter_chunked_shard(actor, policy_version=0, max_chunk_bytes=8, ray_module=ray)
    target = _target(exported, ray, 1, {1: [30.5, -1.25, 1e-30]})
    with pytest.raises(ValueError, match="verified fragments"):
        prepare_full_parameter_chunked_shard(
            actor, target, commit_token="bad", ray_module=ray, verified_fragments={"1": "x"})


def test_the_transport_advertises_the_verified_apply():
    assert "verified_apply" in FULL_PARAMETER_TRANSPORT_FEATURES


async def test_replica_ranks_skip_the_checks_of_verified_fragments(cluster):
    async def run(verified: bool):
        group, ranks, ray = cluster(4, 2)
        _manifests, _plans = await _install(group)
        first = await export_chunked_cut(group, 0, 0, max_chunk_bytes=16)
        await apply_chunked_cut(group, _dp_target(first, ray, 1, _merged), commit_token="dp-apply-1")
        for rank in ranks:
            rank.local_step(rollout_id=1, delta=0.25)
        await record_local_step(group, base_policy_version=1, rollout_id=1)
        deferred = await export_chunked_cut(group, 1, 1, max_chunk_bytes=16, defer_payload=True)
        changed = 0
        target = _dp_target(
            deferred, ray, 2, lambda d: _merged(d) * 3 if d.fragment_id == changed else None)
        claim = None
        if verified:
            claim = {}
            for shard in target.shards:
                for descriptor, refs in zip(shard.fragments, shard.chunk_refs, strict=True):
                    if descriptor.fragment_id == changed:
                        verify_stored_fragment_chunks(
                            descriptor, refs, parameter_layout_hash=shard.parameter_layout_hash,
                            plan_hash=shard.plan_hash, ray_module=ray)
                        claim[changed] = descriptor.payload_hash
        await apply_chunked_cut(group, target, commit_token="dp-apply-2", skip_resident_fragments=True,
                                verified_fragments=claim)
        stats = [rank._external_full_parameter_apply_stats for rank in ranks]
        return _assert_replicas_identical(ranks), stats

    checked, checked_stats = await run(False)
    ahead, ahead_stats = await run(True)
    assert checked.keys() == ahead.keys()
    for name in checked:
        assert torch.equal(ahead[name], checked[name]), name
    assert sum(s["prepare"]["read_fragments"] for s in checked_stats) == 1
    assert sum(s["prepare"]["replica_read_fragments"] for s in checked_stats) >= 1
    for stats in ahead_stats:
        assert stats["prepare"]["read_fragments"] == 0
        assert stats["prepare"]["replica_read_fragments"] == 0
    assert sum(s["prepare"]["verified_ahead_fragments"] for s in ahead_stats) == 1
    # The commits still copy the changed fragment into the owner and its replicas.
    assert [s["commit"]["replica_copied_fragments"] for s in ahead_stats] == [
        s["commit"]["replica_copied_fragments"] for s in checked_stats]
