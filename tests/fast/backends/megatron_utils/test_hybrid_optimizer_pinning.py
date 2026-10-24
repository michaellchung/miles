"""The CPU FP32 masters of Megatron's HybridDeviceOptimizer end up in pinned memory.

CPU only: pinning needs a CUDA driver, so the module's ``_pin`` and
``_is_pinned`` are replaced by a stand-in that records which storages are
"pinned".  ``_FakeHybridDeviceOptimizer`` reproduces the part of Megatron's
``HybridDeviceOptimizer`` (radixark/Megatron-LM a84b105) that builds the CPU
copies: each offloaded parameter is cloned to pinned host memory, then, under
``param_update_in_fp32``, cast with ``.clone().float()`` -- a pageable tensor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import hybrid_optimizer_pinning as pinning


class _PinnedStorages:
    def __init__(self) -> None:
        self.pointers: set[int] = set()
        self.pins = 0

    def pin(self, value: torch.Tensor) -> torch.Tensor:
        self.pins += 1
        pinned = value.clone()
        self.pointers.add(pinned.data_ptr())
        return pinned

    def is_pinned(self, value: torch.Tensor) -> bool:
        return value.data_ptr() in self.pointers


@pytest.fixture
def storages(monkeypatch):
    recorder = _PinnedStorages()
    monkeypatch.setattr(pinning, "_pin", recorder.pin)
    monkeypatch.setattr(pinning, "_is_pinned", recorder.is_pinned)
    monkeypatch.setattr(pinning, "_empty_pinned_cache", lambda: None)
    return recorder


def _fake_class(recorder: _PinnedStorages):
    class _FakeHybridDeviceOptimizer:
        """Megatron's CPU-copy construction, with ``.pin_memory()`` routed to ``recorder``."""

        def __init__(self, params, *, param_update_in_fp32=True, pin_cpu_params=True):
            self.param_groups = [{"params": list(params)}]
            self.param_update_in_fp32 = param_update_in_fp32
            self.pin_cpu_params = pin_cpu_params
            self._init_sub_optimizers()

        def _init_sub_optimizers(self):
            self.cpu_param_groups, self.gpu_params_map_cpu_copy = [], {}
            self.cpu_copys_map_gpu_param, self.param_to_fp32_param = {}, {}
            cpu_group = {"params": []}
            for orig_param in self.param_groups[0]["params"]:
                param = recorder.pin(orig_param.detach().clone().cpu())  # Megatron: .cpu().pin_memory()
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()  # Megatron: pageable FP32 master
                    self.param_to_fp32_param[orig_param] = param
                self.gpu_params_map_cpu_copy[orig_param] = param
                self.cpu_copys_map_gpu_param[param] = orig_param
                cpu_group["params"].append(param)
            self.cpu_param_groups.append(cpu_group)
            self.cpu_optimizers = [SimpleNamespace(param_groups=[{"params": [p]}]) for p in cpu_group["params"]]
            self.param_to_inner_param = dict(self.param_to_fp32_param)

    return _FakeHybridDeviceOptimizer


def _bf16_params():
    return [
        torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
        torch.tensor([0.5, -2.0]).to(torch.bfloat16),
    ]


def test_wrapped_init_pins_every_fp32_master_in_place(storages):
    cls = _fake_class(storages)
    assert pinning.wrap_init_sub_optimizers(cls) is True
    params = _bf16_params()
    optimizer = cls(params)

    masters = optimizer.cpu_param_groups[0]["params"]
    assert all(master.dtype == torch.float32 and storages.is_pinned(master) for master in masters)
    for orig, master in zip(params, masters, strict=True):
        # The same objects stay in every map and sub-optimizer; only the storage moved.
        assert optimizer.param_to_fp32_param[orig] is master
        assert optimizer.gpu_params_map_cpu_copy[orig] is master
        assert optimizer.cpu_copys_map_gpu_param[master] is orig
        assert torch.equal(master, orig.float())
    assert [group["params"][0] for sub in optimizer.cpu_optimizers for group in sub.param_groups] == masters


def test_reinit_after_state_load_pins_the_new_copies(storages):
    cls = _fake_class(storages)
    pinning.wrap_init_sub_optimizers(cls)
    optimizer = cls(_bf16_params())
    first = optimizer.cpu_param_groups[0]["params"]
    optimizer._init_sub_optimizers()  # Megatron's post-load_state_dict hook does this
    second = optimizer.cpu_param_groups[0]["params"]
    assert all(new is not old for new, old in zip(second, first, strict=True))
    assert all(storages.is_pinned(master) for master in second)


def test_wrap_is_idempotent_and_respects_pin_cpu_params(storages):
    cls = _fake_class(storages)
    assert pinning.wrap_init_sub_optimizers(cls) is True
    wrapped = cls._init_sub_optimizers
    assert pinning.wrap_init_sub_optimizers(cls) is False
    assert cls._init_sub_optimizers is wrapped

    optimizer = cls(_bf16_params(), pin_cpu_params=False)
    assert not any(storages.is_pinned(master) for master in optimizer.cpu_param_groups[0]["params"])


def test_already_pinned_copies_are_left_alone(storages):
    cls = _fake_class(storages)
    pinning.wrap_init_sub_optimizers(cls)
    # Without the FP32 cast the CPU copies are Megatron's pinned BF16 clones.
    optimizer = cls(_bf16_params(), param_update_in_fp32=False)
    pins_before = storages.pins
    assert pinning.pin_cpu_optimizer_params(optimizer) == 0
    assert storages.pins == pins_before


def test_pin_reports_bytes_and_counts_shared_tensors_once():
    pinned = set()
    shared = torch.zeros(4, dtype=torch.float32)
    optimizer = SimpleNamespace(
        pin_cpu_params=True, cpu_param_groups=[{"params": [shared, torch.ones(3)]}, {"params": [shared]}]
    )

    def pin(value):
        copy = value.clone()
        pinned.add(copy.data_ptr())
        return copy

    moved = pinning.pin_cpu_optimizer_params(optimizer, pin=pin, is_pinned=lambda value: value.data_ptr() in pinned)
    assert moved == (4 + 3) * 4
    assert all(tensor.data_ptr() in pinned for group in optimizer.cpu_param_groups for tensor in group["params"])
    with pytest.raises(RuntimeError, match="changed its dtype or shape"):
        pinning.pin_cpu_optimizer_params(
            SimpleNamespace(cpu_param_groups=[{"params": [torch.zeros(2)]}]),
            pin=lambda value: value.double(),
            is_pinned=lambda value: False,
        )


def test_install_without_megatron_is_a_no_op(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_megatron(name, *args, **kwargs):
        if name.startswith("megatron"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_megatron)
    assert pinning.install() is False
