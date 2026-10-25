"""CPU tests: with local CUDA graphs for mamba / attn only, the MoE layers (which then run eagerly) keep
their "moe" selective recompute (Megatron a84b10547 turns it off for every local-graph run).

The patch logic is tested on a stand-in MoELayer class (no Megatron needed); one extra test checks the
guard line in the real Megatron source when Megatron is importable.
"""

import sys
import types
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.backends.megatron_utils import cuda_graph_fallback as cgf

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class _StandInMoELayer:
    def __init__(self, config, submodules=None, layer_number=None):
        # same rule as megatron/core/transformer/moe/moe_layer.py (a84b10547)
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )


def _config(impl, modules, recompute=("moe", "layernorm")):
    return SimpleNamespace(cuda_graph_impl=impl, cuda_graph_modules=[SimpleNamespace(name=m) for m in modules],
                           recompute_granularity="selective", recompute_modules=list(recompute))


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch):
    mod = types.ModuleType("megatron.core.transformer.moe.moe_layer")
    cls = type("MoELayer", (_StandInMoELayer,), {"__init__": _StandInMoELayer.__init__})
    mod.MoELayer = cls
    for name in ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.moe"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.moe_layer", mod)
    monkeypatch.setattr(cgf, "_MOE_PATCHED", False)
    cgf.MOE_RECOMPUTE_STATS["moe_layers_recompute_restored"] = 0
    assert cgf.install_moe_recompute_for_eager_moe()
    return cls


@pytest.mark.parametrize(
    "impl,modules,expected",
    [
        ("local", ["mamba", "attn"], True),
        ("local", ["mamba"], True),
        ("local", ["mamba", "attn", "moe_router"], False),
        ("local", ["moe_router", "moe_preprocess"], False),
        ("local", ["moe"], False),
        ("local", [], False),  # empty list = whole layers, MoE included
        ("none", ["mamba"], False),
        ("transformer_engine", ["mamba"], False),
    ],
)
def test_moe_runs_outside_graphs(impl, modules, expected):
    assert cgf.moe_runs_outside_graphs(_config(impl, modules)) is expected


def test_recompute_is_restored_only_when_moe_is_eager(stand_in):
    assert stand_in(_config("local", ["mamba", "attn"])).moe_layer_recompute is True
    assert stand_in(_config("local", ["mamba", "attn", "moe_router"])).moe_layer_recompute is False
    assert stand_in(_config("local", ["mamba"], recompute=("layernorm",))).moe_layer_recompute is False
    assert stand_in(_config("none", [])).moe_layer_recompute is True  # unchanged eager rule
    assert cgf.MOE_RECOMPUTE_STATS["moe_layers_recompute_restored"] == 1
    assert cgf.install_moe_recompute_for_eager_moe()  # idempotent


def test_real_megatron_has_the_guard_line():
    moe_layer = pytest.importorskip("megatron.core.transformer.moe.moe_layer")
    import inspect

    assert cgf._MOE_RECOMPUTE_GUARD in inspect.getsource(moe_layer.MoELayer.__init__)
