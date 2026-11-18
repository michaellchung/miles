"""The K3 KDA table comes from the checkpoint's linear_attn_config, so a layer-renumbering cut keeps its MLA layers;
the K3.1 DSA check reads the same table."""

import importlib.util
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[5]


def _load(name, rel):
    # by file path: the model packages import Megatron, which CPU CI does not install
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kda = _load("_kda_layout_under_test", "miles_plugins/models/kimi_k3/kda_layout.py")
dsa = _load("_dsa_layout_under_test", "miles_plugins/models/kimi_k31/dsa_layout.py")

FULL_KDA = [n for n in range(1, 94) if n % 4 != 0 and n != 93]
FULL_MLA = [n for n in range(1, 94) if n not in FULL_KDA]
# ops/mfu-16l stage-2 cut: original layer 1, then 62..93, renumbered 1..33 (new 33 = original 93, MLA)
STAGE2_ORIG = [1, *range(62, 94)]
STAGE2_KDA = [new for new, orig in enumerate(STAGE2_ORIG, start=1) if orig in FULL_KDA]
STAGE2_MLA = [new for new, orig in enumerate(STAGE2_ORIG, start=1) if orig not in FULL_KDA]


def _text_config(kda_layers=None):
    linear = None if kda_layers is None else {"kda_layers": list(kda_layers), "num_heads": 96}
    return types.SimpleNamespace(linear_attn_config=linear)


def test_released_table_is_the_released_layout():
    assert list(kda.KIMI_K3_KDA_LAYERS) == FULL_KDA


@pytest.mark.parametrize("table", [FULL_KDA, None])
def test_full_model(table):
    layers = kda.kimi_k3_kda_layers(_text_config(table), 93)
    assert list(layers) == FULL_KDA
    assert dsa.checked_dsa_layers(FULL_MLA, num_layers=93, kda_layers=layers) == FULL_MLA


@pytest.mark.parametrize("table", [FULL_KDA, [n for n in FULL_KDA if n <= 16], None])
def test_prefix_cut_keeps_or_trims_the_table(table):
    # the L16 checkpoint: the first 16 layers, with the full or the trimmed table (or none)
    layers = kda.kimi_k3_kda_layers(_text_config(table), 16)
    assert list(layers) == [n for n in FULL_KDA if n <= 16]
    assert dsa.checked_dsa_layers(FULL_MLA, num_layers=16, kda_layers=layers) == [4, 8, 12, 16]


def test_stage2_cut_renumbers_its_layers():
    assert STAGE2_MLA == [4, 8, 12, 16, 20, 24, 28, 32, 33]
    layers = kda.kimi_k3_kda_layers(_text_config(STAGE2_KDA), 33)
    assert list(layers) == STAGE2_KDA and 33 not in layers
    assert dsa.checked_dsa_layers(STAGE2_MLA, num_layers=33, kda_layers=layers) == STAGE2_MLA
    # the released layout calls layer 33 KDA: the bug this table fixes
    with pytest.raises(AssertionError):
        dsa.checked_dsa_layers(STAGE2_MLA, num_layers=33, kda_layers=kda.kimi_k3_kda_layers(_text_config(), 33))


def test_table_without_kda_layers_key_falls_back():
    assert kda.kimi_k3_kda_layers(types.SimpleNamespace(linear_attn_config={"num_heads": 96}), 93) == tuple(FULL_KDA)
    assert kda.kimi_k3_kda_layers(types.SimpleNamespace(), 8) == (1, 2, 3, 5, 6, 7)
