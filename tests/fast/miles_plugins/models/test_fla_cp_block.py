"""``MILES_K31_KDA_FLA_CP_BLOCK``: the switch is read as documented, and ``install`` makes fla's CP pre-process
launchers launch their kernels with the requested BLOCK_SIZE and the column grid recomputed for it, only where fla
chose 64 (K > 64), and nothing otherwise. Uses a stand-in ``fla.ops.cp.chunk_delta_h`` whose launchers follow fla
0.5.2's code (kernel looked up as a module global at call time; ``BLOCK_SIZE = 32 if K <= 64 else 64``), plus, when
the real fla imports, a check that its launchers still reference those globals. That 32 and 64 give bit-equal
results is a GPU property (B300: hm / dhm / dv equal, work/kda-comm2 gpu_check Part C)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "models" / "fla_cp_block.py"
_SPEC = importlib.util.spec_from_file_location("_fla_cp_block_under_test", _PATH)
fla_cp_block = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fla_cp_block)

_FAKE_SRC = """
calls = []

class _Kernel:
    def __init__(self, name):
        self.name = name
    def __getitem__(self, grid):
        def run(**kwargs):
            calls.append((self.name, grid, kwargs["BLOCK_SIZE"]))
        return run

pre_process_fwd_kernel_merged = _Kernel("fwd")
pre_process_bwd_kernel_merged = _Kernel("bwd")

def cdiv(a, b):
    return -(-a // b)

def chunk_gated_delta_rule_fwd_h_pre_process(K, V, HV):
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (cdiv(V, BLOCK_SIZE) + cdiv(K, BLOCK_SIZE), HV)
    pre_process_fwd_kernel_merged[grid](K=K, V=V, BLOCK_SIZE=BLOCK_SIZE)

def chunk_gated_delta_rule_bwd_dhu_pre_process(K, V, HV):
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (cdiv(V, BLOCK_SIZE) + cdiv(K, BLOCK_SIZE), HV)
    pre_process_bwd_kernel_merged[grid](K=K, V=V, BLOCK_SIZE=BLOCK_SIZE)
"""


@pytest.fixture
def fake_fla(monkeypatch):
    module = types.ModuleType(fla_cp_block.FLA_MODULE)
    exec(_FAKE_SRC, module.__dict__)
    monkeypatch.setitem(sys.modules, fla_cp_block.FLA_MODULE, module)
    return module


def _run(module, K, V, HV=12):
    module.calls.clear()
    module.chunk_gated_delta_rule_fwd_h_pre_process(K, V, HV)
    module.chunk_gated_delta_rule_bwd_dhu_pre_process(K, V, HV)
    return list(module.calls)


@pytest.mark.parametrize("raw,want", [(None, None), ("", None), ("0", None), ("64", None), ("32", 32), ("16", 16)])
def test_switch_values(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(fla_cp_block.FLA_CP_BLOCK_ENV, raising=False)
    else:
        monkeypatch.setenv(fla_cp_block.FLA_CP_BLOCK_ENV, raw)
    assert fla_cp_block.fla_cp_block() == want


def test_switch_rejects_other_blocks(monkeypatch):
    monkeypatch.setenv(fla_cp_block.FLA_CP_BLOCK_ENV, "48")
    with pytest.raises(ValueError):
        fla_cp_block.fla_cp_block()


def test_off_leaves_fla_alone(monkeypatch, fake_fla):
    monkeypatch.delenv(fla_cp_block.FLA_CP_BLOCK_ENV, raising=False)
    kernel = fake_fla.pre_process_fwd_kernel_merged
    assert fla_cp_block.install() is None
    assert fake_fla.pre_process_fwd_kernel_merged is kernel
    assert _run(fake_fla, 128, 128) == [("fwd", (4, 12), 64), ("bwd", (4, 12), 64)]


def test_block_32_k128(monkeypatch, fake_fla):
    monkeypatch.setenv(fla_cp_block.FLA_CP_BLOCK_ENV, "32")
    assert fla_cp_block.install() == 32
    # K = V = 128: (4 + 4) column CTAs of 32 instead of (2 + 2) of 64
    assert _run(fake_fla, 128, 128) == [("fwd", (8, 12), 32), ("bwd", (8, 12), 32)]
    # fla already picks 32 for K <= 64: untouched
    assert _run(fake_fla, 64, 64) == [("fwd", (4, 12), 32), ("bwd", (4, 12), 32)]
    # idempotent, and a later install takes the new block
    assert fla_cp_block.install() == 32
    monkeypatch.setenv(fla_cp_block.FLA_CP_BLOCK_ENV, "16")
    assert fla_cp_block.install() == 16
    assert _run(fake_fla, 128, 128) == [("fwd", (16, 12), 16), ("bwd", (16, 12), 16)]
    assert isinstance(fake_fla.pre_process_fwd_kernel_merged.kernel, type(fake_fla._Kernel("x")))


def test_install_refuses_other_fla(monkeypatch, fake_fla):
    monkeypatch.setenv(fla_cp_block.FLA_CP_BLOCK_ENV, "32")
    del fake_fla.pre_process_bwd_kernel_merged
    with pytest.raises(RuntimeError):
        fla_cp_block.install()


def test_real_fla_structure():
    """The installed fla still launches the two kernels from module globals (skipped where fla is absent)."""
    try:
        module = importlib.import_module(fla_cp_block.FLA_MODULE)
    except Exception as exc:  # no fla / no triton backend on this host
        pytest.skip(f"fla not importable: {exc!r}")
    for kernel_name, launcher_name in fla_cp_block.KERNELS.items():
        assert kernel_name in getattr(module, launcher_name).__code__.co_names
        assert hasattr(module, kernel_name)
