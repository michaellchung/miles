"""miles.utils.sync_free: same tensors as the host-syncing calls it replaces (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from miles.utils import sync_free  # noqa: E402

T = torch.tensor


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bits (NaN payloads and signed zeros included)."""

    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False), ("off", False),
                                            (" False ", False), ("no", False), ("yes", True)])
def test_enabled_reads_the_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(sync_free.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(sync_free.ENV_VAR, value)
    assert sync_free.enabled() is expected


# Lists shaped like loss_function's logging values: a count followed by 0-d metrics.
LISTS = [
    [16] + [T(0.25), T(-1.5), T(3.0e-8), T(0.0), T(-0.0), T(1.0e38)],
    [T(4096), T(0.1), T(7, dtype=torch.int32)],
    [3, T(1.5), T(2, dtype=torch.int32)],
    [True, T(1, dtype=torch.int64)],
    [T(1.0, dtype=torch.bfloat16), T(2.5, dtype=torch.float16)],
    [T(1.0, dtype=torch.bfloat16), T(-3.0, dtype=torch.bfloat16)],
    [2, T(0.1, dtype=torch.float64)],
    [T(True), T(False)],
    [16, 1e300, -1e300],
    [T(16777216), 0.5],
    [T(float("nan")), T(float("inf")), T(float("-inf")), 0.1],
    [T(1.0e-45), 5e-324, 1.17e-38],
    [0.1, 0.2, 1 / 3],
    [7],
    [T(2.0)],
]


@pytest.mark.parametrize("values", LISTS, ids=range(len(LISTS)))
def test_scalars_to_tensor_matches_torch_tensor(values):
    assert _same(sync_free.scalars_to_tensor(values, device="cpu"), torch.tensor(values, device="cpu"))


@pytest.mark.parametrize("values", LISTS, ids=range(len(LISTS)))
def test_scalars_to_tensor_switch_off_is_torch_tensor(values, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, "0")
    assert _same(sync_free.scalars_to_tensor(values, device="cpu"), torch.tensor(values, device="cpu"))


def test_scalars_to_tensor_detaches():
    weight = torch.ones((), requires_grad=True)
    result = sync_free.scalars_to_tensor([weight * 2, 3], device="cpu")
    assert not result.requires_grad and result.tolist() == [2.0, 3.0]


@pytest.mark.parametrize(
    "values",
    [[], [[1.0, 2.0]], [T([1.0])], [T([1.0, 2.0])]],
    ids=["empty", "nested", "one-element-1d", "1d"],
)
def test_other_inputs_go_to_torch_tensor(values):
    try:
        expected = torch.tensor(values, device="cpu")
    except (ValueError, TypeError, RuntimeError) as exc:
        with pytest.raises(type(exc)):
            sync_free.scalars_to_tensor(values, device="cpu")
        return
    assert _same(sync_free.scalars_to_tensor(values, device="cpu"), expected)


def test_numpy_numbers_go_to_torch_tensor():
    np = pytest.importorskip("numpy")
    values = [np.float32(0.5), T(1.0)]
    assert _same(sync_free.scalars_to_tensor(values, device="cpu"), torch.tensor(values, device="cpu"))


@pytest.mark.parametrize("value", [1, 0, T(12), T(12, dtype=torch.int32), T(2.5), True, 0.1])
def test_scalar_to_tensor_matches_torch_tensor(value):
    result = sync_free.scalar_to_tensor(value, device="cpu")
    assert _same(result, torch.tensor(value, device="cpu"))
    if isinstance(value, torch.Tensor):
        assert result.data_ptr() != value.data_ptr()  # a copy, as torch.tensor(tensor) makes


def test_default_float_dtype_is_followed():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        assert _same(sync_free.scalars_to_tensor([0.1, 2], device="cpu"), torch.tensor([0.1, 2]))
    finally:
        torch.set_default_dtype(previous)


def test_to_device_async_on_cpu_keeps_values():
    data = torch.tensor([0, 3, 7, 12], dtype=torch.int32)
    assert _same(sync_free.to_device_async(data, "cpu"), data)


def test_source_sha256():
    def sample():
        return 1

    digest = sync_free.source_sha256(sample)
    assert digest == sync_free.source_text_sha256("def sample():\n    return 1\n")
    assert sync_free.source_sha256(len) is None  # builtin: no source
