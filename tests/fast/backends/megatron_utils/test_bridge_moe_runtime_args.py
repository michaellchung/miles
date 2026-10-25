"""CPU tests: MoE runtime flags reach a bridge-built provider, and the unsafe ones are rejected.

Bridge mode builds the model config from the HF checkpoint, so a command-line flag reaches the
model only through ``model_provider._apply_bridge_runtime_config``. These tests run that function
on a stand-in provider with the Nemotron-H bridge defaults; no Megatron model, GPU or Ray.
"""

import argparse
import ast
import dataclasses
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args
from miles.utils.megatron_bridge_utils import (
    BRIDGE_MOE_RUNTIME_ARG_DEFAULTS,
    apply_bridge_moe_runtime_args,
    describe_bridge_moe_runtime_config,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@dataclasses.dataclass
class _NemotronHProvider:
    """The MoE runtime fields of a Nemotron-H bridge provider before Miles copies args onto it.

    ``NemotronHBridge.provider_bridge`` sets the dispatcher to alltoall and turns permute fusion and
    shared-expert overlap on; the other fields keep the TransformerConfig defaults.
    """

    moe_token_dispatcher_type: str = "alltoall"
    moe_flex_dispatcher_backend: str = "deepep"
    moe_flex_dispatcher_num_sms: int | None = None
    moe_deepep_num_sms: int | None = None
    moe_hybridep_num_sms: int | None = None
    moe_enable_deepep: bool = False
    moe_permute_fusion: bool = True
    moe_router_fusion: bool = False
    moe_shared_expert_overlap: bool = True
    overlap_moe_expert_parallel_comm: bool = False
    delay_wgrad_compute: bool = False
    cuda_graph_impl: str = "none"
    cuda_graph_modules: object = "full"
    cuda_graph_warmup_steps: int = 3
    use_te_rng_tracker: bool = False
    mtp_num_layers: int | None = None
    mtp_detach_heads: bool = False


@pytest.fixture(scope="module")
def apply_bridge_runtime_config() -> Callable:
    # The production helper, without importing the GPU-only Megatron modules of model_provider.py.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_bridge_runtime_config"
    )
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_bridge_runtime_config"]


def _megatron_args(**overrides) -> argparse.Namespace:
    """The args of an RL island run (G-AN Xdeepep / s27-ba-a): flex + DeepEP, permute fusion, R3."""
    args = argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        gradient_accumulation_fusion=True,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity="selective",
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=["moe", "layernorm"],
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe="delayed",
        attention_backend="auto",
        moe_token_dispatcher_type="flex",
        cuda_graph_impl="none",
        use_routing_replay=True,
        use_rollout_routing_replay=True,
        **dict(BRIDGE_MOE_RUNTIME_ARG_DEFAULTS),
    )
    args.moe_permute_fusion = True
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_an_rl_island_run_keeps_the_provider_moe_settings(apply_bridge_runtime_config: Callable) -> None:
    """The args of the existing runs must build the model they built before this copy existed."""
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, _megatron_args())

    assert provider.moe_token_dispatcher_type == "flex"
    assert provider.moe_flex_dispatcher_backend == "deepep"
    assert provider.moe_permute_fusion is True
    # args hold the CLI default False (the flag is absent); the provider's True stays
    assert provider.moe_shared_expert_overlap is True
    assert provider.moe_router_fusion is False
    assert provider.moe_flex_dispatcher_num_sms is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("moe_flex_dispatcher_backend", "deepepv2"),
        ("moe_flex_dispatcher_backend", "ncclep"),
        ("moe_flex_dispatcher_num_sms", 32),
        ("moe_deepep_num_sms", 24),
        ("moe_hybridep_num_sms", 16),
        ("moe_enable_deepep", True),
        ("overlap_moe_expert_parallel_comm", True),
        ("delay_wgrad_compute", True),
    ],
)
def test_a_set_flag_reaches_the_provider(apply_bridge_runtime_config: Callable, name: str, value: object) -> None:
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, _megatron_args(**{name: value}))

    assert getattr(provider, name) == value


@pytest.mark.parametrize("name", ["moe_permute_fusion", "moe_shared_expert_overlap"])
def test_a_store_true_flag_turns_a_provider_setting_on(apply_bridge_runtime_config: Callable, name: str) -> None:
    provider = _NemotronHProvider(**{name: False})

    apply_bridge_runtime_config(provider, _megatron_args(**{name: True}))

    assert getattr(provider, name) is True


def test_router_fusion_reaches_the_provider_without_routing_replay(apply_bridge_runtime_config: Callable) -> None:
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(
        provider, _megatron_args(moe_router_fusion=True, use_routing_replay=False, use_rollout_routing_replay=False)
    )

    assert provider.moe_router_fusion is True


def test_args_without_the_moe_flags_leave_the_provider_unchanged(apply_bridge_runtime_config: Callable) -> None:
    """Callers that register only part of the arguments (the other bridge tests) must keep working."""
    args = _megatron_args()
    absent = (*BRIDGE_MOE_RUNTIME_ARG_DEFAULTS, "cuda_graph_impl", "use_routing_replay", "use_rollout_routing_replay")
    for name in absent:
        delattr(args, name)
    provider = _NemotronHProvider()

    apply_bridge_runtime_config(provider, args)

    assert dataclasses.replace(provider, moe_token_dispatcher_type="alltoall") == _NemotronHProvider()


def test_a_set_flag_without_a_provider_field_raises(apply_bridge_runtime_config: Callable) -> None:
    @dataclasses.dataclass
    class _OlderProvider:
        moe_token_dispatcher_type: str = "alltoall"
        moe_router_fusion: bool = False

    with pytest.raises(AttributeError, match="--moe-flex-dispatcher-num-sms is set"):
        apply_bridge_runtime_config(_OlderProvider(), _megatron_args(moe_flex_dispatcher_num_sms=32))


@pytest.mark.parametrize(
    ("use_routing_replay", "use_rollout_routing_replay"),
    [(True, True), (True, False), (False, True)],
    ids=["r3", "r2", "r3-before-validation"],
)
def test_router_fusion_with_routing_replay_is_rejected(
    apply_bridge_runtime_config: Callable, use_routing_replay: bool, use_rollout_routing_replay: bool
) -> None:
    """The fused top-k returns before the replay hook, so replay would be skipped without an error."""
    args = _megatron_args(
        moe_router_fusion=True,
        use_routing_replay=use_routing_replay,
        use_rollout_routing_replay=use_rollout_routing_replay,
    )

    with pytest.raises(ValueError, match="cannot be used with routing replay"):
        apply_bridge_runtime_config(_NemotronHProvider(), args)


def test_a_provider_that_fuses_the_router_is_rejected_under_replay(apply_bridge_runtime_config: Callable) -> None:
    """The check reads the provider, so a bridge that turns router fusion on by itself is caught too."""
    with pytest.raises(ValueError, match="cannot be used with routing replay"):
        apply_bridge_runtime_config(_NemotronHProvider(moe_router_fusion=True), _megatron_args())


@pytest.mark.parametrize("impl", ["transformer_engine", "full_iteration"])
def test_cuda_graphs_are_rejected_in_bridge_mode(apply_bridge_runtime_config: Callable, impl: str) -> None:
    with pytest.raises(ValueError, match=f"--cuda-graph-impl {impl} is not supported"):
        apply_bridge_runtime_config(_NemotronHProvider(), _megatron_args(cuda_graph_impl=impl))


def test_local_cuda_graphs_copy_their_settings_onto_the_provider(
    apply_bridge_runtime_config: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--cuda-graph-impl local: impl, modules, warmup and the TE RNG tracker reach the provider, and the
    eager fallback for unmatched graph inputs is installed (stubbed here: no Megatron on CPU CI)."""
    from miles.backends.megatron_utils import cuda_graph_fallback

    installed = []
    monkeypatch.setattr(cuda_graph_fallback, "install", lambda: installed.append(True) or True)
    provider = _NemotronHProvider()
    args = _megatron_args(
        cuda_graph_impl="local", cuda_graph_modules=["mamba", "attn"], cuda_graph_warmup_steps=2, te_rng_tracker=True
    )

    apply_bridge_runtime_config(provider, args)

    assert provider.cuda_graph_impl == "local"
    assert provider.cuda_graph_modules == ["mamba", "attn"]
    assert provider.cuda_graph_warmup_steps == 2
    assert provider.use_te_rng_tracker is True
    assert installed == [True]
    # the MoE settings of the island run are unchanged
    assert provider.moe_token_dispatcher_type == "flex"
    assert provider.moe_flex_dispatcher_backend == "deepep"


def test_local_cuda_graphs_need_the_te_rng_tracker(apply_bridge_runtime_config: Callable) -> None:
    args = _megatron_args(
        cuda_graph_impl="local", cuda_graph_modules=["mamba"], cuda_graph_warmup_steps=3, te_rng_tracker=False
    )
    with pytest.raises(ValueError, match="needs --te-rng-tracker"):
        apply_bridge_runtime_config(_NemotronHProvider(), args)


def test_the_log_line_names_the_effective_values_and_the_copied_flags() -> None:
    provider = _NemotronHProvider(moe_token_dispatcher_type="flex")
    args = _megatron_args(moe_flex_dispatcher_backend="deepepv2", moe_flex_dispatcher_num_sms=32)
    apply_bridge_moe_runtime_args(provider, args)

    line = json.loads(describe_bridge_moe_runtime_config(provider, args))

    assert line["from_args"] == ["moe_flex_dispatcher_backend", "moe_flex_dispatcher_num_sms", "moe_permute_fusion"]
    assert line["effective"]["moe_token_dispatcher_type"] == "flex"
    assert line["effective"]["moe_flex_dispatcher_backend"] == "deepepv2"
    assert line["effective"]["moe_flex_dispatcher_num_sms"] == 32
    assert line["effective"]["moe_shared_expert_overlap"] is True
    assert line["effective"]["cuda_graph_impl"] == "none"


def test_the_table_holds_megatrons_cli_defaults() -> None:
    """A flag counts as set when args differ from this table, so the table must be the parser's default."""
    megatron_arguments = pytest.importorskip("megatron.training.arguments")
    parser = megatron_arguments.add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))

    assert {name: parser.get_default(name) for name in BRIDGE_MOE_RUNTIME_ARG_DEFAULTS} == dict(
        BRIDGE_MOE_RUNTIME_ARG_DEFAULTS
    )


class TestArgumentValidation:
    """miles_validate_args rejects router fusion with replay before any worker starts (both model paths)."""

    @staticmethod
    def _parse(extra: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(["--rollout-batch-size", "64", "--num-rollout", "1", *extra])
        args.moe_router_fusion = True  # a Megatron flag; the Megatron parser is not built here
        return args

    @pytest.mark.parametrize("flag", ["--use-rollout-routing-replay", "--use-routing-replay"])
    def test_router_fusion_with_replay_is_rejected(self, flag: str) -> None:
        with pytest.raises(ValueError, match="cannot be used with routing replay"):
            miles_validate_args(self._parse([flag]))

    def test_router_fusion_without_replay_is_accepted(self) -> None:
        args = self._parse([])

        miles_validate_args(args)

        assert args.moe_router_fusion is True


def test_the_bridge_provider_gets_the_flags_before_finalize_and_logs_them(monkeypatch, caplog) -> None:
    """The real bridge branch of get_model_provider_func, with a stand-in AutoBridge."""
    pytest.importorskip("megatron.core")
    from miles.backends.megatron_utils import model_provider  # needs Megatron, so imported here

    class _FinalizingProvider(_NemotronHProvider):
        def finalize(self) -> None:
            self.finalized_with = dataclasses.asdict(self)

    provider = _FinalizingProvider()
    auto_bridge = types.SimpleNamespace(
        from_hf_pretrained=lambda path, trust_remote_code: types.SimpleNamespace(
            to_megatron_provider=lambda load_weights: provider
        )
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge", types.SimpleNamespace(AutoBridge=auto_bridge))
    args = _megatron_args(
        moe_flex_dispatcher_num_sms=32,
        megatron_to_hf_mode="bridge",
        hf_checkpoint="/unused",
        custom_model_provider_path=None,
    )

    with caplog.at_level("INFO", logger=model_provider.__name__):
        model_provider.get_model_provider_func(args, role="actor")

    assert provider.finalized_with["moe_token_dispatcher_type"] == "flex"
    assert provider.finalized_with["moe_flex_dispatcher_num_sms"] == 32
    assert provider.finalized_with["moe_shared_expert_overlap"] is True
    (record,) = [r for r in caplog.records if r.getMessage().startswith("bridge MoE runtime config: ")]
    line = json.loads(record.getMessage().removeprefix("bridge MoE runtime config: "))
    assert line["effective"]["moe_flex_dispatcher_num_sms"] == 32
    assert "moe_flex_dispatcher_num_sms" in line["from_args"]
