"""MILES_K31_RECOMPUTE_NUM_LAYERS_PER_STAGE sets each pipeline stage's --recompute-method block layer count."""

import sys
import types

import pytest

from miles_plugins.models.kimi_k3 import stage_recompute as sr


def _count(raw, pp_rank=0, pp_size=3, stage_layers=32, granularity="full", method="block"):
    return sr.stage_recompute_num_layers(
        raw, pp_rank=pp_rank, pp_size=pp_size, stage_layers=stage_layers, granularity=granularity, method=method
    )


def test_unset_leaves_the_config_alone():
    assert _count("") is None and _count("  ") is None
    # unset is accepted whatever the recompute flags are
    assert _count("", granularity="selective", method=None) is None


@pytest.mark.parametrize("raw", ["31,27,24", "31 27 24", " 31, 27 ,24 "])
def test_each_rank_reads_its_own_count(raw):
    assert [_count(raw, pp_rank=r, stage_layers=(32, 29, 32)[r]) for r in range(3)] == [31, 27, 24]


@pytest.mark.parametrize(
    "raw,kwargs,match",
    [
        ("31,27,24", {"granularity": "selective"}, "needs --recompute-granularity full"),
        ("31,27,24", {"method": "uniform"}, "needs --recompute-granularity full"),
        ("31,27", {}, "2 counts for 3 pipeline stages"),
        ("33,27,24", {}, "stage 0 has 32 layers, not 33"),
        ("-1,27,24", {}, "not -1"),
    ],
)
def test_refuses_what_block_recompute_cannot_do(raw, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _count(raw, **kwargs)


def test_apply_sets_recompute_num_layers(monkeypatch):
    config = types.SimpleNamespace(
        pipeline_model_parallel_size=3, recompute_granularity="full", recompute_method="block", recompute_num_layers=32
    )
    monkeypatch.delenv(sr.STAGE_RECOMPUTE_ENV, raising=False)
    assert sr.apply_stage_recompute(config, pp_rank=1, stage_layers=29) is None
    assert config.recompute_num_layers == 32
    monkeypatch.setenv(sr.STAGE_RECOMPUTE_ENV, "31,27,24")
    assert sr.apply_stage_recompute(config, pp_rank=1, stage_layers=29) == 27
    assert config.recompute_num_layers == 27


def test_unset_needs_no_parallel_state(monkeypatch):
    """Converters build the spec without model-parallel state: unset must not ask Megatron for the PP rank."""
    monkeypatch.delenv(sr.STAGE_RECOMPUTE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", None)  # any import of it would raise
    config = types.SimpleNamespace(
        pipeline_model_parallel_size=3,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    assert sr.apply_stage_recompute(config, stage_layers=32) is None
    assert config.recompute_num_layers == 1


def _block_forward(config, num_layers, monkeypatch):
    """Megatron's own TransformerBlock._checkpointed_forward on stand-in layers: which layers run checkpointed."""
    tb = pytest.importorskip("megatron.core.transformer.transformer_block")
    import torch

    checkpointed, ran = [], []

    def fake_checkpoint(fn, distribute, *args):
        checkpointed.append(len(ran))
        return fn(*args)

    monkeypatch.setattr(tb.tensor_parallel, "checkpoint", fake_checkpoint)

    class _Layer:
        def __init__(self, index):
            self.layer_number = index + 1

        def __call__(self, *, hidden_states, context, **_):
            ran.append(self.layer_number - 1)
            return hidden_states + 1, context

    layers = [_Layer(i) for i in range(num_layers)]
    block = types.SimpleNamespace(
        config=config, num_layers_per_pipeline_rank=num_layers, _get_layer=lambda i: layers[i]
    )
    out = tb.TransformerBlock._checkpointed_forward(
        block,
        hidden_states=torch.zeros(2, requires_grad=True),
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
    )
    assert ran == list(range(num_layers)) and torch.equal(out, torch.full((2,), float(num_layers)))
    return checkpointed


def test_megatron_block_recomputes_the_first_n_of_this_stage(monkeypatch):
    """The count apply_stage_recompute sets is the one Megatron's block loop reads at forward time."""
    config = types.SimpleNamespace(
        pipeline_model_parallel_size=3,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=32,
        fp8=False,
        fp4=False,
        distribute_saved_activations=False,
    )
    monkeypatch.setenv(sr.STAGE_RECOMPUTE_ENV, "31,27,24")
    assert sr.apply_stage_recompute(config, pp_rank=2, stage_layers=32) == 24
    assert _block_forward(config, 32, monkeypatch) == list(range(24))
    # stock flags (no switch): uniform with one layer per chunk checkpoints every layer
    config.recompute_method, config.recompute_num_layers = "uniform", 1
    assert _block_forward(config, 5, monkeypatch) == list(range(5))
