"""loss_function's normalizer and logging values are bit-identical with and without host syncs (CPU).

Runs the dispatcher of every snapshot configuration twice (MILES_HOST_SYNC_FREE=1 and
=0) and also against ``torch.tensor`` built from the same logging dict.
"""

import pytest
import torch

from miles.backends.training_utils.loss import loss_function
from miles.utils import sync_free

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state
from .test_loss_snapshot import CONFIGS, SEED, VOCAB_SIZE, process_group  # noqa: F401  (autouse gloo group)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.bool else t.reshape(-1).view(torch.uint8)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(_bits(a), _bits(b)))
    return a == b


def _run(config, *, switch, per_token, monkeypatch):
    monkeypatch.setenv(sync_free.ENV_VAR, switch)
    _name, overrides, batch_size, prompt_lens, response_lens = config
    args = make_args(global_batch_size=batch_size, calculate_per_token_loss=per_token, **overrides)
    make_parallel_state()
    inputs = make_inputs(seed=SEED, batch_size=batch_size, prompt_lens=prompt_lens, response_lens=response_lens,
                         vocab_size=VOCAB_SIZE, args=args)
    batch = make_batch(inputs, args.loss_type)
    logits = deep_clone(inputs["policy_logits"] if args.loss_type != "value_loss" else inputs["value_logits"])
    logits.requires_grad_(True)
    loss, normalizer, log_dict = loss_function(args, batch, 1, logits)
    loss.backward()
    return loss.detach(), normalizer, log_dict, logits.grad


def _needs_megatron(config) -> bool:
    # chunked configurations use Megatron's fused vocab-parallel cross entropy
    return "chunked" in config[0]


@pytest.mark.parametrize("config", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.parametrize("per_token", [False, True])
def test_outputs_identical_with_and_without_host_syncs(config, per_token, monkeypatch):
    if _needs_megatron(config):
        pytest.importorskip("megatron.core.fusions.fused_cross_entropy")
    new = _run(config, switch="1", per_token=per_token, monkeypatch=monkeypatch)
    old = _run(config, switch="0", per_token=per_token, monkeypatch=monkeypatch)
    new_loss, new_norm, new_log, new_grad = new
    old_loss, old_norm, old_log, old_grad = old
    assert _same(new_loss, old_loss) and _same(new_grad, old_grad)
    assert _same(new_norm, old_norm)
    assert new_log["keys"] == old_log["keys"]
    assert _same(new_log["values"], old_log["values"])
    assert not new_log["values"].requires_grad and not new_norm.requires_grad
