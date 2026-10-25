"""CPU tests: after local CUDA graphs exist, a call with an uncaptured input signature runs eagerly
instead of hitting Megatron's ``assert False``; a matching call still replays its graph.

Needs an importable Megatron (radixark a84b10547); skipped otherwise. No GPU: the manager and runners
are stand-ins with the attributes ``CudaGraphManager.__call__`` reads.
"""

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

cg = pytest.importorskip("megatron.core.transformer.cuda_graphs")

from miles.backends.megatron_utils import cuda_graph_fallback  # noqa: E402

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class _Runner:
    def __init__(self, shape):
        self.shape, self.status, self.grad_enabled, self.replays = shape, cg._GraphStatus.FWD_READY, True, 0

    def get_mismatch_errors(self, args, kwargs):
        return [] if tuple(kwargs["hidden_states"].shape) == self.shape else ["shape"]

    def replay_graph_capture(self, is_first_microbatch, args, kwargs):
        self.replays += 1
        return ("graph", tuple(kwargs["hidden_states"].shape))


def _manager(runners):
    manager = cg.CudaGraphManager.__new__(cg.CudaGraphManager)
    torch.nn.Module.__init__(manager)
    manager.eval()  # no DDP pre-forward hooks on this path
    manager.need_backward, manager.reuse_cudagraphs = True, True
    manager.cudagraph_runners, manager.is_first_microbatch = runners, False
    manager.func = lambda hidden_states: ("eager", tuple(hidden_states.shape))
    return manager


@pytest.fixture
def graphs_created(monkeypatch: pytest.MonkeyPatch):
    assert cuda_graph_fallback.install(), "a84b10547 source expected"
    monkeypatch.setattr(cg._CudagraphGlobalRecord, "cudagraph_created", True)
    cuda_graph_fallback.STATS["eager_fallback_calls"] = 0


def test_matching_signature_replays_and_unmatched_runs_eagerly(graphs_created) -> None:
    runner = _Runner((56, 1, 8))
    manager = _manager([runner])
    with torch.no_grad():
        assert manager(None, (), {"hidden_states": torch.zeros(56, 1, 8)}) == ("graph", (56, 1, 8))
        assert manager(None, (), {"hidden_states": torch.zeros(70, 1, 8)}) == ("eager", (70, 1, 8))
    assert runner.replays == 1
    assert cuda_graph_fallback.STATS["eager_fallback_calls"] == 1


def test_install_is_idempotent_and_marks_the_class(graphs_created) -> None:
    assert cuda_graph_fallback.install()
    assert cg.CudaGraphManager._miles_eager_fallback is True
