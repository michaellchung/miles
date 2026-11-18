"""Hide the MoE expert-parallel all-to-alls of one microbatch behind work of the same layer that does not need them.

``MILES_K3_MOE_A2A_OVERLAP=1`` (default off) builds K3's MoE layers as :class:`A2AOverlapMoELayer`. Megatron's
``MoELayer.forward`` still drives the layer (routing, padding mask, logging, latent projections); only three
steps change where they run:

* dispatch: the token all-to-all goes on a side communication stream; while it is in flight the current stream
  dequantizes the frozen MXFP4 experts' fc1 weights (they depend on the weights only, not on the tokens), then
  the probs all-to-all follows on the side stream. ``MILES_K3_MOE_A2A_OVERLAP_PREFETCH=0`` skips the prefetch.
* combine: the combine all-to-all goes on the side stream and the shared experts (``shared_experts_compute``,
  which Megatron runs before the router) run on the current stream while it is in flight.
* backward: autograd runs each node's backward on its forward stream, so the all-to-all gradients also run on
  the side stream. The engine runs ready nodes in decreasing sequence number. The shared experts' nodes are
  created after the combine all-to-all node and before the unpermute that consumes it. Their backward therefore
  runs right after the unpermute's backward has queued the combine-backward all-to-all's input, and overlaps it.

Every kernel sees the same inputs as without the switch, and the dequantization is the same function of the
same bytes. Only the stream and launch order change. One sum would otherwise change: the layer input's
gradient, whose terms arrive in a new order. :class:`_ForkSharedInput` keeps Megatron's order (routed path
first, shared experts last). So the outputs and every gradient are bit-identical.

Launch order is chosen for ``CUDA_DEVICE_MAX_CONNECTIONS=1``, which yeto sets, where every stream shares one
hardware queue. A queued command that waits on unfinished work blocks everything queued behind it. So
nothing that depends on an in-flight all-to-all is queued before the work meant to overlap it: token a2a, then
the dequantization, then the probs a2a (the same reason Megatron's ``token_dispatch`` puts shared fc1 between
its two all-to-alls).

The NCCL collectives on each communicator keep their order, the same on every rank: tokens, probs, combine
on the EP group, and the shared experts' TP collectives after the dispatch instead of before the router. A rank
only waits on another stream for work it queued itself.

``MILES_K3_MOE_A2A_OVERLAP_BWD=1`` (default off, needs the switch above) fixes the backward. Without it the
combine-backward all-to-all is launched by its own autograd node, which the engine runs only after the shared
experts' backward (their nodes have the higher sequence number). Under ``CUDA_DEVICE_MAX_CONNECTIONS=1`` that
all-to-all then waits in the single hardware queue behind every shared-expert kernel, so the two run one after
the other (x5 trace: the all-to-all starts 5.5-5.9 ms after its input is ready, in all 30 layer-microbatches of
both profiled ranks). With the switch the combine output passes through :class:`_LaunchCombineBackward`, created
after the shared experts. Its backward runs right after the unpermute's backward (before the shared experts'),
launches the combine-backward all-to-all on the side stream, and stashes the result. The all-to-all's own node
(:class:`_CombineAllToAll`) then returns the stash. The same backward also dequantizes, on the current stream
inside that window, the frozen experts whose dgrad comes first (``MILES_K3_MOE_A2A_OVERLAP_BWD_PREFETCH``:
``fc2`` by default, ``fc1,fc2`` or ``none``); this needs ``MILES_K31_MXFP4_REDEQUANT_BACKWARD=1`` with the
grouped (Triton or batched) dequantization, which is what saves one packed group per fetch. In the forward the
dispatch window also dequantizes fc2, after fc1. The all-to-all is the same call on the same gradient, the
dequantization the same function of the same bytes, so every output and gradient stays bit-identical.

Not used (``super()`` paths) under Megatron's own overlaps (``moe_shared_expert_overlap``,
``overlap_moe_expert_parallel_comm``, ``overlap_dispatch_backward_with_experts_wgrad``), MoE-module
recompute, CUDA graphs, the inference dispatchers or any dispatcher other than all-to-all, partial
``fwd_execution_map`` calls, or EP 1. On CPU the reordering still happens and the stream steps are no-ops,
so the gloo tests check the reordered computation.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os

import torch
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

logger = logging.getLogger(__name__)

A2A_OVERLAP_ENV = "MILES_K3_MOE_A2A_OVERLAP"
PREFETCH_ENV = "MILES_K3_MOE_A2A_OVERLAP_PREFETCH"
BWD_ENV = "MILES_K3_MOE_A2A_OVERLAP_BWD"
BWD_PREFETCH_ENV = "MILES_K3_MOE_A2A_OVERLAP_BWD_PREFETCH"
_BWD_PREFETCH_CHOICES = {"none": (), "fc2": ("linear_fc2",), "fc1,fc2": ("linear_fc2", "linear_fc1")}
_FULL_EXECUTION = ["route", "expert_compute", "postprocess"]
_comm_streams: dict[int, torch.cuda.Stream] = {}
_reported = False


def a2a_overlap_enabled() -> bool:
    return os.environ.get(A2A_OVERLAP_ENV, "0") == "1"


def prefetch_enabled() -> bool:
    return os.environ.get(PREFETCH_ENV, "1") == "1"


def bwd_overlap_enabled() -> bool:
    return os.environ.get(BWD_ENV, "0") == "1"


def bwd_prefetch_projections() -> tuple[str, ...]:
    """The frozen projections whose backward dequantization runs in the combine-backward window, in order."""
    value = os.environ.get(BWD_PREFETCH_ENV, "fc2")
    if value not in _BWD_PREFETCH_CHOICES:
        raise ValueError(f"{BWD_PREFETCH_ENV} must be one of {sorted(_BWD_PREFETCH_CHOICES)}, got {value!r}")
    return _BWD_PREFETCH_CHOICES[value]


def _comm_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _comm_streams:
        # high priority: the NCCL CTAs get SMs ahead of the compute kernels they overlap
        _comm_streams[index] = torch.cuda.Stream(device=index, priority=-1)
    return _comm_streams[index]


@contextlib.contextmanager
def _on_comm_stream(*inputs: torch.Tensor, wait: bool = True):
    """Run the body on the side stream. With ``wait`` the stream first waits for the work already queued on
    the current stream (the producers of ``inputs``). ``inputs`` are recorded on the side stream, so their
    memory is not reused while it reads them. Yields the side stream (None off CUDA)."""
    if not inputs[0].is_cuda:
        yield None
        return
    comm = _comm_stream(inputs[0].device)
    if wait:
        comm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(comm):
        yield comm
    for tensor in inputs:
        tensor.record_stream(comm)


def _join(comm: torch.cuda.Stream | None, *outputs: torch.Tensor) -> None:
    """The current stream waits for the side stream. ``outputs``, which the side stream allocated, are
    recorded on the current stream that reads them from here on."""
    if comm is None:
        return
    current = torch.cuda.current_stream()
    current.wait_stream(comm)
    for tensor in outputs:
        tensor.record_stream(current)


def _frozen_projection(experts, name: str):
    from miles_plugins.models.kimi_k3.mxfp4_redequant import RedequantFrozenExperts

    # PartiallyTrainableGroupedMLP or a whole-MXFP4 TEGroupedMLP
    frozen = getattr(experts, "frozen", experts)  # config-access-exempt: split MLP half or whole experts
    linear = getattr(frozen, name, None)  # config-access-exempt: linear_fc1 / linear_fc2 by name
    return linear if isinstance(linear, RedequantFrozenExperts) else None


def prefetch_frozen_expert_weights(experts, projections=("linear_fc1",)) -> bool:
    """Dequantize the frozen MXFP4 experts' weights of ``projections`` now, in order; each grouped linear's
    next weight fetch takes them. Returns whether there was anything to prefetch."""
    found = False
    for name in projections:
        linear = _frozen_projection(experts, name)
        if linear is not None:
            linear.prefetch_weights()
            found = True
    return found


def _take_saved_groups(experts, projections) -> list:
    """The packed groups the frozen projections saved for the backward in the forward that just ran (one per
    projection, in ``projections`` order; projections that saved none are skipped)."""
    groups = []
    for name in projections:
        linear = _frozen_projection(experts, name)
        group = linear.take_saved_group() if linear is not None else None
        if group is not None:
            groups.append(group)
    return groups


def _combine_backward_all_to_all(group, grad, output_splits, input_splits):
    """The combine all-to-all's gradient: Megatron's ``_AllToAll.backward`` call."""
    return all_to_all(group, grad, output_splits, input_splits)


class _CombineBackwardHandle:
    """What the two combine nodes share: the backward all-to-all's arguments, its early result, and the packed
    frozen-expert groups to dequantize in its window."""

    __slots__ = ("group", "output_splits", "input_splits", "result", "prefetch_groups")

    def __init__(self, group, output_splits, input_splits, prefetch_groups):
        self.group = group
        self.output_splits = output_splits
        self.input_splits = input_splits
        self.result = None
        self.prefetch_groups = prefetch_groups

    def all_to_all(self, grad):
        return _combine_backward_all_to_all(self.group, grad, self.output_splits, self.input_splits)


class _CombineAllToAll(torch.autograd.Function):
    """Megatron's combine all-to-all (``token_combine``'s ``all_to_all`` call); its backward returns the
    gradient :class:`_LaunchCombineBackward` already sent, and only falls back to sending it itself."""

    @staticmethod
    def forward(ctx, hidden_states, group, output_splits, input_splits, handle):
        ctx.handle = handle
        return all_to_all(group, hidden_states, output_splits, input_splits)

    @staticmethod
    def backward(ctx, grad):
        handle, ctx.handle = ctx.handle, None
        result, handle.result = handle.result, None
        if result is None:
            result = handle.all_to_all(grad)
        return result, None, None, None, None


class _LaunchCombineBackward(torch.autograd.Function):
    """Identity on the combine output, created after the shared experts. Its backward runs before theirs and
    launches the combine-backward all-to-all on the side stream, then dequantizes the frozen experts the
    expert backward needs first, on the current stream, while it is in flight."""

    @staticmethod
    def forward(ctx, combined, handle):
        ctx.handle = handle
        return combined.view_as(combined)

    @staticmethod
    def backward(ctx, grad):
        handle, ctx.handle = ctx.handle, None
        with _on_comm_stream(grad) as comm:
            result = handle.all_to_all(grad)
        if comm is not None:
            # read on the current stream by the expert backward; freed only after that work
            result.record_stream(torch.cuda.current_stream())
        handle.result = result
        groups, handle.prefetch_groups = handle.prefetch_groups, ()
        for packed_group in groups:
            packed_group.prefetch()
        return grad, None


class _ForkSharedInput(torch.autograd.Function):
    """The MoE input, once for the routed path (router, latent projection, permute) and once for the shared
    experts. Backward: the routed path's gradient (accumulated in its own arrival order) plus the shared
    experts'. That is exactly the sum Megatron's graph forms at the layer input: there the shared experts,
    created before the router, deliver last. Moving them later in creation order would otherwise make them
    deliver first, and the bf16/fp32 sum would round differently."""

    @staticmethod
    def forward(ctx, hidden_states):
        return hidden_states.view_as(hidden_states), hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_routed, grad_shared):
        if grad_routed is None:
            return grad_shared
        if grad_shared is None:
            return grad_routed
        return grad_routed + grad_shared


class A2AOverlapMoELayer(MoELayer):
    """``MoELayer`` whose EP all-to-alls overlap the shared experts and the frozen experts' dequantization."""

    _a2a_overlap = False
    _shared_input: torch.Tensor | None = None
    _deferred_shared_input: torch.Tensor | None = None
    _deferred_shared_output: torch.Tensor | None = None

    def _a2a_overlap_applies(self, intermediate_tensors) -> bool:
        dispatcher = self.token_dispatcher
        config = self.config
        return (
            intermediate_tensors is None
            and isinstance(dispatcher, MoEAlltoAllTokenDispatcher)
            # the reimplemented dispatch below is MoEAlltoAllTokenDispatcher.token_dispatch's body
            and type(dispatcher).token_dispatch is MoEAlltoAllTokenDispatcher.token_dispatch
            and type(dispatcher).token_combine is MoEAlltoAllTokenDispatcher.token_combine
            and dispatcher.ep_size > 1
            and dispatcher.shared_experts is None
            and not dispatcher.use_nccl_stream
            and not dispatcher.drop_and_pad
            and not self.shared_expert_overlap
            and not self.moe_layer_recompute
            and not config.overlap_moe_expert_parallel_comm
            and not config.overlap_dispatch_backward_with_experts_wgrad
            and config.cuda_graph_impl == "none"
            and not hasattr(self, "_inference_token_dispatcher")  # config-access-exempt: Megatron version probe
            and list(self.fwd_execution_map) == _FULL_EXECUTION
        )

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None, input_ids=None, **kwargs):
        self._a2a_overlap = self._a2a_overlap_applies(intermediate_tensors)
        _report_once(self._a2a_overlap)
        self._shared_input = None
        if self._a2a_overlap and self.use_shared_expert and torch.is_grad_enabled() and hidden_states.requires_grad:
            hidden_states, self._shared_input = _ForkSharedInput.apply(hidden_states)
        try:
            return super().forward(
                hidden_states,
                intermediate_tensors=intermediate_tensors,
                padding_mask=padding_mask,
                input_ids=input_ids,
                **kwargs,
            )
        finally:
            self._a2a_overlap = False
            self._shared_input = None
            self._deferred_shared_input = None
            self._deferred_shared_output = None

    def shared_experts_compute(self, hidden_states):
        if not self._a2a_overlap:
            return super().shared_experts_compute(hidden_states)
        # Megatron calls this before the router; the shared experts run in combine(), under the combine a2a
        self._deferred_shared_input = hidden_states if self._shared_input is None else self._shared_input
        return None

    def dispatch(self, hidden_states, probs):
        if not self._a2a_overlap:
            return super().dispatch(hidden_states, probs)
        dispatcher = self.token_dispatcher
        # MoEAlltoAllTokenDispatcher.token_dispatch with the all-to-alls on the side stream
        dispatcher.tokens_per_expert = dispatcher._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", dispatcher.tokens_per_expert
        )
        with _on_comm_stream(hidden_states, probs) as comm:
            global_tokens = all_to_all(
                dispatcher.ep_group, hidden_states, dispatcher.output_splits, dispatcher.input_splits
            )
        if prefetch_enabled():
            projections = ("linear_fc1", "linear_fc2") if bwd_overlap_enabled() else ("linear_fc1",)
            prefetch_frozen_expert_weights(self.experts, projections)
        # probs were ready with the tokens: no second wait, which would queue this behind the dequantization
        with _on_comm_stream(probs, wait=False):
            global_probs = all_to_all(dispatcher.ep_group, probs, dispatcher.output_splits, dispatcher.input_splits)
        _join(comm, global_tokens, global_probs)
        return global_tokens, global_probs

    def combine(self, output):
        if not self._a2a_overlap:
            return super().combine(output)
        handle = None
        if bwd_overlap_enabled() and torch.is_grad_enabled() and output.requires_grad:
            dispatcher = self.token_dispatcher
            # this forward's saved groups (each forward saves its own: recompute, or several microbatches in flight)
            groups = _take_saved_groups(self.experts, bwd_prefetch_projections())
            # backward splits = Megatron's _AllToAll.backward: the forward's output/input splits swapped
            handle = _CombineBackwardHandle(
                dispatcher.ep_group, dispatcher.output_splits, dispatcher.input_splits, groups
            )
        with _on_comm_stream(output) as comm:
            if handle is None:
                combined = super().combine(output)
            else:
                dispatcher = self.token_dispatcher
                combined = _CombineAllToAll.apply(
                    output, dispatcher.ep_group, dispatcher.input_splits, dispatcher.output_splits, handle
                )
        shared_input, self._deferred_shared_input = self._deferred_shared_input, None
        if shared_input is not None:
            self._deferred_shared_output = super().shared_experts_compute(shared_input)
        _join(comm, combined)
        if handle is not None:
            # created after the shared experts: its backward runs before theirs (higher sequence number)
            combined = _LaunchCombineBackward.apply(combined, handle)
        return combined

    def postprocess(self, output, shared_expert_output):
        if self._a2a_overlap:
            assert shared_expert_output is None, "A2AOverlapMoELayer defers the shared experts to combine()"
            shared_expert_output, self._deferred_shared_output = self._deferred_shared_output, None
        return super().postprocess(output, shared_expert_output)


def _report_once(active: bool) -> None:
    global _reported
    if _reported:
        return
    _reported = True
    if active:
        print(f"[k3] {A2A_OVERLAP_ENV}=1: MoE EP all-to-alls overlap shared experts / expert dequant", flush=True)
        if bwd_overlap_enabled():
            print(
                f"[k3] {BWD_ENV}=1: combine-backward all-to-all launched before the shared experts' backward; "
                f"backward prefetch {bwd_prefetch_projections()}",
                flush=True,
            )
    else:
        logger.warning(
            "%s=1 but this MoE layer's configuration is not supported; Megatron's order is kept", A2A_OVERLAP_ENV
        )


def a2a_overlap_moe_spec(mlp_spec):
    """Swap ``partial(MoELayer, ...)`` (or a ModuleSpec of MoELayer) for :class:`A2AOverlapMoELayer`."""
    if isinstance(mlp_spec, functools.partial):
        assert mlp_spec.func is MoELayer, mlp_spec.func
        return functools.partial(A2AOverlapMoELayer, *mlp_spec.args, **mlp_spec.keywords)
    assert getattr(mlp_spec, "module", None) is MoELayer, mlp_spec  # config-access-exempt: ModuleSpec
    mlp_spec.module = A2AOverlapMoELayer
    return mlp_spec
