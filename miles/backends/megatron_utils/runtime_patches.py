"""Process-wide patches the Megatron training stack needs before it runs.

Importing this module (once per process; later imports are no-ops):

* wraps ``deep_ep.Buffer.__init__`` so torch_memory_saver does not track the
  DeepEP buffers;
* imports ``miles_plugins.megatron_bridge``, which registers miles' bridge
  subclasses and shims with ``megatron.bridge`` before any ``AutoBridge`` use;
* keeps the CPU FP32 masters of Megatron's HybridDeviceOptimizer in pinned
  memory (``hybrid_optimizer_pinning``).

These used to run in the package ``__init__``, so every process that touched
a light submodule (``megatron_config``, ``checkpoint_tracker``) paid for
``megatron.bridge`` -- including RayWorkerManager, which unpickles placement
group and worker-spec objects whose modules import those submodules.  Modules
that need the patches (the trainer actors, model provider, checkpoint, HF
export and weight-update paths, ``AutoBridge`` call sites) import this module.
"""

import logging

import torch

from miles.backends.megatron_utils import hybrid_optimizer_pinning

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    import miles_plugins.megatron_bridge  # noqa: F401
except Exception as _e:  # best-effort; not every environment uses megatron.bridge
    logging.warning("miles megatron.bridge plugins failed to load: %s", _e)

# HybridDeviceOptimizer (--optimizer-cpu-offload) keeps its CPU FP32 masters in
# pinned memory, so their per-step copy back to the GPU is a DMA copy.
hybrid_optimizer_pinning.install()
