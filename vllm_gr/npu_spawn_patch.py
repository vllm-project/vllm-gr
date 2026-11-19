# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Force the `spawn` multiprocessing start method once NPU is initialized.

Upstream `_maybe_force_spawn()` (vllm/utils/system_utils.py) forces spawn when
CUDA or XPU is already initialized in the parent process, because a forked
child cannot re-initialize those runtimes. torch_npu has the same constraint
("Cannot re-initialize NPU in forked subprocess") but upstream does not know
about NPU, so entry points that create engines after touching the NPU runtime
had to set VLLM_WORKER_MULTIPROC_METHOD by hand. This patch extends the
upstream check with an NPU branch so the decision is made centrally at
worker-creation time, with the same semantics as the CUDA/XPU branches: spawn
is forced only when the NPU runtime was already initialized in this process.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm_gr.npu_spawn_patch")

_ORIGINAL_ATTR = "_vllm_gr_npu_spawn_original"


def _npu_is_initialized() -> bool:
    try:
        import torch

        return bool(hasattr(torch, "npu") and torch.npu.is_initialized())
    except Exception:  # pragma: no cover - torch is always importable here
        return False


def _force_spawn_if_npu_initialized() -> None:
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn" and _npu_is_initialized():
        logger.warning(
            "We must use the `spawn` multiprocessing start method because NPU is "
            "initialized. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'."
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _is_npu_spawn_wrapper(function: object) -> bool:
    return hasattr(function, _ORIGINAL_ATTR)


def is_npu_spawn_patched() -> bool:
    import vllm.utils.system_utils as system_utils
    import vllm.v1.executor.multiproc_executor as multiproc_executor

    target = getattr(system_utils, "_maybe_force_spawn", None)
    return _is_npu_spawn_wrapper(target) and multiproc_executor._maybe_force_spawn is target


def apply_npu_spawn_patch() -> bool:
    """Wrap upstream `_maybe_force_spawn` to also force spawn for NPU.

    Returns True if a wrapper or executor binding was installed by this call.
    An already-wrapped target is reused and its executor binding is verified.
    """
    import vllm.utils.system_utils as system_utils
    import vllm.v1.executor.multiproc_executor as multiproc_executor

    current = getattr(system_utils, "_maybe_force_spawn", None)
    if current is None:
        return False
    if _is_npu_spawn_wrapper(current):
        if multiproc_executor._maybe_force_spawn is current:
            return False
        multiproc_executor._maybe_force_spawn = current
        return True

    def _maybe_force_spawn_with_npu() -> None:
        current()
        _force_spawn_if_npu_initialized()

    setattr(_maybe_force_spawn_with_npu, _ORIGINAL_ATTR, current)
    system_utils._maybe_force_spawn = _maybe_force_spawn_with_npu
    # multiproc_executor imports the symbol directly at module load; rebind
    # that reference too or set_multiprocessing_worker_envs() keeps the original.
    multiproc_executor._maybe_force_spawn = _maybe_force_spawn_with_npu
    return True
