# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Active platform dispatcher and public facade for Prefill Graph Runners."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_gr.v1.worker.prefill_graph_common import (
    DEFAULT_PREFILL_BUCKETS,
    NPU_DEFAULT_PREFILL_BUCKETS,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    PrefillBucketBuffers,
    get_default_buckets,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_PREFILL_BUCKETS",
    "NPU_DEFAULT_PREFILL_BUCKETS",
    "SUPPORTED_NUM_REQS",
    "BasePrefillGraphRunner",
    "BeamPrefillGraphRunner",
    "PrefillBucketBuffers",
    "get_active_prefill_graph_runner_class",
    "get_default_buckets",
    "get_prefill_graph_runner",
    "is_prefill_graph_enabled",
    "is_prefill_phase_batch",
]


def get_active_prefill_graph_runner_class() -> type[BasePrefillGraphRunner]:
    """Returns the platform-specific prefill graph runner class."""
    if f"{current_platform.device_type}".lower() == "npu":
        from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

        return NPUPrefillGraphRunner
    else:
        from vllm_gr.v1.worker.gpu.v1.prefill_graph_runner import (
            GPUPrefillGraphRunner,
        )

        return GPUPrefillGraphRunner


def BeamPrefillGraphRunner(
    model_runner: Any,
    buckets: list[int] | None = None,
    **kwargs: Any,
) -> BasePrefillGraphRunner:
    """Factory function returning the active platform's prefill graph runner."""
    runner_cls = get_active_prefill_graph_runner_class()
    instance: BasePrefillGraphRunner = runner_cls(
        model_runner=model_runner,
        buckets=buckets,
        **kwargs,
    )
    return instance


def get_prefill_graph_runner(runner: Any) -> BasePrefillGraphRunner:
    """Retrieves or lazily initializes the active prefill graph runner on a ModelRunner."""
    if not hasattr(runner, "_prefill_graph_runner"):
        runner._prefill_graph_runner = BeamPrefillGraphRunner(runner)
    return runner._prefill_graph_runner  # type: ignore[no-any-return]
