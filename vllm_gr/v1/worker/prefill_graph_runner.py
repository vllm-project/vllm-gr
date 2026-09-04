# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active platform dispatcher and public facade for Prefill Graph Runners."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_gr.v1.worker.prefill_graph_common import (
    DEFAULT_PREFILL_BUCKETS,
    ENV_ENABLE_PREFILL_GRAPH,
    ENV_PREFILL_GRAPH_BUCKET_STEP,
    ENV_PREFILL_GRAPH_MAX_BUCKET,
    MAX_PAD_RATIO_DEFAULT,
    MAX_PAD_TOKENS_DEFAULT,
    NPU_DEFAULT_PREFILL_BUCKETS,
    PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
    PREFILL_GRAPH_MAX_BUCKET_DEFAULT,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    PrefillBucketBuffers,
    get_default_buckets,
    get_dynamic_thresholds,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

logger = init_logger(__name__)

__all__ = [
    "DEFAULT_PREFILL_BUCKETS",
    "ENV_ENABLE_PREFILL_GRAPH",
    "ENV_PREFILL_GRAPH_BUCKET_STEP",
    "ENV_PREFILL_GRAPH_MAX_BUCKET",
    "MAX_PAD_RATIO_DEFAULT",
    "MAX_PAD_TOKENS_DEFAULT",
    "NPU_DEFAULT_PREFILL_BUCKETS",
    "PREFILL_GRAPH_BUCKET_STEP_DEFAULT",
    "PREFILL_GRAPH_MAX_BUCKET_DEFAULT",
    "SUPPORTED_NUM_REQS",
    "BasePrefillGraphRunner",
    "BeamPrefillGraphRunner",
    "PrefillBucketBuffers",
    "get_active_prefill_graph_runner_class",
    "get_default_buckets",
    "get_dynamic_thresholds",
    "get_prefill_graph_runner",
    "is_prefill_graph_enabled",
    "is_prefill_phase_batch",
]


def get_active_prefill_graph_runner_class() -> type[BasePrefillGraphRunner]:
    """Returns the platform-specific prefill graph runner class."""
    if f"{current_platform.device_type}".lower() == "npu":
        from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

        return NPUPrefillGraphRunner
    else:
        from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

        return GPUPrefillGraphRunner


def BeamPrefillGraphRunner(
    model_runner: Any,
    buckets: list[int] | None = None,
    max_pad_ratio: float | None = None,
    max_pad_tokens: int | None = None,
    **kwargs: Any,
) -> BasePrefillGraphRunner:
    """Factory function returning the active platform's prefill graph runner."""
    runner_cls = get_active_prefill_graph_runner_class()
    instance: BasePrefillGraphRunner = runner_cls(
        model_runner=model_runner,
        buckets=buckets,
        max_pad_ratio=max_pad_ratio,
        max_pad_tokens=max_pad_tokens,
        **kwargs,
    )
    return instance


def get_prefill_graph_runner(runner: Any) -> BasePrefillGraphRunner:
    """Retrieves or lazily initializes the active prefill graph runner on a ModelRunner."""
    if not hasattr(runner, "_prefill_graph_runner"):
        runner._prefill_graph_runner = BeamPrefillGraphRunner(runner, buckets=get_default_buckets())
    return runner._prefill_graph_runner  # type: ignore[no-any-return]
