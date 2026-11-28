# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Active platform dispatcher and public facade for Prefill Graph Runners."""

from __future__ import annotations

from typing import Any

from vllm_gr.platforms import get_current_gr_platform
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
    """Return the V1 prefill graph implementation of the selected GR platform."""
    return get_current_gr_platform().get_prefill_graph_runner_class()


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
