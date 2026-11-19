# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Active platform dispatcher and public facade for Prefill Graph Runners."""

from __future__ import annotations

from typing import Any

from vllm_gr.platforms.graph import graph_for, options_for, shared_prefill_runner_class
from vllm_gr.platforms.plan import read_backend_plan
from vllm_gr.v1.worker.prefill_graph_common import (
    DEFAULT_PREFILL_BUCKETS,
    ENV_ENABLE_PREFILL_GRAPH,
    ENV_PREFILL_GRAPH_BUCKET_STEP,
    ENV_PREFILL_GRAPH_KV_BOUND,
    ENV_PREFILL_GRAPH_MAX_BUCKET,
    NPU_DEFAULT_PREFILL_BUCKETS,
    PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
    PREFILL_GRAPH_MAX_BUCKET_DEFAULT,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    PrefillBucketBuffers,
    get_default_buckets,
    is_prefill_graph_enabled,
    is_prefill_phase_batch,
)

__all__ = [
    "DEFAULT_PREFILL_BUCKETS",
    "ENV_ENABLE_PREFILL_GRAPH",
    "ENV_PREFILL_GRAPH_BUCKET_STEP",
    "ENV_PREFILL_GRAPH_KV_BOUND",
    "ENV_PREFILL_GRAPH_MAX_BUCKET",
    "NPU_DEFAULT_PREFILL_BUCKETS",
    "PREFILL_GRAPH_BUCKET_STEP_DEFAULT",
    "PREFILL_GRAPH_MAX_BUCKET_DEFAULT",
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
    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    factory = getattr(provider, "native_graph_factory", None)
    if factory is None:
        raise RuntimeError(f"Backend {provider.backend_id} has no native Prefill graph factory")
    return shared_prefill_runner_class(factory)


def BeamPrefillGraphRunner(
    model_runner: Any,
    buckets: list[int] | None = None,
    **kwargs: Any,
) -> BasePrefillGraphRunner:
    """Factory function returning the active platform's prefill graph runner."""
    plan = read_backend_plan(getattr(model_runner, "vllm_config", None))
    if plan is not None and plan.prefill_graph_mode == "provider":
        raise RuntimeError(
            "Provider-managed Prefill capture does not use the shared runner factory"
        )
    runner_cls = shared_prefill_runner_class(graph_for(model_runner))
    instance: BasePrefillGraphRunner = runner_cls(
        model_runner=model_runner,
        buckets=buckets,
        **kwargs,
    )
    return instance


def get_prefill_graph_runner(runner: Any) -> BasePrefillGraphRunner:
    """Retrieves or lazily initializes the active prefill graph runner on a ModelRunner."""
    if getattr(runner, "_prefill_graph_runner", None) is None:
        runner._prefill_graph_runner = BeamPrefillGraphRunner(
            runner, buckets=get_default_buckets(options_for(runner))
        )
    return runner._prefill_graph_runner  # type: ignore[no-any-return]
