# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""EngineCore Beam admission checks and interfaces."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Protocol

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.config import GRConfig
from vllm_gr.config.execution import (
    validate_gr_execution_config,
    validate_gr_execution_runtime,
)
from vllm_gr.config.models import (
    CONSTRAINT_TABLE_MAX_DECODE_STEPS,
    configured_candidate_width,
    runtime_constraint_table,
)
from vllm_gr.platforms.ascend.admission import (
    validate_decode_token_budget,
    validate_npu_unconstrained_runner,
)
from vllm_gr.platforms.errors import GRDeviceRequestError
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
)
from vllm_gr.v1.beam.dispatch import (
    GRDeviceStateRef,
    GRDispatchSession,
    GRExecutionCapacity,
    GRRetireMetadata,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
    validate_dispatch,
    validate_dispatch_request,
    validate_dispatch_session,
)
from vllm_gr.v1.core.sched.beam_session import (
    GRAsyncSchedulerInterface,
    GRResourceStatus,
    GRSessionState,
    GRSessionStatus,
    GRStageCompletion,
)


def validate_gr_device_config(config: Any, request: BeamBatchRequest | None = None) -> None:
    """Reject unavailable adapters without importing a device runtime."""
    from vllm_gr.config import GRConfig

    device = getattr(getattr(config, "device_config", None), "device", None)
    if device is not None and str(device).split(":")[0] not in ("cuda", "npu"):
        raise ValueError(
            "Asynchronous scheduling requires a CUDA ModelRunner or NPU ModelRunner V1"
        )
    gr_config = GRConfig.from_vllm_config(config)
    if gr_config.beam_sid_stage_vocab and (device is None or str(device).split(":")[0] != "npu"):
        raise ValueError("beam.sid_stage_vocab requires NPU ModelRunner V1")
    if device is not None and str(device).split(":")[0] == "npu":
        from vllm_gr.platforms.ascend.admission import validate_npu_device_config

        validate_npu_device_config(config, gr_config, request)
    if (
        request is not None
        and device is not None
        and str(device).split(":")[0] not in ("cuda", "npu")
        and (
            request.beam_params.dynamic_beam_widths is not None
            or request.output_options.num_return_sequences != request.beam_params.beam_width
        )
    ):
        raise ValueError("dynamic beam widths and result widths require CUDA or NPU beam_search")


def _reject_candidate_width_below_dynamic_schedule(
    gr_config: Any,
    params: BeamBatchParams,
) -> None:
    """A configured K must cover every dynamic stage width.

    An explicit request-level ``per_parent_beam_widths`` schedule takes
    precedence over the configured scalar and is validated on its own.

    K is forwarded to every stage, so each stage samples K candidates per
    parent and keeps its own ``output_width``: ``K=4`` with ``(2, 2, 2)`` runs
    TopK=4 and retains 2.  The one unsupported combination is a stage that
    retains more beams than its parents supply candidates.  This reads the
    *configured* K so the rejection is a request-level error at admission
    instead of a contract failure deep in the worker.  Leaving K unset keeps
    dynamic Beam widths working exactly as before.
    """

    if params.per_parent_beam_widths is not None:
        return
    widths = params.dynamic_beam_widths
    if widths is None:
        return
    configured = configured_candidate_width(gr_config)
    if configured is None:
        return
    for index, width in enumerate(widths):
        if width > configured:
            raise ValueError(
                f"vllm-gr constraint_table.candidate_width={configured} is below "
                f"dynamic_beam_widths[{index}]={width}: each parent must supply at "
                "least as many candidates as the stage retains"
            )


def validate_gr_generation_budget(gr_config: GRConfig, max_tokens: int) -> None:
    """Fix constrained SID length to table depth; otherwise bound it by capacity."""
    table = runtime_constraint_table(gr_config)
    if table is not None and table.enabled and max_tokens != CONSTRAINT_TABLE_MAX_DECODE_STEPS:
        raise ValueError(
            f"effective generation length {max_tokens} must equal {table.format} depth "
            f"({CONSTRAINT_TABLE_MAX_DECODE_STEPS})"
        )
    if max_tokens > gr_config.beam_max_decode_steps:
        raise ValueError(
            f"Decode steps {max_tokens} exceed runtime limit {gr_config.beam_max_decode_steps}"
        )


def validate_gr_request(
    request: BeamBatchRequest, *, config: Any = None, trie_count: int | None = None
) -> None:
    """Validate shared V1 semantics before native ADD or Worker allocation."""
    validate_dispatch_request(request, trie_count=trie_count)
    if config is not None:
        gr_config = GRConfig.from_vllm_config(config)
        validate_gr_generation_budget(gr_config, request.beam_params.max_tokens)
        _reject_candidate_width_below_dynamic_schedule(gr_config, request.beam_params)
        validate_decode_token_budget(request, config)


class GRModelExecutorInterface(Protocol):
    """Native vLLM ModelExecutor call extended by the A4 Worker patch."""

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        *,
        non_block: bool,
    ) -> Future[ModelRunnerOutput] | ModelRunnerOutput:
        """Use vLLM's entry; non-blocking execution returns a Future."""
        ...


class GREngineCoreInterface(Protocol):
    """GR hooks installed on vLLM EngineCore by monkey patching."""

    def submit_gr_session(self, request: BeamBatchRequest) -> str:
        """Enqueue one frontend submission and return its session ID.

        Both online and offline adapters call this same entry exactly once.
        """
        ...

    def step_with_batch_queue(self) -> Any:
        """Patched native vLLM asynchronous EngineCore iteration.

        A3 adds GR session transitions around the existing schedule,
        ``execute_model(non_block=True)``, queue, and update-from-output flow.
        Consecutive stages of the same session may coexist in the native queue
        when ``can_issue_gr_stage`` permits look-ahead. The upstream return
        contract remains unchanged.
        """
        ...

    def finish_gr_session(self, session_id: str) -> BeamBatchResult:
        """Return the retained terminal result after Scheduler/device drain."""
        ...

    def abort_gr_session(self, session_id: str) -> None: ...


__all__ = [
    "GRAsyncSchedulerInterface",
    "GRDeviceRequestError",
    "GRDeviceStateRef",
    "GREngineCoreInterface",
    "GRExecutionCapacity",
    "GRModelExecutorInterface",
    "GRResourceStatus",
    "GRRetireMetadata",
    "GRRetireResult",
    "GRSessionState",
    "GRSessionStatus",
    "GRStageCompletion",
    "GRStageKind",
    "GRStageMetadata",
    "GRWorkerResult",
    "GRDispatchSession",
    "validate_dispatch",
    "validate_dispatch_session",
    "validate_gr_device_config",
    "validate_gr_execution_config",
    "validate_gr_execution_runtime",
    "validate_gr_generation_budget",
    "validate_gr_request",
    "validate_npu_unconstrained_runner",
]
