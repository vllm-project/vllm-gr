# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime patch orchestration for vllm-gr.

Patch implementations live with their owning runtime components. This module
defines installation plans, validation, execution order, and result reporting.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from vllm_gr import lmcache_patch
from vllm_gr.ops import sliced_logits_patch
from vllm_gr.v1 import ascend_compat

logger = logging.getLogger("vllm_gr.patch")


class BeamPatchStatus(str, Enum):
    """Observable result of one required Beam patch installer."""

    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    FAILED = "failed"


@dataclass(frozen=True)
class BeamPatch:
    """One concrete Beam responsibility installed during runtime startup."""

    name: str
    apply: Callable[[], object]
    verify: Callable[[], str | None]


@dataclass(frozen=True)
class BeamPatchResult:
    name: str
    status: BeamPatchStatus
    detail: str | None = None


@dataclass(frozen=True)
class BeamPatchReport:
    results: tuple[BeamPatchResult, ...]

    @property
    def applied_count(self) -> int:
        return sum(result.status is BeamPatchStatus.APPLIED for result in self.results)

    def statuses(self) -> dict[str, str]:
        return {result.name: result.status.value for result in self.results}


class BeamPatchInitializationError(RuntimeError):
    def __init__(self, report: BeamPatchReport):
        self.report = report
        failure = report.results[-1]
        super().__init__(
            f"required Beam patch {failure.name!r} failed: {failure.detail or 'unknown error'}"
        )


_LAST_BEAM_PATCH_REPORT: BeamPatchReport | None = None


def _call_patch(module_name: str, function_name: str) -> None:
    module = importlib.import_module(module_name)
    getattr(module, function_name)()


def _verified(condition: bool, detail: str) -> str | None:
    return None if condition else detail


def _validate_flat_logprobs_patch() -> str | None:
    from vllm.logprobs import FlatLogprobs

    from vllm_gr.logprobs import append_fast

    return _verified(
        FlatLogprobs.append_fast is append_fast,
        "FlatLogprobs.append_fast does not reference the GR implementation",
    )


def _validate_openai_beam_search_patch() -> str | None:
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    from vllm_gr.entrypoints.openai.serving_engine import beam_search

    return _verified(
        OpenAIServing.beam_search is beam_search,
        "OpenAIServing.beam_search does not reference the GR implementation",
    )


def _validate_chat_sampling_patch() -> str | None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    return _verified(
        ChatCompletionRequest.to_beam_search_params is to_beam_search_params,
        "ChatCompletionRequest.to_beam_search_params is not installed",
    )


def _validate_engine_args_patch() -> str | None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine.arg_utils_gr import (
        _init_engine_args,
        add_cli_args,
        create_engine_config,
        from_cli_args,
    )

    return _verified(
        EngineArgs.add_cli_args is add_cli_args
        and EngineArgs.create_engine_config is create_engine_config
        and EngineArgs.from_cli_args.__func__ is from_cli_args
        and EngineArgs.__init__ is _init_engine_args,
        "EngineArgs GR integration is incomplete",
    )


def _validate_serving_models_patch() -> str | None:
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

    from vllm_gr.entrypoints.openai.serving_models import init_gr

    return _verified(
        OpenAIServingModels.__init__ is init_gr,
        "OpenAIServingModels.__init__ does not reference the GR implementation",
    )


def _validate_request_types_patch() -> str | None:
    from vllm.v1.engine import EngineCoreRequestType

    valid = (
        getattr(getattr(EngineCoreRequestType, "ADD_BATCH", None), "value", None) == b"\x06"
        and getattr(getattr(EngineCoreRequestType, "BEAM_REQUEST_STEP_UPDATE", None), "value", None)
        == b"\x07"
    )
    return _verified(valid, "GR EngineCore request types are missing or use wrong wire values")


def _validate_engine_clients_patch() -> str | None:
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    from vllm_gr.v1.engine.core_client import (
        add_requests_async,
        beam_request_step_update_async,
    )
    from vllm_gr.v1.engine.core_client_patch import (
        inproc_add_requests,
        inproc_beam_request_step_update,
        sync_mp_add_requests,
        sync_mp_beam_request_step_update,
    )

    valid = (
        AsyncMPClient.add_requests_async is add_requests_async
        and AsyncMPClient.beam_request_step_update_async is beam_request_step_update_async
        and SyncMPClient.add_requests is sync_mp_add_requests
        and SyncMPClient.beam_request_step_update is sync_mp_beam_request_step_update
        and InprocClient.add_requests is inproc_add_requests
        and InprocClient.beam_request_step_update is inproc_beam_request_step_update
        and getattr(InprocClient.get_output, "_vllm_gr_inproc_self_advance", False)
    )
    return _verified(valid, "EngineCore client methods are not fully installed")


def _validate_async_llm_patch() -> str | None:
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_gr.v1.engine.async_llm import (
        _add_requests_batch_fn,
        beam_request_step_update_fn,
        prepare_request_fn,
        register_beam_output_fn,
    )

    valid = (
        AsyncLLM.prepare_request is prepare_request_fn
        and AsyncLLM._add_requests_batch is _add_requests_batch_fn
        and AsyncLLM.register_beam_output is register_beam_output_fn
        and AsyncLLM.beam_request_step_update is beam_request_step_update_fn
    )
    return _verified(valid, "AsyncLLM grouped-beam methods are not fully installed")


def _validate_engine_core_entrypoint_patch() -> str | None:
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    return _verified(
        EngineCoreProc.run_engine_core is run_engine_core,
        "EngineCore process entrypoint wrapper is not installed",
    )


def _validate_engine_core_request_patch() -> str | None:
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _apply_beam_step,
        _beam_self_advance,
        _cache_beam_request,
        _handle_beam_request_step_update,
        process_input_sockets,
    )
    from vllm_gr.v1.engine.engine_core_patch import (
        _BEAM_STATE_INIT_MARKER,
        _CLIENT_REQUEST_PATCH_MARKER,
        _ENGINE_STEP_PATCH_MARKER,
        _PIS_PATCH_MARKER,
    )

    valid = (
        getattr(EngineCore.__init__, _BEAM_STATE_INIT_MARKER, False)
        and getattr(EngineCoreProc.__init__, _BEAM_STATE_INIT_MARKER, False)
        and EngineCore._cache_beam_request is _cache_beam_request
        and EngineCoreProc._cache_beam_request is _cache_beam_request
        and EngineCore._handle_beam_request_step_update is _handle_beam_request_step_update
        and EngineCoreProc._handle_beam_request_step_update is _handle_beam_request_step_update
        and EngineCore._apply_beam_step is _apply_beam_step
        and EngineCoreProc._apply_beam_step is _apply_beam_step
        and EngineCore._beam_self_advance is _beam_self_advance
        and EngineCoreProc._beam_self_advance is _beam_self_advance
        and getattr(EngineCoreProc._process_engine_step, _ENGINE_STEP_PATCH_MARKER, False)
        and getattr(EngineCoreProc._handle_client_request, _CLIENT_REQUEST_PATCH_MARKER, False)
        and EngineCoreProc.process_input_sockets is process_input_sockets
        and getattr(EngineCoreProc.process_input_sockets, _PIS_PATCH_MARKER, False)
    )
    return _verified(valid, "EngineCore beam request dispatch is not fully installed")


def _validate_scheduler_patch() -> str | None:
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_gr.v1.engine.engine_core_patch import _SCHEDULER_PATCH_MARKER

    return _verified(
        bool(getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False)),
        "Scheduler beam metadata propagation is not installed",
    )


def _validate_model_runner_patch() -> str | None:
    from vllm_gr.v1.worker.model_runner_patch import (
        discover_active_worker_runners,
        is_worker_runner_patched,
    )

    runners = discover_active_worker_runners()
    if not runners:
        return "No supported active-platform ModelRunner is available"

    failures = [
        f"{runner_name} beam execution hooks are not installed"
        for runner_name, runner_class in runners
        if not is_worker_runner_patched(runner_class)
    ]
    return "; ".join(failures) or None


def _validate_kv_hash_memo_patch() -> str | None:
    """Verify the bounded memo wrapper on hash_block_tokens."""
    from vllm.v1.core import kv_cache_utils

    if not getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return "hash_block_tokens memo wrapper not installed"
    return None


def _validate_attention_backend_registration() -> str | None:
    """Verify the Beam backend is registered for Backend.CUSTOM resolution."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    if not AttentionBackendEnum.CUSTOM.is_overridden():
        return "Beam attention backend is not registered for Backend.CUSTOM"
    try:
        backend_class = AttentionBackendEnum.CUSTOM.get_class()
    except Exception as exc:
        return (
            "Backend.CUSTOM registration does not resolve to the Beam "
            f"attention backend ({type(exc).__name__}: {exc})"
        )
    return _verified(
        backend_class is BeamAttentionBackend,
        "Backend.CUSTOM resolves to "
        f"{backend_class.__module__}.{backend_class.__qualname__} instead "
        "of the Beam attention backend",
    )


def _validate_engine_core_request_group() -> str | None:
    failures = tuple(
        detail
        for detail in (
            _validate_request_types_patch(),
            _validate_engine_core_request_patch(),
        )
        if detail is not None
    )
    return "; ".join(failures) or None


def _validate_batch_and_fork_patch() -> str | None:
    """Verify every target mutated by the grouped Beam engine installer."""

    failures = tuple(
        detail
        for detail in (
            _validate_request_types_patch(),
            _validate_engine_clients_patch(),
            _validate_async_llm_patch(),
            _validate_engine_core_entrypoint_patch(),
            _validate_engine_core_request_patch(),
            _validate_scheduler_patch(),
            _validate_model_runner_patch(),
        )
        if detail is not None
    )
    return "; ".join(failures) or None


def _required_beam_patches() -> tuple[BeamPatch, ...]:
    """Return the six concrete Beam responsibilities used in production."""

    def patch(
        name: str,
        module_name: str,
        function_name: str,
        verify: Callable[[], str | None],
    ) -> BeamPatch:
        return BeamPatch(
            name=name,
            apply=lambda: _call_patch(module_name, function_name),
            verify=verify,
        )

    return (
        patch(
            "flat_logprobs",
            "vllm_gr.logprobs_patch",
            "patch_flat_logprobs",
            _validate_flat_logprobs_patch,
        ),
        patch(
            "openai_beam_search",
            "vllm_gr.entrypoints.openai.beam_search_patch",
            "patch_beam_search",
            _validate_openai_beam_search_patch,
        ),
        patch(
            "chat_sampling_conversion",
            "vllm_gr.entrypoints.openai.beam_search_patch",
            "patch_sampling",
            _validate_chat_sampling_patch,
        ),
        patch(
            "engine_args",
            "vllm_gr.engine.arg_utils_gr",
            "patch_add_cli_args",
            _validate_engine_args_patch,
        ),
        patch(
            "serving_models",
            "vllm_gr.entrypoints.openai.serving_models",
            "patch_OpenAIServingModels_init",
            _validate_serving_models_patch,
        ),
        patch(
            "batch_and_fork",
            "vllm_gr.entrypoints.openai.beam_search_patch",
            "patch_batch_and_fork",
            _validate_batch_and_fork_patch,
        ),
    )


def _engine_core_beam_patches() -> tuple[BeamPatch, ...]:
    """Return only responsibilities required inside an EngineCore process."""

    def patch(
        name: str,
        module_name: str,
        function_name: str,
        verify: Callable[[], str | None],
    ) -> BeamPatch:
        return BeamPatch(
            name=name,
            apply=lambda: _call_patch(module_name, function_name),
            verify=verify,
        )

    return (
        patch(
            "attention_backend",
            "vllm_gr.v1.engine.engine_core_patch",
            "apply_attention_backend_registration",
            _validate_attention_backend_registration,
        ),
        patch(
            "engine_core_requests",
            "vllm_gr.v1.engine.engine_core_patch",
            "apply_engine_core_request_patches",
            _validate_engine_core_request_group,
        ),
        patch(
            "kv_hash_memo",
            "vllm_gr.v1.engine.engine_core_patch",
            "apply_kv_hash_memo_patch",
            _validate_kv_hash_memo_patch,
        ),
        patch(
            "scheduler",
            "vllm_gr.v1.engine.engine_core_patch",
            "apply_scheduler_patch",
            _validate_scheduler_patch,
        ),
        patch(
            "model_runner",
            "vllm_gr.v1.worker.model_runner_patch",
            "apply_worker_patches",
            _validate_model_runner_patch,
        ),
    )


def get_last_beam_patch_report() -> BeamPatchReport | None:
    """Return the most recent per-patch initialization report."""
    return _LAST_BEAM_PATCH_REPORT


def _initialize_beam_patch_plan(patches: tuple[BeamPatch, ...]) -> BeamPatchReport:
    """Install and verify one process-appropriate Beam patch plan."""

    global _LAST_BEAM_PATCH_REPORT

    results: list[BeamPatchResult] = []
    for patch in patches:
        try:
            try:
                already_applied = patch.verify() is None
            except Exception:
                already_applied = False

            if already_applied:
                result = BeamPatchResult(patch.name, BeamPatchStatus.ALREADY_APPLIED)
            else:
                patch.apply()
                verification_error = patch.verify()
                if verification_error is not None:
                    raise RuntimeError(verification_error)
                result = BeamPatchResult(patch.name, BeamPatchStatus.APPLIED)
        except Exception as exc:
            result = BeamPatchResult(
                patch.name,
                BeamPatchStatus.FAILED,
                f"{type(exc).__name__}: {exc}",
            )
            results.append(result)
            _LAST_BEAM_PATCH_REPORT = BeamPatchReport(tuple(results))
            logger.exception("Required Beam patch %s failed", patch.name)
            raise BeamPatchInitializationError(_LAST_BEAM_PATCH_REPORT) from exc

        results.append(result)
        logger.info("Beam patch %s status=%s", patch.name, result.status.value)

    # Applying the model runner patch imports vllm-ascend, whose attention
    # module rebinds Backend.CUSTOM to vllm-ascend's own backend after the
    # attention registration patch was already verified. Re-assert that
    # registration once every patch has been applied so the final process
    # state keeps the verified contract.
    attention_patch = next((patch for patch in patches if patch.name == "attention_backend"), None)
    if attention_patch is not None:
        try:
            attention_patch.apply()
            verification_error = _validate_attention_backend_registration()
            if verification_error is not None:
                raise RuntimeError(verification_error)
        except Exception as exc:
            result = BeamPatchResult(
                attention_patch.name,
                BeamPatchStatus.FAILED,
                f"{type(exc).__name__}: {exc}",
            )
            results.append(result)
            _LAST_BEAM_PATCH_REPORT = BeamPatchReport(tuple(results))
            logger.exception("Required Beam patch %s regressed", attention_patch.name)
            raise BeamPatchInitializationError(_LAST_BEAM_PATCH_REPORT) from exc

    _LAST_BEAM_PATCH_REPORT = BeamPatchReport(tuple(results))
    return _LAST_BEAM_PATCH_REPORT


def initialize_required_beam_patches() -> BeamPatchReport:
    """Install the complete frontend/plugin Beam plan."""
    return _initialize_beam_patch_plan(_required_beam_patches())


def initialize_engine_core_beam_patches() -> BeamPatchReport:
    """Install only capabilities required by an EngineCore process."""
    return _initialize_beam_patch_plan(_engine_core_beam_patches())


def _apply_gr_beam_patches(force: bool = False) -> int:
    """Apply Beam patches through the single production initialization path.

    These patches are independent from the GR core type replacement patches
    and can be safely applied alongside them.
    """
    del force  # Repeated initialization probes every target before applying it.
    patch_count = initialize_required_beam_patches().applied_count

    if sliced_logits_patch.apply_sliced_logits_patch():
        patch_count += 1
    try:
        if lmcache_patch.is_lmcache_adapter_requested() and lmcache_patch.apply_lmcache_adapter():
            patch_count += 1
    except Exception as exc:
        logger.debug("vllm_gr: Skipping LMCache adapter patch: %s", exc)

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    ascend_attn_count = ascend_compat.apply_ascend_attention_backend_patch(force=force)
    # Import/patch NPUWorker before the beam patch tries to access
    # NPUModelRunner. vllm-ascend's worker module deliberately initializes
    # vllm_ascend.ops first, avoiding an import-order cycle through device_op.
    npu_worker_count = ascend_compat.patch_npu_worker_init_device()
    beam_count = _apply_gr_beam_patches(force=force)
    return ascend_attn_count + npu_worker_count + beam_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules."""
    return _apply_patches(force=True)


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
