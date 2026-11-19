# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Runtime patch orchestration for vllm-gr.

Patch implementations live with their owning runtime components. This module
defines installation plans, validation, execution order, and result reporting.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from vllm_gr import pangu_compat
from vllm_gr.ops import sliced_logits_patch
from vllm_gr.platforms.patches import CORE_PATCH_IDS, SelectedPatchClaims
from vllm_gr.v1.attention import selector_patch

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
        from vllm_gr.platforms.errors import BackendNotActiveError
        from vllm_gr.platforms.registry import get_active_backend, resolve_backend_id

        self.report = report
        failure = report.results[-1]
        try:
            origin = resolve_backend_id(get_active_backend().backend_id).origin
        except BackendNotActiveError:
            origin = "no active provider"
        super().__init__(
            f"required Beam patch {failure.name!r} failed: {failure.detail or 'unknown error'} "
            f"({origin})"
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


def _validate_beam_search_dispatch_patch() -> str | None:
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing

    from vllm_gr.entrypoints.openai.beam_search_patch import beam_search_dispatch

    return _verified(
        GenerateBaseServing.beam_search is beam_search_dispatch,
        "GenerateBaseServing.beam_search does not reference the GR dispatcher",
    )


def _validate_admission_rejection_patch() -> str | None:
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing

    from vllm_gr.entrypoints.openai.beam_search_patch import (
        _ADMISSION_REJECTION_PATCH_MARKER,
    )

    return _verified(
        getattr(
            getattr(GenerateBaseServing, "_raise_if_error", None),
            _ADMISSION_REJECTION_PATCH_MARKER,
            False,
        ),
        "GenerateBaseServing._raise_if_error does not reference the GR admission-rejection wrapper",
    )


def _validate_client_request_id_patch() -> str | None:
    """Verify the two sites that pair a caller's name with the engine's."""

    from vllm.entrypoints.serve.engine.serving import BaseServing
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.entrypoints.openai.beam_search_patch import (
        _CLIENT_REQUEST_ID_PATCH_MARKER,
        _SUBMIT_REQUEST_ID_PATCH_MARKER,
    )

    return _verified(
        getattr(
            getattr(BaseServing, "_base_request_id", None),
            _CLIENT_REQUEST_ID_PATCH_MARKER,
            False,
        ),
        "BaseServing._base_request_id does not reference the GR client-request-id wrapper",
    ) or _verified(
        getattr(
            getattr(OutputProcessor, "add_request", None),
            _SUBMIT_REQUEST_ID_PATCH_MARKER,
            False,
        ),
        "OutputProcessor.add_request does not reference the GR client-request-id wrapper",
    )


def _validate_openai_beam_search_patch() -> str | None:
    """Verify every target mutated by the grouped OpenAI Beam installer."""

    failures = tuple(
        detail
        for detail in (
            _validate_beam_search_dispatch_patch(),
            _validate_admission_rejection_patch(),
            _validate_client_request_id_patch(),
            _validate_completion_batch_patch(),
        )
        if detail is not None
    )
    return "; ".join(failures) or None


def _validate_completion_batch_patch() -> str | None:
    """Verify the completions route still owns the batch seam.

    The installer and its verifier are one responsibility and live together:
    the list of upstream helpers the batch path borrows is long, and keeping a
    second copy of it here is how the two drift apart.
    """

    from vllm_gr.entrypoints.openai.completion_batch_patch import (
        _validate_completion_batch_patch as validate,
    )

    return validate()


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


def _validate_request_types_patch() -> str | None:
    from vllm.v1.engine import EngineCoreRequestType

    valid = (
        getattr(getattr(EngineCoreRequestType, "SUBMIT_GR_SESSION", None), "value", None) == b"\x0a"
    )
    return _verified(valid, "GR EngineCore request types are missing or use wrong wire values")


def _validate_engine_clients_patch() -> str | None:
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    from vllm_gr.v1.engine.core_client import submit_gr_session_async
    from vllm_gr.v1.engine.core_client_patch import (
        inproc_abort_requests,
        inproc_add_request,
        inproc_get_output,
        inproc_shutdown,
        inproc_submit_gr_session,
        sync_mp_submit_gr_session,
    )

    valid = (
        getattr(AsyncMPClient, "submit_gr_session_async", None) is submit_gr_session_async
        and getattr(SyncMPClient, "submit_gr_session", None) is sync_mp_submit_gr_session
        and getattr(InprocClient, "submit_gr_session", None) is inproc_submit_gr_session
        and getattr(InprocClient, "get_output", None) is inproc_get_output
        and getattr(InprocClient, "add_request", None) is inproc_add_request
        and getattr(InprocClient, "abort_requests", None) is inproc_abort_requests
        and getattr(InprocClient, "shutdown", None) is inproc_shutdown
    )
    return _verified(valid, "EngineCore client methods are not fully installed")


def _validate_async_llm_patch() -> str | None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine import gr_session_client
    from vllm_gr.v1.engine.async_llm import _PROCESS_OUTPUTS_PATCH_MARKER
    from vllm_gr.v1.engine.core_client_patch import (
        async_llm_add_request,
        llm_engine_add_request,
    )

    valid = (
        bool(getattr(OutputProcessor.process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False))
        and getattr(LLMEngine, "add_request", None) is llm_engine_add_request
        and getattr(AsyncLLM, "_add_request", None) is async_llm_add_request
    )
    for cls, suffix in ((LLMEngine, "sync"), (AsyncLLM, "async")):
        for operation in ("submit", "wait", "abort", "release"):
            name = f"{operation}_gr_session"
            valid = valid and getattr(cls, name, None) is getattr(
                gr_session_client, f"{name}_{suffix}"
            )
    return _verified(valid, "GR frontend lifecycle methods are not fully installed")


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
        _handle_submit_gr_session,
        is_gr_session_retired,
        process_input_sockets,
    )
    from vllm_gr.v1.engine.engine_core_patch import (
        _CLIENT_REQUEST_PATCH_MARKER,
        _ENGINE_STEP_PATCH_MARKER,
        _PIS_PATCH_MARKER,
    )

    valid = (
        getattr(EngineCore, "_handle_submit_gr_session", None) is _handle_submit_gr_session
        and getattr(EngineCore.__init__, "_gr_trie_capacity_patch", False)
        and getattr(EngineCoreProc, "_handle_submit_gr_session", None) is _handle_submit_gr_session
        and getattr(EngineCore, "is_gr_session_retired", None) is is_gr_session_retired
        and getattr(EngineCoreProc._process_engine_step, _ENGINE_STEP_PATCH_MARKER, False)
        and getattr(EngineCoreProc._handle_client_request, _CLIENT_REQUEST_PATCH_MARKER, False)
        and EngineCoreProc.process_input_sockets is process_input_sockets
        and getattr(EngineCoreProc.process_input_sockets, _PIS_PATCH_MARKER, False)
    )
    return _verified(valid, "EngineCore GR session dispatch is not fully installed")


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
        is_active_worker_runner_patched,
    )

    runners = discover_active_worker_runners()
    if not runners:
        return "No supported active-platform ModelRunner is available"

    failures = [
        f"{runner_name} beam execution hooks are not installed"
        for runner_name, runner_class in runners
        if not is_active_worker_runner_patched(runner_class)
    ]
    return "; ".join(failures) or None


def _validate_kv_hash_memo_patch() -> str | None:
    """Verify the bounded memo wrapper on hash_block_tokens."""
    from vllm.v1.core import kv_cache_utils

    if not getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return "hash_block_tokens memo wrapper not installed"
    return None


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

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    from vllm_gr.platforms.loader import activate_current_backend

    # re_apply_patches is also called directly by fresh worker processes.
    provider = activate_current_backend()
    claims = SelectedPatchClaims.from_provider(provider, CORE_PATCH_IDS)
    # ModelConfig construction happens before model loading, so preserve the
    # official Pangu tied-weight layout at the earliest runtime boundary.
    pangu_count = int(pangu_compat.apply_pangu_tied_embeddings_patch())
    # CUSTOM selection is shared by CUDA and Ascend. Install it before a
    # provider imports its native worker; it must not become an NPU-only hook.
    attention_count = selector_patch.apply_custom_attention_backend_patch(force=force)
    backend_count = claims.install(force=force)
    beam_count = _apply_gr_beam_patches(force=force)
    return pangu_count + attention_count + backend_count + beam_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules."""
    return _apply_patches(force=True)
