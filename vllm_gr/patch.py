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
from typing import TYPE_CHECKING

from vllm_gr.v1.attention import selector_patch
from vllm_gr.v1.worker import sliced_logits_patch

if TYPE_CHECKING:
    from vllm_gr.platforms.interface import GRBackendProvider

logger = logging.getLogger("vllm_gr.patch")

# Semantic ownership, independent of which process orchestrates an installer.
# Built-in providers declare their hardware-owned IDs in ``patch_ids``.
# These independent architectural assertions are checked against the authoritative
# docs/design/patch_inventory.yaml; they do not drive installation. See the ownership
# rationale in docs/design/runtime_patch_inventory.md.
CORE_PATCH_IDS = (
    "GR-PATCH-000",
    "GR-PATCH-001",
    "GR-PATCH-003",
    "GR-PATCH-004",
    "GR-PATCH-005",
    "GR-PATCH-007",
    "GR-PATCH-008",
    "GR-PATCH-009",
    "GR-PATCH-010",
    "GR-PATCH-013",
    "GR-PATCH-015",
    "GR-PATCH-022",
)


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


def _module_patch(name: str, module_name: str, installer: str, validator: str) -> BeamPatch:
    """Resolve patch callables when the process reaches this step."""

    def apply() -> None:
        getattr(importlib.import_module(module_name), installer)()

    def verify() -> str | None:
        validate: Callable[[], str | None] = getattr(
            importlib.import_module(module_name), validator
        )
        return validate()

    return BeamPatch(name=name, apply=apply, verify=verify)


def install_engine_session_hooks() -> None:
    """Install session transport for the frontend, in-process Core, and spawned Core."""
    from vllm_gr.v1.engine.core_client_patch import apply_gr_session_patches
    from vllm_gr.v1.engine.engine_core_patch import patch_engine_core_process_entrypoint

    apply_gr_session_patches()
    patch_engine_core_process_entrypoint()


def _validate_engine_session_hooks() -> str | None:
    """Verify every target mutated by the grouped Beam engine installer."""

    from vllm_gr.v1.core.sched.scheduler_patch import validate_scheduler_patch
    from vllm_gr.v1.engine.core_client_patch import (
        validate_async_llm_patch,
        validate_engine_clients_patch,
    )
    from vllm_gr.v1.engine.engine_core_patch import (
        validate_engine_core_entrypoint_patch,
        validate_engine_core_request_patch,
    )
    from vllm_gr.v1.engine.wire import validate_request_types_patch
    from vllm_gr.v1.worker.model_runner_patch import validate_model_runner_patch

    failures = tuple(
        detail
        for detail in (
            validate_request_types_patch(),
            validate_engine_clients_patch(),
            validate_async_llm_patch(),
            validate_engine_core_entrypoint_patch(),
            validate_engine_core_request_patch(),
            validate_scheduler_patch(),
            validate_model_runner_patch(),
        )
        if detail is not None
    )
    return "; ".join(failures) or None


def _required_beam_patches() -> tuple[BeamPatch, ...]:
    """Return the required frontend patch plan."""

    return (
        _module_patch(
            "flat_logprobs",
            "vllm_gr.v1.engine.logprobs_patch",
            "patch_flat_logprobs",
            "validate_flat_logprobs_patch",
        ),
        _module_patch(
            "openai_beam_search",
            "vllm_gr.entrypoints.openai.beam_search_patch",
            "patch_beam_search",
            "validate_openai_beam_search_patch",
        ),
        _module_patch(
            "chat_sampling_conversion",
            "vllm_gr.entrypoints.openai.beam_search_patch",
            "patch_sampling",
            "validate_sampling_patch",
        ),
        _module_patch(
            "engine_args",
            "vllm_gr.engine.arg_utils",
            "patch_add_cli_args",
            "validate_engine_args_patch",
        ),
        BeamPatch(
            name="batch_and_fork",
            apply=install_engine_session_hooks,
            verify=_validate_engine_session_hooks,
        ),
    )


def _engine_core_beam_patches() -> tuple[BeamPatch, ...]:
    """Return only responsibilities required inside an EngineCore process."""

    return (
        _module_patch(
            "engine_core_requests",
            "vllm_gr.v1.engine.engine_core_patch",
            "apply_engine_core_request_patches",
            "validate_engine_core_request_group",
        ),
        _module_patch(
            "kv_hash_memo",
            "vllm_gr.v1.core.kv_cache_utils",
            "apply_kv_hash_memo_patch",
            "validate_kv_hash_memo_patch",
        ),
        _module_patch(
            "scheduler",
            "vllm_gr.v1.core.sched.scheduler_patch",
            "apply_scheduler_patch",
            "validate_scheduler_patch",
        ),
        _module_patch(
            "model_runner",
            "vllm_gr.v1.worker.model_runner_patch",
            "apply_worker_patches",
            "validate_model_runner_patch",
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


def _apply_gr_beam_patches() -> int:
    """Apply Beam patches through the single production initialization path.

    These patches are independent from the GR core type replacement patches
    and can be safely applied alongside them.
    """
    patch_count = initialize_required_beam_patches().applied_count

    if sliced_logits_patch.apply_sliced_logits_patch():
        patch_count += 1

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def _apply_patches(force: bool = False, *, provider: "GRBackendProvider | None" = None) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    if provider is None:
        from vllm_gr.platforms.loader import activate_current_backend

        # Fresh workers can call this before provider activation.
        provider = activate_current_backend()
    # CUSTOM selection is shared by CUDA and Ascend. Install it before a
    # provider imports its native worker; it must not become an NPU-only hook.
    attention_count = selector_patch.apply_custom_attention_backend_patch(force=force)
    try:
        # The selected provider installs its early hooks before the Beam
        # plan can discover ModelRunner classes or create worker processes.
        backend_count: int = provider.apply_runtime_patches(force=force)
    except Exception as exc:
        raise RuntimeError(
            f"Runtime patches for backend {provider.backend_id!r} failed; "
            "initialization may be partial, restart the process"
        ) from exc
    beam_count = _apply_gr_beam_patches()
    return attention_count + backend_count + beam_count


def re_apply_patches(*, provider: "GRBackendProvider | None" = None) -> int:
    """Recheck live runtime targets and apply the required patches."""
    return _apply_patches(force=True, provider=provider)
