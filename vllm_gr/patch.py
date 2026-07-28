# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Monkey-patch mechanism for vllm-gr.

After removing the GR adapter / recommend runtime, the only patches applied
here are the ones still used by the OneRec beam-search path and by the NPU
runtime:

- :func:`_apply_gr_beam_patches` — offline / OpenAI beam-search and
  catalog-constrained generation support.

The previous ``_apply_gr_core_patches`` that replaced vLLM core types
(``EngineCoreRequest`` / ``Request`` / ``NewRequestData``) with GR-extended
variants has been removed together with the recommend runtime.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from vllm_gr.config import GRAttentionBackend, GRConfig

logger = logging.getLogger("vllm_gr.patch")


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    """Read the validated GR backend marker used by the attention selector."""
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


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


def _is_npu_platform() -> bool:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


_LAST_BEAM_PATCH_REPORT: BeamPatchReport | None = None
_ASCEND_ATTN_BACKEND_PATCH_APPLIED = False
_ASCEND_ORIGINAL_CACHED_GET_ATTN = None
_SLICED_LOGITS_PATCHED = False


def _apply_ascend_attention_backend_patch(force: bool = False) -> int:
    """Preserve CUSTOM attention backend on all platforms.

    The vllm-ascend platform plugin ignores any user-specified attention
    backend (other than FLASH_ATTN) in its ``get_attn_backend_cls()``, so
    ``get_attn_backend`` would return an Ascend-specific backend even when the
    config explicitly requests ``AttentionBackendEnum.CUSTOM``.

    To avoid relying on an environment variable, this patch is applied
    **unconditionally** on every platform.  It is a no-op whenever
    ``backend != AttentionBackendEnum.CUSTOM``, so on GPU the CUDA platform's
    native handling is preserved.  When ``backend == CUSTOM`` the patch returns
    ``BeamAttentionBackend`` directly, bypassing any platform-specific
    ``get_attn_backend_cls``.
    """
    global _ASCEND_ATTN_BACKEND_PATCH_APPLIED
    global _ASCEND_ORIGINAL_CACHED_GET_ATTN

    if _ASCEND_ATTN_BACKEND_PATCH_APPLIED and not force:
        return 0

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return 0

    # Ensure BeamAttentionBackend is registered under CUSTOM.
    try:
        import vllm_gr.v1.attention.backends.beam_attn  # noqa: F401
    except Exception:
        return 0

    if not AttentionBackendEnum.CUSTOM.is_overridden():
        _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
        return 0

    patch_count = 0

    # --- Patch selector._cached_get_attn_backend -----------------------
    try:
        import vllm.v1.attention.selector as selector_mod
    except ImportError:
        return 0

    # Save the original on first call to avoid double-wrapping on re-apply.
    if _ASCEND_ORIGINAL_CACHED_GET_ATTN is None:
        _ASCEND_ORIGINAL_CACHED_GET_ATTN = selector_mod._cached_get_attn_backend

    original_cached_fn = _ASCEND_ORIGINAL_CACHED_GET_ATTN

    def _patched_cached_get_attn_backend(backend, attn_selector_config, num_heads=None):
        """Return BeamAttentionBackend when backend is CUSTOM."""
        if backend == AttentionBackendEnum.CUSTOM:
            from vllm_gr.v1.attention.backends.beam_attn import (
                BeamAttentionBackend,
            )

            # Replicate the KV cache layout adjustment from the original
            # selector so that backends requiring a specific layout still
            # get it applied.
            required_layout = BeamAttentionBackend.get_required_kv_cache_layout()
            if required_layout is not None:
                from vllm.v1.attention.backends.utils import (
                    set_kv_cache_layout,
                )

                set_kv_cache_layout(required_layout)

            return BeamAttentionBackend

        # On Ascend NPU the platform plugin resets attention_config.backend to
        # None inside VllmConfig.__post_init__.  The reset happens before the
        # config is serialized to child processes, so EngineCore and Worker
        # processes see backend=None even though the user requested CUSTOM.
        # We store a marker in additional_config (untouched by the platform)
        # and use it as a fallback here.
        if backend is None:
            if _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
                from vllm_gr.v1.attention.backends.beam_attn import (
                    BeamAttentionBackend,
                )

                required_layout = BeamAttentionBackend.get_required_kv_cache_layout()
                if required_layout is not None:
                    from vllm.v1.attention.backends.utils import (
                        set_kv_cache_layout,
                    )

                    set_kv_cache_layout(required_layout)

                return BeamAttentionBackend

        return original_cached_fn(backend, attn_selector_config, num_heads=num_heads)

    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    patch_count += 1

    _ASCEND_ATTN_BACKEND_PATCH_APPLIED = True
    return patch_count


def patch_sliced_logits() -> None:
    global _SLICED_LOGITS_PATCHED
    if _SLICED_LOGITS_PATCHED:
        return
    try:
        from vllm.config import get_current_vllm_config_or_none
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        from vllm_gr.ops.sliced_logits import (
            compute_sliced_logits,
            is_vocab_slice_enabled,
            set_vocab_slice_config,
        )

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and getattr(vllm_config, "additional_config", None)
            and isinstance(vllm_config.additional_config, dict)
        ):
            offset = vllm_config.additional_config.get("vllm_gr_vocab_offset")
            slice_size = vllm_config.additional_config.get("vllm_gr_vocab_slice_size")
            if offset is not None and slice_size is not None:
                set_vocab_slice_config(offset, slice_size)

        orig_init = LogitsProcessor.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not is_vocab_slice_enabled():
                vllm_cfg = get_current_vllm_config_or_none()
                if (
                    vllm_cfg is not None
                    and getattr(vllm_cfg, "additional_config", None)
                    and isinstance(vllm_cfg.additional_config, dict)
                ):
                    offset = vllm_cfg.additional_config.get("vllm_gr_vocab_offset")
                    slice_size = vllm_cfg.additional_config.get("vllm_gr_vocab_slice_size")
                    if offset is not None and slice_size is not None:
                        set_vocab_slice_config(offset, slice_size)

        LogitsProcessor.__init__ = patched_init

        orig_get_logits = LogitsProcessor._get_logits

        def patched_get_logits(self, hidden_states, lm_head, embedding_bias):
            if is_vocab_slice_enabled():
                return compute_sliced_logits(
                    hidden_states,
                    lm_head,
                    embedding_bias,
                    logits_processor=self,
                    orig_get_logits=orig_get_logits,
                )
            return orig_get_logits(self, hidden_states, lm_head, embedding_bias)

        LogitsProcessor._get_logits = patched_get_logits
        _SLICED_LOGITS_PATCHED = True
        logger.info("vllm_gr: Patched LogitsProcessor._get_logits for sliced vocabulary projection")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_sliced_logits: %s", exc)


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
        add_cli_args,
        create_engine_config,
        from_cli_args,
    )

    return _verified(
        EngineArgs.add_cli_args is add_cli_args
        and EngineArgs.create_engine_config is create_engine_config
        and EngineArgs.from_cli_args.__func__ is from_cli_args,
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
        _cache_beam_request,
        _handle_beam_request_step_update,
        process_input_sockets,
    )
    from vllm_gr.v1.engine.engine_core_patch import _BEAM_STATE_INIT_MARKER

    valid = (
        getattr(EngineCore.__init__, _BEAM_STATE_INIT_MARKER, False)
        and getattr(EngineCoreProc.__init__, _BEAM_STATE_INIT_MARKER, False)
        and EngineCore._cache_beam_request is _cache_beam_request
        and EngineCoreProc._cache_beam_request is _cache_beam_request
        and EngineCore._handle_beam_request_step_update is _handle_beam_request_step_update
        and EngineCoreProc._handle_beam_request_step_update is _handle_beam_request_step_update
        and EngineCoreProc.process_input_sockets is process_input_sockets
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
    from vllm_gr.v1.engine.engine_core_patch import (
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
        function_name: str,
        verify: Callable[[], str | None],
    ) -> BeamPatch:
        return BeamPatch(
            name=name,
            apply=lambda: _call_patch(
                "vllm_gr.v1.engine.engine_core_patch",
                function_name,
            ),
            verify=verify,
        )

    return (
        patch(
            "engine_core_requests",
            "apply_engine_core_request_patches",
            _validate_engine_core_request_group,
        ),
        patch(
            "scheduler",
            "apply_scheduler_patch",
            _validate_scheduler_patch,
        ),
        patch(
            "model_runner",
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

    try:
        patch_sliced_logits()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_sliced_logits")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_sliced_logits: %s", exc)

    # Optional: LMCache adapter patch
    try:
        import importlib.util
        from pathlib import Path

        adapter_path = (
            Path(__file__).resolve().parent
            / "3rd_party"
            / "lmcache"
            / "integration"
            / "vllm"
            / "vllm_v1_adapter.py"
        )
        if adapter_path.exists():
            import os
            import sys

            should_patch = False
            if os.getenv("VLLM_GR_LMCACHE_PATCH", "").lower() in (
                "1",
                "true",
                "yes",
                "y",
            ):
                should_patch = True
            elif any("LMCacheConnectorV1" in arg for arg in sys.argv):
                should_patch = True

            if should_patch:
                spec = importlib.util.spec_from_file_location(
                    "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
                    adapter_path,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.patch_installed_lmcache_adapter()
                patch_count += 1
                logger.debug("vllm_gr: Applied LMCache adapter patch")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping LMCache adapter patch: %s", exc)

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def patch_lmcache_adapter():
    """Apply LMCache adapter patch.

    Public helper used by engine_core_patch.py in child processes where
    only the LMCache monkey patch must be installed, without re-running
    the full patch sequence.
    """
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parent
        / "3rd_party"
        / "lmcache"
        / "integration"
        / "vllm"
        / "vllm_v1_adapter.py"
    )
    if not module_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
        module_path,
    )
    if spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_installed_lmcache_adapter()


_NPU_WORKER_PATCH_APPLIED = False


def _patch_npu_worker_init_device() -> int:
    """Wrap NPUWorker.init_device to call re_apply_patches() first.

    When vllm-ascend overrides ``worker_cls`` to ``NPUWorker`` (bypassing
    ``GRNPUWorker``), ``re_apply_patches()`` is not called in worker processes.
    NPUWorker instantiates NPUModelRunner inside ``init_device()``, and model
    layer creation calls ``get_attn_backend()``.  We therefore run
    ``re_apply_patches()`` **before** the original ``init_device()`` so the
    selector patch is in place before any attention backend is resolved.
    """
    global _NPU_WORKER_PATCH_APPLIED

    if not _is_npu_platform():
        return 0

    if _NPU_WORKER_PATCH_APPLIED:
        return 0

    try:
        from vllm_ascend.worker.worker import NPUWorker
    except ImportError as exc:
        logger.warning(
            "Failed to import vllm_ascend.worker.worker.NPUWorker; the "
            "vllm-gr delayed NPU patch retry hook was not installed for this "
            "attempt. ImportError: %s",
            exc,
        )
        logger.debug("NPUWorker import traceback", exc_info=True)
        return 0

    original_init_device = NPUWorker.init_device

    def _patched_init_device(self):
        # Apply patches BEFORE original init_device() because NPUWorker
        # instantiates NPUModelRunner inside init_device(), and model
        # layer creation calls get_attn_backend().
        re_apply_patches()
        return original_init_device(self)

    NPUWorker.init_device = _patched_init_device
    _NPU_WORKER_PATCH_APPLIED = True
    return 1


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-gr."""
    ascend_attn_count = _apply_ascend_attention_backend_patch(force=force)
    # Import/patch NPUWorker before the beam patch tries to access
    # NPUModelRunner. vllm-ascend's worker module deliberately initializes
    # vllm_ascend.ops first, avoiding an import-order cycle through device_op.
    npu_worker_count = _patch_npu_worker_init_device()
    beam_count = _apply_gr_beam_patches(force=force)
    return ascend_attn_count + beam_count + npu_worker_count


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules."""
    return _apply_patches(force=True)


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
