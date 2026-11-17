# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import argparse
import inspect
import json
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from typing import Final

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.config import (
    ConstraintBackendKind,
    GRAttentionBackend,
    GRConfig,
    GRConfigInput,
    gr_config_to_mapping,
    write_canonical_gr_config,
)
from vllm_gr.config.models import resolve_constraint_backend
from vllm_gr.config.parser import (
    load_engine_gr_config,
    merge_gr_settings,
    read_settings,
)
from vllm_gr.engine.startup import resolve_native_settings

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
origin_engine_args_init = EngineArgs.__init__
origin_async_engine_args_init = AsyncEngineArgs.__init__
logger = logging.getLogger("vllm_gr.engine.arg_utils_gr")
_ORIGIN_ENGINE_ARG_POSITIONAL_NAMES: Final = tuple(
    list(inspect.signature(origin_engine_args_init).parameters)[1:]
)


def _gr_engine_args_init(original):
    def initialize(
        self: EngineArgs,
        *args: object,
        config: GRConfigInput | None = None,
        _gr_startup_document: Mapping | None = None,
        _gr_mode: bool = False,
        _gr_explicit_native_fields: set[str] | None = None,
        gr: Mapping | None = None,
        **kwargs: object,
    ) -> None:
        """Forward the GR file path to the shared driver startup path."""
        additional = kwargs.get("additional_config")
        selected = (
            _gr_mode
            or config is not None
            or _gr_startup_document is not None
            or gr is not None
            or isinstance(additional, Mapping)
            and any(
                key == "vllm_gr" or isinstance(key, str) and key.startswith("vllm_gr_")
                for key in additional
            )
        )
        if not selected:
            original(self, *args, **kwargs)
            self._gr_selected = False
            return
        document = _gr_startup_document
        if document is None and config is not None:
            document = read_settings(config)
        positional = dict(zip(_ORIGIN_ENGINE_ARG_POSITIONAL_NAMES, args))
        values = resolve_native_settings(
            document, {**positional, **kwargs}, explicit_fields=_gr_explicit_native_fields
        )
        original(
            self, *args, **{name: value for name, value in values.items() if name not in positional}
        )
        self.config = config
        self._gr_startup_document = document
        self.gr = gr
        self._gr_selected = True

    return initialize


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    model_group = parser.add_argument_group(title="vLLM-GR")
    model_group.add_argument(
        "--gr-config",
        dest="gr",
        type=json.loads,
        action="append",
        default=None,
        help="GR settings as JSON, or use --gr.beam_width '[128,64,32]'.",
    )
    return parser


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    selected = bool(getattr(args, "_gr_mode", False) or getattr(args, "gr", None))
    explicit = getattr(args, "_gr_explicit_native_fields", None)
    if selected:
        args = argparse.Namespace(
            **resolve_native_settings(None, vars(args), explicit_fields=explicit)
        )
    engine_args = origin_from_cli_args(cls, args)
    engine_args._gr_selected = selected
    # --config was expanded by GRArgumentParser before EngineArgs construction.
    engine_args.config = None
    overrides = None
    for block in getattr(args, "gr", None) or ():
        if not isinstance(block, dict):
            raise ValueError("--gr-config requires a JSON object")
        overrides = merge_gr_settings(overrides or {}, block)
    engine_args.gr = overrides
    return engine_args


def _apply_npu_default_block_size(config) -> None:
    """Use an Ascend-compatible default without overriding user intent.

    vLLM-Ascend 0.30 sets the block size to 128 itself when prefix caching or
    chunked prefill is enabled, but leaves the platform default in place
    otherwise, while the beam attention backends require the 128 kernel
    geometry. This rule closes that gap; it was validated on the 0.30.0rc2
    runtime.
    """
    try:
        from vllm.platforms import current_platform
    except Exception:
        return
    if getattr(current_platform, "device_type", None) != "npu":
        return

    cache_config = getattr(config, "cache_config", None)
    model_config = getattr(config, "model_config", None)
    if cache_config is None:
        return
    if getattr(cache_config, "user_specified_block_size", False):
        return
    # Hybrid attention/state models have model-specific cache sizing rules in
    # vLLM-Ascend and must not be forced onto the dense-attention default.
    if model_config is not None and getattr(model_config, "is_hybrid", False):
        return
    if cache_config.block_size != 128:
        logger.info(
            "Using Ascend-compatible default KV cache block size 128 "
            "(vLLM resolved the unspecified value to %s).",
            cache_config.block_size,
        )
        cache_config.block_size = 128


def _is_npu_platform() -> bool:
    """Return whether the active platform is an Ascend NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _is_cuda_platform() -> bool:
    """Return whether the active platform is CUDA."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "cuda"


def _resolve_constraint_backend(config: GRConfig) -> GRConfig:
    """Resolve ``constraint.table.backend='auto'`` to the active device."""
    if (
        config.constraint is None
        or config.constraint.table is None
        or config.constraint.table.backend is not ConstraintBackendKind.AUTO
    ):
        return config
    from vllm.platforms import current_platform

    return resolve_constraint_backend(config, current_platform.device_type)


def _beam_worker_decision_default(
    *,
    beam_width_configured: bool,
    worker_decision: bool | None,
) -> bool | None:
    """Resolve an omitted accelerator control while preserving explicit False."""
    if beam_width_configured and (_is_cuda_platform() or _is_npu_platform()):
        if worker_decision is None:
            return True
    return worker_decision


def create_engine_config(self, usage_context=None, headless=False):
    requested_backend = getattr(self, "attention_backend", None)
    if requested_backend is None:
        requested_backend = getattr(getattr(self, "attention_config", None), "backend", None)
    requested_backend_name = getattr(requested_backend, "name", requested_backend)
    config = origin_create_engine_config(self, usage_context, headless)
    additional_input = getattr(config, "additional_config", None)
    has_gr_additional = isinstance(additional_input, Mapping) and any(
        key == "vllm_gr" or isinstance(key, str) and key.startswith("vllm_gr_")
        for key in additional_input
    )
    if (
        not getattr(self, "_gr_selected", False)
        and not has_gr_additional
        and not any(
            getattr(self, name, None) is not None
            for name in ("config", "gr", "_gr_startup_document")
        )
    ):
        return config
    _apply_npu_default_block_size(config)
    path = getattr(self, "config", None)
    overrides = getattr(self, "gr", None)
    additional = getattr(config, "additional_config", None)
    if (
        additional is not None
        and not isinstance(additional, Mapping)
        and path is None
        and overrides is None
    ):
        # Preserve opaque upstream plugin state when this engine has no GR input.
        return config
    effective_gr_config = load_engine_gr_config(
        path,
        getattr(config, "additional_config", None),
        overrides,
        startup_document=getattr(self, "_gr_startup_document", None),
        worker_decision_default=_beam_worker_decision_default(
            beam_width_configured=True, worker_decision=None
        ),
    )
    attention = getattr(config, "attention_config", None)
    backend = getattr(attention, "backend", None)
    # Ascend resets the native selector to auto during VllmConfig creation.
    # Preserve explicit CUSTOM intent so our worker selector can honor it.
    if (
        backend is None
        and isinstance(requested_backend_name, str)
        and requested_backend_name.upper() == "CUSTOM"
    ):
        backend = GRAttentionBackend.CUSTOM
    if backend is not None:
        resolved_backend = (
            GRAttentionBackend.CUSTOM if getattr(backend, "name", backend) == "CUSTOM" else None
        )
        if effective_gr_config.attention_backend != resolved_backend:
            effective_gr_config = replace(effective_gr_config, attention_backend=resolved_backend)
    worker_decision = _beam_worker_decision_default(
        beam_width_configured=effective_gr_config.beam_max_width is not None,
        worker_decision=effective_gr_config.beam_worker_decision,
    )
    if worker_decision != effective_gr_config.beam_worker_decision:
        effective_gr_config = replace(effective_gr_config, beam_worker_decision=worker_decision)
    effective_gr_config = _resolve_constraint_backend(effective_gr_config)

    if (
        effective_gr_config.constrained_generation is False
        and effective_gr_config.beam_max_width is not None
        and _is_npu_platform()
    ):
        from vllm_gr.v1.engine.beam_search_v1 import validate_npu_unconstrained_runner

        validate_npu_unconstrained_runner(config)

    if effective_gr_config.constrained_generation is False:
        from vllm_gr.ops.sliced_logits import set_vocab_slice_config

        set_vocab_slice_config(None, None)

    # vLLM also allows opaque, hashable objects for other plugins. They
    # cannot carry GR state, but a plain engine needs no transport override.
    if (
        isinstance(getattr(config, "additional_config", None), MutableMapping)
        or effective_gr_config != GRConfig()
    ):
        write_canonical_gr_config(config, effective_gr_config)

    if effective_gr_config.attention_backend is GRAttentionBackend.CUSTOM:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    if effective_gr_config.beam_max_width is not None:
        from vllm_gr.v1.engine.beam_search_v1 import validate_gr_execution_config

        validate_gr_execution_config(config)
        compilation = getattr(config, "compilation_config", None)
        if (
            _is_cuda_platform()
            and not getattr(config, "use_v2_model_runner", False)
            and compilation is not None
            and compilation.backend == "inductor"
        ):
            # Inductor can benchmark an extra, halved reduction block on each
            # AOT load. Q/K norm rounding then changes across independent
            # engines, including Beam candidate order and scores. Keep the
            # compiled reduction geometry; record this in the native config
            # so the AOT cache key includes it. Explicit overrides still win.
            compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)

    # On Ascend, vllm-ascend wraps EngineCoreProc.run_engine_core and validates
    # additional_config (pydantic extra="forbid") in the spawned EngineCore
    # process *before* the vllm-gr plugin loads. Install the GR-key sanitizer
    # here -- after origin_create_engine_config and every general plugin, and
    # before EngineCoreClient spawns the process -- so it is the outermost
    # entrypoint and filters GR keys during vllm-ascend's init_ascend_config.
    if _is_npu_platform():
        from vllm.v1.engine.core import EngineCoreProc

        from vllm_gr.v1.ascend_compat import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)

    logger.info(
        "Resolved vLLM-GR configuration: %s",
        json.dumps(gr_config_to_mapping(effective_gr_config), sort_keys=True),
    )
    return config


_init_engine_args = _gr_engine_args_init(origin_engine_args_init)
_init_async_engine_args = _gr_engine_args_init(origin_async_engine_args_init)


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = staticmethod(add_cli_args)
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    EngineArgs.__init__ = _init_engine_args
    AsyncEngineArgs.__init__ = _init_async_engine_args
