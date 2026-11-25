# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import logging
from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs

from vllm_gr.config import (
    ConstraintBackendKind,
    GRAttentionBackend,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    write_canonical_gr_config,
)
from vllm_gr.config.models import (
    resolve_constraint_backend,
)
from vllm_gr.config.parser import (
    GR_CONFIG_KEY,
    validate_additional_keys,
)
from vllm_gr.platforms.config import BackendConfigService, engine_config_service

origin_create_engine_config = EngineArgs.create_engine_config
logger = logging.getLogger(__name__)


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
    service: BackendConfigService,
    *,
    beam_width_configured: bool,
    worker_decision: bool | None,
) -> bool | None:
    """Resolve an omitted accelerator control while preserving explicit False."""
    if beam_width_configured and service.default_worker_decision:
        if worker_decision is None:
            return True
    return worker_decision


def create_engine_config(self, usage_context=None, headless=False):
    requested_backend = getattr(self, "attention_backend", None)
    if requested_backend is None:
        requested_backend = getattr(getattr(self, "attention_config", None), "backend", None)
    requested_backend_name = getattr(requested_backend, "name", requested_backend)
    additional = getattr(self, "additional_config", None)
    if not isinstance(additional, Mapping) or not any(
        key == GR_CONFIG_KEY or isinstance(key, str) and key.startswith("vllm_gr_")
        for key in additional
    ):
        return origin_create_engine_config(self, usage_context, headless)
    # Upstream can add kernel selectors to its dictionary during construction.
    # Give each creation its own copy, including when creation raises.
    validate_additional_keys(additional)
    native_args = copy(self)
    native_args.additional_config = dict(additional)
    if GR_CONFIG_KEY in additional:
        effective_gr_config = gr_config_from_mapping(
            native_args.additional_config.pop(GR_CONFIG_KEY), base_dir=Path.cwd()
        )
    else:
        effective_gr_config = load_gr_config()
    config = origin_create_engine_config(native_args, usage_context, headless)
    service = engine_config_service()
    service.apply_native_defaults(config)
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
        service,
        beam_width_configured=effective_gr_config.beam_max_width is not None,
        worker_decision=effective_gr_config.beam_worker_decision,
    )
    if worker_decision != effective_gr_config.beam_worker_decision:
        effective_gr_config = replace(effective_gr_config, beam_worker_decision=worker_decision)
    effective_gr_config = _resolve_constraint_backend(effective_gr_config)

    service.validate_engine(config, effective_gr_config)

    write_canonical_gr_config(config, effective_gr_config)

    if effective_gr_config.attention_backend is GRAttentionBackend.CUSTOM:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    if effective_gr_config.beam_max_width is not None:
        from vllm_gr.config.execution import validate_gr_execution_config

        validate_gr_execution_config(config)

    service.finalize_native_config(config, effective_gr_config)

    from vllm_gr.platforms.planning import admit_backend_plan

    admit_backend_plan(config)

    logger.info(
        "Resolved vLLM-GR configuration: %s",
        json.dumps(gr_config_to_mapping(effective_gr_config), sort_keys=True),
    )
    return config


def patch_create_engine_config() -> None:
    EngineArgs.create_engine_config = create_engine_config


def validate_engine_args_patch() -> str | None:
    if EngineArgs.create_engine_config is create_engine_config:
        return None
    return "EngineArgs.create_engine_config GR integration is incomplete"
