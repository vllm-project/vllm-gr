# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-side device policy shared by Beam frontends and EngineCore."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from vllm_gr.config import GRConfig
from vllm_gr.config.models import (
    ConstraintMode,
    configured_constraint_mode,
    runtime_constraint_table,
)
from vllm_gr.platforms.admission import beam_limits, validate_beam_batch
from vllm_gr.platforms.errors import GRDeviceRequestError

if TYPE_CHECKING:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


def validate_gr_device_config(config: Any, request: BeamBatchRequest | None = None) -> None:
    """Check execution support without importing a device runtime."""
    limits = beam_limits(config)
    gr = GRConfig.from_vllm_config(config)
    if gr.beam_sid_stage_vocab and not limits.supports_sid_stage_vocab:
        raise ValueError("The admitted GR platform does not support beam.sid_stage_vocab")
    from vllm_gr.platforms import get_current_gr_platform
    from vllm_gr.platforms.errors import UnsupportedBackendError

    try:
        platform = get_current_gr_platform()
    except UnsupportedBackendError:
        platform = None
    from vllm_gr.platforms.identity import qualified_class_name

    same_platform = platform is not None and (
        limits.platform_class == qualified_class_name(type(platform))
        or limits.platform_class is None
        and limits.device_type == platform.device_type
    )
    if same_platform and platform is not None:
        platform.validate_gr_request(config, request)
    if configured_constraint_mode(gr) is ConstraintMode.NONE and not limits.supports_unconstrained:
        raise ValueError("The admitted GR platform does not support unconstrained beam_search")
    if (
        not limits.supports_dynamic_width
        and request is not None
        and (
            request.beam_params.dynamic_beam_widths is not None
            or request.output_options.num_return_sequences != request.beam_params.beam_width
        )
    ):
        raise GRDeviceRequestError("The admitted GR platform does not support dynamic Beam widths")


def engine_request_validator(config: Any) -> Callable[[BeamBatchRequest], None] | None:
    """Bind the installed platform's optional check once during EngineCore startup."""
    from vllm_gr.platforms import get_current_gr_platform

    return partial(get_current_gr_platform().validate_gr_request, config)


def validate_decode_token_budget(request: BeamBatchRequest, config: Any) -> None:
    validate_beam_batch(request, beam_limits(config))


def decode_batch_capacity(config: Any, beam_width: int) -> int | None:
    """Return the physical Decode batch cap used for frontend grouping."""
    budget = beam_limits(config).max_decode_rows
    return budget // beam_width if budget is not None else None


def validate_dynamic_width_support(config: Any) -> None:
    if not beam_limits(config).supports_dynamic_width:
        raise ValueError(
            "The admitted GR platform does not support dynamic_beam_widths or result_width"
        )


def validate_frontend_constraint_mode(
    config: Any, gr_config: GRConfig, catalog_strategy: str | None
) -> None:
    limits = beam_limits(config)
    if configured_constraint_mode(gr_config) is ConstraintMode.NONE:
        if not limits.supports_unconstrained:
            raise ValueError("The admitted GR platform does not support unconstrained beam_search")
        return

    table = runtime_constraint_table(gr_config)
    backend = limits.constraint_backend
    if backend is None:
        raise ValueError("The admitted GR platform does not support a canonical constraint table")
    if table is None or not table.enabled or table.backend.value != backend:
        raise ValueError(
            f"beam_search requires an enabled canonical {backend.upper()} constraint table"
        )
    if catalog_strategy is not None:
        raise ValueError("catalog_strategy is not applicable to the constraint_table backend")
