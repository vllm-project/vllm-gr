# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""NPU-specific Beam admission checks, loaded only for NPU execution."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any

from vllm_gr.config.models import (
    ConstraintMode,
    configured_constraint_mode,
    runtime_constraint_table,
)
from vllm_gr.platforms.errors import GRDeviceRequestError

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig
    from vllm_gr.platforms.admission import BeamLimits
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


# Full-vocabulary Beam has its own validated horizon. SID projection and
# Constraint Table generation retain their independent three-stage contract.
NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS = 6
_NPU_KV_REORDER_BUFFER_BYTES = 95 * 1024


def _validate_extended_unconstrained_kv_capacity(limits: BeamLimits, capacity: int) -> None:
    """Keep newly admitted suffixes within the native KV reorder buffer."""
    heads, head_size = limits.local_kv_heads, limits.head_size
    if heads is None or head_size is None:
        raise ValueError(
            "NPU unconstrained beam_search above three tokens requires model KV-head geometry"
        )
    if type(heads) is not int or heads < 1 or type(head_size) is not int or head_size < 1:
        raise ValueError("NPU unconstrained beam_search requires positive integer KV-head geometry")
    # SelectUnsharedKV divides its 95 KiB region by head size when tiling,
    # but copies the full configured suffix for every selected head. FP16
    # and BF16 both occupy two bytes. Check pool capacity even for shorter
    # requests; request.max_tokens does not change the physical KV stride.
    tile_heads = _NPU_KV_REORDER_BUFFER_BYTES // 2 // head_size
    copy_bytes = min(heads, tile_heads) * capacity * head_size * 2
    if tile_heads == 0 or copy_bytes > _NPU_KV_REORDER_BUFFER_BYTES:
        raise ValueError(
            "NPU unconstrained beam_search configured KV suffix exceeds "
            "the SelectUnsharedKV 95 KiB buffer; reduce beam.max_decode_steps "
            "or use a model with fewer KV heads or a smaller head size"
        )


def validate_npu_unconstrained_runner(config: Any) -> None:
    """Check the catalog-free NPU adapter before model or request allocation."""
    _validate_runner_generation("v2" if getattr(config, "use_v2_model_runner", False) else "v1")


def _validate_runner_generation(generation: str) -> None:
    if generation != "v1":
        raise ValueError(
            "NPU unconstrained beam_search requires ModelRunner V1; set VLLM_USE_V2_MODEL_RUNNER=0"
        )


def validate_npu_device_config(
    limits: BeamLimits, gr_config: GRConfig, request: BeamBatchRequest | None = None
) -> None:
    """Validate NPU deployment and request capabilities before allocation."""
    parent_widths = gr_config.deployment.candidate_width
    unconstrained = configured_constraint_mode(gr_config) is ConstraintMode.NONE
    if (
        unconstrained
        and parent_widths is not None
        and parent_widths != gr_config.deployment.beam_width
    ):
        raise ValueError("independent per-step candidate_width schedules require CUDA")
    if unconstrained:
        _validate_runner_generation(limits.runner_generation)
        # Keep admission bounded to the validated decision/KV/graph horizon;
        # this does not admit CUDA's arbitrary generation horizons.
        if gr_config.beam_max_decode_steps > NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS or (
            request is not None
            and request.beam_params.max_tokens > NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS
        ):
            raise ValueError(
                "NPU unconstrained beam_search supports "
                f"1..{NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS} generated tokens"
            )
        if gr_config.beam_max_decode_steps > 3:
            _validate_extended_unconstrained_kv_capacity(limits, gr_config.beam_max_decode_steps)
        # Public matching schedules describe K == W at each stage. Their
        # maximum is workspace capacity, not a fixed K for dynamic requests.
        # Keep the fixed-width check for internal scalar configurations and
        # explicit request overrides below.
        configured_k = gr_config.beam_candidate_width if parent_widths is None else None
        candidate_width_error = (
            "NPU unconstrained beam_search requires candidate_width "
            "to match each stage's beam width"
        )
        if configured_k is not None and configured_k != gr_config.beam_max_width:
            # A startup K/W mismatch is a deployment fault even if the
            # request also has an unsupported width schedule.
            raise ValueError(candidate_width_error)
        if request is not None:
            temperature = float(request.beam_params.temperature)
            if temperature > 0.0:
                try:
                    device_temperature = struct.unpack("=f", struct.pack("=f", temperature))[0]
                except OverflowError:
                    device_temperature = math.inf
                if not math.isfinite(device_temperature) or device_temperature == 0.0:
                    raise GRDeviceRequestError(
                        "NPU unconstrained beam_search requires positive temperature "
                        "to be representable as a finite nonzero float32 value"
                    )
            for candidate_width in (configured_k, request.beam_params.candidate_width):
                if candidate_width is not None and (
                    candidate_width != gr_config.beam_max_width
                    or (
                        request.beam_params.dynamic_beam_widths is not None
                        and any(
                            width != candidate_width
                            for width in request.beam_params.dynamic_beam_widths
                        )
                    )
                ):
                    raise GRDeviceRequestError(candidate_width_error)
            if request.beam_params.per_parent_beam_widths is not None:
                raise GRDeviceRequestError(
                    "NPU per_parent_beam_widths requires an enabled canonical NPU constraint table"
                )
    elif request is not None:
        table = runtime_constraint_table(gr_config)
        widths = request.beam_params.per_parent_beam_widths
        # A later stage can exceed capacity even when the bootstrap fits.
        # Refuse the entire request before native preprocessing or Worker allocation.
        if table is not None and widths is not None:
            max_top_k = table.max_top_k
            assert max_top_k is not None
            if any(width > max_top_k for width in widths):
                raise GRDeviceRequestError(
                    f"NPU per_parent_beam_widths exceeds constraint_table.max_top_k={max_top_k}"
                )
