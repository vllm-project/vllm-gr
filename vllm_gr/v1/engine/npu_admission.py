# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-specific Beam admission checks, loaded only for NPU execution."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any

from vllm_gr.v1.engine.beam_search_v1 import GRDeviceRequestError

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


# Full-vocabulary Beam has its own validated horizon. SID projection and
# Constraint Table generation retain their independent three-stage contract.
NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS = 6
_NPU_KV_REORDER_BUFFER_BYTES = 95 * 1024


def _validate_extended_unconstrained_kv_capacity(config: Any, capacity: int) -> None:
    """Keep newly admitted suffixes within the native KV reorder buffer."""
    try:
        model = config.model_config
        heads = model.get_num_kv_heads(getattr(config, "parallel_config", None))
        head_size = model.get_head_size()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "NPU unconstrained beam_search_v1 above three tokens requires model KV-head geometry"
        ) from exc
    if type(heads) is not int or heads < 1 or type(head_size) is not int or head_size < 1:
        raise ValueError(
            "NPU unconstrained beam_search_v1 requires positive integer KV-head geometry"
        )
    # SelectUnsharedKV divides its 95 KiB region by head size when tiling,
    # but copies the full configured suffix for every selected head. FP16
    # and BF16 both occupy two bytes. Check pool capacity even for shorter
    # requests; request.max_tokens does not change the physical KV stride.
    tile_heads = _NPU_KV_REORDER_BUFFER_BYTES // 2 // head_size
    copy_bytes = min(heads, tile_heads) * capacity * head_size * 2
    if tile_heads == 0 or copy_bytes > _NPU_KV_REORDER_BUFFER_BYTES:
        raise ValueError(
            "NPU unconstrained beam_search_v1 configured KV suffix exceeds "
            "the SelectUnsharedKV 95 KiB buffer; reduce beam.max_decode_steps "
            "or use a model with fewer KV heads or a smaller head size"
        )


def validate_npu_unconstrained_runner(config: Any) -> None:
    """Check the catalog-free NPU adapter before model or request allocation."""
    if getattr(config, "use_v2_model_runner", False):
        raise ValueError(
            "NPU unconstrained beam_search_v1 requires ModelRunner V1; "
            "set VLLM_USE_V2_MODEL_RUNNER=0"
        )


def validate_npu_device_config(
    config: Any, gr_config: GRConfig, request: BeamBatchRequest | None = None
) -> None:
    """Validate NPU deployment and request capabilities before allocation."""
    if gr_config.constrained_generation is False:
        validate_npu_unconstrained_runner(config)
        # Keep admission bounded to the validated decision/KV/graph horizon;
        # this does not admit CUDA's arbitrary generation horizons.
        if gr_config.beam_max_decode_steps > NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS or (
            request is not None
            and request.beam_params.max_tokens > NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS
        ):
            raise ValueError(
                "NPU unconstrained beam_search_v1 supports "
                f"1..{NPU_UNCONSTRAINED_MAX_GENERATED_TOKENS} generated tokens"
            )
        if gr_config.beam_max_decode_steps > 3:
            _validate_extended_unconstrained_kv_capacity(config, gr_config.beam_max_decode_steps)
        table = gr_config.constraint_table
        configured_k = None if table is None else table.candidate_width
        candidate_width_error = (
            "NPU unconstrained beam_search_v1 requires candidate_width "
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
                        "NPU unconstrained beam_search_v1 requires positive temperature "
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
