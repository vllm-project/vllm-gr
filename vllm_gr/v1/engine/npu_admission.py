# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-specific Beam admission checks, loaded only for NPU execution."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any

from vllm_gr.config.models import CONSTRAINT_TABLE_MAX_DECODE_STEPS
from vllm_gr.v1.engine.beam_search_v1 import GRDeviceRequestError

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


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
        # The existing NPU decision/KV operators have only been validated
        # for the three-stage suffix. Full-vocabulary selection must not
        # accidentally admit CUDA's arbitrary generation horizons.
        if gr_config.beam_max_decode_steps > CONSTRAINT_TABLE_MAX_DECODE_STEPS or (
            request is not None
            and request.beam_params.max_tokens > CONSTRAINT_TABLE_MAX_DECODE_STEPS
        ):
            raise ValueError("NPU unconstrained beam_search_v1 supports 1..3 generated tokens")
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
