# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Normalize frontend Beam parameters and resolve deployment width defaults."""

from typing import Any

from msgspec.structs import asdict
from vllm.sampling_params import BeamSearchParams as NativeBeamSearchParams

from vllm_gr.config import GRConfig
from vllm_gr.config.beam_widths import SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS
from vllm_gr.entrypoints.beam_search.exceptions import BeamSearchRequestError
from vllm_gr.platforms.beam_requests import uses_retained_candidate_widths
from vllm_gr.sampling_params import BeamSearchParams


def resolve_request_widths(
    config: GRConfig, params: NativeBeamSearchParams, *, device: object = None
) -> BeamSearchParams:
    """Keep runtime capacity separate from request length and width schedules."""
    values = asdict(params)
    for name in ("dynamic_beam_widths", "per_parent_beam_widths"):
        if values.get(name) is not None:
            values[name] = list(values[name])
    if config.beam_max_width is not None and params.beam_width != config.beam_max_width:
        raise BeamSearchRequestError(
            f"beam_width must equal the fixed runtime width {config.beam_max_width}"
        )
    options = config.deployment
    steps = (
        params.max_tokens
        - int(values.get("begin_token") is not None)
        - int(values.get("end_token") is not None)
    )
    retained = options.beam_width
    expanded = options.candidate_width
    retained_default = retained if retained and len(set(retained)) > 1 else None
    expanded_default = (
        expanded
        if expanded and (len(set(expanded)) > 1 or retained and max(retained) > max(expanded))
        else None
    )
    updates: dict[str, Any] = {}
    for name, default in (
        ("dynamic_beam_widths", retained_default),
        ("per_parent_beam_widths", expanded_default),
        ("result_width", options.result_width),
    ):
        if values.get(name) is not None or default is None:
            continue
        if isinstance(default, tuple):
            if steps not in SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS or steps > len(default):
                raise BeamSearchRequestError(
                    f"{steps} effective generation steps cannot use the engine's "
                    f"{len(default)}-entry {name} default; dynamic requests support "
                    "one to three steps. Engine max_decode_steps is capacity, "
                    "not a request schedule."
                )
            updates[name] = list(default[:steps])
        else:
            updates[name] = default
    if "per_parent_beam_widths" in updates and uses_retained_candidate_widths(config, device):
        widths = updates.get("dynamic_beam_widths", values.get("dynamic_beam_widths"))
        if updates["per_parent_beam_widths"] == widths:
            del updates["per_parent_beam_widths"]
    # Construction validates result_width against this request's final retained
    # width, and validates explicit schedules too. msgspec.replace skips that.
    return BeamSearchParams(**{**values, **updates})


def apply_beam_request_defaults(
    vllm_config: Any, params: NativeBeamSearchParams
) -> BeamSearchParams:
    return resolve_request_widths(
        GRConfig.from_vllm_config(vllm_config),
        params,
        device=getattr(getattr(vllm_config, "device_config", None), "device", None),
    )
