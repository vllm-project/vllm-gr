# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend attention services for tests that exercise shared Beam attention.

Shared Beam attention asks the selected backend for its answers instead of
reading module-level platform constants. Tests therefore supply a service with
explicit answers rather than patching ``NPU_DEVICE`` / ``CUDA_DEVICE``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def attention_service(
    execution: Any,
    *,
    backend_id: str = "cuda",
    prefill: Any = None,
    decode_mode: str = "cascade",
    supports_sinks: bool = False,
    supports_fp8: bool = False,
    supports_batch_invariance: bool = True,
    supports_short_suffix: bool = True,
    requires_flash_attn_version: bool = True,
    requires_layer_kv_scale: bool = True,
    returns_bsnd: bool = False,
    prefill_version: Any = None,
    capture_standard: Any = None,
    has_standard_capture: bool = False,
    row_selector: Any = None,
) -> Any:
    """Return a Beam attention service with explicit, overridable answers.

    ``prefill`` defaults to ``execution``; pass a different module to model a
    backend whose Decode path does not serve Prefill. ``prefill_version`` forces
    the resolved Prefill FlashAttention version, and ``capture_standard`` stands
    in for the backend's native Prefill capture.
    """

    def resolve_prefill_version(requested: Any, decode_version: Any, **kwargs: Any) -> Any:
        if prefill_version is not None:
            return prefill_version
        if backend_id != "cuda":
            # Ascend has one Prefill attention version; an explicit selection
            # is a configuration error rather than a silently ignored one.
            if requested not in (None, "auto"):
                raise ValueError("beam.prefill_flash_attn_version is CUDA-only")
            return decode_version
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            resolve_prefill_flash_attn_version,
        )

        if requested is None:
            return decode_version
        return resolve_prefill_flash_attn_version(requested, decode_version, **kwargs)

    return SimpleNamespace(
        backend_id=backend_id,
        execution_module=lambda: execution,
        prefill_module=lambda: execution if prefill is None else prefill,
        decode_mode=lambda: decode_mode,
        supports_sinks=lambda: supports_sinks,
        supports_fp8_kv_cache=lambda: supports_fp8,
        supports_batch_invariance=lambda: supports_batch_invariance,
        supports_short_suffix_attention=lambda: supports_short_suffix,
        supports_prefill_version_selection=lambda: supports_short_suffix,
        requires_flash_attn_version=lambda: requires_flash_attn_version,
        requires_layer_kv_scale=lambda: requires_layer_kv_scale,
        uses_bsnd_prefill=lambda attn_metadata: returns_bsnd,
        beam_row_selector=lambda: row_selector,
        supports_compute_capability=lambda capability: True,
        validate_impl_features=lambda **kwargs: None,
        resolve_prefill_version=resolve_prefill_version,
        update_standard_graph_params=lambda *args: None,
        capture_standard_prefill=capture_standard or (lambda *args: False),
        has_standard_prefill_capture=lambda: has_standard_capture,
    )


__all__ = ["attention_service"]
