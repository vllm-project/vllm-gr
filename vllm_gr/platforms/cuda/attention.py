# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Cuda Beam attention and physical metadata policy."""

from __future__ import annotations

import functools
import os
from typing import Any

from vllm_gr.v1.attention.backends.beam_dense_prefill import (
    DENSE_PREFILL_MODES as _DENSE_PREFILL_MODES,
)
from vllm_gr.v1.attention.backends.beam_dense_prefill import (
    ENV_DENSE_PREFILL,
)

from ..attention import AttentionExecutor, MetadataProfile
from ..plan import BackendPlan


class CudaMetadataProfile(MetadataProfile):
    def scheduler_metadata(self, *args: Any, **kwargs: Any) -> Any:
        from vllm.v1.attention.backends.fa_utils import (
            is_flash_attn_varlen_func_available,
        )

        if not is_flash_attn_varlen_func_available():
            raise RuntimeError(
                "BeamAttentionMetadataBuilder requires FlashAttention scheduler support"
            )
        from vllm.v1.attention.backends.fa_utils import (
            get_scheduler_metadata as _get_scheduler_metadata,
        )

        return _get_scheduler_metadata(*args, **kwargs)


class CudaAttention(AttentionExecutor):
    """Own CUDA Beam attention selection and admissibility.

    Shared Beam attention asks this service instead of reading platform
    constants. FlashAttention availability still lives in the execution module;
    this type only reports it to shared code under a backend-neutral contract.
    """

    metadata_profile = CudaMetadataProfile(
        allocates_fixed_buffers=True,
        uses_v2_prefill_capture=True,
        forces_single_reduction_split=True,
        writes_seqlens_per_tensor=True,
    )

    uses_pool_pointers = True
    backend_id = "cuda"

    def __init__(self, *, plan: BackendPlan | None = None, kv_layout: str | None = None) -> None:
        choices = plan.choices.to_dict() if plan is not None else {}
        super().__init__(kv_layout=choices.get("kv_layout", kv_layout))
        self._planned = plan is not None
        self._dense_prefill: str | None = choices.get("dense_prefill_mode")
        self.requires_fresh_prefill = self._dense_prefill is not None

    def execution_module(self) -> Any:
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return beam_attn_gpu

    def supports_sinks(self) -> bool:
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return bool(beam_attn_gpu.is_available() and beam_attn_gpu.supports_sinks())

    def supports_fp8_kv_cache(self) -> bool:
        from vllm.platforms import current_platform

        # FP8 KV cache needs an FA3 decode path on the 90-class devices; the
        # capability test belongs to the backend, not to shared attention.
        if self.execution_module().get_flash_attn_version() != 3:
            return False
        return bool(current_platform.is_device_capability_family(90))

    def supports_batch_invariance(self) -> bool:
        return True

    def supports_sliding_window(self) -> bool:
        return True

    def requires_flash_attn_version(self) -> bool:
        return True

    def supports_short_suffix_device(self, device: Any) -> bool:
        return bool(device.type == "cuda")

    def supports_short_suffix_attention(self) -> bool:
        return True

    def short_suffix_head_size_supported(self, head_size: int) -> bool:
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            short_suffix_head_size_supported,
        )

        return bool(short_suffix_head_size_supported(head_size))

    def supports_prefill_version_selection(self) -> bool:
        return True

    def supports_compute_capability(self, capability: Any) -> bool:
        from vllm.platforms.interface import DeviceCapability

        return bool(capability >= DeviceCapability(8, 0))

    def resolve_prefill_version(
        self,
        requested: Any,
        decode_version: Any,
        **kwargs: Any,
    ) -> Any:
        if requested is None:
            return decode_version
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            resolve_prefill_flash_attn_version,
        )

        return resolve_prefill_flash_attn_version(requested, decode_version, **kwargs)

    def write_suffix_kv(self, pool: Any, key: Any, value: Any, session: Any, metadata: Any) -> None:
        decode_step_buf = getattr(session.context, "decode_step_buf", None)
        if decode_step_buf is not None and decode_step_buf.is_cuda:
            from vllm_gr.v1.attention.backends.beam_graph_kernels import (
                cache_unshared_kv_graph,
            )

            cache_unshared_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                key,
                value,
                session.slot,
                decode_step_buf,
                active_mask=getattr(metadata, "beam_active_mask", None),
            )
        else:
            super().write_suffix_kv(pool, key, value, session, metadata)

    def dense_prefill_mode(self, config: Any) -> str | None:
        if self._planned:
            return self._dense_prefill
        return _resolve_dense_prefill_mode(config)

    def validate_short_suffix(self, *args: Any) -> None:
        from vllm_gr.v1.attention.backends.beam_attn_gpu import (
            validate_short_suffix_attention,
        )

        validate_short_suffix_attention(*args)


def _resolve_dense_prefill_mode(config: Any) -> str | None:
    """Opt-in dense CUDA Prefill mode, or None to keep the paged path.

    Captured Prefill graphs hold fresh prompts only when prefix caching,
    chunked Prefill and KV connectors are all off; otherwise a request can carry
    context that only the paged cache holds, so the mode is ignored. Each call
    still checks its CPU lengths (eager batches and non-Beam decode rows also
    reach the Prefill entry).
    """
    from vllm import envs

    raw = os.environ.get(ENV_DENSE_PREFILL, "off").strip().lower()
    if raw in ("", "off"):
        return None
    if raw not in _DENSE_PREFILL_MODES:
        _warn_invalid_dense_prefill_mode(raw)
        return None
    if config is None:
        return None
    from vllm_gr.v1.attention.backends.beam_dense_prefill import (
        admitted_dense_prefill_mode,
    )

    return admitted_dense_prefill_mode(
        raw,
        prefix_caching=config.cache_config.enable_prefix_caching,
        chunked_prefill=config.scheduler_config.enable_chunked_prefill,
        kv_transfer=getattr(config, "kv_transfer_config", None) is not None,
        batch_invariant=envs.VLLM_BATCH_INVARIANT,
    )


@functools.lru_cache(maxsize=8)
def _warn_invalid_dense_prefill_mode(raw: str) -> None:
    # Resolved once per attention layer; warn once per process.
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "vllm_gr: ignoring %s=%s (expected off, fa2 or cudnn)", ENV_DENSE_PREFILL, raw
    )
