# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

import functools
from copy import copy
from typing import Any, NoReturn, cast

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as _get_flash_attn_version,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

from vllm_gr.v1.attention.backends.beam_attention_state import (
    AttentionState,
    clear_empty_prefix_states,
    normalize_flash_attn_state,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    require_beam_decode_session,
    resolve_beam_pool,
)
from vllm_gr.v1.attention.backends.beam_dense_prefill import in_fresh_prefill_capture

logger = init_logger(__name__)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
else:

    def _flash_attn_varlen_unavailable(
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable


class GPUBeamAttentionBatch:
    """Zero-copy request-major attention view over contiguous Beam slots."""

    def __init__(self, sessions: tuple[Any, ...], *, width: int | None = None) -> None:
        if not sessions:
            raise ValueError("GPU Beam attention batch must not be empty")
        first = sessions[0]
        self.sessions = sessions
        self.context = first.context
        self.slot = 0
        self.batch_size = len(sessions)
        self.request_width = first.beam_width if width is None else width
        self.beam_width = self.batch_size * self.request_width
        self.session_id = first.session_id
        if tuple(session.slot for session in sessions) != tuple(range(self.batch_size)):
            raise RuntimeError("GPU Beam batch requires contiguous ordered KV slots")
        if any(session.context is not self.context for session in sessions):
            raise RuntimeError("GPU Beam batch sessions must share one context")
        geometry = (first.beam_width, first.decode_step)
        if any((session.beam_width, session.decode_step) != geometry for session in sessions):
            raise RuntimeError("GPU Beam batch members changed width or Decode step")
        if not 1 <= self.request_width <= self.context.max_beams:
            raise RuntimeError("GPU Beam batch requires the configured exact Beam width")

        self.pools = []
        from vllm_gr.v1.beam.dynamic_beam import get_stage_pools

        for source in get_stage_pools(self.context, self.request_width):
            pool = copy(source)
            storage_rows = self.batch_size * self.request_width
            pool.max_batch = 1
            pool.max_beams = storage_rows
            pool.unshared_key = source.unshared_key[: self.batch_size].reshape(
                1, storage_rows, *source.unshared_key.shape[2:]
            )
            pool.unshared_value = source.unshared_value[: self.batch_size].reshape(
                1, storage_rows, *source.unshared_value.shape[2:]
            )
            self.pools.append(pool)

    @property
    def is_active(self) -> bool:
        return all(session.is_active for session in self.sessions)

    @property
    def decode_step(self) -> int:
        return int(self.sessions[0].decode_step)

    @property
    def suffix_k_buf(self) -> torch.Tensor:
        rows = self.beam_width * self.context.max_decode_steps
        return self.context.suffix_k_buf[:rows]

    @property
    def suffix_v_buf(self) -> torch.Tensor:
        rows = self.beam_width * self.context.max_decode_steps
        return self.context.suffix_v_buf[:rows]


def _num_splits(default: int) -> int:
    """Use deterministic FlashAttention reductions when vLLM requests them."""
    return 1 if envs.VLLM_BATCH_INVARIANT else default


def is_available() -> bool:
    return bool(is_flash_attn_varlen_func_available())


def get_flash_attn_version(
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
) -> int | None:
    return cast(
        int | None,
        _get_flash_attn_version(
            requires_alibi=requires_alibi,
            head_size=head_size,
        ),
    )


def resolve_prefill_flash_attn_version(
    requested: int | str | None,
    decode_version: int,
    *,
    head_size: int,
    kv_cache_dtype: str,
    requires_alibi: bool,
    has_sinks: bool,
    batch_invariant: bool,
) -> int:
    """Resolve once before graph capture, without mutating vLLM's decode config.

    Auto is deliberately limited to FA2 -> FA3 on eligible devices. FA4,
    quantized KV and special attention features need separate qualification.
    Explicit requests fail early instead of silently benchmarking a fallback.
    """
    from vllm_gr.config.validation import validate_prefill_flash_attn_version

    validate_prefill_flash_attn_version(requested)
    if requested is None or requested == decode_version:
        return decode_version
    if requested == "auto" and decode_version != 2:
        return decode_version

    target = 3 if requested == "auto" else int(requested)
    reason = None
    if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        reason = "independent prefill selection requires unquantized KV"
    elif batch_invariant:
        reason = "independent prefill selection is not enabled with batch invariance"
    elif requires_alibi or has_sinks:
        reason = "independent prefill selection does not support ALiBi or sinks"
    elif head_size <= 0 or head_size > 256 or head_size % 8:
        reason = "prefill head size must be a multiple of 8 in [8, 256]"
    else:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        if not is_fa_version_supported(target):
            reason = fa_version_unsupported_reason(target) or "backend unavailable"
    if reason is not None:
        if requested == "auto":
            return decode_version
        raise ValueError(f"Cannot select prefill FlashAttention {target}: {reason}")
    return target


def supports_sinks() -> bool:
    return bool(is_available() and flash_attn_supports_sinks())


def cache_unshared_kv(
    unshared_key: Any,
    unshared_value: Any,
    curr_key: Any,
    curr_value: Any,
    slot: int,
    decode_step: int,
    beam_session: object = None,  # unused on GPU; xllm-only contract
) -> None:
    """Cascade-mode pool write: slot/0-based semantics, unchanged behavior."""
    from vllm_gr.v1.attention.backends.beam_kernels import (
        cache_unshared_kv as _cache_unshared_kv,
    )

    del beam_session
    _cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        slot,
        decode_step,
    )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# Minimum token count for the cuDNN dense Prefill; cuDNN SDPA rejects
# single-token K/V and short shapes gain nothing over FA2.
_CUDNN_DENSE_MIN_TOKENS = 64


def _all_fresh_prompts(attn_metadata: Any) -> bool:
    """Whether every request's K/V context is exactly its scheduled tokens.

    The pure-Prefill metadata also serves eager batches and non-Beam decode
    rows, whose earlier tokens live only in the paged cache. Missing CPU
    mirrors count as "not fresh" so the paged path stays the default.
    """
    starts = getattr(attn_metadata, "query_start_loc_cpu", None)
    lengths = getattr(attn_metadata, "seq_lens_cpu", None)
    num_reqs = getattr(attn_metadata, "num_reqs", 0)
    if not num_reqs or starts is None or lengths is None:
        return False
    if len(starts) < num_reqs + 1 or len(lengths) < num_reqs:
        return False
    return all(lengths[i] == starts[i + 1] - starts[i] for i in range(num_reqs))


@functools.lru_cache(maxsize=1)
def _warn_cudnn_unavailable(reason: str) -> None:
    logger.warning(
        "vllm_gr: cuDNN dense Prefill unavailable (%s); using dense FA2", reason.splitlines()[0]
    )


def _dense_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    *,
    scale: float,
    mode: str,
    causal: bool,
    sliding_window_size: list[int] | tuple[int, int] | None,
    logits_soft_cap: float,
    s_aux: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
) -> bool:
    """Attend over the contiguous fresh K/V instead of the paged cache.

    Without cached context, request K/V rows align with its query rows, so
    ``cu_seqlens_k == cu_seqlens_q``. Reading the paged cache forces FA2's
    split-KV kernel (BlockM=64); the dense FA2 kernel (BlockM=128) and cuDNN
    are faster on long prompts. Returns False when the layer configuration,
    capture scope or request lengths do not allow it, leaving the caller on the
    paged path. See ``beam_dense_prefill`` for the capture/replay contract.
    """
    if (
        not causal
        or alibi_slopes is not None
        or s_aux is not None
        or logits_soft_cap
        or (sliding_window_size is not None and tuple(sliding_window_size) != (-1, -1))
        or key.dtype != query.dtype
        # A graph freezes this choice: outside a fresh-Prefill capture (e.g. a
        # native decode capture from seq_len == 1 dummies) replay would see
        # cached context, so only the paged path is safe.
        or (torch.cuda.is_current_stream_capturing() and not in_fresh_prefill_capture())
        or not _all_fresh_prompts(attn_metadata)
    ):
        return False
    tokens = query.shape[0]
    key, value = key[:tokens], value[:tokens]
    if (
        mode == "cudnn"
        and getattr(attn_metadata, "num_reqs", 0) == 1
        and tokens >= _CUDNN_DENSE_MIN_TOKENS
    ):
        # One request: padded tail rows past cu_seqlens_q[-1] come after every
        # real row, so causal masking keeps them out of the real outputs.
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(
                    query.unsqueeze(0).transpose(1, 2),
                    key.unsqueeze(0).transpose(1, 2),
                    value.unsqueeze(0).transpose(1, 2),
                    is_causal=True,
                    scale=scale,
                    enable_gqa=True,
                )
        except RuntimeError as error:
            # Backend selection fails before any launch when this torch/cuDNN
            # build cannot serve the shape; dense FA2 below still applies.
            _warn_cudnn_unavailable(str(error))
        else:
            output[:tokens] = out[0].transpose(0, 1)
            return True
    cu_seqlens_q = attn_metadata.query_start_loc
    max_q_len = attn_metadata.max_query_len
    output[:tokens] = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_q_len,
        softmax_scale=scale,
        causal=True,
        fa_version=2,
    )
    return True


def prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    dense_key: torch.Tensor | None = None,
    dense_value: torch.Tensor | None = None,
    dense_mode: str | None = None,
) -> None:
    """Dedicated pure prefill attention execution for the prefill graph path.

    ``dense_key``/``dense_value`` are this layer's freshly projected K/V. The
    caller passes them only when every captured Prefill is a fresh prompt (no
    prefix caching, no chunked Prefill); see ``_dense_prefill_attention``.
    """
    cu_seqlens_q = getattr(attn_metadata, "query_start_loc", None)
    max_q_len = getattr(attn_metadata, "max_query_len", 0)
    seqused_k = getattr(attn_metadata, "seq_lens", None)
    max_seq_len = getattr(attn_metadata, "max_seq_len", 0)
    block_table = (
        getattr(attn_metadata, "block_table_tensor", None)
        if getattr(attn_metadata, "block_table_tensor", None) is not None
        else getattr(attn_metadata, "block_table", None)
    )
    causal = getattr(attn_metadata, "causal", True)
    assert cu_seqlens_q is not None, "query_start_loc must not be None for prefill attention"
    assert seqused_k is not None, "seq_lens must not be None for prefill attention"

    # Compute only the real rows: FA2 silently ignores query rows beyond
    # cu_seqlens_q[-1], but slicing keeps the fast path aligned with the NPU
    # FIA convention (queryT == last(actual_seq_qlen)).
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    query = query[:num_actual_tokens]

    if dense_key is not None and dense_value is not None and dense_mode is not None:
        if _dense_prefill_attention(
            output,
            query,
            dense_key,
            dense_value,
            attn_metadata,
            scale=scale,
            mode=dense_mode,
            causal=causal,
            sliding_window_size=sliding_window_size,
            logits_soft_cap=logits_soft_cap,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
        ):
            return

    std_out = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q_len,
        seqused_k=seqused_k,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        window_size=sliding_window_size,
        block_table=block_table,
        softcap=logits_soft_cap,
        fa_version=vllm_flash_attn_version,
        num_splits=_num_splits(0),
        s_aux=s_aux,
    )

    output[: std_out.shape[0]] = std_out


def short_suffix_head_size_supported(head_size: int) -> bool:
    """Return whether FlashAttention 2 can produce prefix states for this size.

    FA2 accepts head sizes that are multiples of 8 up to 256. Sizes outside the
    FA2-aligned MMA fast path (64/128/256) still use the generic Triton merge
    kernel, so they are not rejected; see ``merge_short_suffix_attention``.
    """
    return 0 < head_size <= 256 and head_size % 8 == 0


def validate_short_suffix_attention(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    fa_version: int,
    sliding_window: list[int] | tuple[int, int] | None,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
) -> None:
    """Validate static layer semantics once, before graph capture/execution.

    ``fa_version`` only gates the *prefix* backend that produced the incoming
    state: normalization already made FA2/FA3/FA4 states interchangeable. The
    fused suffix arithmetic never follows it -- ``merge_short_suffix_attention``
    picks its kernel from shape/dtype alone and keeps FA2's intermediate rounding
    as the reference for every version (asserted by
    ``test_fused_suffix_uses_the_fa2_emulating_kernel_for_every_prefix_backend``).
    """
    if (
        num_kv_heads <= 0
        or num_heads % num_kv_heads
        or not short_suffix_head_size_supported(head_size)
    ):
        raise ValueError(
            "Short suffix attention requires integral GQA groups and a "
            "FlashAttention-compatible head size (multiple of 8, at most 256)"
        )
    if fa_version not in (2, 3, 4):
        raise ValueError(
            "Short suffix attention requires an available FlashAttention 2, 3 or 4 backend"
        )
    if (
        sliding_window not in (None, (-1, -1), [-1, -1])
        or softcap
        or alibi_slopes is not None
        or sinks is not None
    ):
        raise ValueError(
            "Short suffix attention does not support sliding window, softcap, ALiBi or sinks"
        )


def _is_pure_beam_decode(attn_metadata: BeamAttentionMetadata, beam_width: int) -> bool:
    """Only fuse batches whose complete output is one query per beam.

    For a fixed-membership multi-request batch, beam_width is the session's
    flattened B*W width. Mixed standard/beam rows and prefix recomputation must
    retain the generic path's indexed output writes.
    """
    return (
        beam_width > 0
        and attn_metadata.std_indices is None
        and attn_metadata.prefix_indices is not None
        and attn_metadata.suffix_max_q_len == 1
        and attn_metadata.num_prefix_indices == beam_width
        and attn_metadata.num_actual_tokens == beam_width
    )


def prefix_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    *,
    scale: float,
    fa_version: int,
    sliding_window_size: list[int] | tuple[int, int] | None,
    logits_soft_cap: float,
    alibi_slopes: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
    plan: Any = None,
) -> AttentionState:
    """Compute a canonical state; suffix fusion is independent of the plan."""
    from vllm_gr.v1.attention.backends.beam_prefix_policy import (
        available_prefix_versions,
        device_fingerprint,
        multi_processor_count,
    )

    plain = (
        attn_metadata.prefix_block_table is not None
        and alibi_slopes is None
        and s_aux is None
        and not logits_soft_cap
        and sliding_window_size in (None, (-1, -1), [-1, -1])
        and query.dtype in (torch.float16, torch.bfloat16)
        and key_cache.dtype == query.dtype == value_cache.dtype
    )
    # Only a bucketed call carries a meaningful prefix length: without a length
    # bucket ``prefix_max_seq_len`` is the model bound, so deriving a plan from it
    # would segment a 1K prefix as if it were 8K (measured: 5% slower on
    # 1024/B1/W128).
    if (
        plan is None
        and plain
        and attn_metadata.max_num_splits == 0
        and getattr(attn_metadata, "prefix_length_bucket", 0)
    ):
        from vllm_gr.v1.attention.backends.beam_prefix_policy import (
            PrefixShape,
            resolve_prefix_plan,
        )

        widths = attn_metadata.prefix_seqlens_q_cpu
        if widths and all(w == attn_metadata.prefix_max_q_len for w in widths):
            shape = PrefixShape(
                batch=len(widths),
                width=widths[0],
                length=attn_metadata.prefix_max_seq_len,
                heads=query.shape[1],
                kv_heads=key_cache.shape[-2],
                dim=query.shape[2],
                dtype=str(query.dtype),
                block_size=key_cache.shape[1],
            )
            plan = resolve_prefix_plan(
                shape,
                default_version=fa_version,
                fingerprint=device_fingerprint(query.device.index),
                supported_versions=available_prefix_versions(query.device.index),
                sm_count=multi_processor_count(query.device.index),
                explicit_version=attn_metadata.explicit_fa_version,
                policy=attn_metadata.prefix_policy,
                min_chain_us=attn_metadata.prefix_min_chain_us,
                calibrations=attn_metadata.prefix_calibrations,
                batch_invariant=envs.VLLM_BATCH_INVARIANT,
            )
            if plan is not None and plan.segments > 1:
                # A segment layout may never address pages the graph did not
                # allocate; drop the plan instead of raising inside capture.
                pages = -(-attn_metadata.prefix_max_seq_len // key_cache.shape[1])
                chunk = max(1, -(-pages // plan.segments))
                columns = attn_metadata.prefix_block_table.shape[1]
                if chunk * plan.segments > columns:
                    plan = None
    scheduler = attn_metadata.prefix_scheduler_metadata
    splits = _num_splits(attn_metadata.max_num_splits)
    if plan is not None:
        if not plain:
            raise ValueError("Prefix plans require plain paged FP16/BF16 attention")
        if envs.VLLM_BATCH_INVARIANT:
            from vllm_gr.v1.attention.backends.beam_prefix_policy import PrefixPlan

            plan = PrefixPlan(fa_version, 1)
        fa_version, splits = plan.fa_version, plan.num_splits
        # A schedule belongs to one backend/split configuration. Never reuse
        # an AOT schedule built for the layer's original plan.
        scheduler = None
        if plan.segments > 1:
            from vllm_gr.v1.attention.backends.beam_prefix_segmented import (
                segmented_prefix,
            )

            return segmented_prefix(
                query,
                key_cache,
                value_cache,
                cu_query=attn_metadata.prefix_cu_seqlens_q_device,
                lengths=attn_metadata.prefix_seqlens_kv_device,
                block_table=attn_metadata.prefix_block_table,
                max_query=attn_metadata.prefix_max_q_len,
                max_length=attn_metadata.prefix_max_seq_len,
                scale=scale,
                fa_version=fa_version,
                segments=plan.segments,
                flash=flash_attn_varlen_func,
            )
    out, lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=attn_metadata.prefix_cu_seqlens_q_device,
        seqused_k=attn_metadata.prefix_seqlens_kv_device,
        max_seqlen_q=attn_metadata.prefix_max_q_len,
        max_seqlen_k=attn_metadata.prefix_max_seq_len,
        softmax_scale=scale,
        causal=False,
        alibi_slopes=alibi_slopes,
        window_size=sliding_window_size,
        block_table=attn_metadata.prefix_block_table,
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        fa_version=fa_version,
        scheduler_metadata=scheduler,
        num_splits=splits,
        s_aux=s_aux,
    )

    state = normalize_flash_attn_state(
        out,
        lse,
        fa_version=fa_version,
        num_kv_heads=key_cache.shape[-2],
        max_query_len=attn_metadata.prefix_max_q_len,
        paged_kv=attn_metadata.prefix_block_table is not None,
    )
    if attn_metadata.prefix_block_table is not None:
        state = clear_empty_prefix_states(
            state,
            attn_metadata.prefix_cu_seqlens_q_device,
            attn_metadata.prefix_seqlens_kv_device,
        )
    return state


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    beam_pool: object = None,  # Per-layer pool, passed explicitly by forward()
    prefill_flash_attn_version: int | None = None,
) -> None:
    if attn_metadata.is_beam_decode or attn_metadata.prefix_indices is not None:
        beam_session = require_beam_decode_session(attn_metadata)
        if attn_metadata.prefix_indices is not None:
            pool = resolve_beam_pool(attn_metadata, beam_pool)

    if attn_metadata.std_indices is not None:
        if attn_metadata.prefix_indices is None:
            std_query = query[: attn_metadata.num_std_indices]
        else:
            std_query = query[attn_metadata.std_indices]
        # Standard rows may share a dispatch with Beam decode rows. Apply the
        # override only to the standard multi-token path, never prefix/suffix.
        std_fa_version = (
            prefill_flash_attn_version
            if prefill_flash_attn_version is not None and attn_metadata.std_max_q_len > 1
            else vllm_flash_attn_version
        )
        std_out = flash_attn_varlen_func(
            q=std_query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.std_cu_seqlens_q_device,
            max_seqlen_q=attn_metadata.std_max_q_len,
            seqused_k=attn_metadata.std_seqlens_kv_device,
            max_seqlen_k=attn_metadata.std_max_seq_len,
            softmax_scale=scale,
            causal=attn_metadata.causal,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.std_block_table,
            softcap=logits_soft_cap,
            fa_version=std_fa_version,
            num_splits=_num_splits(0),
        )
        if attn_metadata.prefix_indices is None:
            output[: attn_metadata.num_std_indices] = std_out
        else:
            output[attn_metadata.std_indices] = std_out

    if attn_metadata.prefix_indices is not None:
        if attn_metadata.std_indices is None:
            beam_query = query[: attn_metadata.num_prefix_indices]
        else:
            beam_query = query[attn_metadata.prefix_indices]

        prefix_out, prefix_lse = prefix_attention(
            beam_query,
            key_cache,
            value_cache,
            attn_metadata,
            scale=scale,
            fa_version=vllm_flash_attn_version,
            sliding_window_size=sliding_window_size,
            logits_soft_cap=logits_soft_cap,
            alibi_slopes=alibi_slopes,
            s_aux=s_aux,
        )

        slot = beam_session.slot
        H = pool.num_kv_heads
        D = pool.head_dim
        W = beam_session.beam_width
        max_steps = pool.max_decode_steps
        # Pure incremental short decode: consume the pool directly and
        # merge into the final output, avoiding packed KV and suffix FA2.
        decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
        if getattr(
            beam_session.context, "enable_short_suffix_attention", False
        ) and _is_pure_beam_decode(attn_metadata, W):
            from vllm_gr.v1.attention.backends.beam_short_suffix import (
                merge_short_suffix_attention,
            )

            merge_short_suffix_attention(
                output[:W],
                beam_query,
                prefix_out,
                prefix_lse,
                pool.unshared_key,
                pool.unshared_value,
                slot,
                decode_step_buf,
                scale,
                getattr(attn_metadata, "beam_active_mask", None),
            )
            return

        suffix_k_buf = beam_session.suffix_k_buf
        suffix_v_buf = beam_session.suffix_v_buf

        # Copy the pool's valid suffix KV into the packed suffix buffer.
        #
        # Graph-compatible path: a Triton kernel reads decode_step from
        # the device buffer and computes the packed destination offsets at
        # runtime.  The eager path below uses host-side Python slices
        # (`: step + 1`), which CUDA Graph capture would bake in — a
        # graph captured at step=1 would forever copy only 2 KV rows.
        decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
        if decode_step_buf is not None and decode_step_buf.is_cuda:
            from vllm_gr.v1.attention.backends.beam_graph_kernels import (
                copy_suffix_kv_graph,
            )

            copy_suffix_kv_graph(
                pool.unshared_key,
                pool.unshared_value,
                suffix_k_buf,
                suffix_v_buf,
                slot,
                decode_step_buf,
                beam_width=W,
                active_mask=getattr(attn_metadata, "beam_active_mask", None),
            )
        else:
            # Eager fallback: host-int slice copy (packed layout)
            step = beam_session.decode_step  # host int
            pool_k = pool.unshared_key[slot, :W, :, : step + 1, :]
            pool_v = pool.unshared_value[slot, :W, :, : step + 1, :]

            # Copy to suffix_buf in packed layout: [W*(step+1), H, D]
            # Beam 0's KV rows first, then beam 1's, etc.
            total_kv = W * (step + 1)
            suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, H, D)
            suffix_v_buf[:total_kv] = pool_v.permute(0, 2, 1, 3).reshape(-1, H, D)

        # CUDA Graph-compatible: pass the full suffix_buf (fixed shape)
        # instead of a variable-length view. FA2 uses cu_seqlens_k to
        # determine the actual KV boundaries — the padding zeros beyond
        # the packed prefix don't affect the output.
        # max_seqlen_k is set to max_steps (fixed upper bound).
        # Verified: FA2 produces identical output with over-set max_seqlen_k
        # when cu_seqlens_k correctly delimits each batch element's KV.
        suffix_out, suffix_lse = flash_attn_varlen_func(
            q=beam_query,
            k=suffix_k_buf,
            v=suffix_v_buf,
            cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
            cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
            seqused_k=None,
            max_seqlen_q=attn_metadata.suffix_max_q_len,
            max_seqlen_k=max_steps,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=None,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            fa_version=vllm_flash_attn_version,
            num_splits=_num_splits(0),
        )

        suffix_out, suffix_lse = normalize_flash_attn_state(
            suffix_out,
            suffix_lse,
            fa_version=vllm_flash_attn_version,
            num_kv_heads=attn_metadata.num_kv_heads,
            max_query_len=attn_metadata.suffix_max_q_len,
        )

        out = torch.empty_like(beam_query)
        merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)

        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out
