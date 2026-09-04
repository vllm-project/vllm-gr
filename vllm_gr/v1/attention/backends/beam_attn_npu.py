# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU execution primitives for the Beam Attention backend."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch_npu
from vllm.logger import init_logger

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
)
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
    merge_tnd_attention_outputs,
)
from vllm_gr.v1.worker.prefill_graph_common import SUPPORTED_NUM_REQS

logger = init_logger(__name__)


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: object = None,  # unused on cascade; xllm-only contract
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


def is_available() -> bool:
    return True


def get_flash_attn_version(
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
) -> int:
    del requires_alibi, head_size
    return 0


def supports_sinks() -> bool:
    return False


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
    del kv_cache_dtype, k_scale, v_scale
    torch_npu._npu_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )


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
) -> None:
    if attn_metadata.std_indices is not None:
        if attn_metadata.prefix_indices is None:
            std_query = query[: attn_metadata.num_std_indices]
        else:
            std_query = query[attn_metadata.std_indices]
        std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=std_query,
            key=key_cache.view(*key_cache.shape[:-2], -1),
            value=value_cache.view(*value_cache.shape[:-2], -1),
            atten_mask=attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
            block_table=attn_metadata.std_block_table,
            input_layout="TND",
            block_size=attn_metadata.block_size,
            actual_seq_qlen=attn_metadata.std_cu_seqlens_q_cpu,
            actual_seq_kvlen=attn_metadata.std_seqlens_kv_cpu,
            num_key_value_heads=attn_metadata.num_kv_heads,
            num_query_heads=attn_metadata.num_heads,
            softmax_scale=scale,
            sparse_mode=3 if attn_metadata.causal else 0,
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
        prefix_out, prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
            query=beam_query,
            key=key_cache.view(*key_cache.shape[:-2], -1),
            value=value_cache.view(*value_cache.shape[:-2], -1),
            block_table=attn_metadata.prefix_block_table,
            input_layout="TND",
            atten_mask=None,
            block_size=attn_metadata.block_size,
            actual_seq_qlen=attn_metadata.prefix_cu_seqlens_q_cpu,
            actual_seq_kvlen=attn_metadata.prefix_seqlens_kv_cpu,
            num_key_value_heads=attn_metadata.num_kv_heads,
            num_query_heads=attn_metadata.num_heads,
            softmax_scale=scale,
            return_softmax_lse=True,
            sparse_mode=0,
        )

        # --- Incremental decode path: read suffix KV from pool slice ---
        beam_session = getattr(attn_metadata, "beam_session", None)
        is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

        if is_beam_decode and beam_session is not None and beam_session.is_active:
            # Incremental decode: read suffix KV from pool slice
            # instead of extract_suffix_kv_pytorch from paged cache.
            # Prefer the explicitly-passed beam_pool; fall back to metadata
            # for backward compatibility with callers that still set
            # beam_layer_idx on the metadata.
            if beam_pool is not None:
                pool = beam_pool
            else:
                layer_idx = getattr(attn_metadata, "beam_layer_idx", 0)
                pool = beam_session.pools[layer_idx]
            step = beam_session.decode_step  # host int, no .item()
            slot = beam_session.slot
            H = pool.num_kv_heads
            D = pool.head_dim
            W = beam_session.beam_width

            # Read pool slice: [W, H, step+1, D]
            pool_k = pool.unshared_key[slot, :W, :, : step + 1, :]
            pool_v = pool.unshared_value[slot, :W, :, : step + 1, :]

            # Copy to pre-allocated fixed buffer (avoiding new allocation)
            # Reshape: [W, H, step+1, D] -> [W*(step+1), H, D]
            total_kv = W * (step + 1)
            suffix_k_buf = beam_session.suffix_k_buf
            suffix_v_buf = beam_session.suffix_v_buf
            suffix_k_buf[:total_kv] = pool_k.permute(0, 2, 1, 3).reshape(-1, H, D)
            suffix_v_buf[:total_kv] = pool_v.permute(0, 2, 1, 3).reshape(-1, H, D)

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                beam_query,
                suffix_k_buf[:total_kv],
                suffix_v_buf[:total_kv],
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.suffix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.suffix_cu_seqlens_kv_cpu,
                softmax_scale=scale,
                atten_mask=attn_metadata.npu_causal_atten_mask,
                block_table=None,
                block_size=0,
                sparse_mode=3,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=True,
            )
        else:
            # Standard (recompute) path: extract_suffix_kv from paged cache
            contig_k, contig_v = extract_suffix_kv_pytorch(
                key_cache,
                value_cache,
                attn_metadata.suffix_slot_mapping_out,
            )

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                beam_query,
                contig_k,
                contig_v,
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.suffix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.suffix_cu_seqlens_kv_cpu,
                softmax_scale=scale,
                atten_mask=attn_metadata.npu_causal_atten_mask,
                block_table=None,
                block_size=0,
                sparse_mode=3,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=True,
            )

        out, _ = merge_tnd_attention_outputs(
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
        )
        if attn_metadata.std_indices is None:
            output[: attn_metadata.num_prefix_indices] = out
        else:
            output[attn_metadata.prefix_indices] = out


# --- Pure-prefill fast path (prefill graph capture/replay) ---

# Module-level state for the standalone NPU prefill-graph runner. The runner
# sets this around its own ``torch.npu.graph`` capture so the prefill fast
# path can wrap the FIA call in a graph task group (vllm-ascend's
# full-graph recipe: static ops captured plain, FIA in a task group, host
# seq-lens lists pre-built outside the capture).
_GRAPH_CAPTURE_STATE: dict[str, Any] = {"capturing": False, "entry": None}

_WEAK_TENSOR_UNAVAILABLE_WARNED = False


def _make_weak_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Weak-tensor alias sharing the data without pinning the original.

    The captured tensors are stored as weak tensors (``torch_npu._C._weak_ref_tensor``,
    vllm-ascend's weak-tensor method): the task-update only needs their
    ADDRESSES — the replayed graph recomputes all content in-graph — while the
    pool accounting can free the originals, collapsing the per-bucket memory.

    Falls back to the tensor itself (strong) when the Ascend API is missing.
    """
    global _WEAK_TENSOR_UNAVAILABLE_WARNED
    try:
        return torch_npu._C._weak_ref_tensor(tensor)
    except AttributeError:
        if not _WEAK_TENSOR_UNAVAILABLE_WARNED:
            _WEAK_TENSOR_UNAVAILABLE_WARNED = True
            logger.warning(
                "torch_npu._C._weak_ref_tensor is unavailable on this stack; "
                "falling back to strong tensor references."
            )
        return tensor


@dataclass
class NPUGraphLayerTask:
    """Per-layer FIA task-group state captured inside an NPU prefill graph.

    Tensor fields hold WEAK tensors by default: the task-update only binds
    their addresses and the replayed graph recomputes all content in-graph,
    so the originals need not be pinned. Strong refs are the automatic
    fallback only when the Ascend weak-tensor API is unavailable.
    """

    handle: Any
    output: Any
    query: Any
    key: Any
    value: Any
    mask: Any
    lse: Any
    block_size: int
    num_heads: int
    num_kv_heads: int
    scale: float
    sparse_mode: int


@dataclass
class NPUPrefillGraphEntry:
    """Registry entry for one captured prefill graph (one per bucket).

    The FIA workspace is shared by every layer of the bucket (vllm-ascend
    convention: one workspace per num_tokens) because layers execute
    sequentially both in-graph and on the update stream. The runner may also
    bind a cross-bucket shared workspace here (largest-first capture).

    ``max_kvlen`` is the maximum supported per-request KV length (the model's
    max context): the workspace is sized for it rather than for the dummy
    capture-time lengths, so replay updates with large cached/chunked-prefill
    contexts stay within the reserved budget.
    """

    bucket: int
    workspace: Any = None
    max_kvlen: int = 0
    layers: list[NPUGraphLayerTask] = field(default_factory=list)


@contextmanager
def prefill_graph_capture(entry: NPUPrefillGraphEntry):
    """Marks the prefill fast path as graph-capturing and registers layer tasks."""
    prev = (_GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"])
    _GRAPH_CAPTURE_STATE["capturing"] = True
    _GRAPH_CAPTURE_STATE["entry"] = entry
    try:
        yield
    finally:
        _GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"] = prev


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
) -> None:
    """Dedicated pure prefill attention execution for NPU without beam branching.

    Consumes the common-fields fast-path metadata (query_start_loc, seq_lens,
    block_table): for pure prefill batches all the grouped std_* fields are
    None, so they must not be consulted here.
    """
    del sliding_window_size, logits_soft_cap, vllm_flash_attn_version, s_aux, alibi_slopes
    num_reqs = attn_metadata.num_reqs
    if num_reqs <= 0:
        return
    if attn_metadata.is_prefill_graph_exec and _GRAPH_CAPTURE_STATE["capturing"]:
        _prefill_attention_graph(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            scale=scale,
        )
        return
    qsl = attn_metadata.query_start_loc_cpu
    sl = attn_metadata.seq_lens_cpu
    num_actual_tokens = attn_metadata.num_actual_tokens
    # vllm-ascend pads the batch to num_tokens_padded; the FIA-v2 TND layout
    # requires queryT == last(actual_seq_qlen), so compute only the real rows.
    query = query[:num_actual_tokens]
    # Skip the leading zero of the cu-style query_start_loc (matches the
    # std_cu_seqlens_q_cpu convention: np.cumsum without the leading 0) and
    # trim the zero padding beyond num_reqs.
    actual_seq_qlen = qsl[1 : num_reqs + 1]
    actual_seq_kvlen = sl[:num_reqs]
    std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query=query,
        key=key_cache.view(*key_cache.shape[:-2], -1),
        value=value_cache.view(*value_cache.shape[:-2], -1),
        atten_mask=attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
        block_table=attn_metadata.block_table,
        input_layout="TND",
        block_size=attn_metadata.block_size,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        num_key_value_heads=attn_metadata.num_kv_heads,
        num_query_heads=attn_metadata.num_heads,
        softmax_scale=scale,
        sparse_mode=3 if attn_metadata.causal else 0,
    )
    output[:num_actual_tokens] = std_out


def _fia_v2_graph_kwargs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    scale: float,
    qlen_list: list,
    kvlen_list: list,
) -> dict:
    """Shared FIA-v2 kwargs for the graph capture, workspace sizing, and update."""
    return {
        "query": query,
        "key": key,
        "value": value,
        "atten_mask": attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
        "block_table": attn_metadata.block_table,
        "input_layout": "TND",
        "block_size": attn_metadata.block_size,
        "actual_seq_qlen": qlen_list,
        "actual_seq_kvlen": kvlen_list,
        "num_key_value_heads": attn_metadata.num_kv_heads,
        "num_query_heads": attn_metadata.num_heads,
        "softmax_scale": scale,
        "sparse_mode": 3 if attn_metadata.causal else 0,
    }


def _prefill_attention_graph(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    scale: float,
) -> None:
    """Capture-time prefill attention: FIA-v2 wrapped in a graph task group.

    The seq-lens lists are pre-built in the metadata (host side, outside the
    capture region) and the workspace is sized here. Static ops (including
    reshape_and_cache) stay plain-captured; only the FIA call is task-grouped
    so ``update_graph_params`` can rewrite the host lists per replay.

    No per-layer events: the runner orders the WHOLE update after the
    previous replay's completion and the WHOLE replay after the update's
    completion with stream-level waits (``wait_stream`` pairs), which makes
    per-layer event granularity redundant.
    """
    entry = _GRAPH_CAPTURE_STATE["entry"]
    assert entry is not None, "NPU prefill graph capture requires an active entry"
    num_reqs = attn_metadata.num_reqs
    qsl = attn_metadata.query_start_loc_cpu
    sl = attn_metadata.seq_lens_cpu
    qlen_list = qsl[1 : num_reqs + 1]
    kvlen_list = sl[:num_reqs]
    key = key_cache.view(*key_cache.shape[:-2], -1)
    value = value_cache.view(*value_cache.shape[:-2], -1)
    kw = _fia_v2_graph_kwargs(query, key, value, attn_metadata, scale, qlen_list, kvlen_list)
    if entry.workspace is None:
        # Size the workspace for the maximum supported per-request KV length
        # (the model's max context), not the dummy capture-time lengths: the
        # task-update rebinds the per-request kvlens at every replay, and
        # cached/chunked prefill can carry contexts far beyond the capture
        # dummy. max_kvlen == 0 (unknown) falls back to the capture lengths.
        # The request count is capped at the supported maximum
        # (min(SUPPORTED_NUM_REQS[-1], num_reqs)) for the same reason.
        ws_num_reqs = min(SUPPORTED_NUM_REQS[-1], num_reqs)
        if entry.max_kvlen > 0:
            bucket = qlen_list[-1]
            base_len = bucket // ws_num_reqs
            rem = bucket % ws_num_reqs
            ws_qlen_list = []
            curr = 0
            for i in range(ws_num_reqs):
                curr += base_len + (1 if i < rem else 0)
                ws_qlen_list.append(curr)
            ws_kvlen_list = [entry.max_kvlen] * ws_num_reqs
        else:
            ws_qlen_list = qlen_list
            ws_kvlen_list = kvlen_list
        ws_kw = _fia_v2_graph_kwargs(
            query, key, value, attn_metadata, scale, ws_qlen_list, ws_kvlen_list
        )
        entry.workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**ws_kw)
        if entry.workspace is None:
            raise RuntimeError("FIA-v2 workspace sizing failed during prefill graph capture")
    workspace = entry.workspace
    lse = torch.empty(1, dtype=output.dtype, device=output.device)
    stream = torch.npu.current_stream()
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(**kw, workspace=workspace, out=[output, lse])
    handle = torch.npu.graph_task_group_end(stream)
    mask = (
        attn_metadata.npu_causal_atten_mask
        if attn_metadata.npu_causal_atten_mask is not None
        else None
    )
    entry.layers.append(
        NPUGraphLayerTask(
            handle=handle,
            output=_make_weak_graph_tensor(output),
            query=_make_weak_graph_tensor(query),
            key=_make_weak_graph_tensor(key),
            value=_make_weak_graph_tensor(value),
            mask=_make_weak_graph_tensor(mask) if mask is not None else mask,
            lse=_make_weak_graph_tensor(lse),
            block_size=attn_metadata.block_size,
            num_heads=attn_metadata.num_heads,
            num_kv_heads=attn_metadata.num_kv_heads,
            scale=scale,
            sparse_mode=3 if attn_metadata.causal else 0,
        )
    )


def update_graph_params(
    update_stream: Any,
    entry: NPUPrefillGraphEntry,
    qlen_list: list,
    kvlen_list: list,
    block_table: torch.Tensor,
) -> bool:
    """Rewrites the per-layer FIA task-group params before a graph replay.

    Called by the NPU prefill-graph runner with the fresh per-request host
    lists and the current batch's block table. The replayed graph executes
    each task group with these updated params (the update may change the
    request count and re-bind the block table).

    Returns True on success. The caller MUST NOT replay the graph on False:
    a partially-updated graph would execute with stale per-layer params and
    the output would be silently wrong.

    Deliberately ignores non-prefill-graph callers (e.g. vllm-ascend's
    full-graph update hook, which passes different arguments).
    """
    if not isinstance(entry, NPUPrefillGraphEntry):
        logger.debug("update_graph_params: non-prefill-graph entry; skipping.")
        return False
    if entry.workspace is None:
        logger.warning("NPU prefill graph update: entry has no workspace; skipping.")
        return False
    # CANN's RIC task update requires the update stream to be the CURRENT
    # stream (vllm-ascend wraps update_attn_params the same way); issuing the
    # update on a non-current stream fails with 507000.
    serving_stream = torch.npu.current_stream()
    with torch.npu.stream(update_stream):
        # Order this update's device work AFTER the previous replay: without
        # this dependency the update's FIA executions would run on the update
        # stream concurrently with the previous replay on the serving stream,
        # and both write the same layer output buffers (out=[output, lse]) —
        # corrupting the replay's results.
        update_stream.wait_stream(serving_stream)
        for task in entry.layers:
            output = task.output
            query = task.query
            key = task.key
            value = task.value
            lse = task.lse
            if output is None or query is None or key is None or value is None or lse is None:
                logger.warning(
                    "NPU prefill graph update: layer tensors are missing; "
                    "aborting the update (the replay would execute with stale params)."
                )
                return False
            mask = task.mask
            torch.npu.graph_task_update_begin(update_stream, task.handle)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query=query,
                key=key,
                value=value,
                atten_mask=mask,
                block_table=block_table,
                input_layout="TND",
                block_size=task.block_size,
                actual_seq_qlen=qlen_list,
                actual_seq_kvlen=kvlen_list,
                num_key_value_heads=task.num_kv_heads,
                num_query_heads=task.num_heads,
                softmax_scale=task.scale,
                sparse_mode=task.sparse_mode,
                workspace=entry.workspace,
                out=[output, lse],
            )
            torch.npu.graph_task_update_end(update_stream)
    return True
