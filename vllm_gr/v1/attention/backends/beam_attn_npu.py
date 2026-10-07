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


def p2d_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    layer_idx: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    vllm_flash_attn_version: int = 2,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    use_manual_suffix: bool = True,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """NPU execution for the P2D (Prefill + 2 Decodes) graph passes.

    Pass 0 (monolithic graph only): writes the scheduled chunk into the paged
    KV cache (reshape_and_cache with the staged slot mapping) and runs FIA
    full-context attention over the paged cache with the per-request chunk
    qlens and full-prompt kvlens (computed + scheduled), so cached prefixes
    are attended correctly.

    Passes 1 & 2: prefix attention over the paged KV cache (block-table FIA)
    plus suffix attention over the beam's own history (1 or 2 chronological
    tokens); the query is always the trailing history token, so the causal
    window is fully open.

    ``use_manual_suffix`` selects the suffix implementation:
      True  — plain torch ops (no mask; capturable without a task group);
      False — FIA-v2 in a task group with the causal tile (cascade parity).
    """
    if attn_metadata.p2d_pass == 0:
        _p2d_prefill_pass0(
            output=output,
            query=query,
            key=key,
            value=value,
            attn_metadata=attn_metadata,
            scale=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            key_cache=key_cache,
            value_cache=value_cache,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return

    pool = attn_metadata.p2d_pool
    num_beams_total = query.shape[0]
    group_size = num_heads // num_kv_heads

    if key_cache is None or value_cache is None:
        raise RuntimeError("NPU P2D prefix attention requires the paged KV cache tensors.")

    # Write this pass's suffix KV into the static pool slot (slot 0 for
    # pass 1, slot 1 for pass 2), mirroring the GPU pool layout.
    step = 0 if attn_metadata.p2d_pass == 1 else 1
    pool.suffix_kv[layer_idx, 0, :num_beams_total, step].copy_(key, non_blocking=True)
    pool.suffix_kv[layer_idx, 1, :num_beams_total, step].copy_(value, non_blocking=True)

    capturing = _GRAPH_CAPTURE_STATE["capturing"]
    prefix_qlen_list = list(range(1, num_beams_total + 1))
    prefix_kvlen_list = attn_metadata.prefix_kvlen_list

    # ----------------------------------------------------------------
    # Prefix attention over the paged KV cache (non-causal, lse).
    # ----------------------------------------------------------------
    prefix_kw = {
        "query": query,
        "key": key_cache.view(*key_cache.shape[:-2], -1),
        "value": value_cache.view(*value_cache.shape[:-2], -1),
        "atten_mask": None,
        "block_table": attn_metadata.block_table,
        "input_layout": "TND",
        "block_size": attn_metadata.block_size,
        "actual_seq_qlen": prefix_qlen_list,
        "actual_seq_kvlen": prefix_kvlen_list,
        "num_key_value_heads": num_kv_heads,
        "num_query_heads": num_heads,
        "softmax_scale": scale,
        "sparse_mode": 0,
    }
    if capturing:
        entry = _GRAPH_CAPTURE_STATE["entry"]
        tasks = _p2d_task_list(entry, attn_metadata.p2d_pass)
        prefix_out = torch.empty(
            (num_beams_total, num_heads, head_size),
            dtype=query.dtype,
            device=query.device,
        )
        prefix_lse = torch.empty(
            (num_beams_total, num_heads, 1),
            dtype=torch.float32,
            device=query.device,
        )
        workspace = _get_p2d_task_workspace(entry, prefix_kw, kind="prefix")
        stream = torch.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            **prefix_kw, workspace=workspace, out=[prefix_out, prefix_lse]
        )
        handle = torch.npu.graph_task_group_end(stream)
        tasks.append(
            NPUDecodeLayerTask(
                handle=handle,
                kind="prefix",
                p2d_pass=attn_metadata.p2d_pass,
                output=_make_weak_graph_tensor(prefix_out),
                query=_make_weak_graph_tensor(query),
                key=_make_weak_graph_tensor(prefix_kw["key"]),
                value=_make_weak_graph_tensor(prefix_kw["value"]),
                mask=None,
                lse=_make_weak_graph_tensor(prefix_lse),
                workspace=workspace,
                block_size=attn_metadata.block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                scale=scale,
                sparse_mode=0,
                qlen_list=prefix_qlen_list,
                kvlen_list=list(prefix_kvlen_list),
            )
        )
    else:
        prefix_out, prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
            **prefix_kw,
            return_softmax_lse=True,
        )

    # ----------------------------------------------------------------
    # Suffix attention over the beam's own history (open causal window).
    # ----------------------------------------------------------------
    if use_manual_suffix:
        suffix_out, suffix_lse = _p2d_suffix_manual(
            pool=pool,
            layer_idx=layer_idx,
            query=query,
            scale=scale,
            p2d_pass=attn_metadata.p2d_pass,
            num_beams_total=num_beams_total,
            group_size=group_size,
        )
    else:
        suffix_out, suffix_lse = _p2d_suffix_fia(
            attn_metadata=attn_metadata,
            pool=pool,
            layer_idx=layer_idx,
            query=query,
            scale=scale,
            p2d_pass=attn_metadata.p2d_pass,
            num_beams_total=num_beams_total,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            capturing=capturing,
        )

    merged, _ = merge_tnd_attention_outputs(prefix_out, prefix_lse, suffix_out, suffix_lse)
    output[:num_beams_total] = merged


def _p2d_prefill_pass0(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    """Monolithic pass 0: chunk KV write + paged full-context prefill attention.

    Writes the scheduled chunk into the paged KV cache (staged slot mapping)
    and runs FIA causal attention over the paged cache with per-request chunk
    qlens and full-prompt kvlens (computed + scheduled), so cached prefixes
    are attended correctly. In graph mode the FIA call is task-grouped with
    kind "prefill_pass0" so ``update_monolithic_graph_params`` can rebind the
    host lists + block table per replay.
    """
    if key_cache is None or value_cache is None:
        raise RuntimeError("NPU P2D pass 0 requires the paged KV cache tensors.")
    if attn_metadata.slot_mapping is not None:
        reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=attn_metadata.slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    pass0_qlen_list = attn_metadata.pass0_qlen_list
    pass0_kvlen_list = attn_metadata.pass0_kvlen_list
    if not pass0_qlen_list or not pass0_kvlen_list:
        raise RuntimeError("NPU P2D pass 0 requires the host qlen/kvlen lists.")

    kw = {
        "query": query,
        "key": key_cache.view(*key_cache.shape[:-2], -1),
        "value": value_cache.view(*value_cache.shape[:-2], -1),
        "atten_mask": attn_metadata.npu_causal_atten_mask,
        "block_table": attn_metadata.block_table,
        "input_layout": "TND",
        "block_size": attn_metadata.block_size,
        "actual_seq_qlen": pass0_qlen_list,
        "actual_seq_kvlen": pass0_kvlen_list,
        "num_key_value_heads": num_kv_heads,
        "num_query_heads": num_heads,
        "softmax_scale": scale,
        "sparse_mode": 3,
    }
    num_tokens = query.shape[0]

    if _GRAPH_CAPTURE_STATE["capturing"]:
        entry = _GRAPH_CAPTURE_STATE["entry"]
        tasks = _p2d_task_list(entry, 0)
        lse = torch.empty(
            (num_tokens, num_heads, 1),
            dtype=torch.float32,
            device=query.device,
        )
        workspace = _get_p2d_task_workspace(entry, kw, kind="prefill_pass0")
        stream = torch.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            **kw, workspace=workspace, out=[output, lse]
        )
        handle = torch.npu.graph_task_group_end(stream)
        causal_mask = attn_metadata.npu_causal_atten_mask
        tasks.append(
            NPUDecodeLayerTask(
                handle=handle,
                kind="prefill_pass0",
                p2d_pass=0,
                output=_make_weak_graph_tensor(output),
                query=_make_weak_graph_tensor(query),
                key=_make_weak_graph_tensor(kw["key"]),
                value=_make_weak_graph_tensor(kw["value"]),
                mask=_make_weak_graph_tensor(causal_mask) if causal_mask is not None else None,
                lse=_make_weak_graph_tensor(lse),
                workspace=workspace,
                block_size=attn_metadata.block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                scale=scale,
                sparse_mode=3,
                qlen_list=list(pass0_qlen_list),
                kvlen_list=list(pass0_kvlen_list),
            )
        )
        return

    std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(**kw)
    output[: std_out.shape[0]] = std_out


def _p2d_task_list(entry: Any, p2d_pass: int) -> list:
    """Returns the task registry list for a P2D pass under the active capture."""
    if isinstance(entry, NPUMonolithicGraphEntry):
        return entry.prefill_tasks if p2d_pass == 0 else entry.decode_tasks
    if isinstance(entry, NPUDecodeGraphEntry):
        assert p2d_pass != 0, "P2D pass 0 requires a monolithic graph entry"
        return entry.tasks
    raise AssertionError("NPU P2D graph capture requires an active decode/monolithic entry")


def _p2d_suffix_manual(
    pool: Any,
    layer_idx: int,
    query: torch.Tensor,
    scale: float,
    p2d_pass: int,
    num_beams_total: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual suffix attention with plain torch ops (no mask, fully open).

    The suffix query is always the trailing token of the beam history and the
    keys are the history in chronological order, so the causal window covers
    every suffix token: a mask would be a no-op (equivalent to the GPU
    ``causal=True`` call under the end-aligned varlen convention).
    """
    if p2d_pass == 1:
        # Single key per beam: attention output is the value itself and the
        # lse is the (scaled) score — no softmax needed.
        k1 = pool.suffix_kv[layer_idx, 0, :num_beams_total, 0]
        v1 = pool.suffix_kv[layer_idx, 1, :num_beams_total, 0]
        if group_size > 1:
            k1 = k1.repeat_interleave(group_size, dim=-2)
            v1 = v1.repeat_interleave(group_size, dim=-2)
        scores = (query.float() * k1.float()).sum(-1, keepdim=True) * scale
        return v1, scores

    # Pass 2: two chronological keys per beam [t0, t1]; query is t1.
    k2 = pool.suffix_kv[layer_idx, 0, :num_beams_total, 0:2]
    v2 = pool.suffix_kv[layer_idx, 1, :num_beams_total, 0:2]
    if group_size > 1:
        k2 = k2.repeat_interleave(group_size, dim=-2)
        v2 = v2.repeat_interleave(group_size, dim=-2)
    scores = torch.einsum("whd,wshd->whs", query.float(), k2.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    suffix_out = torch.einsum("whs,wshd->whd", weights, v2.float()).to(query.dtype)
    suffix_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return suffix_out, suffix_lse


def _p2d_suffix_fia(
    attn_metadata: Any,
    pool: Any,
    layer_idx: int,
    query: torch.Tensor,
    scale: float,
    p2d_pass: int,
    num_beams_total: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    capturing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suffix attention via FIA-v2 (cascade parity; used when the manual
    implementation is disabled via ``use_manual_p2d_suffix``)."""
    if p2d_pass == 1:
        k = pool.suffix_kv[layer_idx, 0, :num_beams_total, 0]
        v = pool.suffix_kv[layer_idx, 1, :num_beams_total, 0]
        kvlen_list = list(range(1, num_beams_total + 1))
    else:
        k = pool.suffix_kv[layer_idx, 0, :num_beams_total, 0:2].reshape(-1, num_kv_heads, head_size)
        v = pool.suffix_kv[layer_idx, 1, :num_beams_total, 0:2].reshape(-1, num_kv_heads, head_size)
        kvlen_list = list(range(2, 2 * num_beams_total + 1, 2))
    qlen_list = list(range(1, num_beams_total + 1))

    kw = {
        "query": query,
        "key": k,
        "value": v,
        "atten_mask": attn_metadata.npu_causal_atten_mask,
        "block_table": None,
        "block_size": 0,
        "input_layout": "TND",
        "actual_seq_qlen": qlen_list,
        "actual_seq_kvlen": kvlen_list,
        "num_key_value_heads": num_kv_heads,
        "num_query_heads": num_heads,
        "softmax_scale": scale,
        "sparse_mode": 3,
        "inner_precise": 0,
    }
    if capturing:
        entry = _GRAPH_CAPTURE_STATE["entry"]
        tasks = _p2d_task_list(entry, p2d_pass)
        suffix_out = torch.empty(
            (num_beams_total, num_heads, head_size),
            dtype=query.dtype,
            device=query.device,
        )
        suffix_lse = torch.empty(
            (num_beams_total, num_heads, 1),
            dtype=torch.float32,
            device=query.device,
        )
        workspace = _get_p2d_task_workspace(entry, kw, kind="suffix")
        stream = torch.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            **kw, workspace=workspace, out=[suffix_out, suffix_lse]
        )
        handle = torch.npu.graph_task_group_end(stream)
        causal_mask = attn_metadata.npu_causal_atten_mask
        tasks.append(
            NPUDecodeLayerTask(
                handle=handle,
                kind="suffix_fia",
                p2d_pass=p2d_pass,
                output=_make_weak_graph_tensor(suffix_out),
                query=_make_weak_graph_tensor(query),
                key=_make_weak_graph_tensor(k),
                value=_make_weak_graph_tensor(v),
                mask=_make_weak_graph_tensor(causal_mask) if causal_mask is not None else None,
                lse=_make_weak_graph_tensor(suffix_lse),
                workspace=workspace,
                block_size=0,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                scale=scale,
                sparse_mode=3,
                qlen_list=qlen_list,
                kvlen_list=kvlen_list,
            )
        )
        return suffix_out, suffix_lse
    return torch_npu.npu_fused_infer_attention_score_v2(
        **kw,
        return_softmax_lse=True,
    )


def _get_p2d_task_workspace(
    entry: Any,
    fia_kw: dict,
    kind: str,
) -> Any:
    """Returns (sizing on first use) the per-kind FIA workspace for the P2D graph.

    The monolithic entry keeps separate workspaces for the pass-0 prefill
    shape (bucket queries x max_kvlen) and the decode-prefix shape (W queries
    x max_kvlen); the plain decode entry keeps prefix/suffix workspaces.
    """
    if isinstance(entry, NPUMonolithicGraphEntry):
        if kind == "prefill_pass0":
            if entry.prefill_workspace is None:
                entry.prefill_workspace = _size_fia_workspace(entry, fia_kw, kind="prefill_pass0")
            return entry.prefill_workspace
        if kind == "prefix":
            if entry.decode_prefix_workspace is None:
                entry.decode_prefix_workspace = _size_fia_workspace(entry, fia_kw, kind="prefix")
            return entry.decode_prefix_workspace
        if entry.suffix_workspace is None:
            entry.suffix_workspace = _size_fia_workspace(entry, fia_kw, kind="suffix")
        return entry.suffix_workspace
    if isinstance(entry, NPUDecodeGraphEntry):
        if kind == "prefix":
            if entry.prefix_workspace is None:
                entry.prefix_workspace = _size_fia_workspace(entry, fia_kw, kind="prefix")
            return entry.prefix_workspace
        if entry.suffix_workspace is None:
            entry.suffix_workspace = _size_fia_workspace(entry, fia_kw, kind="suffix")
        return entry.suffix_workspace
    raise AssertionError("NPU P2D graph capture requires an active decode/monolithic entry")


def _size_fia_workspace(
    entry: Any,
    fia_kw: dict,
    kind: str,
) -> Any:
    """Sizes the FIA workspace for the given task kind.

    Prefix/prefill workspaces are sized for the maximum supported per-request
    KV length (the model's max context): replays rebind the per-row kvlens and
    cached/chunked contexts can exceed the capture-time dummy lengths.
    """
    ws_kw = dict(fia_kw)
    max_kvlen = entry.max_kvlen or max(ws_kw["actual_seq_kvlen"])
    if kind == "prefix":
        num_rows = ws_kw["query"].shape[0]
        ws_kw["actual_seq_qlen"] = list(range(1, num_rows + 1))
        ws_kw["actual_seq_kvlen"] = [max_kvlen] * num_rows
    elif kind == "prefill_pass0":
        num_rows = len(ws_kw["actual_seq_qlen"])
        ws_kw["actual_seq_kvlen"] = [max_kvlen] * num_rows
    workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**ws_kw)
    if workspace is None:
        raise RuntimeError(f"FIA-v2 workspace sizing failed during P2D {kind} capture")
    return workspace


@dataclass
class NPUDecodeLayerTask:
    """Per-layer FIA task-group state captured inside an NPU P2D decode graph.

    ``kind`` is "prefix" (rebindable block table + prompt kvlens) or
    "suffix_fia" (static lists, no block table). The update dispatches the
    right parameter set per task.
    """

    handle: Any
    kind: str
    p2d_pass: int
    output: Any
    query: Any
    key: Any
    value: Any
    mask: Any
    lse: Any
    workspace: Any
    block_size: int
    num_heads: int
    num_kv_heads: int
    scale: float
    sparse_mode: int
    qlen_list: list
    kvlen_list: list


@dataclass
class NPUDecodeGraphEntry:
    """Registry entry for one captured P2D decode graph (one per (num_reqs, beam_width))."""

    max_kvlen: int = 0
    prefix_workspace: Any = None
    suffix_workspace: Any = None
    tasks: list[NPUDecodeLayerTask] = field(default_factory=list)


@contextmanager
def decode_graph_capture(entry: NPUDecodeGraphEntry):
    """Marks the P2D decode fast path as graph-capturing and registers layer tasks."""
    assert not _GRAPH_CAPTURE_STATE["capturing"], "nested NPU graph capture"
    prev = (_GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"])
    _GRAPH_CAPTURE_STATE["capturing"] = True
    _GRAPH_CAPTURE_STATE["entry"] = entry
    try:
        yield
    finally:
        _GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"] = prev


def _update_p2d_task(
    update_stream: Any,
    task: NPUDecodeLayerTask,
    qlen_list: list,
    kvlen_list: list,
    task_block_table: torch.Tensor | None,
    atten_mask: Any,
    block_size: int,
    sparse_mode: int,
    extra_kw: dict[str, Any] | None = None,
) -> bool:
    """Rebinds one P2D FIA task group with fresh params. False on missing tensors."""
    output = task.output
    query = task.query
    key = task.key
    value = task.value
    lse = task.lse
    if output is None or query is None or key is None or value is None or lse is None:
        logger.warning(
            "NPU P2D graph update: layer tensors are missing; "
            "aborting the update (the replay would execute with stale params)."
        )
        return False
    torch.npu.graph_task_update_begin(update_stream, task.handle)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        query=query,
        key=key,
        value=value,
        atten_mask=atten_mask,
        block_table=task_block_table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_qlen=qlen_list,
        actual_seq_kvlen=kvlen_list,
        num_key_value_heads=task.num_kv_heads,
        num_query_heads=task.num_heads,
        softmax_scale=task.scale,
        sparse_mode=sparse_mode,
        workspace=task.workspace,
        out=[output, lse],
        **(extra_kw or {}),
    )
    torch.npu.graph_task_update_end(update_stream)
    return True


def update_decode_graph_params(
    update_stream: Any,
    entry: NPUDecodeGraphEntry,
    prefix_qlen_list: list,
    prefix_kvlen_list: list,
    block_table: torch.Tensor,
) -> bool:
    """Rewrites the decode graph's FIA task-group params before a replay.

    Prefix tasks are rebound with the fresh per-beam qlen/kvlen host lists and
    the current (beam-replicated) block table; suffix FIA tasks are re-sent
    with their capture-time static lists. Tasks in manual-suffix graphs are
    prefix-only, so the update stays single-kind there.

    Returns True on success; the caller MUST NOT replay on False.
    """
    if not isinstance(entry, NPUDecodeGraphEntry):
        logger.debug("update_decode_graph_params: non-decode-graph entry; skipping.")
        return False
    serving_stream = torch.npu.current_stream()
    with torch.npu.stream(update_stream):
        update_stream.wait_stream(serving_stream)
        for task in entry.tasks:
            if task.kind == "prefix":
                ok = _update_p2d_task(
                    update_stream,
                    task,
                    prefix_qlen_list,
                    prefix_kvlen_list,
                    block_table,
                    None,
                    task.block_size,
                    0,
                )
            else:
                ok = _update_p2d_task(
                    update_stream,
                    task,
                    task.qlen_list,
                    task.kvlen_list,
                    None,
                    task.mask,
                    0,
                    3,
                    extra_kw={"inner_precise": 0},
                )
            if not ok:
                return False
    return True


@dataclass
class NPUMonolithicGraphEntry:
    """Registry entry for one captured monolithic P2D graph
    (one per (bucket, num_reqs, beam_width)).

    Pass 0 tasks live in ``prefill_tasks`` (kind "prefill_pass0"); passes 1/2
    tasks live in ``decode_tasks`` (kinds "prefix"/"suffix_fia"). Three FIA
    workspaces: the pass-0 prefill shape (bucket queries x max_kvlen), the
    decode-prefix shape (W queries x max_kvlen), and the suffix shape.
    """

    bucket: int = 0
    num_reqs: int = 1
    beam_width: int = 128
    max_kvlen: int = 0
    prefill_workspace: Any = None
    decode_prefix_workspace: Any = None
    suffix_workspace: Any = None
    prefill_tasks: list[NPUDecodeLayerTask] = field(default_factory=list)
    decode_tasks: list[NPUDecodeLayerTask] = field(default_factory=list)


@contextmanager
def monolithic_graph_capture(entry: NPUMonolithicGraphEntry):
    """Marks the P2D monolithic fast path as graph-capturing and registers tasks."""
    assert not _GRAPH_CAPTURE_STATE["capturing"], "nested NPU graph capture"
    prev = (_GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"])
    _GRAPH_CAPTURE_STATE["capturing"] = True
    _GRAPH_CAPTURE_STATE["entry"] = entry
    try:
        yield
    finally:
        _GRAPH_CAPTURE_STATE["capturing"], _GRAPH_CAPTURE_STATE["entry"] = prev


def update_monolithic_graph_params(
    update_stream: Any,
    entry: NPUMonolithicGraphEntry,
    pass0_qlen_list: list,
    pass0_kvlen_list: list,
    prefix_qlen_list: list,
    prefix_kvlen_list: list,
    pass0_block_table: torch.Tensor,
    prefix_block_table: torch.Tensor,
) -> bool:
    """Rewrites the monolithic graph's FIA task-group params before a replay.

    Pass-0 tasks are rebound with the per-request chunk qlens and full-prompt
    kvlens (computed + scheduled) plus the request-row staged block table;
    decode-prefix tasks with the per-beam qlen/kvlen lists plus the
    beam-replicated staged block table; suffix FIA tasks are re-sent with
    their capture-time static lists.

    Returns True on success; the caller MUST NOT replay on False.
    """
    if not isinstance(entry, NPUMonolithicGraphEntry):
        logger.debug("update_monolithic_graph_params: non-monolithic entry; skipping.")
        return False
    serving_stream = torch.npu.current_stream()
    with torch.npu.stream(update_stream):
        update_stream.wait_stream(serving_stream)
        for task in list(entry.prefill_tasks) + list(entry.decode_tasks):
            if task.kind == "prefill_pass0":
                ok = _update_p2d_task(
                    update_stream,
                    task,
                    pass0_qlen_list,
                    pass0_kvlen_list,
                    pass0_block_table,
                    task.mask,
                    task.block_size,
                    3,
                )
            elif task.kind == "prefix":
                ok = _update_p2d_task(
                    update_stream,
                    task,
                    prefix_qlen_list,
                    prefix_kvlen_list,
                    prefix_block_table,
                    None,
                    task.block_size,
                    0,
                )
            else:
                ok = _update_p2d_task(
                    update_stream,
                    task,
                    task.qlen_list,
                    task.kvlen_list,
                    None,
                    task.mask,
                    0,
                    3,
                    extra_kw={"inner_precise": 0},
                )
            if not ok:
                return False
    return True
