# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-neutral ModelRunner helpers for grouped Beam execution."""

from __future__ import annotations

from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds

logger = init_logger(__name__)

_WORKER_EXECUTE_PATCH_MARKER = "_vllm_gr_beam_execute_patch"
_WORKER_PREPARE_PATCH_MARKER = "_vllm_gr_beam_prepare_patch"
_WORKER_BOOKKEEPING_PATCH_MARKER = "_vllm_gr_beam_bookkeeping_patch"


def _get_positions_buffers(runner: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        if isinstance(runner.positions, torch.Tensor):
            # vLLM-Ascend 0.22 uses a plain device tensor in PCP paths.
            gpu_positions = runner.positions
        else:
            gpu_positions = getattr(runner.positions, "gpu", None)
            cpu_positions = getattr(runner.positions, "cpu", None)
    if cpu_positions is None:
        # vLLM-Ascend 0.22 keeps a dedicated pinned host buffer even when
        # ``positions`` itself is a plain device tensor.
        cpu_positions = getattr(runner, "_positions_cpu_buf", None)
    return gpu_positions, cpu_positions


def _write_position_segments(
    token_offset: int,
    seg_starts: List[int],
    seg_lens: List[int],
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
) -> None:
    """Write concatenated per-beam position-id segments in one vectorized pass.

    Each active beam contributes ``range(st_b, st_b + len_b)``. The previous
    implementation emitted one ``torch.arange`` per beam (up to ``beam_width``
    GPU kernel launches per call); here we build the whole concatenation on CPU
    with a handful of vectorized ops and a single host->device copy for
    ``gpu_pos``. The result is element-wise identical to the per-beam writes.
    """
    if not seg_lens:
        return
    lens = torch.tensor(seg_lens, dtype=torch.int64)
    total = int(lens.sum().item())
    if total == 0:
        return
    starts = torch.tensor(seg_starts, dtype=torch.int64)
    # Output offset where each segment begins (prefix sum minus own length).
    seg_off = torch.cumsum(lens, dim=0) - lens
    gidx = torch.arange(total, dtype=torch.int64)
    seg_off_rep = torch.repeat_interleave(seg_off, lens)
    starts_rep = torch.repeat_interleave(starts, lens)
    positions = starts_rep + (gidx - seg_off_rep)
    if gpu_pos is not None:
        gpu_pos[token_offset : token_offset + total] = positions
    if cpu_pos is not None:
        cpu_pos[token_offset : token_offset + total] = positions


def _process_beam_decode_request(
    mdata: Dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
    new_logits_indices: List[int],
    is_incremental: bool = False,
) -> Tuple[int, int, int, List[int]]:
    """Traces allocation arrays and mutates position buffers for a speculative request.

    When is_incremental=True, each beam inputs 1 token with position_id = prefix_len + decode_step.
    """
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_width"]
    decode_steps = mdata["beam_decode_steps"]
    cache_len = mdata["cache_len"]
    active_beam_ids = []  # Track structural beam locations

    # --- Incremental decode path ---
    if is_incremental:
        decode_step = mdata.get("decode_step", 0)
        seg_starts: List[int] = []
        seg_lens: List[int] = []

        for b in range(beam_width):
            # Each beam: 1 token, position = prefix_len + decode_step
            seg_starts.append(prefix_len + decode_step)
            seg_lens.append(1)
            new_logits_indices.append(token_offset + b)
            active_beam_ids.append(b)

        # Batched position-id write
        _write_position_segments(token_offset, seg_starts, seg_lens, gpu_pos, cpu_pos)

        return beam_width, beam_width, token_offset + beam_width, active_beam_ids

    # --- Standard (recompute) path ---
    chunk_budget = seq_len
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    b_cache_len = cache_len
    seg_starts = []
    seg_lens = []
    for b in range(beam_width):
        if chunk_budget <= 0:
            break

        b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
            b, prefix_len, b_cache_len, decode_steps, chunk_budget
        )

        if b_uncached > 0:
            st = cache_len if (prefix_len >= b_cache_len) else (prefix_len + b_past_suffix)
            seg_starts.append(st)
            seg_lens.append(b_uncached)

            if b_past_suffix + b_uncached >= decode_steps:
                new_logits_indices.append(curr_offset + b_uncached - 1)
                active_beam_ids.append(b_physical)

                logits_row_cnt += 1

            valid_rows_for_req += 1
            curr_offset += b_uncached
            chunk_budget -= b_uncached
            b_cache_len += b_uncached

    # Batched position-id write: one vectorized build (CPU) + single H2D copy
    # instead of one torch.arange per beam (up to beam_width GPU launches/call).
    _write_position_segments(token_offset, seg_starts, seg_lens, gpu_pos, cpu_pos)

    return valid_rows_for_req, logits_row_cnt, curr_offset, active_beam_ids


def _remux_logprobs_tensors(
    request_row_configs: List[Tuple[Any, int, int]],
    max_chained_dim: int,
    target_K: int,
    orig_lps: torch.Tensor,
    orig_lp_ids: Optional[torch.Tensor],
    orig_ranks: Optional[torch.Tensor],
    st_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Merges and aligns sampler tensor slices sequentially across runtime waves."""
    if max_chained_dim == 0 or orig_lps.numel() == 0:
        return st_ids, orig_lps, orig_lp_ids, orig_ranks

    new_st_ids, new_lps, new_lp_ids, new_ranks = [], [], [], []
    gpu_row_cursor = 0

    for req_id, _, W_logits, *_ in request_row_configs:
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        if W_logits == 0:
            dummy_st = torch.tensor([[-1]], dtype=st_ids.dtype, device=st_ids.device)
            new_st_ids.append(dummy_st)

            new_lps.append(
                torch.full(
                    (1, max_chained_dim),
                    float("-inf"),
                    dtype=orig_lps.dtype,
                    device=orig_lps.device,
                )
            )
            if orig_lp_ids is not None:
                new_lp_ids.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_lp_ids.dtype, device=orig_lp_ids.device
                    )
                )
            if orig_ranks is not None:
                new_ranks.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_ranks.dtype, device=orig_ranks.device
                    )
                )
            continue

        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        new_st_ids.append(st_ids[gpu_row_cursor : gpu_row_cursor + 1, :])

        req_lps = orig_lps[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
            1, W_logits * target_K
        )
        if W_logits * target_K < max_chained_dim:
            req_lps = torch.nn.functional.pad(
                req_lps, (0, max_chained_dim - (W_logits * target_K)), value=float("-inf")
            )
        new_lps.append(req_lps)

        if orig_lp_ids is not None:
            req_lp_ids = orig_lp_ids[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
                1, W_logits * target_K
            )
            if W_logits * target_K < max_chained_dim:
                req_lp_ids = torch.nn.functional.pad(
                    req_lp_ids, (0, max_chained_dim - (W_logits * target_K)), value=0
                )
            new_lp_ids.append(req_lp_ids)

        if orig_ranks is not None:
            req_ranks = orig_ranks[gpu_row_cursor : gpu_row_cursor + W_logits].reshape(1, W_logits)
            if W_logits < max_chained_dim:
                req_ranks = torch.nn.functional.pad(
                    req_ranks, (0, max_chained_dim - W_logits), value=0
                )
            new_ranks.append(req_ranks)

        gpu_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None,
    )


class BeamReqIdsProxy(list):
    def __init__(self, original_list, req_id_map):
        super().__init__(original_list)
        self.req_id_map = req_id_map

    def __getitem__(self, index):
        try:
            return self.req_id_map[index]
        except IndexError as e:
            logger.error("Out of bounds in BeamReqIdsProxy for index %s", index)
            raise e

    def copy(self):
        return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __getitem__(self, idx):
        try:
            return self.original[self.batch_idx_map[idx]]
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray getitem for index %s", idx)
            raise e

    def __setitem__(self, idx, value):
        try:
            self.original[self.batch_idx_map[idx]] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray setitem for index %s", idx)
            raise e


class Mock2DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __setitem__(self, key, value):
        idx, slc = key
        try:
            self.original[self.batch_idx_map[idx], slc] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray setitem for key %s", key)
            raise e

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, key):
        idx, slc = key
        try:
            return self.original[self.batch_idx_map[idx], slc]
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray getitem for key %s", key)
            raise e


class InputBatchProxy:
    def __init__(self, obj, req_map, idx_map, logits_map):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_req_id_map", req_map)
        object.__setattr__(self, "_batch_idx_map", idx_map)
        object.__setattr__(self, "_logit_idx_map", logits_map)

    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        idx_map = object.__getattribute__(self, "_batch_idx_map")
        logits_map = object.__getattribute__(self, "_logit_idx_map")
        req_map = object.__getattribute__(self, "_req_id_map")

        if name == "sampling_metadata":
            s_meta = getattr(obj, "sampling_metadata", None)
            if s_meta is None or logits_map is None:
                return s_meta

            orig_req_len = len(obj.req_ids) if hasattr(obj, "req_ids") else 1
            for attr in [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "frequency_penalties",
                "presence_penalties",
                "repetition_penalties",
            ]:
                val = getattr(s_meta, attr, None)
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    if val.shape[0] == orig_req_len and val.shape[0] != len(logits_map):
                        object.__setattr__(s_meta, attr, val[logits_map])
            return s_meta

        elif name == "req_ids":
            return BeamReqIdsProxy(obj.req_ids, req_map)
        elif name == "num_tokens_no_spec":
            return Mock1DArray(obj.num_tokens_no_spec, idx_map)
        elif name == "num_tokens":
            # NPU's _bookkeeping_sync writeback loop also updates num_tokens
            # (the GPU path does not). Map it to the logical batch slot like
            # num_tokens_no_spec. Inert when never accessed (e.g. GPU).
            return Mock1DArray(obj.num_tokens, idx_map)
        elif name == "token_ids_cpu":
            return Mock2DArray(obj.token_ids_cpu, idx_map)
        elif name == "is_token_ids":
            return (
                Mock2DArray(obj.is_token_ids, idx_map)
                if hasattr(obj, "is_token_ids")
                else Mock2DArray(obj.token_ids_cpu, idx_map)
            )
        return getattr(obj, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)


def _beam_remap_inputs(runner, scheduler_output, num_scheduled_tokens) -> List[int]:
    """Expand a packed beam-decode batch into per-beam rows.

    Device-neutral. Overwrites per-beam RoPE position ids (Job B), builds the
    per-beam ``logits_indices`` list, installs an ``InputBatchProxy`` on
    ``runner.input_batch`` and stashes ``scheduler_output._gr_request_row_configs``
    (Job C). Job A (populating ``beam_info["req_ids"]``) is left to the caller
    because its ordering relative to the original ``_prepare_inputs`` differs
    between GPU and NPU.

    Assumes ``scheduler_output.beam_data`` is non-empty (caller checks).

    Returns:
        The new per-beam logits-index list (caller converts to a tensor).
    """
    beam_data = getattr(scheduler_output, "beam_data", {})

    runner.input_batch.logprob_token_ids = {}
    new_logits_indices: List[int] = []
    orig_req_ids = runner.input_batch.req_ids

    token_offset = 0
    row_to_req_id, row_to_batch_idx, logits_to_batch_idx, request_row_configs = ([], [], [], [])
    input_row_idx = 0

    gpu_positions, cpu_positions = _get_positions_buffers(runner)

    for batch_idx, req_id in enumerate(orig_req_ids):
        seq_len = num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
        if req_id is None:
            row_to_req_id.append(None)
            row_to_batch_idx.append(batch_idx)
            input_row_idx += 1
            token_offset += seq_len
            continue

        mdata = beam_data.get(req_id)
        if mdata and mdata.get("is_beam_decode", False):
            # Check if incremental decode is active
            is_incremental = "decode_step" in mdata
            beams_count, beams_count_with_logits, next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    cpu_positions,
                    new_logits_indices,
                    is_incremental=is_incremental,
                )
            )
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)
            logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)

            request_row_configs.append(
                (req_id, input_row_idx, beams_count_with_logits, active_beam_ids)
            )
            input_row_idx += beams_count
        else:
            num_computed_tokens = runner.input_batch.num_computed_tokens_cpu[batch_idx]
            num_prompt_tokens = runner.input_batch.num_prompt_tokens[batch_idx]

            is_last_chunk_prefill = bool(
                num_computed_tokens < num_prompt_tokens
                and num_computed_tokens + seq_len >= num_prompt_tokens
            )
            beams_count_with_logits = 0
            beams_count = 1
            if is_last_chunk_prefill:
                new_logits_indices.append(token_offset + seq_len - 1)
                beams_count_with_logits = 1
                logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)

            request_row_configs.append((req_id, input_row_idx, beams_count_with_logits, []))
            input_row_idx += beams_count

        token_offset += seq_len

    runner.input_batch = InputBatchProxy(
        runner.input_batch, row_to_req_id, row_to_batch_idx, logits_to_batch_idx
    )
    scheduler_output._gr_request_row_configs = request_row_configs

    return new_logits_indices


def _beam_collapse_bookkeeping(
    runner, orig_input_batch, request_row_configs, beam_data, sampler_output
) -> Tuple[List[Any], Dict[Any, int], int]:
    """Collapse per-beam sampler output back onto logical (session) requests.

    Device-neutral. Remuxes the sampler logprobs tensors in place so that each
    logical request owns one flattened ``W * K`` row, and resizes the sampler
    output tensors accordingly.

    Returns:
        ``(collapsed_req_ids, collapsed_req_id_to_index, num_logical_requests)``.
    """
    collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
    collapsed_req_id_to_index = {req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)}
    num_logical_requests = len(collapsed_req_ids_copy)

    if sampler_output is not None:
        st_ids = sampler_output.sampled_token_ids
        lp_tensors = sampler_output.logprobs_tensors
        if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
            orig_lps = lp_tensors.logprobs
            orig_lp_ids = lp_tensors.logprob_token_ids
            orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)
            actual_K = orig_lps.shape[1]

            max_chained_dim = 0
            for req_id, _, W, *_ in request_row_configs:
                if req_id is not None and req_id in beam_data:
                    m_width = beam_data[req_id].get("beam_width", 1)
                    needed_dim = m_width * actual_K
                else:
                    needed_dim = W * actual_K
                if needed_dim > max_chained_dim:
                    max_chained_dim = needed_dim

            final_st_ids, final_lps, final_lp_ids, final_ranks = _remux_logprobs_tensors(
                request_row_configs,
                max_chained_dim,
                actual_K,
                orig_lps,
                orig_lp_ids,
                orig_ranks,
                st_ids,
            )
            if not hasattr(runner, "_beam_logprobs_cache"):
                runner._beam_logprobs_cache = {}

            req_id_to_config = {r[0]: (r[2], r[3]) for r in request_row_configs if r[0] is not None}

            for idx, req_id in enumerate(collapsed_req_ids_copy):
                mdata = beam_data.get(req_id)
                if mdata and mdata.get("is_beam_decode", False):
                    logits_num, active_beams = req_id_to_config.get(req_id, (0, []))
                    total_beam_width = mdata.get("beam_width", 1)

                    if logits_num > 0:
                        # FAST-PATH: This chunk contains ALL beams directly.
                        if logits_num == total_beam_width:
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]
                            continue

                        # SLOW-PATH: Handle split chunked prefills across steps.
                        actual_lps = final_lps[idx, : logits_num * actual_K].view(
                            logits_num, actual_K
                        )
                        actual_lp_ids = (
                            final_lp_ids[idx, : logits_num * actual_K].view(logits_num, actual_K)
                            if final_lp_ids is not None
                            else None
                        )
                        actual_ranks = (
                            final_ranks[idx, :logits_num] if final_ranks is not None else None
                        )

                        if req_id not in runner._beam_logprobs_cache:
                            runner._beam_logprobs_cache[req_id] = {
                                "lps": torch.full(
                                    (total_beam_width, actual_K),
                                    float("-inf"),
                                    device=orig_lps.device,
                                    dtype=orig_lps.dtype,
                                ),
                                "lp_ids": torch.zeros(
                                    (total_beam_width, actual_K),
                                    dtype=orig_lp_ids.tensor.dtype
                                    if hasattr(orig_lp_ids, "tensor")
                                    else torch.long,
                                    device=orig_lps.device,
                                )
                                if orig_lp_ids is not None
                                else None,
                                "ranks": torch.zeros(
                                    total_beam_width, dtype=torch.long, device=orig_lps.device
                                )
                                if orig_ranks is not None
                                else None,
                                "completed_beams": set(),
                            }

                        cache_entry = runner._beam_logprobs_cache[req_id]

                        for local_row_idx, physical_beam_id in enumerate(active_beams):
                            if physical_beam_id not in cache_entry["completed_beams"]:
                                cache_entry["lps"][physical_beam_id] = actual_lps[local_row_idx]
                                if actual_lp_ids is not None:
                                    cache_entry["lp_ids"][physical_beam_id] = actual_lp_ids[
                                        local_row_idx
                                    ]
                                if actual_ranks is not None:
                                    cache_entry["ranks"][physical_beam_id] = actual_ranks[
                                        local_row_idx
                                    ]
                                cache_entry["completed_beams"].add(physical_beam_id)

                        if len(cache_entry["completed_beams"]) >= total_beam_width:
                            final_lps[idx].copy_(cache_entry["lps"].view(-1))
                            if final_lp_ids is not None:
                                final_lp_ids[idx].copy_(cache_entry["lp_ids"].view(-1))
                            if final_ranks is not None:
                                final_ranks[idx, :total_beam_width].copy_(cache_entry["ranks"])

                            del runner._beam_logprobs_cache[req_id]

            try:
                st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
                lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

                if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                    lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
                if orig_ranks is not None and final_ranks is not None:
                    clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                    lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(
                        clean_ranks_slice
                    )
            except Exception as e:
                logger.critical("Inplace copy failed during beam collapse: %s", e)
                raise RuntimeError("Inplace copy failed during beam collapse") from e

    return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests


def _get_or_create_beam_session(runner, beam_data: dict, scheduler_output=None) -> Optional[Any]:
    """Get or create per-request BeamSearchSession for incremental decode.

    Manages per-request sessions via runner._beam_sessions dict, keyed by
    session_id (== req_id).  Each session references a shared
    BeamAttentionContext (runner._beam_context) which owns the pools and
    the BeamSessionRegistry.

    Session lifecycle:
    - Creation: on the first beam decode step (decode_step=0) for a req_id.
    - Step progression: step_complete (select_unshared_kv + advance) on
      subsequent steps.
    - Teardown: the authoritative signal is
      ``SchedulerOutput.beam_finished_session_ids``, populated by
      ``patched_schedule`` from
      ``Scheduler._beam_finished_session_ids`` which is written by
      ``_handle_beam_request_step_update`` when it receives a
      ``BeamRequestStepUpdate(beam_width=0)`` — the beam search loop's
      explicit "generation finished" message.  This is distinct from
      ``finished_req_ids`` which fires on every decode step.
    - Fallback: if allocate fails and the registry is full, stale sessions
      (whose req_ids are no longer in beam_data) are released as a safety
      net for unexpected abort/error paths.
    """
    runner._beam_sessions = getattr(runner, "_beam_sessions", {})

    # 1. Clean up finished sessions (beam_width=0 teardown signal)
    finished_ids = getattr(scheduler_output, "beam_finished_session_ids", [])
    for sid in finished_ids:
        session = runner._beam_sessions.pop(sid, None)
        if session is not None:
            session.end_session()

    # 2. Check if we have any beam decode requests with decode_step
    has_incremental = any(
        mdata.get("is_beam_decode", False) and "decode_step" in mdata
        for mdata in beam_data.values()
    )
    if not has_incremental:
        return None

    # 3. Ensure worker-level context exists (lazy creation)
    if not hasattr(runner, "_beam_context"):
        # Get model config for pool dimensions
        model_config = getattr(runner, "model_config", None)
        if model_config is None:
            vllm_config = getattr(runner, "vllm_config", None)
            if vllm_config is not None:
                model_config = vllm_config.model_config

        if model_config is None:
            logger.warning("Cannot create BeamAttentionContext: model_config not available")
            return None

        parallel_config = getattr(runner, "parallel_config", None)
        if parallel_config is None:
            vllm_config = getattr(runner, "vllm_config", None)
            if vllm_config is not None:
                parallel_config = vllm_config.parallel_config

        if parallel_config is None:
            logger.warning("Cannot create BeamAttentionContext: parallel_config not available")
            return None

        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
        num_layers = model_config.get_num_layers(parallel_config)
        device = runner.device if hasattr(runner, "device") else torch.device("cuda")

        # Determine max_batch: the maximum number of concurrent beam search sessions.
        # This is NOT the scheduler's max_num_seqs (which includes all request types).
        # For now, hardcode to 1 for max_batch=1 testing.  Future: read from
        # beam-specific config or dynamically size the pool.
        max_batch = 1

        # Determine max_beams and max_decode_steps from the first beam request
        # These will be used for pool pre-allocation.  In the future, we may
        # need to support heterogeneous beam widths across requests.
        first_mdata = None
        for md in beam_data.values():
            if md.get("is_beam_decode", False) and "decode_step" in md:
                first_mdata = md
                break

        max_beams = first_mdata["beam_width"] if first_mdata else 128
        max_decode_steps = first_mdata.get("max_decode_steps", 0) if first_mdata else 0
        if max_decode_steps <= 0 and first_mdata is not None:
            max_decode_steps = max(first_mdata.get("beam_decode_steps", 0), 64)
        if max_decode_steps <= 0:
            max_decode_steps = 64

        from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext

        runner._beam_context = BeamAttentionContext(
            max_batch=max_batch,
            max_beams=max_beams,
            num_kv_heads=num_kv_heads,
            max_decode_steps=max_decode_steps,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=model_config.dtype,
        )

    # 4. Per-request session management
    active_session = None

    for req_id, mdata in beam_data.items():
        if not mdata.get("is_beam_decode") or "decode_step" not in mdata:
            continue

        session = runner._beam_sessions.get(req_id)
        mdata_step = mdata.get("decode_step", 0)

        if session is not None:
            # Existing session: step_complete or reset
            if mdata_step == 0:
                # First decode step of a request: full reset.
                # There is no previous beam decision yet, so select_unshared_kv
                # must not run, and decode KV must be written to pool slice 0.
                # (parent_beam_indices is non-None even on the first step, so it
                # cannot be used to distinguish the first step — decode_step can.)
                session.begin_session(mdata["beam_width"], mdata["prefix_len"])
            else:
                # Subsequent decode steps: apply the previous step's beam decision.
                # step_complete (select_unshared_kv + advance_step) must run BEFORE
                # the new KV is written, using the current decode_step to select
                # the KV range to reorder.
                parent_beam_ids = mdata.get("parent_beam_ids")
                max_decode_steps = mdata.get("max_decode_steps", 0)
                if max_decode_steps <= 0:
                    max_decode_steps = max(mdata.get("beam_decode_steps", 0), 64)

                is_last_step = mdata_step >= max_decode_steps - 1
                state = session._state
                state.is_last_step = is_last_step

                if parent_beam_ids is not None and not is_last_step:
                    parent = torch.tensor(parent_beam_ids, dtype=torch.int32, device=runner.device)
                    session.step_complete(parent)
                else:
                    # Last step or no parent: just advance decode_step
                    state.decode_step += 1
        else:
            # New session
            from vllm_gr.v1.beam.beam_search_session import BeamSearchSession

            try:
                slot = runner._beam_context.registry.allocate(
                    req_id, mdata["beam_width"], mdata["prefix_len"]
                )
            except RuntimeError:
                # Fallback: release stale sessions whose req_ids are no
                # longer in beam_data.  This handles unexpected abort/error
                # paths where the beam_width=0 cleanup signal never arrived.
                # The primary teardown path is beam_finished_session_ids
                # (set by _handle_beam_request_step_update on beam_width=0).
                active_req_ids = set(beam_data.keys())
                stale_ids = [
                    sid for sid in list(runner._beam_sessions.keys()) if sid not in active_req_ids
                ]
                for sid in stale_ids:
                    old_session = runner._beam_sessions.pop(sid)
                    old_session.end_session()
                # Retry allocation after cleanup
                slot = runner._beam_context.registry.allocate(
                    req_id, mdata["beam_width"], mdata["prefix_len"]
                )
            session = BeamSearchSession(session_id=req_id, slot=slot, context=runner._beam_context)
            session.begin_session(mdata["beam_width"], mdata["prefix_len"])
            # Set initial decode_step from scheduler
            session._state.decode_step = mdata_step
            runner._beam_sessions[req_id] = session

        if active_session is None:
            active_session = session

    return active_session


def make_patched_execute_model(_orig_execute_model):
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    if getattr(_orig_execute_model, _WORKER_EXECUTE_PATCH_MARKER, False):
        return _orig_execute_model

    @wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        beam_info = {"beam_data": self._current_beam_data, "req_ids": []}

        # Manage BeamSearchSession for incremental decode
        beam_session = _get_or_create_beam_session(self, self._current_beam_data, scheduler_output)
        if beam_session is not None:
            beam_info["beam_session"] = beam_session

        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        try:
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_beam_data = {}
            self._current_beam_info = None
            BEAM_DATA_VAR.reset(token)

    setattr(patched_execute_model, _WORKER_EXECUTE_PATCH_MARKER, True)
    return patched_execute_model


def is_worker_runner_patched(runner_class) -> bool:
    """Return whether all grouped-beam hooks are present on a runner class."""
    return bool(
        getattr(runner_class.execute_model, _WORKER_EXECUTE_PATCH_MARKER, False)
        and getattr(runner_class._prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False)
        and getattr(
            runner_class._bookkeeping_sync,
            _WORKER_BOOKKEEPING_PATCH_MARKER,
            False,
        )
    )
