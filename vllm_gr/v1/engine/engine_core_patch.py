# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""
    import threading

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        self.beam_cache_lock = threading.Lock()

    return patched_init


# ---------------------------------------------------------------------------
# Helpers: add enum members to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------


def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent.

    This uses object.__new__() to bypass enum's normal creation pathway,
    which is necessary because we're monkey-patching at runtime and cannot
    modify the original class definition in vLLM's codebase.

    While this approach manipulates enum internals, it:
    - Is a standard technique used by enum extension libraries (e.g. aenum)
    - Includes validation to detect incompatibilities early
    - Is contained to initialization time (not runtime performance critical)
    - Is tested to ensure correctness

    Args:
        name: The name of the new enum member (e.g., "ADD_BATCH")
        value: The byte value for the enum member (e.g., b"\\x05")

    Raises:
        RuntimeError: If the enum's internal structure is incompatible
    """
    # Skip if already added
    if hasattr(EngineCoreRequestType, name):
        return

    # Validate that the enum has expected internal structure
    required_attrs = ["_value2member_map_", "_member_map_", "_member_names_"]
    for attr in required_attrs:
        if not hasattr(EngineCoreRequestType, attr):
            raise RuntimeError(
                f"EngineCoreRequestType missing expected attribute '{attr}'. "
                f"This indicates an incompatible Python/vLLM version. "
                f"Please report this issue with your Python version."
            )

    # Verify the value type matches existing members
    if EngineCoreRequestType._value2member_map_:
        existing_value_type = type(next(iter(EngineCoreRequestType._value2member_map_.keys())))
        if not isinstance(value, existing_value_type):
            raise TypeError(
                f"Value type mismatch: expected {existing_value_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Create the member instance bypassing enum's __new__
    # (This is the standard approach for runtime enum extension)
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value

    # Register in enum's internal maps
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)

    # Set as class attribute (must use type.__setattr__ to bypass enum's __setattr__)
    type.__setattr__(EngineCoreRequestType, name, member)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------




def apply_engine_core_child_patches():
    """Apply all patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _cache_beam_request,
        _handle_beam_request_step_update,
        process_input_sockets,
    )


    logger.info("[vllm-gr] apply_engine_core_child_patches called.")

    # Add new enum members: ADD_BATCH, BEAM_REQUEST_STEP_UPDATE
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x06")

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_beam_request_step_update on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_request_step_update = _handle_beam_request_step_update

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    from vllm.sampling_params import SamplingParams

    # Force the greedy verification check to be a no-op to allow n > 1 under greedy beam search parameters
    SamplingParams._verify_greedy_sampling = lambda self: None

    # Apply scheduler patches in the child process
    apply_scheduler_patch()

    # Apply worker patches (e.g., GPUModelRunner)
    apply_worker_patches()

    logger.debug("Engine-core patches applied in child process.")


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. This indicates a recursive "
            "call, likely due to patching being applied multiple times. "
            "Check that patch_batch_and_fork() is only called once."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the saved original stashed by patch_batch_and_fork() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        apply_engine_core_child_patches()
        original_run(*args, **kwargs)
    finally:
        _run_engine_core_active = False


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from functools import wraps

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_patched_for_beam_requests", False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        block_size = self.cache_config.block_size

        # TODO: Add support for cache > prefix
        def patch_get_computed_blocks(req):
            bn, tn = _original_get_computed_blocks(req)
            if req and getattr(req, "is_beam_decode", False):
                pl = getattr(req, "prefix_len", 0) // block_size * block_size
                if tn > pl:
                    sliced_blocks = tuple(group[: pl // block_size] for group in bn.blocks)
                    return self.kv_cache_manager.create_kv_cache_blocks(sliced_blocks), pl
            return bn, tn

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        try:
            output = _original_schedule(self)

            # Propagate beam request attributes from Request objects to SchedulerOutput
            beam_data = {}
            if hasattr(output, "num_scheduled_tokens"):
                for req_id in output.num_scheduled_tokens.keys():
                    req = self.requests.get(req_id)
                    if req and getattr(req, "is_beam_decode", False):
                        beam_data[req_id] = {
                            "is_beam_decode": True,
                            "beam_beam_width": getattr(req, "beam_beam_width", 1),
                            "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
                            "prefix_len": getattr(req, "prefix_len", 0),
                            "cache_len": req.num_computed_tokens
                            - output.num_scheduled_tokens[req_id],
                        }
            output.beam_data = beam_data
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks

    Scheduler.schedule = patched_schedule
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")


# ============================================================================
# TENSOR AND BUFFER GEOMETRY EXTRACTION HELPERS
# ============================================================================


def _get_positions_buffers(runner: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        gpu_positions = getattr(runner.positions, "gpu", None)
        cpu_positions = getattr(runner.positions, "cpu", None)
    return gpu_positions, cpu_positions


def _compute_beam_bounds(
    b: int,
    b_prefix_len: int,
    b_cache_len: int,
    decode_steps: int,
    chunk_budget: int,
) -> Tuple[int, int]:
    """Calculates active lookahead allocations and historical trace lengths."""
    delta = max(0, b_cache_len - b_prefix_len)
    remaining_prefix = b_prefix_len - b_cache_len
    b_physical = delta // decode_steps
    b_uncached = 0
    b_suffix_len = 0

    b_past_suffix = max(0, min(decode_steps, delta - b_physical * decode_steps))
    if remaining_prefix >= 0:
        # Scenario A: Prefix Incomplete - compute remaining prefix + suffix of first beam
        b_uncached = min(chunk_budget, remaining_prefix + decode_steps)
    else:
        # Scenario B: Compute only suffix of beams
        b_uncached = min(chunk_budget, max(decode_steps - b_past_suffix, 0))
    b_suffix_len = b_uncached + b_past_suffix
    return b_uncached, b_suffix_len, b_past_suffix, b_physical


def _overwrite_position_ids(
    st: int,
    b_uncached: int,
    curr_offset: int,
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
) -> None:
    """Overwrites positional indexing structures in place on targets."""
    if gpu_pos is not None:
        gpu_pos[curr_offset : curr_offset + b_uncached] = torch.arange(
            st, st + b_uncached, dtype=gpu_pos.dtype, device=gpu_pos.device
        )
    if cpu_pos is not None:
        cpu_pos[curr_offset : curr_offset + b_uncached] = torch.arange(
            st, st + b_uncached, dtype=cpu_pos.dtype, device=torch.device("cpu")
        )


def _process_beam_decode_request(
    mdata: Dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
    new_logits_indices: List[int],
) -> Tuple[int, int, int]:
    """Traces allocation arrays and mutates position buffers for a speculative request."""
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_beam_width"]
    decode_steps = mdata["beam_decode_steps"]
    cache_len = mdata["cache_len"]
    active_beam_ids = []  # Track structural beam locations

    chunk_budget = seq_len
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    b_cache_len = cache_len
    for b in range(beam_width):
        if chunk_budget <= 0:
            break

        b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
            b, prefix_len, b_cache_len, decode_steps, chunk_budget
        )

        if b_uncached > 0:
            st = cache_len if (prefix_len >= b_cache_len) else (prefix_len + b_past_suffix)
            _overwrite_position_ids(st, b_uncached, curr_offset, gpu_pos, cpu_pos)

            if b_past_suffix + b_uncached >= decode_steps:
                new_logits_indices.append(curr_offset + b_uncached - 1)
                active_beam_ids.append(b_physical)

                logits_row_cnt += 1

            valid_rows_for_req += 1
            curr_offset += b_uncached
            chunk_budget -= b_uncached
            b_cache_len += b_uncached

    return valid_rows_for_req, logits_row_cnt, curr_offset, active_beam_ids


# ============================================================================
# CORE WORKER PIPELINE WELDER LOGIC
# ============================================================================


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
        # --------------------------------------------------------------------
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        # --------------------------------------------------------------------
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

        # --------------------------------------------------------------------
        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        # --------------------------------------------------------------------
        # Retain leader token track identity
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

        # Advance the pointer cursor past the complete multi-beam block size
        gpu_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None,
    )


# ============================================================================
# INLINE COMPILATION PROXIES FOR HIGH-PERFORMANCE INTERCEPTION
# ============================================================================


class BeamReqIdsProxy(list):
    def __init__(self, original_list, req_id_map):
        super().__init__(original_list)
        self.req_id_map = req_id_map

    def __getitem__(self, index):
        return (
            self.req_id_map[index]
            if index < len(self.req_id_map)
            else (self.req_id_map[-1] if self.req_id_map else None)
        )

    def copy(self):
        return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __getitem__(self, idx):
        return (
            self.original[self.batch_idx_map[idx]]
            if idx < len(self.batch_idx_map)
            else self.original[0]
        )

    def __setitem__(self, idx, value):
        if idx < len(self.batch_idx_map):
            self.original[self.batch_idx_map[idx]] = value


class Mock2DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __setitem__(self, key, value):
        idx, slc = key
        if idx < len(self.batch_idx_map):
            self.original[self.batch_idx_map[idx], slc] = value

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, key):
        idx, slc = key
        actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
        return self.original[actual_idx, slc]


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


# ============================================================================
# MONKEY PATCH APPLICATOR REGISTRY
# ============================================================================


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
            W, W_logits, next_offset, active_beam_ids = _process_beam_decode_request(
                mdata,
                seq_len,
                token_offset,
                gpu_positions,
                cpu_positions,
                new_logits_indices,
            )
            row_to_req_id.extend([req_id] * W)
            row_to_batch_idx.extend([batch_idx] * W)
            logits_to_batch_idx.extend([batch_idx] * W_logits)

            request_row_configs.append((req_id, input_row_idx, W_logits, active_beam_ids))
            input_row_idx += W
        else:
            num_computed_tokens = runner.input_batch.num_computed_tokens_cpu[batch_idx]
            num_prompt_tokens = runner.input_batch.num_prompt_tokens[batch_idx]

            is_last_chunk_prefill = bool(
                num_computed_tokens < num_prompt_tokens
                and num_computed_tokens + seq_len >= num_prompt_tokens
            )
            W_logits_count = 0
            W = 1
            if is_last_chunk_prefill:
                new_logits_indices.append(token_offset + seq_len - 1)
                W_logits_count = 1
                logits_to_batch_idx.extend([batch_idx] * W)
            row_to_req_id.extend([req_id] * W)
            row_to_batch_idx.extend([batch_idx] * W)

            request_row_configs.append((req_id, input_row_idx, W_logits_count, []))
            input_row_idx += W

        token_offset += seq_len

    runner.input_batch = InputBatchProxy(
        runner.input_batch, row_to_req_id, row_to_batch_idx, logits_to_batch_idx
    )
    scheduler_output._gr_request_row_configs = request_row_configs

    return new_logits_indices


def _beam_collapse_bookkeeping(runner,
    orig_input_batch, request_row_configs, beam_data, sampler_output
) -> Tuple[List[Any], Dict[Any, int], int]:
    """Collapse per-beam sampler output back onto logical (session) requests.

    Device-neutral. Remuxes the sampler logprobs tensors in place so that each
    logical request owns one flattened ``W * K`` row, and resizes the sampler
    output tensors accordingly.

    Returns:
        ``(collapsed_req_ids, collapsed_req_id_to_index, num_logical_requests)``.
    """
    collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
    collapsed_req_id_to_index = {
        req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)
    }
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
                    m_width = beam_data[req_id].get("beam_beam_width", 1)
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

            req_id_to_config = {
                r[0]: (r[2], r[3]) for r in request_row_configs if r[0] is not None
            }

            for idx, req_id in enumerate(collapsed_req_ids_copy):
                mdata = beam_data.get(req_id)
                if mdata and mdata.get("is_beam_decode", False):
                    W_logits, active_beams = req_id_to_config.get(req_id, (0, []))
                    total_beam_width = mdata.get("beam_beam_width", 1)

                    if W_logits > 0:
                        # ---------------------------------------------------------------------
                        # FAST-PATH: This chunk contains ALL beams directly (No Split occurred)
                        # ---------------------------------------------------------------------
                        if W_logits == total_beam_width:
                            # Clean up stale entries left behind by older preempted/aborted steps
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]

                            # Skip all caching loops! final_lps[idx] is already fully populated.
                            continue

                        # ---------------------------------------------------------------------
                        # SLOW-PATH: Handled split chunked prefills across steps
                        # ---------------------------------------------------------------------
                        # Extract this chunk's output elements
                        actual_lps = final_lps[idx, : W_logits * actual_K].view(
                            W_logits, actual_K
                        )
                        actual_lp_ids = (
                            final_lp_ids[idx, : W_logits * actual_K].view(W_logits, actual_K)
                            if final_lp_ids is not None
                            else None
                        )
                        actual_ranks = (
                            final_ranks[idx, :W_logits] if final_ranks is not None else None
                        )

                        # Initialize pre-allocated tensor cache matrix structures only if needed
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

                        # Scatter write chunk data lines directly to their physical beam slots
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

                        # Evict and overwrite target boundaries only when ALL unique structural slots are filled
                        if len(cache_entry["completed_beams"]) >= total_beam_width:
                            final_lps[idx].copy_(cache_entry["lps"].view(-1))
                            if final_lp_ids is not None:
                                final_lp_ids[idx].copy_(cache_entry["lp_ids"].view(-1))
                            if final_ranks is not None:
                                final_ranks[idx, :total_beam_width].copy_(cache_entry["ranks"])

                            del runner._beam_logprobs_cache[req_id]

            # Safely resizes the original tensor wrapper and copies the elements in place
            st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
            lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

            if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
            if orig_ranks is not None and final_ranks is not None:
                clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(
                    clean_ranks_slice
                )

    return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests


# ---------------------------------------------------------------------------
# Factory functions (module-level so each captures its own original via
# closure argument, avoiding late-binding bugs when patching multiple runners).
# ---------------------------------------------------------------------------


def make_patched_execute_model(_orig_execute_model):
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    from vllm_gr.v1.attention.backends.beam_attn import BEAM_DATA_VAR

    @wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        beam_info = {"beam_data": self._current_beam_data, "req_ids": []}
        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        try:
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_beam_data = {}
            self._current_beam_info = None
            BEAM_DATA_VAR.reset(token)

    return patched_execute_model


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """GPU ``_prepare_inputs`` wrapper: ``(scheduler_output, num_scheduled_tokens)``
    returning ``(logits_indices, spec_decode_metadata)``."""

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        logits_indices, spec_decode_metadata = _orig_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # Job A: the GPU attention builder runs after _prepare_inputs returns, so
        # populating req_ids here (post-orig) is in time.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return logits_indices, spec_decode_metadata

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        return (
            torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device),
            spec_decode_metadata,
        )

    return patched_prepare_inputs


def make_patched_bookkeeping_sync_gpu(_orig_bookkeeping_sync):
    """GPU ``_bookkeeping_sync`` wrapper: returns the 7-tuple including
    ``num_nans_in_logits``."""

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(self, 
            orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        return (
            num_nans_in_logits,
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    return patched_bookkeeping_sync


def make_patched_prepare_inputs_npu(_orig_prepare_inputs):
    """NPU ``_prepare_inputs`` wrapper.

    Signature: ``(scheduler_output, intermediate_tensors=None)``.
    Returns the NPU 7-tuple
    ``(attn_metadata, num_scheduled_tokens, num_input_tokens,
    num_tokens_across_dp, logits_indices, spec_decode_metadata,
    max_num_scheduled_tokens)``.

    Two NPU-specific differences vs the GPU wrapper:

    - The NPU attention metadata builder runs *inside* the original
      ``_prepare_inputs``, so Job A (publishing ``req_ids`` to the contextvar)
      must happen *before* delegating, not after.
    - ``_prepare_inputs`` is called from ``execute_model``, while sampling and
      ``_bookkeeping_sync`` run in a separate ``sample_tokens`` call. If an
      early return skips bookkeeping, a stale ``InputBatchProxy`` could survive
      to the next step, so we defensively un-proxy at entry.

    The per-beam RoPE position overwrite needs no extra device copy: NPU
    ``self.positions`` is a ``CpuGpuBuffer`` and ``_overwrite_position_ids``
    writes the ``.gpu`` (device) tensor in place, which ``_preprocess`` reads.
    The physical ``slot_mapping`` is materialized from linear positions inside
    the original before the overwrite, so it is unaffected.
    """

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, intermediate_tensors=None):
        # Drop a stale proxy left over from a prior step (see docstring).
        if isinstance(self.input_batch, InputBatchProxy):
            self.input_batch = object.__getattribute__(self.input_batch, "_obj")

        # Job A: publish req_ids BEFORE the original builds attention metadata.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        out = _orig_prepare_inputs(self, scheduler_output, intermediate_tensors)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return out

        (
            attn_metadata,
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            logits_indices,
            spec_decode_metadata,
            max_num_scheduled_tokens,
        ) = out

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)

        return (
            attn_metadata,
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            logits_indices,
            spec_decode_metadata,
            max_num_scheduled_tokens,
        )

    return patched_prepare_inputs


def make_patched_bookkeeping_sync_npu(_orig_bookkeeping_sync):
    """NPU ``_bookkeeping_sync`` wrapper: returns the 6-tuple (no
    ``num_nans_in_logits``).

    NPU calls ``_bookkeeping_sync(scheduler_output, sampler_output, logits,
    hidden_states, num_scheduled_tokens, spec_decode_metadata)`` positionally,
    so ``sampler_output`` is ``args[0]``.
    """

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(self,
            orig_input_batch, request_row_configs, beam_data, sampler_output
        )

        return (
            [[] for _ in range(n)] if logprobs_lists is not None else None,
            [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    return patched_bookkeeping_sync


def apply_worker_patches():
    """Applies runtime overrides onto GPUModelRunner and NPUModelRunner execution entries."""
    from vllm_gr.utils.npu_adapter import is_npu_available

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("Failed to import GPUModelRunner segments, skipping GPU patches.")
        GPUModelRunner = None

    npu_available = is_npu_available()
    if npu_available:
        try:
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        except ImportError:
            logger.warning("Failed to import NPUModelRunner, skipping NPU patches.")
            NPUModelRunner = None
    else:
        NPUModelRunner = None

    # Return early if nothing to patch
    if GPUModelRunner is None and NPUModelRunner is None:
        logger.warning("No supported model runner found for patching.")
        return

    # Track which runners have already been patched
    runners_to_patch = []
    if GPUModelRunner is not None and not getattr(
        GPUModelRunner, "_patched_for_beam_beam_pos_ids", False
    ):
        runners_to_patch.append(("GPUModelRunner", GPUModelRunner))
    if NPUModelRunner is not None and not getattr(
        NPUModelRunner, "_patched_for_beam_beam_pos_ids", False
    ):
        runners_to_patch.append(("NPUModelRunner", NPUModelRunner))

    if not runners_to_patch:
        return

    # Apply patches to each runner
    for runner_name, runner_class in runners_to_patch:
        runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)

        if runner_name == "GPUModelRunner":
            runner_class._prepare_inputs = make_patched_prepare_inputs_gpu(
                runner_class._prepare_inputs
            )
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_gpu(
                runner_class._bookkeeping_sync
            )
        elif runner_name == "NPUModelRunner":
            runner_class._prepare_inputs = make_patched_prepare_inputs_npu(
                runner_class._prepare_inputs
            )
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_npu(
                runner_class._bookkeeping_sync
            )

        runner_class._patched_for_beam_beam_pos_ids = True
        logger.info(f"{runner_name} hooks patched successfully.")
