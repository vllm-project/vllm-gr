# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.attention.backends.beam_attn import _compute_beam_bounds

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Greedy-sampling exemption for beam decode
# ---------------------------------------------------------------------------
# Beam decode legitimately runs greedy (temperature=0) with n == beam_width > 1.
# Upstream SamplingParams._verify_greedy_sampling rejects that combination, and
# msgspec invokes __post_init__ (→ _verify_greedy_sampling) when the child
# process deserializes a BeamRequestStepUpdate, so the check fires on the wire
# decode path — not just on explicit construction.
#
# Instead of blanking the method process-wide (which also disarmed the check
# for every ordinary greedy request), we keep the original method and install a
# thin wrapper that suppresses the n>1 rejection *only* while a beam-update
# decode is in flight (see _beam_decode_decoding below). All other requests
# continue to use the upstream verification unchanged.
_beam_decode_decoding: ContextVar[bool] = ContextVar("_beam_decode_decoding", default=False)
_original_verify_greedy_sampling: Any = None
_greedy_patch_applied: bool = False


@contextmanager
def beam_update_decode_scope() -> Iterator[None]:
    """Mark a beam-update deserialization as in flight.

    While active, the patched _verify_greedy_sampling skips the n>1 check so a
    greedy beam-width>1 SamplingParams can be rebuilt from the wire. The scope
    is intentionally tight: it must wrap *only* the decode call, never the rest
    of request handling.
    """
    token = _beam_decode_decoding.set(True)
    try:
        yield
    finally:
        _beam_decode_decoding.reset(token)


def _install_scoped_greedy_sampling_patch() -> None:
    """Install the beam-scoped _verify_greedy_sampling wrapper (idempotent)."""
    global _original_verify_greedy_sampling, _greedy_patch_applied
    if _greedy_patch_applied:
        return

    from vllm.sampling_params import SamplingParams

    _original_verify_greedy_sampling = SamplingParams._verify_greedy_sampling
    original = _original_verify_greedy_sampling

    def _verify_greedy_sampling(self) -> None:
        if _beam_decode_decoding.get():
            # Beam decode is rebuilding a greedy n>1 SamplingParams from the
            # wire; the combination is intentional. Skip only this rejection.
            return
        original(self)

    SamplingParams._verify_greedy_sampling = _verify_greedy_sampling
    _greedy_patch_applied = True


# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        self.beam_cache_lock = threading.Lock()

    return patched_init


# ---------------------------------------------------------------------------
# Helpers: add enum members to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------


def _add_enum_member_to(enum_cls: type, name: str, value: bytes) -> None:
    """Dynamically add a member to ``enum_cls`` if absent and value is free.

    This uses object.__new__() to bypass enum's normal creation pathway,
    which is necessary because we're monkey-patching at runtime and cannot
    modify the original class definition in vLLM's codebase.

    All compatibility checks run before *any* shared state is mutated, so a
    failed call leaves the enum unchanged (atomic). The four internal writes
    (value map, name map, name list, class attribute) are additionally wrapped
    in a defensive rollback in case an unexpected error slips past validation.

    Args:
        enum_cls: The enum class to extend (e.g. ``EngineCoreRequestType``).
        name: The name of the new enum member (e.g., "ADD_BATCH")
        value: The byte value for the enum member (e.g., b"\\x05")

    Raises:
        RuntimeError: If the enum's internal structure is incompatible, or if
            ``name``/``value`` collides with an existing member or class
            attribute (idempotent re-registration of the same name+value is
            allowed).
        TypeError: If ``value``'s type does not match existing members.
    """
    # Validate that the enum has the expected internal structure *and* that the
    # runtime-extension technique we rely on still works on it. We probe the
    # actual capabilities (read, write, member construction) rather than just
    # attribute presence, so an incompatible upstream change surfaces as a clear
    # RuntimeError instead of a low-level AttributeError/TypeError mid-write.
    member_map = getattr(enum_cls, "_member_map_", None)
    value2member_map = getattr(enum_cls, "_value2member_map_", None)
    member_names = getattr(enum_cls, "_member_names_", None)

    def _missing(attr: str) -> RuntimeError:
        return RuntimeError(
            f"{enum_cls.__name__} missing expected attribute '{attr}'. "
            f"This indicates an incompatible Python/vLLM version."
        )

    if member_map is None:
        raise _missing("_member_map_")
    if value2member_map is None:
        raise _missing("_value2member_map_")
    if member_names is None:
        raise _missing("_member_names_")

    # The value map must support both reverse lookup (`.get`) and registration
    # (`__setitem__`); likewise the name map. The name list must be appendable.
    for label, obj in (
        ("_value2member_map_", value2member_map),
        ("_member_map_", member_map),
    ):
        if not callable(getattr(obj, "get", None)) or not hasattr(obj, "__setitem__"):
            raise RuntimeError(
                f"{enum_cls.__name__}.{label} is not a writable mapping "
                f"(type {type(obj).__name__}); runtime enum extension is not "
                f"supported on this Python/vLLM version."
            )
    append_fn = getattr(member_names, "append", None)
    if not callable(append_fn):
        raise RuntimeError(
            f"{enum_cls.__name__}._member_names_ is not appendable "
            f"(type {type(member_names).__name__}); runtime enum extension is "
            f"not supported on this Python/vLLM version."
        )

    # Verify object.__new__(enum_cls) can still build a bare member. Some enum
    # metaclasses override __new__/__init__ in ways that break this technique.
    try:
        object.__new__(enum_cls)  # dry-run; the real member is built below
    except TypeError as exc:
        raise RuntimeError(
            f"object.__new__({enum_cls.__name__}) failed: {exc}. The runtime "
            f"enum-extension technique is not compatible with this enum."
        ) from exc

    # Verify the value type matches existing members
    if value2member_map:
        existing_value_type = type(next(iter(value2member_map.keys())))
        if not isinstance(value, existing_value_type):
            raise TypeError(
                f"Value type mismatch: expected {existing_value_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # ---- Conflict checks (read-only, all before any shared-state write) ----
    existing_by_name = member_map.get(name)
    if existing_by_name is not None:
        # Same name already registered.
        if existing_by_name.value == value:
            # Idempotent: same name + same value → no-op success.
            return
        # Same name, different value → incompatible.
        raise RuntimeError(
            f"Cannot register enum member {name!r}={value!r} on "
            f"{enum_cls.__name__}: name already in use with value "
            f"{existing_by_name.value!r}. vLLM version may have changed; "
            f"please align the wire value before proceeding."
        )

    existing_by_value = value2member_map.get(value)
    if existing_by_value is not None:
        # Same value, different name → would silently overwrite the value
        # reverse-lookup and misroute wire decoding.
        raise RuntimeError(
            f"Cannot register enum member {name!r}={value!r} on "
            f"{enum_cls.__name__}: value already in use by member "
            f"{existing_by_value.name!r}. vLLM version may have changed; "
            f"pick a free wire byte."
        )

    # A non-member class attribute with this name would shadow/be shadowed.
    for klass in enum_cls.__mro__:
        if name in klass.__dict__ and klass.__dict__[name] is not member_map.get(name):
            raise RuntimeError(
                f"Cannot register enum member {name!r} on {enum_cls.__name__}: "
                f"name already used as a non-member class attribute."
            )

    # ---- Atomic registration (defensive rollback on unexpected failure) ----
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value

    wrote_value_map = wrote_name_map = appended_name = False
    try:
        value2member_map[value] = member
        wrote_value_map = True
        member_map[name] = member
        wrote_name_map = True
        enum_cls._member_names_.append(name)
        appended_name = True
        # Set as class attribute (must use type.__setattr__ to bypass enum's
        # __setattr__).
        type.__setattr__(enum_cls, name, member)
    except BaseException:
        # Reverse-order rollback of whatever took effect.
        if appended_name and enum_cls._member_names_ and enum_cls._member_names_[-1] == name:
            enum_cls._member_names_.pop()
        if wrote_name_map and name in member_map:
            del member_map[name]
        if wrote_value_map and value in value2member_map:
            del value2member_map[value]
        raise


def _add_enum_member(name: str, value: bytes) -> None:
    """Add a member to the upstream ``EngineCoreRequestType``.

    Thin wrapper over :func:`_add_enum_member_to` preserving the historical
    two-argument signature used by all production call sites.
    """
    _add_enum_member_to(EngineCoreRequestType, name, value)


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

    if getattr(EngineCore, "_patched_for_beam_requests", False):
        return

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

    # Install a beam-scoped greedy-sampling exemption (replaces the former
    # process-wide no-op): ordinary greedy requests keep upstream verification;
    # only a beam-update wire decode (wrapped by beam_update_decode_scope) may
    # run greedy with n>1.
    _install_scoped_greedy_sampling_patch()

    # Apply scheduler patches in the child process
    apply_scheduler_patch()

    # Apply worker patches (e.g., GPUModelRunner)
    apply_worker_patches()

    EngineCore._patched_for_beam_requests = True
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
                            "beam_width": getattr(req, "beam_width", 1),
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
) -> Tuple[int, int, int, List[int]]:
    """Traces allocation arrays and mutates position buffers for a speculative request."""
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_width"]
    decode_steps = mdata["beam_decode_steps"]
    cache_len = mdata["cache_len"]
    active_beam_ids = []  # Track structural beam locations

    chunk_budget = seq_len
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    b_cache_len = cache_len
    seg_starts: List[int] = []
    seg_lens: List[int] = []
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
            # No row expansion/remap needed → return the original object unchanged,
            # avoiding a pointless copy.
            if orig_req_len == len(logits_map):
                return s_meta

            # Projection needed: shallow-copy the metadata and rebind the beam-row
            # tensors on the COPY only. The original SamplingMetadata (and thus the
            # underlying InputBatch) is never mutated, so a subsequent ordinary
            # request — or a second read in the same step — still sees the original
            # unprojected tensors. Reading `val` from `s_meta` (not the copy) is what
            # keeps every access safe: the original never changes shape, so
            # `val[logits_map]` is always well-formed.
            projected = copy.copy(s_meta)
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
                    if val.shape[0] == orig_req_len:
                        object.__setattr__(projected, attr, val[logits_map])
            return projected

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
            beams_count, beams_count_with_logits, next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    cpu_positions,
                    new_logits_indices,
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
                        # ---------------------------------------------------------------------
                        # FAST-PATH: This chunk contains ALL beams directly (No Split occurred)
                        # ---------------------------------------------------------------------
                        if logits_num == total_beam_width:
                            # Clean up stale entries left behind by older preempted/aborted steps
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]

                            # Skip all caching loops! final_lps[idx] is already fully populated.
                            continue

                        # ---------------------------------------------------------------------
                        # SLOW-PATH: Handled split chunked prefills across steps
                        # ---------------------------------------------------------------------
                        # Extract this chunk's output elements
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
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

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
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
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
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
            sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
            return res

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
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
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
        GPUModelRunner, "_patched_for_beam_pos_ids", False
    ):
        runners_to_patch.append(("GPUModelRunner", GPUModelRunner))
    if NPUModelRunner is not None and not getattr(
        NPUModelRunner, "_patched_for_beam_pos_ids", False
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

        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
