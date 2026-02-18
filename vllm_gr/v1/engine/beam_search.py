# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs

from vllm_gr.v1.engine.types import (
    BeamSearchRequest,
    BeamSearchResult,
    BeamSearchSequenceResult,
)

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore

logger = init_logger(__name__)


def _get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    seq_len = len(tokens)
    # If there are no tokens, or if all tokens are effectively EOS (seq_len
    # becomes 0 after EOS adjustment), avoid division by zero. In these
    # edge cases we simply return the cumulative log probability without
    # applying a length penalty.
    if seq_len == 0:
        return cumulative_logprob
    if tokens[-1] == eos_token_id:
        seq_len -= 1
        if seq_len <= 0:
            return cumulative_logprob
    return cumulative_logprob / (seq_len**length_penalty)


class _Beam:
    __slots__ = ("tokens", "cum_logprob")

    def __init__(self, tokens: list[int], cum_logprob: float):
        self.tokens = tokens
        self.cum_logprob = cum_logprob


_STEP_TIMEOUT_S = 120  # seconds before a pending step is considered stuck

# Below this candidate count, pure-Python sort is faster than numpy due to
# per-call overhead (array creation, vstack, argpartition, GC pressure).
# bw=64 → 4096 candidates (Python wins), bw=512 → 262144 (numpy wins).
_NUMPY_THRESHOLD = 10_000


class BeamSearchController:
    """Non-blocking beam search controller (state machine).

    Instead of monopolizing the engine core's busy loop, this controller
    submits one step's worth of requests to the scheduler, then yields
    control.  The engine core feeds step outputs back via on_outputs().
    """

    def __init__(self, engine_core: EngineCore, req: BeamSearchRequest):
        self.engine_core = engine_core
        self.scheduler = engine_core.scheduler
        self.block_hasher = engine_core.request_block_hasher

        self._req = req
        self._step = 0

        # Build initial beam, prepending begin_token if provided
        initial_tokens = list(req.prompt_token_ids)
        if req.sid_begin_token_id is not None:
            initial_tokens.append(req.sid_begin_token_id)

        self._all_beams = [
            _Beam(tokens=initial_tokens, cum_logprob=0.0)
        ]
        self._completed: list[_Beam] = []

        # Adjust max_tokens to account for begin/end tokens
        pre_calc = (1 if req.sid_begin_token_id is not None else 0) + \
                   (1 if req.sid_end_token_id is not None else 0)
        self._effective_max_tokens = req.max_tokens - pre_calc
        logger.debug(
            "BeamSearchController: req_id=%s, max_tokens=%d, pre_calc=%d, "
            "effective_max_tokens=%d, beam_width=%d, "
            "sid_begin=%s, sid_end=%s",
            req.request_id, req.max_tokens, pre_calc,
            self._effective_max_tokens, req.beam_width,
            req.sid_begin_token_id, req.sid_end_token_id,
        )
        if self._effective_max_tokens < 0:
            raise ValueError(
                f"max_tokens ({req.max_tokens}) is too small to "
                f"accommodate begin/end tokens (need at least {pre_calc})"
            )

        self._sampling_params = SamplingParams(
            logprobs=req.beam_width,
            max_tokens=1,
            temperature=req.temperature, detokenize=False
        )

        # Per-step tracking
        self._pending_rids: set[str] = set()
        self._beam_req_ids: list[str] = []
        self._output_by_rid: dict[str, tuple[list[int], object]] = {}
        self._step_t0: float = 0.0
        self._step_t1: float = 0.0

        self._done = False
        self._result: BeamSearchResult | None = None

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def result(self) -> BeamSearchResult:
        assert self._result is not None
        return self._result

    @property
    def client_index(self) -> int:
        return self._req.client_index

    @property
    def request_id(self) -> str:
        return self._req.request_id

    @property
    def is_timed_out(self) -> bool:
        """True if the current step has been pending longer than the timeout."""
        return (
            bool(self._pending_rids)
            and self._step_t0 > 0
            and (time.monotonic() - self._step_t0) > _STEP_TIMEOUT_S
        )

    def submit_step(self) -> None:
        """Create Request objects for current beams and add to scheduler."""
        from vllm.v1.request import Request

        # If effective_max_tokens is 0, finish immediately without
        # submitting any scheduler requests (honour the max_tokens contract).
        if self._step >= self._effective_max_tokens:
            self._finish()
            return

        self._step_t0 = time.monotonic()
        self._beam_req_ids = []
        self._output_by_rid = {}
        self._pending_rids = set()

        logger.debug(
            "submit_step: req_id=%s, step=%d, n_beams=%d",
            self._req.request_id, self._step, len(self._all_beams),
        )

        for i, beam in enumerate(self._all_beams):
            rid = f"{self._req.request_id}-s{self._step}-b{i}"
            self._beam_req_ids.append(rid)
            self._pending_rids.add(rid)
            r = Request(
                request_id=rid,
                prompt_token_ids=beam.tokens,
                sampling_params=self._sampling_params,
                pooling_params=None,
                eos_token_id=self._req.eos_token_id,
                block_hasher=self.block_hasher,
            )
            self.scheduler.add_request(r)

        self._step_t1 = time.monotonic()

    def on_outputs(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        """Scan step outputs for this controller's pending request IDs.

        When all beams for the current step have finished, perform beam
        selection and either submit the next step or mark as done.
        """
        if not self._pending_rids:
            return

        for _, engine_outputs in outputs.items():
            for out in engine_outputs.outputs:
                if out.request_id in self._pending_rids:
                    self._output_by_rid[out.request_id] = (
                        out.new_token_ids,
                        out.new_logprobs,
                    )
                    self._pending_rids.discard(out.request_id)

        if self._pending_rids:
            return

        # All beams for this step finished — run beam selection
        t2 = time.monotonic()
        self._beam_selection()
        t3 = time.monotonic()

        logger.debug(
            "beam step %d: %d beams, "
            "create=%.1fms, step=%.1fms, select=%.1fms, "
            "total=%.1fms",
            self._step,
            len(self._beam_req_ids),
            (self._step_t1 - self._step_t0) * 1000,
            (t2 - self._step_t1) * 1000,
            (t3 - t2) * 1000,
            (t3 - self._step_t0) * 1000,
        )

        self._step += 1

        logger.debug(
            "on_outputs: step now=%d, effective_max=%d, "
            "remaining_beams=%d, completed=%d, will_%s",
            self._step,
            self._effective_max_tokens,
            len(self._all_beams),
            len(self._completed),
            "submit_next" if (self._all_beams and self._step < self._effective_max_tokens) else "finish",
        )

        if self._all_beams and self._step < self._effective_max_tokens:
            self.submit_step()
        else:
            self._finish()

    def _beam_selection(self) -> None:
        """Select top beams from candidates using numpy vectorization."""
        # For small candidate counts, numpy overhead exceeds its
        # vectorization benefit.  Dispatch to pure-Python path.
        n_candidates = len(self._all_beams) * self._req.beam_width
        if n_candidates <= _NUMPY_THRESHOLD:
            self._beam_selection_python()
            return

        req = self._req

        # Phase 1 — Collect numpy arrays from outputs
        cum_logprobs: list[float] = []
        all_top_logprobs: list[np.ndarray] = []
        all_top_token_ids: list[np.ndarray] = []
        beam_map: list[int] = []  # maps numpy row → original beam index
        k_per_beam: int = -1

        for i, beam in enumerate(self._all_beams):
            rid = self._beam_req_ids[i]
            if rid not in self._output_by_rid:
                continue
            new_token_ids, new_logprobs = self._output_by_rid[rid]
            if new_logprobs is None:
                continue

            lp_token_ids = new_logprobs.logprob_token_ids
            lp_values = new_logprobs.logprobs

            top_token_ids = (
                lp_token_ids[0] if lp_token_ids.ndim == 2 else lp_token_ids
            )
            top_logprobs = (
                lp_values[0] if lp_values.ndim == 2 else lp_values
            )

            k = len(top_token_ids)
            if k_per_beam == -1:
                k_per_beam = k
            elif k != k_per_beam:
                # Non-uniform sizes — fall back to pure-Python path
                self._beam_selection_python()
                return

            cum_logprobs.append(beam.cum_logprob)
            all_top_logprobs.append(np.asarray(top_logprobs))
            all_top_token_ids.append(np.asarray(top_token_ids))
            beam_map.append(i)

        if not cum_logprobs:
            self._all_beams = []
            return

        n_valid = len(cum_logprobs)

        # Phase 2 — Vectorized candidate generation
        cum_arr = np.array(cum_logprobs)               # (n_valid,)
        lp_mat = np.vstack(all_top_logprobs)            # (n_valid, k)
        tid_mat = np.vstack(all_top_token_ids)          # (n_valid, k)
        combined = cum_arr[:, np.newaxis] + lp_mat      # (n_valid, k)

        # Mask out -inf values
        combined[~np.isfinite(combined)] = -np.inf

        # Phase 3 — Top-k partial sort
        flat = combined.ravel()
        n_total = flat.shape[0]
        top_k = min(req.beam_width * 3, n_total)

        # argpartition is O(n) — much faster than full sort for large arrays
        if top_k < n_total:
            part_idx = np.argpartition(-flat, top_k)[:top_k]
        else:
            part_idx = np.arange(n_total)

        # Sort only the top_k candidates
        top_vals = flat[part_idx]
        sorted_order = np.argsort(-top_vals)
        sorted_flat_idx = part_idx[sorted_order]

        # Convert flat indices back to (beam_row, token_offset)
        k = k_per_beam
        beam_rows = sorted_flat_idx // k
        token_offsets = sorted_flat_idx % k

        # Phase 4 — Build beams (dedup/EOS logic, ~beam_width iterations)
        new_beams: list[_Beam] = []
        seen_sequences: set[tuple[int, ...]] = set()

        for idx in range(len(sorted_flat_idx)):
            if len(new_beams) >= req.beam_width:
                break

            row = int(beam_rows[idx])
            col = int(token_offsets[idx])
            cum_lp = float(flat[sorted_flat_idx[idx]])
            if cum_lp == float("-inf"):
                continue

            parent_idx = beam_map[row]
            token_id = int(tid_mat[row, col])

            parent = self._all_beams[parent_idx]
            new_tokens = parent.tokens + [token_id]
            seq_tuple = tuple(new_tokens)

            if seq_tuple not in seen_sequences:
                if not req.ignore_eos and token_id == req.eos_token_id:
                    self._completed.append(
                        _Beam(tokens=new_tokens, cum_logprob=cum_lp)
                    )
                else:
                    new_beams.append(
                        _Beam(tokens=new_tokens, cum_logprob=cum_lp)
                    )
                    seen_sequences.add(seq_tuple)

        self._all_beams = new_beams

    def _beam_selection_python(self) -> None:
        """Pure-Python fallback for non-uniform logprob sizes."""
        req = self._req
        all_candidates: list[tuple[float, int, int]] = []

        for i, beam in enumerate(self._all_beams):
            rid = self._beam_req_ids[i]
            if rid not in self._output_by_rid:
                continue
            new_token_ids, new_logprobs = self._output_by_rid[rid]
            if new_logprobs is None:
                continue

            lp_token_ids = new_logprobs.logprob_token_ids
            lp_values = new_logprobs.logprobs

            top_token_ids = (
                lp_token_ids[0] if lp_token_ids.ndim == 2 else lp_token_ids
            )
            top_logprobs = (
                lp_values[0] if lp_values.ndim == 2 else lp_values
            )

            for j in range(len(top_token_ids)):
                tid = int(top_token_ids[j])
                lp = float(top_logprobs[j])
                if lp == float("-inf"):
                    continue

                all_candidates.append((beam.cum_logprob + lp, i, tid))

        if not all_candidates:
            self._all_beams = []
            return

        all_candidates.sort(reverse=True)

        new_beams: list[_Beam] = []
        seen_sequences: set[tuple[int, ...]] = set()

        for cum_lp, parent_idx, token_id in all_candidates:
            if len(new_beams) >= req.beam_width:
                break

            parent = self._all_beams[parent_idx]
            new_tokens = parent.tokens + [token_id]
            seq_tuple = tuple(new_tokens)

            if seq_tuple not in seen_sequences:
                if not req.ignore_eos and token_id == req.eos_token_id:
                    self._completed.append(
                        _Beam(tokens=new_tokens, cum_logprob=cum_lp)
                    )
                else:
                    new_beams.append(
                        _Beam(tokens=new_tokens, cum_logprob=cum_lp)
                    )
                    seen_sequences.add(seq_tuple)

        self._all_beams = new_beams

    def _finish(self) -> None:
        """Build the final BeamSearchResult."""
        logger.debug(
            "_finish: req_id=%s, completed=%d, remaining_beams=%d",
            self._req.request_id,
            len(self._completed),
            len(self._all_beams),
        )
        req = self._req

        if not self._completed and not self._all_beams:
            logger.warning(
                "_finish: req_id=%s, all beams were pruned — "
                "returning empty result",
                req.request_id,
            )
            self._result = BeamSearchResult(
                request_id=req.request_id,
                sequences=[],
            )
            self._done = True
            return

        # Track which beams completed naturally via EOS BEFORE appending
        # sid_end_token_id, to avoid misidentifying finish_reason when
        # sid_end_token_id == eos_token_id.
        eos_completed: set[int] = {id(b) for b in self._completed}

        # Append end_token to all beams if provided
        if req.sid_end_token_id is not None:
            for beam in self._completed:
                beam.tokens.append(req.sid_end_token_id)
            for beam in self._all_beams:
                beam.tokens.append(req.sid_end_token_id)

        def sort_key(b: _Beam) -> float:
            return _get_beam_search_score(
                b.tokens, b.cum_logprob, req.eos_token_id, req.length_penalty
            )

        all_candidates = self._completed + self._all_beams
        all_candidates.sort(key=sort_key, reverse=True)
        best = all_candidates[: req.n_best]

        prompt_len = len(req.prompt_token_ids)
        sequences = []
        for beam in best:
            if not req.ignore_eos and id(beam) in eos_completed:
                finish_reason = "stop"
                stop_reason: int | str | None = req.eos_token_id
            else:
                finish_reason = "length"
                stop_reason = None

            output_tokens = beam.tokens[prompt_len:]
            # Strip the EOS token from output unless explicitly requested.
            if (
                finish_reason == "stop"
                and not req.include_stop_str_in_output
                and output_tokens
                and output_tokens[-1] == req.eos_token_id
            ):
                output_tokens = output_tokens[:-1]

            sequences.append(
                BeamSearchSequenceResult(
                    token_ids=output_tokens,
                    cumulative_logprob=beam.cum_logprob,
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
            )

        self._result = BeamSearchResult(
            request_id=req.request_id,
            sequences=sequences,
        )
        self._done = True
