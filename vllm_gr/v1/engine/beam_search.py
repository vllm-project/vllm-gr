# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from typing import TYPE_CHECKING

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
    if tokens[-1] == eos_token_id:
        seq_len -= 1
    return cumulative_logprob / (seq_len**length_penalty)


class _Beam:
    __slots__ = ("tokens", "cum_logprob")

    def __init__(self, tokens: list[int], cum_logprob: float):
        self.tokens = tokens
        self.cum_logprob = cum_logprob


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
            temperature=req.temperature,
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

    def submit_step(self) -> None:
        """Create Request objects for current beams and add to scheduler."""
        from vllm.v1.request import Request

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

        logger.info(
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
        """Select top beams from candidates (identical to original logic)."""
        req = self._req
        all_candidates: list[dict] = []

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

                all_candidates.append({
                    "logprob": beam.cum_logprob + lp,
                    "parent_idx": i,
                    "token_id": tid,
                })

        if not all_candidates:
            self._all_beams = []
            return

        all_candidates.sort(key=lambda x: x["logprob"], reverse=True)

        new_beams: list[_Beam] = []
        seen_sequences: set[tuple[int, ...]] = set()

        for cand in all_candidates:
            if len(new_beams) >= req.beam_width:
                break

            parent = self._all_beams[cand["parent_idx"]]
            new_tokens = parent.tokens + [cand["token_id"]]
            seq_tuple = tuple(new_tokens)

            if seq_tuple not in seen_sequences:
                if not req.ignore_eos and cand["token_id"] == req.eos_token_id:
                    self._completed.append(
                        _Beam(
                            tokens=new_tokens,
                            cum_logprob=cand["logprob"],
                        )
                    )
                else:
                    new_beams.append(
                        _Beam(
                            tokens=new_tokens,
                            cum_logprob=cand["logprob"],
                        )
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
            if (
                not req.ignore_eos
                and beam.tokens
                and beam.tokens[-1] == req.eos_token_id
            ):
                finish_reason = "stop"
                stop_reason: int | str | None = req.eos_token_id
            else:
                finish_reason = "length"
                stop_reason = None

            sequences.append(
                BeamSearchSequenceResult(
                    token_ids=beam.tokens[prompt_len:],
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
