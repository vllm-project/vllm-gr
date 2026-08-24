# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side beam-search session state machine.

Mirrors the frontend ``OpenAIServing.beam_search`` loop, but runs entirely
inside the EngineCore process: each wave is submitted as a regular scheduled
request and the wave outputs are intercepted before they reach the output
queue.  When all beams finish, a single final ``EngineCoreOutputs`` is
emitted for the session, carrying the per-beam results and the beam-search
timing stats.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.utils import random_uuid
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.outputs import LogprobsLists

from vllm_gr.beam_search_decision_utils import (
    BeamCandidateCache,
    execute_wave_decision,
    extract_and_dedup_segment,
    finalize_best_beams,
)
from vllm_gr.logger import init_logger
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.v1.engine.types import BeamSearchRequest
from vllm_gr.v1.metrics.stats import RequestStateStats

if TYPE_CHECKING:
    from vllm.utils.catalog import Catalog as _Catalog  # noqa: F401

    Catalog = Any
else:
    Catalog = Any

logger = init_logger(__name__)

# Key under the final EngineCoreOutput.kv_transfer_params carrying the
# serialized RequestStateStats beam fields for the frontend metrics recording.
BEAM_STATS_HEADER_KEY = "_gr_beam_stats"
# Suffix of the single final output's request id (the frontend registers one
# queue for it and unpacks the packed beam tokens).
BEAM_FINAL_SUFFIX = "-beam-final"
# kv_transfer_params keys describing the packed final tokens.
BEAM_LENS_KEY = "_gr_beam_lens"
BEAM_TOKENS_STRIDE_KEY = "_gr_beam_tokens_stride"
BEAM_CUMS_KEY = "_gr_beam_cums"
BEAM_FINISH_FLAGS_KEY = "_gr_beam_finish_flags"
BEAM_STOP_REASONS_KEY = "_gr_beam_stop_reasons"
# Per-beam reconstructed FlatLogprobs serialized as
# [start_indices, end_indices, token_ids, logprobs, ranks, decoded_tokens].
BEAM_LOGPROBS_KEY = "_gr_beam_logprobs"


def _flat_logprobs_from_lists(new_logprobs: LogprobsLists, num_logprobs: int) -> FlatLogprobs:
    """Convert ``LogprobsLists`` into the flat layout the decision helpers use.

    Replicates ``LogprobsProcessor._update_sample_logprobs`` with the
    vLLM-GR ``FlatLogprobs.append_fast`` semantics (decoded tokens are left
    as ``None`` — no detokenizer lives in the engine process).
    """
    from vllm_gr.logprobs import append_fast as gr_append_fast

    flat = FlatLogprobs()
    token_ids_arr, logprobs_arr, ranks_arr, _ = new_logprobs
    for ranks_row, logprobs_row, token_ids_row in zip(ranks_arr, logprobs_arr, token_ids_arr):
        token_ids = token_ids_row.tolist()
        logprobs = logprobs_row.tolist()
        rank = ranks_row.tolist()
        ranks: itertools.chain[int] = itertools.chain((rank,), range(1, num_logprobs + 1))
        gr_append_fast(flat, token_ids, logprobs, ranks, itertools.repeat(None))
    return flat


class BeamSearchSession:
    """One engine-side beam-search request lifecycle."""

    _vocab_slice_configured: bool = False

    def __init__(
        self,
        engine_core: Any,
        request: BeamSearchRequest,
        catalog: Catalog | None,
        catalog_pending: bool = False,
    ) -> None:
        self.engine_core = engine_core
        self.session_id = request.session_id
        self.client_index = request.client_index
        self.prompt_token_ids = list(request.prompt_token_ids)
        self.beam_width = request.beam_width
        self.max_tokens = request.max_tokens
        self.ignore_eos = request.ignore_eos
        self.include_stop_str_in_output = request.include_stop_str_in_output
        self.catalog_strategy = request.catalog_strategy
        self.catalog = catalog
        # When a catalog was requested but its trie is still being built in a
        # background thread, the wave decisions wait for ``_catalog_ready``.
        self._catalog_pending = catalog_pending
        self._catalog_ready = threading.Event()
        if not catalog_pending:
            self._catalog_ready.set()
        self.lora_request = request.lora_request
        self.trace_headers = dict(request.trace_headers or {})
        self.priority = request.priority
        self.data_parallel_rank = request.data_parallel_rank
        self.eos_token_id = request.eos_token_id
        self.sid_begin_token_id = request.sid_begin_token_id
        self.sid_end_token_id = request.sid_end_token_id
        self.beam_search_params = request.sampling_params
        self.logprobs_num = self.beam_width
        assert self.logprobs_num > 0, "logprobs_num (beam_width) must be > 0"

        # Mirrors the frontend: sliced-logits token ids must be remapped to
        # full-vocab ids inside extract_and_dedup_segment.  The config is
        # process-global and only needs to be resolved once.
        if not BeamSearchSession._vocab_slice_configured:
            from vllm_gr.config.models import GRConfig
            from vllm_gr.ops.sliced_logits import set_vocab_slice_config

            gr_config = GRConfig.from_vllm_config(getattr(engine_core, "vllm_config", None))
            if gr_config.vocab_offset is not None and gr_config.vocab_slice_size is not None:
                set_vocab_slice_config(gr_config.vocab_offset, gr_config.vocab_slice_size)
            BeamSearchSession._vocab_slice_configured = True

        self.tokenized_length = len(self.prompt_token_ids)

        self.initial_tokens = list(self.prompt_token_ids)
        self.initial_logprobs = FlatLogprobs()
        if self.sid_begin_token_id is not None:
            self.initial_tokens.append(self.sid_begin_token_id)
            self.initial_logprobs.append_fast([self.sid_begin_token_id], [0.0], [None], [None])

        pre_calc = (1 if self.sid_begin_token_id is not None else 0) + (
            1 if self.sid_end_token_id is not None else 0
        )
        self.max_waves = self.max_tokens - pre_calc

        self.request_id_batch = f"{self.session_id}-{random_uuid()}"

        self.all_beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                orig_prompt=tokens_input(self.initial_tokens),
                tokens=list(self.initial_tokens),
                cum_logprob=0,
                logprobs=self.initial_logprobs,
                lora_request=self.lora_request,
            )
        ]
        self.all_beams[0]._lp_parent = None
        self.all_beams[0]._lp_step_data = None
        self.completed: list[BeamSearchSequence] = []

        # Prefill phase state.
        self.prefill_id: str | None = None
        self.beam_request_ids: list[str] = []
        self._pending_prefill_outputs = 0
        self._prefill_collected: dict[str, EngineCoreOutput] = {}

        # Decode phase state.
        self.wave: int = 0  # Number of generated tokens per active beam.
        self.fork_info: list[tuple[int, int]] | None = None

        # Timing (mirrors the frontend beam-search metrics semantics).
        self.session_start = time.perf_counter()
        self.decode_start: float | None = None
        self.generation_time = 0.0
        self.num_generation_tokens = 0
        self.sort_time = 0.0
        self._wave_submit_ts: float | None = None

        self.finished = False
        self.aborted = False

        # Serializes the session's state transitions (wave decisions / wave
        # submissions) against abort(): the input thread and the step loop
        # both reach these paths.
        self._transition_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _submit_prefill_wave(self) -> None:
        """Submit the single initial prefill request (ADD path).

        Mirrors the frontend: the first wave starts from one beam and the
        wave-0 selection expands it to ``beam_width`` beams.
        """
        with self._transition_lock:
            self._submit_prefill_wave_locked()

    def _submit_prefill_wave_locked(self) -> None:
        from vllm.v1.engine import EngineCoreRequestType

        request_id_item = f"{self.request_id_batch}-beam-0"
        ec_req = EngineCoreRequest(
            request_id=request_id_item,
            prompt_token_ids=self.initial_tokens,
            mm_features=None,
            sampling_params=self.beam_search_params,
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=self.lora_request,
            cache_salt=None,
            data_parallel_rank=self.data_parallel_rank,
            trace_headers=self.trace_headers,
            priority=self.priority,
            client_index=self.client_index,
        )
        preprocessed = self.engine_core.preprocess_add_request(ec_req)
        self.engine_core._cache_beam_request(preprocessed[0])
        self.engine_core.input_queue.put_nowait((EngineCoreRequestType.ADD, preprocessed))
        self.beam_request_ids.append(request_id_item)
        self.prefill_id = request_id_item
        self._pending_prefill_outputs = 1
        self._wave_submit_ts = time.perf_counter()

    def _submit_decode_wave(self) -> EngineCoreOutputs | None:
        """Submit one packed decode wave request for all active beams."""
        from vllm.v1.engine import EngineCoreRequestType
        from vllm.v1.request import Request

        assert self.prefill_id is not None
        if self.decode_start is None:
            # Mirrors the frontend: the decode phase starts when the first
            # decode wave is submitted (before its execution).
            self.decode_start = time.perf_counter()
        # Mirrors the frontend's per-decode-wave accounting: count the active
        # beams for THIS wave only (the prefill wave and pruned widths are
        # excluded, so the total reflects actually generated tokens).
        self.num_generation_tokens += len(self.all_beams)
        with self.engine_core.beam_cache_lock:
            session_cached = self.engine_core.beam_cache.get(self.prefill_id)
        if session_cached is None:
            logger.error("BEAM_SEARCH: session %s prefill cache entry missing", self.session_id)
            return self._finish_with_error()

        decode_sampling_params = clone_beam_decode_sampling_params(self.beam_search_params)
        if self.eos_token_id is not None:
            decode_sampling_params.update_from_generation_config({}, self.eos_token_id)

        beam_token_ids = list(session_cached["all_token_ids"])
        prefix_len = len(self.initial_tokens)
        for beam in self.all_beams:
            beam_token_ids.extend(beam.tokens[prefix_len:])

        req = Request(
            request_id=self.session_id,
            prompt_token_ids=beam_token_ids,
            sampling_params=decode_sampling_params,
            pooling_params=None,
            client_index=self.client_index,
            arrival_time=time.time(),
            lora_request=self.lora_request,
            cache_salt=session_cached["cache_salt"],
            priority=self.priority,
            trace_headers=self.trace_headers,
            prompt_embeds=session_cached["prompt_embeds"],
            prompt_is_token_ids=session_cached["prompt_is_token_ids"],
            mm_features=session_cached["mm_features"],
            block_hasher=None,
        )
        req.block_hashes = list(session_cached["block_hashes"])
        if self.engine_core.request_block_hasher is not None:
            req._block_hasher = self.engine_core.request_block_hasher
            req.update_block_hashes()

        req.is_beam = True
        req.beam_width = self.beam_width
        req.is_beam_decode = True
        req.beam_decode_steps = self.wave
        req.prefix_len = prefix_len

        self.engine_core.input_queue.put_nowait((EngineCoreRequestType.ADD, (req, 0)))
        self._wave_submit_ts = time.perf_counter()
        return None

    # ------------------------------------------------------------------
    # Wave output handling (called from the patched step loop)
    # ------------------------------------------------------------------

    def on_wave_output(self, engine_core_outputs: EngineCoreOutputs) -> EngineCoreOutputs | None:
        """Consume a wave's outputs; return the final outputs to forward, if done."""
        with self._transition_lock:
            if self.aborted or self.finished:
                return None

            if self.wave == 0:
                return self._on_prefill_wave_output(engine_core_outputs)
            return self._on_decode_wave_output(engine_core_outputs)

    def _on_prefill_wave_output(
        self, engine_core_outputs: EngineCoreOutputs
    ) -> EngineCoreOutputs | None:
        for output in engine_core_outputs.outputs:
            if output.request_id in self.beam_request_ids:
                self._prefill_collected[output.request_id] = output
                self._pending_prefill_outputs -= 1

        # Wait until every beam's prefill output arrived (or an error occurred).
        error = any(
            output.finish_reason == FinishReason.ERROR or output.num_nans_in_logits > 0
            for output in self._prefill_collected.values()
        )
        if error or self._pending_prefill_outputs <= 0:
            if error:
                return self._finish_with_error()
            per_beam: list[BeamCandidateCache | None] = []
            for request_id in self.beam_request_ids:
                output = self._prefill_collected.get(request_id)
                if output is None or output.new_logprobs is None:
                    per_beam.append(None)
                    continue
                flat = _flat_logprobs_from_lists(output.new_logprobs, self.logprobs_num)
                per_beam.append(
                    extract_and_dedup_segment(
                        flat.token_ids,
                        flat.logprobs,
                        flat.ranks,
                        flat.decoded_tokens,
                        flat.start_indices[0],
                        flat.end_indices[0],
                        self.logprobs_num,
                    )
                )
            return self._decide_wave(per_beam)
        return None

    def _on_decode_wave_output(
        self, engine_core_outputs: EngineCoreOutputs
    ) -> EngineCoreOutputs | None:
        for output in engine_core_outputs.outputs:
            if output.request_id != self.session_id:
                continue
            if output.finish_reason == FinishReason.ERROR or output.num_nans_in_logits > 0:
                return self._finish_with_error()
            if output.new_logprobs is None:
                return None
            flat = _flat_logprobs_from_lists(output.new_logprobs, self.logprobs_num)
            total = len(flat.token_ids)
            if total == 0:
                return None
            stride_k = total // self.logprobs_num
            per_beam: list[BeamCandidateCache | None] = []
            for b_idx in range(len(self.fork_info or self.all_beams)):
                start_offset = b_idx * stride_k
                end_offset = start_offset + stride_k
                per_beam.append(
                    extract_and_dedup_segment(
                        flat.token_ids,
                        flat.logprobs,
                        flat.ranks,
                        flat.decoded_tokens,
                        start_offset,
                        end_offset,
                        self.logprobs_num,
                    )
                )
            return self._decide_wave(per_beam)
        return None

    # ------------------------------------------------------------------
    # Decision (mirrors the frontend loop body)
    # ------------------------------------------------------------------

    def _decide_wave(self, per_beam: list[BeamCandidateCache | None]) -> EngineCoreOutputs | None:
        if self._catalog_pending:
            # The catalog trie is still being built in the background; the
            # prefill wave ran in parallel, so this usually returns at once.
            self._catalog_ready.wait()
            if self.catalog is None:
                logger.error("BEAM_SEARCH: session %s catalog build failed", self.session_id)
                return self._finish_with_error()
            self._catalog_pending = False

        # Catalog filtering (engine-side catalog).
        valid_tokens_sets = None
        if self.catalog is not None:
            valid_tokens_sets = [
                self.catalog.valid(beam.tokens[self.tokenized_length :]) for beam in self.all_beams
            ]

        decision = execute_wave_decision(
            all_beams=self.all_beams,
            per_beam_candidates=per_beam,
            catalog=self.catalog,
            catalog_strategy=self.catalog_strategy,
            valid_tokens_sets=valid_tokens_sets,
            beam_width=self.beam_width,
            eos_token_id=self.eos_token_id,
            sid_end_token_id=self.sid_end_token_id,
            ignore_eos=self.ignore_eos,
            include_stop_str_in_output=self.include_stop_str_in_output,
            initial_logprobs=self.initial_logprobs,
        )
        self.completed.extend(decision.new_completed_eos)
        self.fork_info = decision.fork_info

        if self.wave > 0 and self._wave_submit_ts is not None:
            self.generation_time += time.perf_counter() - self._wave_submit_ts
            self._wave_submit_ts = None

        if not decision.new_beams:
            self.all_beams = []
            return self._finish_session()

        self.all_beams = decision.new_beams
        self.wave += 1
        if self.wave >= self.max_waves:
            return self._finish_session()

        return self._submit_decode_wave()

    # ------------------------------------------------------------------
    # Finish / abort
    # ------------------------------------------------------------------

    def _final_output_id(self) -> str:
        """Request id of the single packed final output, so the frontend
        registers one output queue and unpacks the per-beam tokens."""
        return f"{self.session_id}{BEAM_FINAL_SUFFIX}"

    def _finish_with_error(self) -> EngineCoreOutputs:
        self._cleanup_cache()
        self.finished = True
        return EngineCoreOutputs(
            engine_index=self.engine_core.engine_index,
            # The emitted output is keyed by the final output id, so the
            # terminal signal must reference it as well (the session id is
            # kept for the client's reqs_in_flight cleanup).
            finished_requests={self.session_id, self._final_output_id()},
            outputs=[
                EngineCoreOutput(
                    request_id=self._final_output_id(),
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                )
            ],
        )

    def _finish_session(self) -> EngineCoreOutputs:
        best_beams, self.sort_time = finalize_best_beams(
            all_beams=self.all_beams,
            completed=self.completed,
            beam_width=self.beam_width,
            initial_logprobs=self.initial_logprobs,
            sid_end_token_id=self.sid_end_token_id,
        )

        decode_time = (
            (time.perf_counter() - self.decode_start) if self.decode_start is not None else 0.0
        )
        prefill_time = (
            (self.decode_start - self.session_start)
            if self.decode_start is not None
            else (time.perf_counter() - self.session_start)
        )
        stats = RequestStateStats(
            beam_search_overhead=decode_time - self.generation_time,
            beam_search_prefill_time=prefill_time,
            beam_search_decode_time=decode_time,
            beam_search_sort_time=self.sort_time,
            num_generation_tokens=self.num_generation_tokens,
            num_beams=self.beam_width,
        )

        # Pack all beams into ONE final output (padded to a fixed stride) so
        # the frontend processes a single output instead of one per beam.
        trimmed: list[list[int]] = []
        for beam in best_beams:
            trimmed.append(
                beam.tokens[self.tokenized_length : -1]
                if beam.tokens[-1] == self.eos_token_id and not self.ignore_eos
                else beam.tokens[self.tokenized_length :]
            )
        stride = max((len(tokens) for tokens in trimmed), default=0)
        packed: list[int] = []
        for tokens in trimmed:
            packed.extend(tokens)
            packed.extend([-1] * (stride - len(tokens)))

        final_output = EngineCoreOutput(
            request_id=self._final_output_id(),
            new_token_ids=packed,
            finish_reason=FinishReason.STOP,
            kv_transfer_params={
                BEAM_STATS_HEADER_KEY: asdict(stats),
                BEAM_LENS_KEY: [len(tokens) for tokens in trimmed],
                BEAM_TOKENS_STRIDE_KEY: stride,
                BEAM_CUMS_KEY: [beam.cum_logprob for beam in best_beams],
                BEAM_FINISH_FLAGS_KEY: [
                    1 if beam.finish_reason == "stop" else 0 for beam in best_beams
                ],
                BEAM_STOP_REASONS_KEY: [beam.stop_reason for beam in best_beams],
                # Per-beam reconstructed logprobs (finalize_best_beams already
                # rebuilt beam.logprobs via the parent-pointer tree); the
                # frontend rebuilds FlatLogprobs objects from these lists.
                BEAM_LOGPROBS_KEY: [
                    [
                        list(lp.start_indices),
                        list(lp.end_indices),
                        list(lp.token_ids),
                        list(lp.logprobs),
                        list(lp.ranks),
                        list(lp.decoded_tokens),
                    ]
                    for beam in best_beams
                    for lp in [beam.logprobs]
                ],
            },
        )

        self._cleanup_cache()
        self.finished = True
        return EngineCoreOutputs(
            engine_index=self.engine_core.engine_index,
            # Terminal signal keyed to the emitted output id; the session id
            # is kept for the client's reqs_in_flight cleanup.
            finished_requests={self.session_id, self._final_output_id()},
            outputs=[final_output],
            timestamp=time.time(),
        )

    def _cleanup_cache(self) -> None:
        with self.engine_core.beam_cache_lock:
            if self.prefill_id is not None:
                self.engine_core.beam_cache.pop(self.prefill_id, None)
            for request_id in self.beam_request_ids:
                self.engine_core.beam_cache.pop(request_id, None)

    def abort(self) -> None:
        """Abort the session and the in-flight wave request."""
        with self._transition_lock:
            if self.aborted or self.finished:
                return
            self.aborted = True
            self.engine_core.aborts_queue.put_nowait(self.session_id)
            for request_id in self.beam_request_ids:
                self.engine_core.aborts_queue.put_nowait(request_id)
            self._cleanup_cache()
