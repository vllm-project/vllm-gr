# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the engine-side beam-search session state machine."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.outputs import LogprobsLists

from vllm_gr.beam_search_decision_utils import (
    extract_and_dedup_segment,
    select_top_indices,
)
from vllm_gr.v1.beam.search_engine import (
    BEAM_LOGPROBS_KEY,
    BEAM_STATS_HEADER_KEY,
    BeamSearchSession,
)
from vllm_gr.v1.engine.types import BeamSearchRequest

BEAM_WIDTH = 2
TOPK = 2  # logprobs per beam == beam_width
MAX_TOKENS = 5

# Deterministic candidate scores: parent beam p, candidate j -> score.
# Parent 1 is always dominated so both active beams descend from parent 0.
SCORES = {
    0: [-0.1, -0.3, -0.6],
    1: [-10.0, -10.0, -10.0],
}


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put_nowait(self, item: Any) -> None:
        self.items.append(item)


class FakeEngineCore:
    preprocess_add_request: Any = None
    _cache_beam_request: Any = None

    def __init__(self) -> None:
        self.input_queue = _FakeQueue()
        self.aborts_queue = _FakeQueue()
        self.output_queue = _FakeQueue()
        self.beam_cache: dict[str, dict[str, Any]] = {}
        self.beam_cache_lock = threading.Lock()
        self._gr_beam_sessions_lock = threading.Lock()
        self.request_block_hasher = None
        self.vllm_config = SimpleNamespace(
            additional_config={},
        )
        self.engine_index = 0
        self._gr_beam_sessions: dict[str, Any] = {}
        self._gr_catalog_cache: dict[Any, Any] = {}
        self._gr_catalog_cache_sizes: dict[Any, int] = {}
        self._gr_catalog_cache_size = 0
        self._gr_catalog_cache_budget: int | None = None


class FakeRequest:
    def __init__(self, request_id: str, prompt_token_ids: list[int], client_index: int = 0):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.client_index = client_index


def _fake_preprocess(self: Any, ec_req: Any) -> tuple[FakeRequest, int]:
    return (FakeRequest(ec_req.request_id, ec_req.prompt_token_ids, ec_req.client_index), 0)


def _fake_cache_beam_request(self: Any, request: Any) -> None:
    self.beam_cache[request.request_id] = {
        "all_token_ids": list(request.prompt_token_ids),
        "block_hashes": [],
        "cache_salt": None,
        "mm_features": None,
        "prompt_embeds": None,
        "prompt_is_token_ids": None,
        "lora_request": None,
        "eos_token_id": None,
        "num_prompt_tokens": len(request.prompt_token_ids),
    }


def _make_request(prompt: list[int]) -> BeamSearchRequest:
    return BeamSearchRequest(
        session_id="session-0",
        prompt_token_ids=prompt,
        beam_width=BEAM_WIDTH,
        max_tokens=MAX_TOKENS,
        sampling_params=SamplingParams(logprobs=TOPK, max_tokens=1, detokenize=False),
        eos_token_id=None,
        ignore_eos=True,
        priority=42,
    )


def _prefill_output(request_id: str, beam_idx: int) -> EngineCoreOutput:
    sampled, *topk = [100 + beam_idx * 10 + j for j in range(TOPK + 1)]
    token_ids = np.array([[sampled, *topk]], dtype=np.int64)
    logprobs = np.array([SCORES[beam_idx]], dtype=np.float32)
    ranks = np.array([[1]], dtype=np.int64)
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[sampled],
        finish_reason=FinishReason.LENGTH,
        new_logprobs=LogprobsLists(token_ids, logprobs, ranks, None),
    )


def _decode_output(
    request_id: str,
    beam_token_ids: list[int],
    parents: list[int],
) -> EngineCoreOutput:
    """Build the remuxed flat decode-wave logprobs for the active beams."""
    flat_ids = []
    flat_lps = []
    for idx, tid in enumerate(beam_token_ids):
        parent = parents[idx]
        scores = list(SCORES[parent])
        if idx > 0:
            scores = [scores[0], scores[1] - 0.01 * idx, scores[2]]
        sampled = tid
        topk = [tid + 10 + j for j in range(1, TOPK + 1)]
        flat_ids.extend([sampled, *topk])
        flat_lps.extend(scores)
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=beam_token_ids,
        finish_reason=FinishReason.LENGTH,
        new_logprobs=LogprobsLists(
            np.array([flat_ids], dtype=np.int64),
            np.array([flat_lps], dtype=np.float32),
            np.array([1], dtype=np.int64),
            None,
        ),
    )


def _simulate_reference() -> list[list[int]]:
    """Replay the decision pipeline on the same scores to get the expected
    final beam token sequences (prefix-trimmed)."""
    from vllm_gr.v1.beam.search_engine import _flat_logprobs_from_lists

    logprobs_num = BEAM_WIDTH
    beam_tokens: list[list[int]] = [[] for _ in range(BEAM_WIDTH)]
    beam_cums: list[float] = [0.0] * BEAM_WIDTH
    beam_parents: list[int] = list(range(BEAM_WIDTH))

    def decide(
        cand_tokens: list[int],
        cand_scores: list[float],
        cand_beam_idx: list[int],
    ) -> tuple[list[list[int]], list[float], list[int]]:
        idx = select_top_indices(np.array(cand_scores), BEAM_WIDTH, "fill")
        new_tokens = [list(beam_tokens[int(cand_beam_idx[i])]) + [cand_tokens[i]] for i in idx]
        new_cums = [float(cand_scores[i]) for i in idx]
        new_parents = [int(cand_beam_idx[i]) for i in idx]
        return new_tokens, new_cums, new_parents

    # Wave 0 (prefill): a single initial beam's block [sampled, top1, top2]
    # expands to BEAM_WIDTH beams (mirrors the frontend ADD_BATCH wave).
    sampled, *topk = [100 + j for j in range(TOPK + 1)]
    block = extract_and_dedup_segment(
        [sampled, *topk], SCORES[0], [1, 2, 3], None, 0, TOPK + 1, logprobs_num
    )
    cand_tokens = list(block[0])
    cand_scores = list(block[1])
    cand_beam_idx = [0] * len(block[0])
    beam_tokens, beam_cums, beam_parents = decide(cand_tokens, cand_scores, cand_beam_idx)

    for wave in range(1, MAX_TOKENS):
        flat_ids: list[int] = []
        flat_lps: list[float] = []
        for idx in range(BEAM_WIDTH):
            parent = beam_parents[idx]
            scores = list(SCORES[parent])
            if idx > 0:
                scores = [scores[0], scores[1] - 0.01 * idx, scores[2]]
            sampled = 200 + wave * 10 + idx
            flat_ids.extend([sampled, sampled + 10 + 1, sampled + 10 + 2])
            flat_lps.extend(scores)
        new_logprobs = LogprobsLists(
            np.array([flat_ids], dtype=np.int64),
            np.array([flat_lps], dtype=np.float32),
            np.array([[1]], dtype=np.int64),
            None,
        )
        flat = _flat_logprobs_from_lists(new_logprobs, logprobs_num)
        total = len(flat.token_ids)
        stride = total // logprobs_num
        cand_tokens = []
        cand_scores = []
        cand_beam_idx = []
        for idx in range(BEAM_WIDTH):
            block = extract_and_dedup_segment(
                flat.token_ids,
                flat.logprobs,
                flat.ranks,
                flat.decoded_tokens,
                idx * stride,
                (idx + 1) * stride,
                logprobs_num,
            )
            cand_tokens.extend(block[0])
            cand_scores.extend(beam_cums[idx] + s for s in block[1])
            cand_beam_idx.extend([idx] * len(block[0]))
        beam_tokens, beam_cums, beam_parents = decide(cand_tokens, cand_scores, cand_beam_idx)
    return beam_tokens


def test_session_full_loop() -> None:
    engine = FakeEngineCore()
    engine.preprocess_add_request = _fake_preprocess.__get__(engine)
    engine._cache_beam_request = _fake_cache_beam_request.__get__(engine)
    session = BeamSearchSession(engine, _make_request([1, 2, 3]), catalog=None)
    engine._gr_beam_sessions["session-0"] = session

    # Prefill wave submission: one request (mirrors the frontend ADD_BATCH wave).
    session._submit_prefill_wave()
    assert len(engine.input_queue.items) == 1
    assert session.prefill_id == session.beam_request_ids[0]
    assert session.prefill_id in engine.beam_cache

    # Prefill wave output -> wave-0 selection expands to W beams, decode wave 1 enqueued.
    result = session.on_wave_output(
        EngineCoreOutputs(
            engine_index=0,
            outputs=[_prefill_output(session.beam_request_ids[0], 0)],
        )
    )
    assert result is None
    assert session.wave == 1
    assert len(session.all_beams) == BEAM_WIDTH
    assert len(engine.input_queue.items) == 2
    packed = engine.input_queue.items[-1][1][0]
    assert packed.request_id == "session-0"
    assert packed.is_beam_decode is True
    assert packed.beam_width == BEAM_WIDTH

    # Decode waves 1..MAX_TOKENS-2, then the final wave finishes the session.
    for wave in range(1, MAX_TOKENS):
        expected_tokens = [200 + wave * 10 + idx for idx in range(BEAM_WIDTH)]
        parents = session.fork_info or [(i, 0) for i in range(BEAM_WIDTH)]
        parent_idx = [p[0] for p in parents]
        out = _decode_output("session-0", expected_tokens, parent_idx)
        result = session.on_wave_output(EngineCoreOutputs(engine_index=0, outputs=[out]))
        if wave < MAX_TOKENS - 1:
            assert result is None
            assert session.wave == wave + 1
            packed = engine.input_queue.items[-1][1][0]
            expected_packed = []
            for beam in session.all_beams:
                expected_packed.extend(beam.tokens[3:])
            assert packed.prompt_token_ids == [1, 2, 3] + expected_packed

    # Final wave finished: outputs emitted, stats attached, cache cleaned.
    assert result is not None
    assert session.finished
    assert result.finished_requests == {"session-0", "session-0-beam-final"}
    # One packed final output carrying all beams.
    assert [o.request_id for o in result.outputs] == ["session-0-beam-final"]
    payload = result.outputs[0].kv_transfer_params
    assert payload is not None
    ref = _simulate_reference()
    stride = payload["_gr_beam_tokens_stride"]
    lens = payload["_gr_beam_lens"]
    packed = list(result.outputs[0].new_token_ids)
    unpacked = [packed[i * stride : i * stride + n] for i, n in enumerate(lens)]
    assert unpacked == ref
    stats = payload[BEAM_STATS_HEADER_KEY]
    assert stats["num_beams"] == BEAM_WIDTH
    # Decode waves only, actual active beam count (mirrors the frontend).
    assert stats["num_generation_tokens"] == BEAM_WIDTH * (MAX_TOKENS - 1)
    # Every beam's reconstructed logprobs are serialized.
    logprobs_rows = payload[BEAM_LOGPROBS_KEY]
    assert len(logprobs_rows) == BEAM_WIDTH
    for row in logprobs_rows:
        assert len(row) == 6
        assert row[2]  # token_ids non-empty for the generated sequence
    assert not engine.beam_cache


def test_session_finish_with_no_surviving_beams() -> None:
    engine = FakeEngineCore()
    engine.preprocess_add_request = _fake_preprocess.__get__(engine)
    engine._cache_beam_request = _fake_cache_beam_request.__get__(engine)
    session = BeamSearchSession(engine, _make_request([1, 2, 3]), catalog=None)
    engine._gr_beam_sessions["session-0"] = session

    session._submit_prefill_wave()
    # A prefill output without logprobs yields no candidates, so no beam
    # survives the wave-0 selection (catalog dead-end equivalent).
    result = session.on_wave_output(
        EngineCoreOutputs(
            engine_index=0,
            outputs=[
                EngineCoreOutput(
                    request_id=session.beam_request_ids[0],
                    new_token_ids=[],
                    finish_reason=FinishReason.LENGTH,
                    new_logprobs=None,
                )
            ],
        )
    )
    assert result is not None
    assert session.finished
    assert result.finished_requests == {"session-0", "session-0-beam-final"}
    # A single empty final output so the frontend queue finishes.
    assert [o.request_id for o in result.outputs] == ["session-0-beam-final"]
    assert result.outputs[0].new_token_ids == []
    assert result.outputs[0].finish_reason == FinishReason.STOP
    assert result.outputs[0].kv_transfer_params is not None
    assert result.outputs[0].kv_transfer_params["_gr_beam_lens"] == []
    assert not engine.beam_cache


def test_finish_session_packs_partial_survival() -> None:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
    from vllm.inputs import tokens_input
    from vllm.logprobs import FlatLogprobs

    engine = FakeEngineCore()
    session = BeamSearchSession(engine, _make_request([1]), catalog=None)

    beam = BeamSearchSequence(
        orig_prompt=tokens_input([1]),
        tokens=[1, 7, 8],
        cum_logprob=-0.5,
        logprobs=FlatLogprobs(),
    )
    beam._lp_parent = None
    beam._lp_step_data = ([7, 8], [-0.1, -0.2], [1, 2], [None, None])
    session.all_beams = [beam]
    session.completed = []

    result = session._finish_session()
    assert session.finished
    assert len(result.outputs) == 1
    assert result.outputs[0].request_id == "session-0-beam-final"
    assert result.finished_requests == {"session-0", "session-0-beam-final"}
    payload = result.outputs[0].kv_transfer_params
    assert payload["_gr_beam_lens"] == [2]
    assert payload["_gr_beam_tokens_stride"] == 2
    assert list(result.outputs[0].new_token_ids) == [7, 8]
    assert payload[BEAM_STATS_HEADER_KEY] is not None
    # Reconstructed per-beam logprobs are serialized (6-list rows).
    logprobs_rows = payload[BEAM_LOGPROBS_KEY]
    assert len(logprobs_rows) == 1
    assert logprobs_rows[0][2] == [7, 8]
    assert logprobs_rows[0][3] == [-0.1, -0.2]


def test_route_beam_session_outputs_forwards_scheduler_stats() -> None:
    """A step whose outputs are all intercepted by a beam session must still
    deliver its scheduler_stats to the frontend (prefix cache hit metrics)."""
    from types import SimpleNamespace

    from vllm.v1.metrics.stats import SchedulerStats

    from vllm_gr.v1.engine.core import _route_beam_session_outputs

    engine = FakeEngineCore()
    session = SimpleNamespace(
        session_id="sess",
        client_index=0,
        beam_request_ids=["sess-beam-0"],
        on_wave_output=lambda eco: None,
    )
    engine._gr_beam_sessions = {"sess": session}

    step_stats = SchedulerStats()
    wave_output = EngineCoreOutput(
        request_id="sess-beam-0",
        new_token_ids=[1],
        finish_reason=FinishReason.LENGTH,
    )
    routed = _route_beam_session_outputs(
        engine,
        {
            0: EngineCoreOutputs(
                engine_index=0,
                outputs=[wave_output],
                scheduler_stats=step_stats,
            )
        },
    )
    # The wave output was consumed by the session; the step's stats survive.
    assert 0 in routed
    assert routed[0].outputs == []
    assert routed[0].scheduler_stats is step_stats
    assert "sess" in engine._gr_beam_sessions


def test_route_beam_session_outputs_merges_finals_with_normals() -> None:
    """Beam finals and normal outputs from the same step are delivered
    together, keeping the step's scheduler_stats."""
    from types import SimpleNamespace

    from vllm.v1.metrics.stats import SchedulerStats

    from vllm_gr.v1.engine.core import _route_beam_session_outputs

    engine = FakeEngineCore()

    def finish(eco: Any) -> EngineCoreOutputs:
        return EngineCoreOutputs(
            engine_index=0,
            finished_requests={"sess", "sess-beam-final"},
            outputs=[
                EngineCoreOutput(
                    request_id="sess-beam-final",
                    new_token_ids=[7, 8],
                    finish_reason=FinishReason.STOP,
                ),
            ],
        )

    session = SimpleNamespace(
        session_id="sess",
        client_index=0,
        beam_request_ids=["sess-beam-0"],
        on_wave_output=finish,
    )
    engine._gr_beam_sessions = {"sess": session}

    step_stats = SchedulerStats()
    normal_output = EngineCoreOutput(
        request_id="other-req",
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
    )
    wave_output = EngineCoreOutput(
        request_id="sess-beam-0",
        new_token_ids=[1],
        finish_reason=FinishReason.LENGTH,
    )
    routed = _route_beam_session_outputs(
        engine,
        {
            0: EngineCoreOutputs(
                engine_index=0,
                outputs=[normal_output, wave_output],
                finished_requests={"other-req"},
                scheduler_stats=step_stats,
            )
        },
    )
    assert "sess" not in engine._gr_beam_sessions
    assert 0 in routed
    ids = [o.request_id for o in routed[0].outputs]
    assert ids == ["other-req", "sess-beam-final"]
    # The step's normal finished-request bookkeeping must survive alongside
    # the synthetic beam-final ids.
    # The step's normal finished-request bookkeeping and the session's own
    # terminal ids must survive alongside the synthetic beam-final ids.
    assert routed[0].finished_requests == {"other-req", "sess", "sess-beam-final"}
    assert routed[0].scheduler_stats is step_stats


def test_route_beam_session_outputs_preserves_finished_requests_without_outputs() -> None:
    """A step whose only non-session content is finished-request bookkeeping
    must still forward it (terminal signal for output-less finishes)."""
    from types import SimpleNamespace

    from vllm_gr.v1.engine.core import _route_beam_session_outputs

    engine = FakeEngineCore()
    session = SimpleNamespace(
        session_id="sess",
        client_index=0,
        beam_request_ids=["sess-beam-0"],
        on_wave_output=lambda eco: None,
    )
    engine._gr_beam_sessions = {"sess": session}

    wave_output = EngineCoreOutput(
        request_id="sess-beam-0",
        new_token_ids=[1],
        finish_reason=FinishReason.LENGTH,
    )
    routed = _route_beam_session_outputs(
        engine,
        {
            0: EngineCoreOutputs(
                engine_index=0,
                outputs=[wave_output],
                finished_requests={"other-req"},
            )
        },
    )
    assert 0 in routed
    assert routed[0].outputs == []
    assert routed[0].finished_requests == {"other-req"}


def test_route_beam_session_outputs_after_session_removed() -> None:
    """A session removed from the map (abort) before routing must not crash
    the routing: its in-flight wave output is treated as a normal output."""
    from types import SimpleNamespace

    from vllm_gr.v1.engine.core import _route_beam_session_outputs

    engine = FakeEngineCore()
    session = SimpleNamespace(
        session_id="sess",
        client_index=0,
        beam_request_ids=["sess-beam-0"],
        on_wave_output=lambda eco: None,
    )
    engine._gr_beam_sessions = {"sess": session}
    # Simulate the abort path removing the session before the step loop runs.
    engine._gr_beam_sessions.pop("sess")

    wave_output = EngineCoreOutput(
        request_id="sess-beam-0",
        new_token_ids=[1],
        finish_reason=FinishReason.LENGTH,
    )
    routed = _route_beam_session_outputs(
        engine,
        {
            0: EngineCoreOutputs(
                engine_index=0,
                outputs=[wave_output],
            )
        },
    )
    assert 0 in routed
    assert [o.request_id for o in routed[0].outputs] == ["sess-beam-0"]


def test_handle_beam_search_request_no_generation_waves() -> None:
    """A budget that covers only reserved begin/end tokens must not run any
    model step: the handler finishes immediately with the reserved tokens."""
    from vllm_gr.v1.engine.core import _handle_beam_search_request

    def run_case(request: BeamSearchRequest, expected_tokens: list[int]) -> None:
        engine = FakeEngineCore()
        _handle_beam_search_request(engine, request)
        assert engine.input_queue.items == []
        assert len(engine.output_queue.items) == 1
        client_index, outputs = engine.output_queue.items[0]
        assert client_index == 0
        assert [o.request_id for o in outputs.outputs] == ["session-0-beam-final"]
        assert outputs.finished_requests == {"session-0", "session-0-beam-final"}
        packed = list(outputs.outputs[0].new_token_ids)
        lens = outputs.outputs[0].kv_transfer_params["_gr_beam_lens"]
        stride = outputs.outputs[0].kv_transfer_params["_gr_beam_tokens_stride"]
        assert packed == expected_tokens
        assert lens == [len(expected_tokens)]
        assert stride == len(expected_tokens)
        assert engine._gr_beam_sessions == {}
        assert engine.beam_cache == {}

    # begin/end reserved tokens only.
    request = BeamSearchRequest(
        session_id="session-0",
        prompt_token_ids=[1, 2],
        beam_width=2,
        max_tokens=2,
        sampling_params=SamplingParams(),
        sid_begin_token_id=5,
        sid_end_token_id=7,
    )
    run_case(request, [5, 7])

    # No reserved tokens at all.
    request = BeamSearchRequest(
        session_id="session-0",
        prompt_token_ids=[1, 2],
        beam_width=2,
        max_tokens=0,
        sampling_params=SamplingParams(),
    )
    run_case(request, [])


def test_engine_catalog_cache_is_byte_bounded() -> None:
    """The engine-side catalog cache evicts the oldest version before
    inserting a new one when the byte budget (or version cap) is exceeded."""
    from vllm_gr.v1.engine.core import _get_gr_engine_catalog

    engine = FakeEngineCore()
    # Tiny budget: catalog A = 3 tokens x 128B = 384B; B = 5 x 128B = 640B.
    engine._gr_catalog_cache_budget = 256

    _get_gr_engine_catalog(engine, 100, [[1, 2], [3]])
    assert 100 in engine._gr_catalog_cache
    b = _get_gr_engine_catalog(engine, 200, [[9, 9], [8, 8], [7]])
    # A was evicted before inserting B (no byte-budget overflow).
    assert 100 not in engine._gr_catalog_cache
    assert 200 in engine._gr_catalog_cache
    assert engine._gr_catalog_cache_size == 5 * 128
    assert _get_gr_engine_catalog(engine, 100, None) is None
    assert _get_gr_engine_catalog(engine, 200, None) is b


def test_session_abort() -> None:
    engine = FakeEngineCore()
    engine.preprocess_add_request = _fake_preprocess.__get__(engine)
    engine._cache_beam_request = _fake_cache_beam_request.__get__(engine)
    session = BeamSearchSession(engine, _make_request([1]), catalog=None)
    session._submit_prefill_wave()
    session.abort()
    assert session.aborted
    assert session.session_id in engine.aborts_queue.items
    assert not engine.beam_cache
    # Outputs after abort are dropped.
    result = session.on_wave_output(EngineCoreOutputs(engine_index=0, outputs=[]))
    assert result is None


def test_flat_logprobs_from_lists_mirrors_frontend_layout() -> None:
    from vllm_gr.v1.beam.search_engine import _flat_logprobs_from_lists

    token_ids = np.array([[5, 101, 102, 103]], dtype=np.int64)
    logprobs = np.array([[-1.0, -0.1, -0.2, -0.3]], dtype=np.float32)
    ranks = np.array([7], dtype=np.int64)
    flat = _flat_logprobs_from_lists(LogprobsLists(token_ids, logprobs, ranks, None), 3)
    assert flat.token_ids == [5, 101, 102, 103]
    assert flat.logprobs == pytest.approx([-1.0, -0.1, -0.2, -0.3])
    # ranks: islice-padded to token_ids length (gr append_fast semantics).
    assert flat.ranks == [7, 1, 2, 3]
    assert flat.start_indices == [0]
    assert flat.end_indices == [4]


def test_catalog_sequence_round_trip() -> None:
    from vllm_gr.utils.catalog import Catalog

    sequences = [[4, 5, 6], [4, 5, 7], [8]]
    catalog = Catalog()
    catalog.load_from_sequences(sequences)
    assert catalog.to_sequences() == [[4, 5, 6], [4, 5, 7], [8]]
    assert catalog.valid([4, 5]) == {6, 7}
    assert catalog.valid([9]) == set()
    assert catalog.is_terminal([4, 5, 6])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
