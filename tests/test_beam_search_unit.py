# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for engine-core beam search (no GPU / no model required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import msgspec
import numpy as np
import pytest

from vllm_gr.v1.engine.beam_search import (
    BeamSearchController,
    _Beam,
    _NUMPY_THRESHOLD,
    _get_beam_search_score,
)
from vllm_gr.v1.engine.types import (
    BeamSearchRequest,
    BeamSearchResult,
    BeamSearchSequenceResult,
    PatchedEngineCoreOutputs,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class MockLogprobs:
    """Minimal stand-in for the logprobs object returned by the engine."""

    def __init__(self, token_ids: list[int], logprobs: list[float]):
        self.logprob_token_ids = np.array(token_ids, dtype=np.int64)
        self.logprobs = np.array(logprobs, dtype=np.float64)


def make_request(**overrides) -> BeamSearchRequest:
    """Factory returning a BeamSearchRequest with sensible defaults."""
    defaults = dict(
        request_id="test-req-0",
        prompt_token_ids=[10, 20, 30],
        eos_token_id=2,
        beam_width=3,
        max_tokens=5,
        n_best=1,
        temperature=0.0,
        length_penalty=1.0,
        ignore_eos=False,
        include_stop_str_in_output=False,
        sid_begin_token_id=None,
        sid_end_token_id=None,
        client_index=0,
    )
    defaults.update(overrides)
    return BeamSearchRequest(**defaults)


def _mock_engine_core() -> MagicMock:
    """Create a mock EngineCore with the attributes BeamSearchController needs."""
    engine_core = MagicMock()
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.add_request = MagicMock()
    engine_core.request_block_hasher = MagicMock()
    return engine_core


def make_controller(req=None, **req_overrides) -> BeamSearchController:
    """Create a BeamSearchController backed by a mock engine_core."""
    if req is None:
        req = make_request(**req_overrides)
    return BeamSearchController(_mock_engine_core(), req)


def setup_beam_step(
    ctrl: BeamSearchController,
    beams_data: list[tuple[list[int], float, list[tuple[int, float]]]],
) -> None:
    """Populate controller internals so _beam_selection can run.

    Each element of *beams_data* is
        (tokens, cum_logprob, [(token_id, logprob), ...])
    """
    ctrl._all_beams = []
    ctrl._beam_req_ids = []
    ctrl._output_by_rid = {}

    for i, (tokens, cum_logprob, candidates) in enumerate(beams_data):
        rid = f"setup-s0-b{i}"
        beam = _Beam(tokens=list(tokens), cum_logprob=cum_logprob)
        ctrl._all_beams.append(beam)
        ctrl._beam_req_ids.append(rid)

        token_ids = [c[0] for c in candidates]
        logprobs = [c[1] for c in candidates]
        ctrl._output_by_rid[rid] = (
            [token_ids[0]],  # new_token_ids (unused by selection)
            MockLogprobs(token_ids, logprobs),
        )


# ---------------------------------------------------------------------------
# 1. BeamSearchRequest serialization round-trip
# ---------------------------------------------------------------------------

class TestBeamSearchRequestSerialization:
    def test_round_trip(self):
        req = make_request(
            sid_begin_token_id=100,
            sid_end_token_id=200,
            ignore_eos=True,
            length_penalty=0.8,
        )
        data = msgspec.msgpack.encode(req)
        dec = msgspec.msgpack.Decoder(BeamSearchRequest)
        decoded = dec.decode(data)

        assert decoded.request_id == req.request_id
        assert decoded.prompt_token_ids == req.prompt_token_ids
        assert decoded.eos_token_id == req.eos_token_id
        assert decoded.beam_width == req.beam_width
        assert decoded.max_tokens == req.max_tokens
        assert decoded.n_best == req.n_best
        assert decoded.temperature == req.temperature
        assert decoded.length_penalty == req.length_penalty
        assert decoded.ignore_eos == req.ignore_eos
        assert decoded.include_stop_str_in_output == req.include_stop_str_in_output
        assert decoded.sid_begin_token_id == 100
        assert decoded.sid_end_token_id == 200
        assert decoded.client_index == req.client_index


# ---------------------------------------------------------------------------
# 2. BeamSearchResult serialization round-trip
# ---------------------------------------------------------------------------

class TestBeamSearchResultSerialization:
    def test_round_trip(self):
        seq1 = BeamSearchSequenceResult(
            token_ids=[1, 2, 3],
            cumulative_logprob=-1.5,
            finish_reason="stop",
            stop_reason=2,
        )
        seq2 = BeamSearchSequenceResult(
            token_ids=[4, 5],
            cumulative_logprob=-2.0,
            finish_reason="length",
            stop_reason=None,
        )
        result = BeamSearchResult(
            request_id="res-1",
            sequences=[seq1, seq2],
        )
        data = msgspec.msgpack.encode(result)
        dec = msgspec.msgpack.Decoder(BeamSearchResult)
        decoded = dec.decode(data)

        assert decoded.request_id == "res-1"
        assert len(decoded.sequences) == 2
        assert decoded.sequences[0].token_ids == [1, 2, 3]
        assert decoded.sequences[0].cumulative_logprob == pytest.approx(-1.5)
        assert decoded.sequences[0].finish_reason == "stop"
        assert decoded.sequences[0].stop_reason == 2
        assert decoded.sequences[1].finish_reason == "length"
        assert decoded.sequences[1].stop_reason is None


# ---------------------------------------------------------------------------
# 3. PatchedEngineCoreOutputs serialization round-trip
# ---------------------------------------------------------------------------

class TestPatchedEngineCoreOutputsSerialization:
    def test_round_trip_with_beam_result(self):
        seq = BeamSearchSequenceResult(
            token_ids=[7, 8],
            cumulative_logprob=-0.5,
            finish_reason="stop",
            stop_reason=2,
        )
        br = BeamSearchResult(
            request_id="br-1",
            sequences=[seq],
        )
        out = PatchedEngineCoreOutputs(
            engine_index=0,
            beam_search_result=br,
        )
        data = msgspec.msgpack.encode(out)
        dec = msgspec.msgpack.Decoder(PatchedEngineCoreOutputs)
        decoded = dec.decode(data)

        assert decoded.beam_search_result is not None
        assert decoded.beam_search_result.request_id == "br-1"
        assert len(decoded.beam_search_result.sequences) == 1
        assert decoded.beam_search_result.sequences[0].token_ids == [7, 8]

    def test_regular_fields(self):
        out = PatchedEngineCoreOutputs(engine_index=3)
        data = msgspec.msgpack.encode(out)
        dec = msgspec.msgpack.Decoder(PatchedEngineCoreOutputs)
        decoded = dec.decode(data)

        assert decoded.engine_index == 3
        assert decoded.outputs == []
        assert decoded.beam_search_result is None
        assert decoded.scheduler_stats is None


# ---------------------------------------------------------------------------
# 4. Basic beam selection
# ---------------------------------------------------------------------------

class TestBeamSelectionBasic:
    """Verify top beam_width candidates are selected by cumulative logprob."""

    @staticmethod
    def _run(ctrl):
        """Set up 2 beams × 3 candidates and run selection."""
        #
        # Beam 0: cum_logprob = -1.0
        #   candidates: token 50 (-0.1), token 51 (-0.5), token 52 (-1.0)
        #   → cum: -1.1, -1.5, -2.0
        #
        # Beam 1: cum_logprob = -0.5
        #   candidates: token 60 (-0.2), token 61 (-0.8), token 62 (-1.5)
        #   → cum: -0.7, -1.3, -2.0
        #
        # Sorted: -0.7 (beam1,60), -1.1 (beam0,50), -1.3 (beam1,61)
        #          top 3 with beam_width=3
        setup_beam_step(ctrl, [
            ([10, 20, 30], -1.0, [(50, -0.1), (51, -0.5), (52, -1.0)]),
            ([10, 20, 30], -0.5, [(60, -0.2), (61, -0.8), (62, -1.5)]),
        ])
        ctrl._beam_selection()
        return ctrl

    def test_python_path(self):
        import vllm_gr.v1.engine.beam_search as bsmod
        original = bsmod._NUMPY_THRESHOLD
        try:
            # Force Python path by setting threshold very high
            bsmod._NUMPY_THRESHOLD = 10_000_000
            ctrl = make_controller(beam_width=3)
            self._run(ctrl)

            assert len(ctrl._all_beams) == 3
            cum_lps = [b.cum_logprob for b in ctrl._all_beams]
            # Should be sorted descending by cumulative logprob
            assert cum_lps == sorted(cum_lps, reverse=True)
            assert cum_lps[0] == pytest.approx(-0.7)
            assert cum_lps[1] == pytest.approx(-1.1)
            assert cum_lps[2] == pytest.approx(-1.3)
            # Verify token appended
            assert ctrl._all_beams[0].tokens[-1] == 60
            assert ctrl._all_beams[1].tokens[-1] == 50
            assert ctrl._all_beams[2].tokens[-1] == 61
        finally:
            bsmod._NUMPY_THRESHOLD = original

    def test_numpy_path(self):
        import vllm_gr.v1.engine.beam_search as bsmod
        original = bsmod._NUMPY_THRESHOLD
        try:
            # Force numpy path by setting threshold to 0
            bsmod._NUMPY_THRESHOLD = 0
            ctrl = make_controller(beam_width=3)
            self._run(ctrl)

            assert len(ctrl._all_beams) == 3
            cum_lps = [b.cum_logprob for b in ctrl._all_beams]
            assert cum_lps == sorted(cum_lps, reverse=True)
            assert cum_lps[0] == pytest.approx(-0.7)
            assert cum_lps[1] == pytest.approx(-1.1)
            assert cum_lps[2] == pytest.approx(-1.3)
            assert ctrl._all_beams[0].tokens[-1] == 60
            assert ctrl._all_beams[1].tokens[-1] == 50
            assert ctrl._all_beams[2].tokens[-1] == 61
        finally:
            bsmod._NUMPY_THRESHOLD = original


# ---------------------------------------------------------------------------
# 5. Beam selection EOS handling
# ---------------------------------------------------------------------------

class TestBeamSelectionEos:
    """Highest-scoring candidate is EOS → moves to _completed."""

    @pytest.mark.parametrize("use_numpy", [False, True], ids=["python", "numpy"])
    def test_eos_moves_to_completed(self, use_numpy):
        import vllm_gr.v1.engine.beam_search as bsmod
        original = bsmod._NUMPY_THRESHOLD
        try:
            bsmod._NUMPY_THRESHOLD = 0 if use_numpy else 10_000_000
            ctrl = make_controller(beam_width=3, eos_token_id=2)

            # Beam 0: best candidate is EOS (token 2) with highest cum logprob
            # cum -0.5 + (-0.1) = -0.6  ← EOS, best overall
            # Beam 1: non-EOS candidates
            setup_beam_step(ctrl, [
                ([10, 20, 30], -0.5, [(2, -0.1), (51, -0.5), (52, -1.0)]),
                ([10, 20, 30], -1.0, [(60, -0.2), (61, -0.8), (62, -1.5)]),
            ])
            ctrl._beam_selection()

            # EOS beam should be in _completed, not _all_beams
            assert len(ctrl._completed) == 1
            assert ctrl._completed[0].tokens[-1] == 2
            assert ctrl._completed[0].cum_logprob == pytest.approx(-0.6)
            # Remaining beams should not contain EOS
            for beam in ctrl._all_beams:
                assert beam.tokens[-1] != 2
        finally:
            bsmod._NUMPY_THRESHOLD = original


# ---------------------------------------------------------------------------
# 6. Beam selection ignore_eos
# ---------------------------------------------------------------------------

class TestBeamSelectionIgnoreEos:
    """With ignore_eos=True, EOS token stays in _all_beams."""

    @pytest.mark.parametrize("use_numpy", [False, True], ids=["python", "numpy"])
    def test_eos_kept_when_ignored(self, use_numpy):
        import vllm_gr.v1.engine.beam_search as bsmod
        original = bsmod._NUMPY_THRESHOLD
        try:
            bsmod._NUMPY_THRESHOLD = 0 if use_numpy else 10_000_000
            ctrl = make_controller(beam_width=3, eos_token_id=2, ignore_eos=True)

            setup_beam_step(ctrl, [
                ([10, 20, 30], -0.5, [(2, -0.1), (51, -0.5), (52, -1.0)]),
                ([10, 20, 30], -1.0, [(60, -0.2), (61, -0.8), (62, -1.5)]),
            ])
            ctrl._beam_selection()

            assert len(ctrl._completed) == 0
            # The EOS token beam should remain in _all_beams
            eos_beams = [b for b in ctrl._all_beams if b.tokens[-1] == 2]
            assert len(eos_beams) == 1
        finally:
            bsmod._NUMPY_THRESHOLD = original


# ---------------------------------------------------------------------------
# 7. Length penalty scoring
# ---------------------------------------------------------------------------

class TestLengthPenaltyScoring:
    def test_normal_case(self):
        tokens = [10, 20, 30, 40]  # len=4, no EOS at end
        score = _get_beam_search_score(tokens, -4.0, eos_token_id=2, length_penalty=1.0)
        assert score == pytest.approx(-4.0 / 4)

    def test_empty_tokens(self):
        score = _get_beam_search_score([], -2.5, eos_token_id=2, length_penalty=1.0)
        assert score == pytest.approx(-2.5)

    def test_eos_as_last_token(self):
        # tokens=[10, 20, 2], EOS at end → effective len = 2
        tokens = [10, 20, 2]
        score = _get_beam_search_score(tokens, -6.0, eos_token_id=2, length_penalty=1.0)
        assert score == pytest.approx(-6.0 / 2)

    def test_single_eos_token(self):
        # Only one token and it's EOS → seq_len-1 = 0, returns cum_logprob
        score = _get_beam_search_score([2], -3.0, eos_token_id=2, length_penalty=1.0)
        assert score == pytest.approx(-3.0)

    def test_length_penalty_zero(self):
        # penalty=0 → seq_len^0 = 1 → score = cum_logprob
        tokens = [10, 20, 30]
        score = _get_beam_search_score(tokens, -9.0, eos_token_id=2, length_penalty=0.0)
        assert score == pytest.approx(-9.0)


# ---------------------------------------------------------------------------
# 8. SID begin / end tokens
# ---------------------------------------------------------------------------

class TestSidBeginEndTokens:
    def test_begin_token_appended_on_init(self):
        ctrl = make_controller(
            sid_begin_token_id=100,
            sid_end_token_id=200,
            prompt_token_ids=[10, 20, 30],
        )
        # Initial beam should have begin token appended to prompt
        assert ctrl._all_beams[0].tokens == [10, 20, 30, 100]

    def test_end_token_appended_on_finish(self):
        ctrl = make_controller(
            sid_begin_token_id=100,
            sid_end_token_id=200,
            prompt_token_ids=[10, 20, 30],
            beam_width=2,
            n_best=1,
            max_tokens=5,
        )
        # Manually set up beams as if we've done some steps
        ctrl._all_beams = [
            _Beam(tokens=[10, 20, 30, 100, 40], cum_logprob=-1.0),
            _Beam(tokens=[10, 20, 30, 100, 50], cum_logprob=-2.0),
        ]
        ctrl._completed = []

        # Trigger finish
        ctrl._finish()

        assert ctrl._done is True
        result = ctrl._result
        assert result is not None
        assert len(result.sequences) >= 1
        # The output tokens (after prompt) should end with the end token 200
        for seq in result.sequences:
            assert seq.token_ids[-1] == 200


# ---------------------------------------------------------------------------
# 9. Effective max_tokens adjustment
# ---------------------------------------------------------------------------

class TestEffectiveMaxTokensAdjustment:
    def test_no_sid_tokens(self):
        ctrl = make_controller(max_tokens=10)
        assert ctrl._effective_max_tokens == 10

    def test_begin_only(self):
        ctrl = make_controller(max_tokens=10, sid_begin_token_id=100)
        assert ctrl._effective_max_tokens == 9

    def test_end_only(self):
        ctrl = make_controller(max_tokens=10, sid_end_token_id=200)
        assert ctrl._effective_max_tokens == 9

    def test_both_sid_tokens(self):
        ctrl = make_controller(max_tokens=10, sid_begin_token_id=100, sid_end_token_id=200)
        assert ctrl._effective_max_tokens == 8

    def test_max_tokens_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            make_controller(max_tokens=1, sid_begin_token_id=100, sid_end_token_id=200)


# ---------------------------------------------------------------------------
# 10. Enum patching
# ---------------------------------------------------------------------------

class TestEnumPatching:
    def test_beam_search_enum_member(self):
        from vllm.v1.engine import EngineCoreRequestType

        # Run only the enum-patching portion of apply_engine_core_child_patches
        if not hasattr(EngineCoreRequestType, "BEAM_SEARCH"):
            beam_val = b"\x06"
            beam_member = object.__new__(EngineCoreRequestType)
            beam_member._name_ = "BEAM_SEARCH"
            beam_member._value_ = beam_val
            EngineCoreRequestType._value2member_map_[beam_val] = beam_member
            EngineCoreRequestType._member_map_["BEAM_SEARCH"] = beam_member
            EngineCoreRequestType._member_names_.append("BEAM_SEARCH")
            type.__setattr__(EngineCoreRequestType, "BEAM_SEARCH", beam_member)

        assert hasattr(EngineCoreRequestType, "BEAM_SEARCH")
        assert EngineCoreRequestType.BEAM_SEARCH.value == b"\x06"
        assert EngineCoreRequestType.BEAM_SEARCH.name == "BEAM_SEARCH"
        # Verify it can be looked up by value
        assert EngineCoreRequestType(b"\x06") is EngineCoreRequestType.BEAM_SEARCH
