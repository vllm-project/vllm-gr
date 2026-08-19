# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import msgspec
import pytest

from vllm_gr.beam_output import BeamOutputSpec
from vllm_gr.entrypoints.openai.protocol import to_beam_search_params
from vllm_gr.entrypoints.openai.serving_engine import _batch_decode_equivalent
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam import beam_search_controller as controller_module
from vllm_gr.v1.beam.beam_search_controller import BeamSearchController
from vllm_gr.v1.beam.controller_types import (
    BeamControllerFinalResult,
    BeamControllerFinalResultLite,
    BeamControllerRequest,
    BeamControllerStepOutput,
)


def _request(spec: BeamOutputSpec, *, beam_width: int = 4) -> BeamControllerRequest:
    return BeamControllerRequest(
        beam_session_id="session",
        external_request_id="request",
        beam_width=beam_width,
        max_tokens=1,
        prompt_token_ids=[10],
        initial_tokens=[10],
        prompt_length=1,
        eos_token_id=2,
        output_spec=spec,
    )


def _finish(spec: BeamOutputSpec, *, candidate_width: int = 4):
    controller = BeamSearchController.from_request(
        _request(spec, beam_width=max(4, spec.result_width))
    )
    token_ids = list(range(11, 11 + candidate_width))
    decision = controller.consume_output(
        BeamControllerStepOutput(
            token_ids=token_ids,
            logprobs=[-float(i) / 10 for i in range(candidate_width)],
            ranks=list(range(1, candidate_width + 1)),
            decoded_tokens=[None] * candidate_width,
        )
    )
    assert decision.finished
    return controller.finalize()


def test_default_contract_preserves_result_width_but_omits_logprobs() -> None:
    params = BeamSearchParams(beam_width=128, max_tokens=3)
    assert params.output_spec == BeamOutputSpec(128, "none", 0)


@pytest.mark.parametrize("result_width", [1, 2, 4])
def test_lite_result_width_and_version(result_width: int, monkeypatch) -> None:
    monkeypatch.setattr(
        controller_module,
        "reconstruct_beam_logprobs",
        lambda *args, **kwargs: pytest.fail("Lite finalize reconstructed logprobs"),
    )
    result = _finish(BeamOutputSpec(result_width, "none", 0))
    assert isinstance(result, BeamControllerFinalResultLite)
    assert len(result.beams) == result_width
    wire = result.to_wire()
    assert wire["type"] == "beam_final_lite"
    assert wire["version"] == 1
    assert isinstance(BeamControllerFinalResult.from_wire(wire), BeamControllerFinalResultLite)


def test_selected_and_top_k_logprobs_contracts() -> None:
    selected = _finish(BeamOutputSpec(2, "selected", 0))
    assert not isinstance(selected, BeamControllerFinalResultLite)
    assert all(len(beam.logprobs["token_ids"]) == 1 for beam in selected.beams)

    top_k = _finish(BeamOutputSpec(2, "top_k", 2))
    assert all(1 <= len(beam.logprobs["token_ids"]) <= 3 for beam in top_k.beams)


def test_lite_wire_is_independent_of_candidate_width() -> None:
    narrow = _finish(BeamOutputSpec(4, "none", 0), candidate_width=4).to_wire()
    wide = _finish(BeamOutputSpec(4, "none", 0), candidate_width=128).to_wire()
    assert len(msgspec.msgpack.encode(narrow)) == len(msgspec.msgpack.encode(wide))


def test_unknown_lite_version_is_rejected() -> None:
    wire = _finish(BeamOutputSpec(1, "none", 0)).to_wire()
    wire["version"] = 2
    with pytest.raises(ValueError, match="unsupported beam final result version"):
        BeamControllerFinalResult.from_wire(wire)


@pytest.mark.parametrize(
    ("logprobs", "top_logprobs", "mode", "top_k"),
    [(False, None, "none", 0), (True, None, "selected", 0), (True, 3, "top_k", 3)],
)
def test_openai_adapter_maps_logprobs_contract(
    logprobs: bool, top_logprobs: int | None, mode: str, top_k: int
) -> None:
    request = SimpleNamespace(
        vllm_xargs=None,
        n=4,
        temperature=0.0,
        ignore_eos=False,
        length_penalty=1.0,
        include_stop_str_in_output=False,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        _DEFAULT_SAMPLING_PARAMS={"temperature": 1.0},
    )
    params = to_beam_search_params(request, 3, {})
    assert params.output_spec == BeamOutputSpec(4, mode, top_k)


def test_batch_decode_equivalence_and_fallback() -> None:
    class Tokenizer:
        def __init__(self, equivalent: bool) -> None:
            self.equivalent = equivalent

        def decode(self, tokens):
            return ":".join(map(str, tokens))

        def batch_decode(self, batches):
            values = [self.decode(tokens) for tokens in batches]
            return values if self.equivalent else [f"bad:{value}" for value in values]

    batches = [[1, 2], [3]]
    assert _batch_decode_equivalent(Tokenizer(True), batches) == ["1:2", "3"]
    tokenizer = Tokenizer(False)
    assert _batch_decode_equivalent(tokenizer, batches) == ["1:2", "3"]
    assert tokenizer._vllm_gr_batch_decode_equivalent is False


def test_invalid_output_combinations_fail_early() -> None:
    with pytest.raises(Exception, match="result_width"):
        BeamSearchParams(beam_width=4, max_tokens=1, result_width=5)
    with pytest.raises(Exception, match="top_logprobs"):
        BeamSearchParams(beam_width=4, max_tokens=1, logprobs_mode="none", top_logprobs=1)


def test_lite_cancel_and_error_keep_lite_wire_contract() -> None:
    cancelled = BeamSearchController.from_request(_request(BeamOutputSpec(1, "none", 0)))
    cancelled.cancel()
    cancel_result = cancelled.finalize()
    assert isinstance(cancel_result, BeamControllerFinalResultLite)
    assert cancel_result.to_wire()["type"] == "beam_final_lite"
    assert cancel_result.finish_reason == "cancelled"

    failed = BeamSearchController.from_request(_request(BeamOutputSpec(1, "none", 0)))
    failed.consume_output(
        BeamControllerStepOutput(
            token_ids=[11, 12],
            logprobs=[-0.1],
            ranks=[1, 2],
            decoded_tokens=[None, None],
        )
    )
    error_result = failed.finalize()
    assert isinstance(error_result, BeamControllerFinalResultLite)
    assert error_result.to_wire()["type"] == "beam_final_lite"
    assert error_result.finish_reason == "error"


def test_lite_result_supports_beam_width_128() -> None:
    result = _finish(BeamOutputSpec(128, "none", 0), candidate_width=128)
    assert isinstance(result, BeamControllerFinalResultLite)
    assert len(result.beams) == 128


def test_equal_scores_have_stable_tie_breaking() -> None:
    def finish_equal():
        controller = BeamSearchController.from_request(_request(BeamOutputSpec(4, "none", 0)))
        controller.consume_output(
            BeamControllerStepOutput(
                token_ids=[14, 13, 12, 11],
                logprobs=[-0.1] * 4,
                ranks=[1, 2, 3, 4],
                decoded_tokens=[None] * 4,
            )
        )
        return [beam.token_ids for beam in controller.finalize().beams]

    assert finish_equal() == finish_equal()
