# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.constraints.strategies import ConstraintTableStrategy
from vllm_gr.entrypoints import gr as offline_gr
from vllm_gr.entrypoints.openai import serving_engine
from vllm_gr.v1.engine.types import GR_REQUEST_METADATA_KEY, GRRequestMetadata

_DIGEST = "sha256:" + "a" * 64


class _OutputQueue:
    def __init__(self, output):
        self.output = output

    async def get(self):
        return self.output


def _persistent_final_decision() -> dict:
    sequences = np.asarray([[11, 21, 31], [12, 22, 32]], dtype=np.int32)
    parents = np.asarray([[0, 0, 0], [0, 1, 1]], dtype=np.int32)
    logprobs = np.asarray(
        [[-0.1, -0.2, -0.3], [-0.2, -0.2, -0.3]], dtype=np.float32
    )
    scores = np.asarray([-0.6, -0.7], dtype=np.float32)
    packed = np.concatenate(
        (
            sequences.reshape(-1),
            parents.reshape(-1),
            logprobs.reshape(-1).view(np.int32),
            scores.view(np.int32),
        )
    ).tobytes()
    return {
        "buffer_format": "gr_final_i32_f32_v1",
        "shape": [2, 3],
        "buffer": packed,
        "is_last_step": True,
    }


def test_offline_prepares_one_persistent_final_only_request() -> None:
    states = {}

    def process_inputs(**kwargs):
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            external_req_id=kwargs["request_id"],
            sampling_params=kwargs["params"],
        )

    def assign_request_id(request):
        request.request_id = "native-id"

    def add_request(request, *_args):
        states[request.request_id] = SimpleNamespace()

    engine = SimpleNamespace(
        input_processor=SimpleNamespace(
            process_inputs=process_inputs,
            assign_request_id=assign_request_id,
        ),
        output_processor=SimpleNamespace(
            add_request=Mock(side_effect=add_request), request_states=states
        ),
        get_supported_tasks=lambda: ("generate",),
    )
    beam = SimpleNamespace(
        tokens=[10],
        lora_request=None,
        get_prompt=lambda: {"prompt_token_ids": [10]},
    )
    request = offline_gr._prepare_persistent_request(
        engine,
        SimpleNamespace(beams=[beam]),
        0,
        SamplingParams(logprobs=2, max_tokens=1),
        3,
        "a" * 64,
    )

    assert request.request_id == "native-id"
    assert request.sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
    assert request.sampling_params.max_tokens == 3
    assert request.sampling_params.logprobs is None
    metadata = request.sampling_params.extra_args[GR_REQUEST_METADATA_KEY]
    assert metadata["request_id"] == "native-id"
    assert metadata["constraint_digest"] == _DIGEST
    assert states["native-id"].gr_persistent_request is True


def test_offline_collection_aborts_live_request_on_failure() -> None:
    states = {"native-id": object()}
    abort = Mock(side_effect=lambda ids, internal: states.pop(ids[0]))
    engine = SimpleNamespace(
        step=Mock(side_effect=RuntimeError("step failed")),
        abort_request=abort,
        output_processor=SimpleNamespace(request_states=states),
    )
    request = SimpleNamespace(request_id="native-id", external_req_id="external-id")

    with pytest.raises(RuntimeError, match="step failed"):
        offline_gr._collect_persistent_output(engine, request)

    abort.assert_called_once_with(["native-id"], internal=True)


@pytest.mark.asyncio
async def test_online_submission_uses_native_add_and_marks_lifecycle() -> None:
    state = SimpleNamespace()
    sent = []

    class Client:
        output_processor = SimpleNamespace(request_states={"native-id": state})

        async def prepare_request(self, _id, _prompt, params, **_kwargs):
            request = SimpleNamespace(request_id="native-id", sampling_params=params)
            return _OutputQueue(SimpleNamespace(finished=True)), request

        async def _add_requests_batch(self, requests):
            sent.extend(requests)

    metadata = GRRequestMetadata(
        request_id="external-id",
        beam_width=2,
        generation_steps=3,
        prefix_len=1,
        result_width=2,
        constraint_digest=_DIGEST,
    )
    outputs, ids = await serving_engine._add_batch_step(
        Client(),
        ({"prompt_token_ids": [10]},),
        (None,),
        "external-id",
        SamplingParams(logprobs=2, max_tokens=1),
        None,
        gr_metadata=metadata,
    )

    assert len(outputs) == 1
    assert ids == ["native-id"]
    assert [request.request_id for request in sent] == ["native-id"]
    assert state.gr_persistent_request is True
    wire = sent[0].sampling_params.extra_args[GR_REQUEST_METADATA_KEY]
    assert wire["request_id"] == "native-id"


@pytest.mark.asyncio
async def test_online_collection_failure_aborts_native_request() -> None:
    class FailingQueue:
        async def get(self):
            raise RuntimeError("collector failed")

    state = SimpleNamespace()
    abort_calls = []

    class Client:
        output_processor = SimpleNamespace(request_states={"native-id": state})

        async def prepare_request(self, _id, _prompt, params, **_kwargs):
            return FailingQueue(), SimpleNamespace(
                request_id="native-id", sampling_params=params
            )

        async def _add_requests_batch(self, _requests):
            return None

        async def abort(self, request_ids, internal=False):
            abort_calls.append((list(request_ids), internal))

    metadata = GRRequestMetadata(
        request_id="external-id",
        beam_width=2,
        generation_steps=3,
        prefix_len=1,
        result_width=2,
        constraint_digest=_DIGEST,
    )
    with pytest.raises(RuntimeError, match="collector failed"):
        await serving_engine._add_batch_step(
            Client(),
            ({"prompt_token_ids": [10]},),
            (None,),
            "external-id",
            SamplingParams(logprobs=2, max_tokens=1),
            None,
            gr_metadata=metadata,
        )

    assert state.gr_persistent_request is True
    assert abort_calls == [(["native-id"], True)]


def test_constraint_table_protocol_requires_persistent_scheduler() -> None:
    strategy = ConstraintTableStrategy("triples.json")
    strategy.digest = "a" * 64
    coordinator = BeamConstraintCoordinator(strategy)
    assert coordinator.strategy is strategy
    with pytest.raises(RuntimeError, match="persistent GR ASYNC scheduler"):
        serving_engine.validate_constraint_table_protocol(
            is_constraint_table=True,
            beam_protocol_available=False,
            required="the persistent GR ASYNC scheduler",
        )
