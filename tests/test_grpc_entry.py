# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# ruff: noqa: E402 -- Optional gRPC dependencies must be checked before imports.
"""Full-schema compatibility, binary conversion and actual gRPC transport tests."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

grpc = pytest.importorskip("grpc", minversion="1.84.0", reason="install vllm-gr[grpc]")
pytest.importorskip("google.protobuf", minversion="7.35.1", reason="install vllm-gr[grpc]")
pytest.importorskip("smg_grpc_servicer", reason="install vllm-gr[grpc]")

from google.protobuf import descriptor_pb2

from tests.test_gr_admission_queue import core as _admission_core_fixture
from tests.test_gr_admission_queue import scheduler as _admission_scheduler_fixture
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc
from vllm_gr.entrypoints.grpc.serving import BeamVllmEngineServicer
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchFailure,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
    BeamSequenceColumns,
)

scheduler = _admission_scheduler_fixture
grpc_admission_core = _admission_core_fixture


def message(index: int = 0, *, binary: bool = True) -> Any:
    value = pb.GenerateRequest(
        request_id=f"id-{index}",
        tokenized=pb.TokenizedInput(original_text="metadata"),
        sampling_params=pb.SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True, n=2),
        beam_search=pb.BeamSearchOptions(beam_width=2),
    )
    if binary:
        value.tokenized.input_ids_i64_le = np.array([1, 4, 3], dtype="<i8").tobytes()
    else:
        value.tokenized.input_ids.extend([1, 4, 3])
    return value


def engine() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8192, get_vocab_size=lambda: 128),
        input_processor=SimpleNamespace(
            tokenizer=SimpleNamespace(max_token_id=127, truncation_side="left")
        ),
        errored=False,
    )


def test_all_original_messages_fields_and_methods_are_retained() -> None:
    # Use the pinned runtime dependency rather than a second schema snapshot.
    original_pb = pytest.importorskip("smg_grpc_proto.generated.vllm_engine_pb2")
    original_common = pytest.importorskip("smg_grpc_proto.generated.common_pb2")
    from vllm_gr.entrypoints.grpc.generated import common_pb2

    original_engine = descriptor_pb2.FileDescriptorProto.FromString(
        original_pb.DESCRIPTOR.serialized_pb
    )
    candidate = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)
    assert original_common.DESCRIPTOR.serialized_pb == common_pb2.DESCRIPTOR.serialized_pb

    def compare_messages(old: Any, new: Any) -> None:
        by_name = {item.name: item for item in new}
        for item in old:
            current = by_name[item.name]
            fields = {field.number: field for field in current.field}
            for field in item.field:
                expected = descriptor_pb2.FieldDescriptorProto()
                expected.CopyFrom(field)
                actual = descriptor_pb2.FieldDescriptorProto()
                actual.CopyFrom(fields[field.number])
                # Ignore compiler differences in materialized canonical JSON names;
                # protobuf field names and all wire properties must still match.
                expected.ClearField("json_name")
                actual.ClearField("json_name")
                assert actual == expected, f"Changed wire field: {item.name}.{field.name}"
            compare_messages(item.nested_type, current.nested_type)
            for enum in item.enum_type:
                assert enum == next(e for e in current.enum_type if e.name == enum.name)

    compare_messages(original_engine.message_type, candidate.message_type)
    methods = {m.name: m for m in candidate.service[0].method}
    for method in original_engine.service[0].method:
        assert methods[method.name] == method
    # Original default-pool bindings and extended private-pool bindings coexist.
    assert original_pb.DESCRIPTOR.pool is not pb.DESCRIPTOR.pool
    old = original_pb.GenerateRequest(request_id="old", text="hello")
    new = pb.GenerateRequest.FromString(old.SerializeToString())
    assert new.request_id == "old" and new.text == "hello"
    assert not new.HasField("beam_search")


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
def test_full_request_to_numpy_batch(binary: bool) -> None:
    servicer = BeamVllmEngineServicer(engine(), 0.0)
    requests = [message(i, binary=binary) for i in range(4)]
    requests[0].sampling_params.seed = 7
    requests[0].sampling_params.truncate_prompt_tokens = 2
    batch, ids = servicer.build_batch(requests)
    assert ids == [f"id-{i}" for i in range(4)]
    assert len(batch.items) == 4 and batch.beam_params.beam_width == 2
    tokens = batch.items[0].request.prompt_token_ids
    assert isinstance(tokens, np.ndarray) and tokens.dtype == np.int64
    np.testing.assert_array_equal(tokens, [4, 3])
    assert batch.items[0].request.sampling_params.seed == 7
    assert batch.items[0].request.external_req_id == "id-0"


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "backend,table_format,trie_id",
    [("cuda", "constraint_table_v1", 0), ("npu", "constraint_table_v2", 1)],
)
def test_constrained_length_is_checked_before_grpc_token_decoding(
    binary: bool, backend: str, table_format: str, trie_id: int
) -> None:
    from tests.test_beam_search_v1_input_processor import _canonical_gr_config

    config = _canonical_gr_config(backend=backend)
    config["constraint_table"]["format"] = table_format
    config["beam"]["max_decode_steps"] = 7
    runtime = engine()
    runtime.vllm_config = SimpleNamespace(
        additional_config={"vllm_gr": config}, device_config=SimpleNamespace(device=backend)
    )
    servicer = BeamVllmEngineServicer(runtime, 0.0)
    decode = Mock(wraps=servicer.decode_tokens)
    servicer.decode_tokens = decode
    request = message(binary=binary)
    request.trie_id = trie_id
    for length in (1, 2, 4, 7):
        request.sampling_params.max_tokens = length
        with pytest.raises(
            ValueError,
            match=f"effective generation length {length} must equal {table_format} depth",
        ):
            servicer.build_batch([request])
        decode.assert_not_called()
    request.sampling_params.max_tokens = 3
    batch, _ = servicer.build_batch([request])
    assert batch.beam_params.max_tokens == 3
    assert batch.items[0].trie_id == trie_id
    decode.assert_called_once()


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("trie_id", [None, 0, 1, 2**32 - 1])  # type: ignore[misc]
def test_per_input_trie_selector_wire_and_batch_mapping(binary: bool, trie_id: int | None) -> None:
    request = message(binary=binary)
    if trie_id is not None:
        request.trie_id = trie_id
    decoded = pb.GenerateRequest.FromString(request.SerializeToString())
    assert decoded.HasField("trie_id") == (trie_id is not None)
    batch, _ = BeamVllmEngineServicer(engine(), 0.0).build_batch([decoded])
    assert batch.items[0].trie_id == (0 if trie_id is None else trie_id)


@pytest.mark.parametrize("trie_id", [0, 1])  # type: ignore[misc]
def test_npu_trie_selector_matches_http_validation(trie_id: int) -> None:
    backend = engine()
    backend.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="npu:0"))
    request = message()
    request.trie_id = trie_id
    servicer = BeamVllmEngineServicer(backend, 0.0)
    batch, _ = servicer.build_batch([request])
    assert batch.items[0].trie_id == trie_id


@pytest.mark.parametrize("result_width", [None, 1, 2])  # type: ignore[misc]
def test_beam_options_map_to_execution_and_output_contracts(result_width: int | None) -> None:
    servicer = BeamVllmEngineServicer(engine(), 0.0)
    request = message()
    request.sampling_params.n = 4
    request.beam_search.CopyFrom(
        pb.BeamSearchOptions(
            beam_width=4,
            dynamic_beam_widths=[4, 4, 2],
            length_penalty=0.7,
            constraint_resource_id="default",
            include_parent_beam_indices=True,
        )
    )
    if result_width is not None:
        request.beam_search.result_width = result_width
    batch, _ = servicer.build_batch([request])
    assert batch.beam_params.beam_width == 4
    assert batch.beam_params.dynamic_beam_widths == (4, 4, 2)
    assert batch.beam_params.length_penalty == pytest.approx(0.7)
    assert batch.beam_params.constraint_resource_id == "default"
    assert batch.output_options.num_return_sequences == (
        2 if result_width is None else result_width
    )
    assert batch.output_options.include_parent_beam_indices
    assert batch.items[0].request.sampling_params.n == 1


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "tokenizer_max,token_id,accepted",
    [(63, 127, True), (63, 128, False), (255, 255, True), (255, 256, False), (None, 256, True)],
)
def test_vocabulary_validation_matches_native_input_processor(
    binary: bool, tokenizer_max: int | None, token_id: int, accepted: bool
) -> None:
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    backend = engine()
    tokenizer = (
        None
        if tokenizer_max is None
        else SimpleNamespace(max_token_id=tokenizer_max, truncation_side="left")
    )
    backend.input_processor.tokenizer = tokenizer
    native = SimpleNamespace(
        tokenizer=tokenizer,
        model_config=backend.model_config,
        _validate_prompt_len=lambda *args: None,
    )
    request = message(binary=binary)
    request.tokenized.ClearField("input_ids")
    request.tokenized.ClearField("input_ids_i64_le")
    if binary:
        request.tokenized.input_ids_i64_le = np.array([token_id], dtype="<i8").tobytes()
    else:
        request.tokenized.input_ids.append(token_id)

    def native_validate() -> None:
        InputProcessor._validate_model_input(
            native, {"type": "token", "prompt_token_ids": [token_id]}, "decoder"
        )

    servicer = BeamVllmEngineServicer(backend, 0.0)
    if accepted:
        native_validate()
        batch, _ = servicer.build_batch([request])
        assert batch.items[0].request.prompt_token_ids[0] == token_id
    else:
        with pytest.raises(VLLMValidationError, match="vocabulary"):
            native_validate()
        with pytest.raises(ValueError, match="vocabulary"):
            servicer.build_batch([request])


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("side", ["left", "right", None])  # type: ignore[misc]
def test_prompt_truncation_matches_native_renderer(binary: bool, side: str | None) -> None:
    from vllm.renderers.params import TokenizeParams

    backend = engine()
    tokenizer = None if side is None else SimpleNamespace(truncation_side=side, max_token_id=127)
    backend.input_processor.tokenizer = tokenizer
    request = message(binary=binary)
    request.sampling_params.truncate_prompt_tokens = 2
    # An out-of-vocabulary token in the discarded portion must not fail validation.
    tokens = [128, 4, 3] if side == "left" else [1, 4, 128]
    request.tokenized.ClearField("input_ids")
    request.tokenized.ClearField("input_ids_i64_le")
    if binary:
        request.tokenized.input_ids_i64_le = np.array(tokens, dtype="<i8").tobytes()
    else:
        request.tokenized.input_ids.extend(tokens)
    expected = TokenizeParams(max_total_tokens=8192, truncate_prompt_tokens=2)._token_truncation(
        tokenizer, tokens
    )
    batch, _ = BeamVllmEngineServicer(backend, 0.0).build_batch([request])
    np.testing.assert_array_equal(batch.items[0].request.prompt_token_ids, expected)


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "max_length,truncate,accepted", [(6, None, True), (5, None, False), (5, 2, True), (4, 2, False)]
)
def test_prompt_and_generation_length_limit_after_truncation(
    binary: bool, max_length: int, truncate: int | None, accepted: bool
) -> None:
    backend = engine()
    backend.model_config.max_model_len = max_length
    servicer = BeamVllmEngineServicer(backend, 0.0)
    request = message(binary=binary)
    if truncate is not None:
        request.sampling_params.truncate_prompt_tokens = truncate
    if accepted:
        batch, _ = servicer.build_batch([request])
        assert len(batch.items[0].request.prompt_token_ids) == (truncate or 3)
    else:
        with pytest.raises(ValueError, match="exceeds model length"):
            servicer.build_batch([request])


def test_dynamic_beam_default_result_width_over_grpc() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            request = message()
            request.sampling_params.n = 4
            request.beam_search.beam_width = 4
            request.beam_search.dynamic_beam_widths.extend([4, 4, 2])
            response = await stub.GenerateBatch(
                pb.BatchGenerateRequest(requests=[request]), timeout=5
            )
            assert submitted[0].output_options.num_return_sequences == 2
            assert len(response.items[0].beams.output_offsets) == 3
            request.beam_search.result_width = 3
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await stub.GenerateBatch(pb.BatchGenerateRequest(requests=[request]), timeout=5)
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert len(submitted) == 1

    asyncio.run(run())


def _columns_result(session_id: str, request: Any) -> BeamBatchResult:
    """Equal-length columnar terminal result for binary-response wiring."""
    tokens = np.arange(6, dtype="<i4").reshape(2, 3)
    scores = np.array([-1.0, -2.0], dtype="<f4")
    parents = np.array([[0, 0, 0], [0, 1, 0]], dtype="<i4")
    parent_bytes = (
        parents.tobytes() if request.output_options.include_parent_beam_indices else None
    )
    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=session_id,
        failure=None,
        output=BeamBatchOutput(
            items=tuple(
                BeamRequestOutput(
                    item_index=index,
                    request_id=item.request.request_id,
                    sequences=BeamSequenceColumns(
                        count=2,
                        steps=3,
                        token_ids=tokens.tobytes(),
                        cumulative_logprobs=scores.tobytes(),
                        parent_beam_indices=parent_bytes,
                    ),
                )
                for index, item in enumerate(request.items)
            )
        ),
    )


def _install_columns_wait(backend: SimpleNamespace) -> None:
    async def wait(session_id: str, request: Any) -> BeamBatchResult:
        return _columns_result(session_id, request)

    backend.wait_gr_session = wait


@pytest.mark.l2  # type: ignore[misc]
def test_binary_beam_output_is_opt_in_over_grpc() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            _install_columns_wait(backend)
            response = await stub.GenerateBatch(
                pb.BatchGenerateRequest(requests=[message()]), timeout=5
            )
            assert submitted[0].output_options.num_return_sequences == 2
            item = response.items[0]
            assert not item.HasField("binary_beams")
            assert list(item.beams.output_ids) == [0, 1, 2, 3, 4, 5]
            assert list(item.beams.output_offsets) == [0, 3, 6]
            assert list(item.beams.cumulative_logprobs) == [-1.0, -2.0]
            assert not list(item.beams.parent_beam_indices)
            assert list(item.beams.parent_offsets) == [0, 0, 0]

    asyncio.run(run())


@pytest.mark.l2  # type: ignore[misc]
def test_binary_beam_output_round_trips_columns_over_grpc() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            _install_columns_wait(backend)
            member = message()
            member.beam_search.include_parent_beam_indices = True
            request = pb.BatchGenerateRequest(requests=[member], binary_beam_output=True)
            response = await stub.GenerateBatch(request, timeout=5)
            assert submitted[0].output_options.include_parent_beam_indices
            item = response.items[0]
            assert item.HasField("binary_beams")
            assert not item.HasField("beams")
            encoded = item.binary_beams
            assert (encoded.count, encoded.steps, encoded.prompt_tokens) == (2, 3, 3)
            np.testing.assert_array_equal(
                np.frombuffer(encoded.output_ids_i32_le, dtype="<i4").reshape(2, 3),
                np.arange(6, dtype="<i4").reshape(2, 3),
            )
            np.testing.assert_array_equal(
                np.frombuffer(encoded.cumulative_logprobs_f32_le, dtype="<f4"),
                np.array([-1.0, -2.0], dtype="<f4"),
            )
            np.testing.assert_array_equal(
                np.frombuffer(encoded.parent_beam_indices_i32_le, dtype="<i4").reshape(2, 3),
                np.array([[0, 0, 0], [0, 1, 0]], dtype="<i4"),
            )

    asyncio.run(run())


@pytest.mark.l2  # type: ignore[misc]
def test_binary_beam_output_falls_back_for_small_results_over_grpc() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):

            async def wait(session_id: str, request: Any) -> BeamBatchResult:
                result = _columns_result(session_id, request)
                assert result.output is not None
                columns_item = result.output.items[0]
                small_item = BeamRequestOutput(
                    item_index=1,
                    request_id=request.items[1].request.request_id,
                    sequences=(
                        BeamOutputSequence(
                            token_ids=(7,),
                            cumulative_logprob=-3.0,
                            parent_beam_indices=(),
                            finish_reason="stop",
                            stop_token_id=7,
                        ),
                    ),
                )
                return BeamBatchResult(
                    protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                    batch_id=session_id,
                    failure=None,
                    output=BeamBatchOutput(items=(columns_item, small_item)),
                )

            backend.wait_gr_session = wait
            request = pb.BatchGenerateRequest(
                requests=[message(0), message(1)], binary_beam_output=True
            )
            response = await stub.GenerateBatch(request, timeout=5)
            first, second = response.items
            assert first.HasField("binary_beams")
            assert not first.binary_beams.HasField("parent_beam_indices_i32_le")
            assert not first.HasField("beams")
            assert second.HasField("beams")
            assert not second.HasField("binary_beams")
            assert list(second.beams.output_ids) == [7]
            assert list(second.beams.finish_reasons) == ["stop"]
            assert list(second.beams.stop_token_ids) == [7]

    asyncio.run(run())

@pytest.mark.parametrize("fault", ["stream", "penalty", "mm", "kv", "mixed", "duplicate"])  # type: ignore[misc]
def test_incompatible_inputs_are_rejected_not_ignored(fault: str) -> None:
    servicer = BeamVllmEngineServicer(engine(), 0.0)
    requests = [message(0), message(1)]
    if fault == "stream":
        requests[0].stream = True
    elif fault == "penalty":
        requests[0].sampling_params.presence_penalty = 0.5
    elif fault == "mm":
        requests[0].mm_inputs.SetInParent()
    elif fault == "kv":
        requests[0].kv_transfer_params.remote_host = "producer"
    elif fault == "mixed":
        requests[1].sampling_params.max_tokens = 2
    else:
        requests[1].request_id = requests[0].request_id
    with pytest.raises(ValueError):
        servicer.build_batch(requests)


@asynccontextmanager
async def connection() -> AsyncIterator[tuple[Any, SimpleNamespace, list[Any], Any]]:
    """Real TCP transport with a deterministic, observable engine backend."""
    backend = engine()
    submitted = []

    async def submit(request: Any) -> Any:
        submitted.append(request)
        return request

    async def wait(session_id: str, request: Any) -> Any:
        return BeamBatchResult(
            protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
            batch_id=session_id,
            failure=None,
            output=BeamBatchOutput(
                items=tuple(
                    BeamRequestOutput(
                        item_index=i,
                        request_id=item.request.request_id,
                        sequences=(
                            BeamOutputSequence(
                                token_ids=(1, 2, 3),
                                cumulative_logprob=-1.0,
                                parent_beam_indices=(),
                                finish_reason="length",
                                stop_token_id=None,
                            ),
                            BeamOutputSequence(
                                token_ids=(4,),
                                cumulative_logprob=None,
                                parent_beam_indices=(0,),
                                finish_reason="stop",
                                stop_token_id=4,
                            ),
                        ),
                    )
                    for i, item in enumerate(request.items)
                )
            ),
        )

    backend.submit_gr_session = submit
    backend.wait_gr_session = wait
    backend.release_gr_session = AsyncMock()
    backend.abort_gr_session = AsyncMock()
    server = grpc.aio.server()
    servicer = BeamVllmEngineServicer(backend, 0.0)
    rpc.add_VllmEngineServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{port}", options=(("grpc.enable_http_proxy", 0),)
        ) as channel:
            yield rpc.VllmEngineStub(channel), backend, submitted, servicer
    finally:
        await server.stop(0)


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("method", ["Generate", "GenerateBatch"])  # type: ignore[misc]
@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_trie_selection_over_grpc(binary: bool, method: str, device: str) -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            backend.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=device))
            if method == "Generate":
                request = message(binary=binary)
                request.trie_id = 1
                replies = [reply async for reply in stub.Generate(request, timeout=5)]
                assert len(replies) == 2
                assert [item.trie_id for item in submitted[0].items] == [1]
            else:
                requests = [message(i, binary=binary) for i in range(4)]
                requests[1].trie_id = 1
                requests[2].trie_id = 0
                requests[3].trie_id = 1
                reply = await stub.GenerateBatch(
                    pb.BatchGenerateRequest(requests=requests), timeout=5
                )
                assert [item.request_id for item in reply.items] == [f"id-{i}" for i in range(4)]
                assert [item.trie_id for item in submitted[0].items] == [0, 1, 0, 1]
            backend.release_gr_session.assert_awaited_once()
            assert not servicer._beam_tasks

    asyncio.run(run())


@pytest.mark.parametrize("trie_count", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("method", ["Generate", "GenerateBatch"])  # type: ignore[misc]
@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_invalid_trie_uses_core_admission_and_service_recovers(
    trie_count: int, method: str, device: str
) -> None:
    from queue import Queue

    from vllm_gr.v1.engine.core import _handle_submit_gr_session
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()

    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            backend.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=device))
            original_submit = backend.submit_gr_session
            core = SimpleNamespace(_gr_trie_count=trie_count, input_queue=Queue())

            async def submit(request: Any) -> Any:
                # Exercise the real Core admission boundary against the loaded count.
                _handle_submit_gr_session(core, request)
                return await original_submit(request)

            backend.submit_gr_session = submit
            request = message()
            request.trie_id = trie_count

            async def invoke() -> Any:
                if method == "Generate":
                    return [reply async for reply in stub.Generate(request, timeout=5)]
                return await stub.GenerateBatch(
                    pb.BatchGenerateRequest(requests=[request]), timeout=5
                )

            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await invoke()
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "trie_id is outside the loaded constraint table" in caught.value.details()
            assert not submitted and core.input_queue.empty()
            assert not servicer._beam_tasks
            request.ClearField("trie_id")
            await invoke()
            assert len(submitted) == 1 and submitted[0].items[0].trie_id == 0
            assert not core.input_queue.empty()

    asyncio.run(run())


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("invalid_index", [1, 3])  # type: ignore[misc]
@pytest.mark.parametrize("cleanup_delay", [0.0, 0.01])  # type: ignore[misc]
def test_mixed_batch_queue_rejection_is_atomic_and_service_recovers(
    grpc_admission_core: Any, binary: bool, invalid_index: int, cleanup_delay: float
) -> None:
    """Route a later member's rejection through Core output and the RPC collector."""
    import msgspec
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.output_processor import OutputProcessor

    from tests.test_gr_admission_queue import worker_receipt
    from vllm_gr.v1.engine.gr_async_scheduler import GR_BATCH_RESULT_ATTR
    from vllm_gr.v1.engine.gr_session_client import (
        submit_gr_session_async,
        wait_gr_session_async,
    )

    core = grpc_admission_core
    core._gr_trie_count = core.scheduler._gr_trie_count = 2
    core.add_request = Mock(wraps=core.add_request)
    core.preprocess_add_request = Mock(wraps=core.preprocess_add_request)
    free_blocks = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    def finish_batch(request: Any) -> None:
        # Keep real admission/retirement; only the model's receipts are synthetic.
        for _ in range(20):
            if not core.scheduler.has_requests():
                break
            scheduled = core.scheduler.schedule()
            output = worker_receipt(core.scheduler, scheduled)
            if output.req_ids:
                tokens = output.sampled_token_ids[0]
                output.req_ids = list(scheduled.num_scheduled_tokens)
                output.req_id_to_index = {rid: i for i, rid in enumerate(output.req_ids)}
                output.sampled_token_ids = [list(tokens) for _ in output.req_ids]
            for session_id, result in getattr(output, GR_BATCH_RESULT_ATTR, {}).items():
                sequences = tuple(
                    msgspec.structs.replace(sequence, parent_beam_indices=())
                    for sequence in result.output.items[0].sequences
                )
                getattr(output, GR_BATCH_RESULT_ATTR)[session_id] = msgspec.structs.replace(
                    result,
                    output=BeamBatchOutput(
                        tuple(
                            BeamRequestOutput(item.item_index, item.request.request_id, sequences)
                            for item in request.items
                        )
                    ),
                )
            outputs = core.scheduler.update_from_output(scheduled, output)
            for routed_output in outputs.items():
                core.output_queue.put_nowait(routed_output)
        assert not core.scheduler.has_requests()

    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            backend.output_processor = OutputProcessor(None, log_stats=False)
            backend._run_output_handler = lambda: None
            delivered = []

            async def send(request: Any) -> None:
                submitted.append(request)
                for item in request.items:
                    item.request.client_index = 3
                core._handle_submit_gr_session(request)
                core._process_input_queue()
                if core.scheduler.has_requests():
                    finish_batch(request)
                while not core.output_queue.empty():
                    client_index, output = core.output_queue.get_nowait()
                    assert client_index == 3
                    # Exercise the untyped terminal payload after an IPC round trip.
                    output = msgspec.msgpack.decode(
                        msgspec.msgpack.encode(output), type=EngineCoreOutputs
                    )
                    delivered.append(output)
                    backend.output_processor.process_outputs(output.outputs)

            async def submit(request: Any) -> Any:
                return await submit_gr_session_async(backend, request)

            async def wait(session_id: str, handle: Any) -> Any:
                return await wait_gr_session_async(backend, session_id, handle)

            backend.engine_core = SimpleNamespace(submit_gr_session_async=send)
            backend.submit_gr_session = submit
            backend.wait_gr_session = wait
            original_close = servicer.client.close

            async def close(session: Any) -> None:
                # Also exercise a handler whose cleanup yields after sending the error.
                if cleanup_delay:
                    await asyncio.sleep(cleanup_delay)
                await original_close(session)

            servicer.client.close = close

            async def assert_clean() -> None:
                async def wait_for_handler_cleanup() -> None:
                    # context.abort can deliver the RPC error before the handler's
                    # finally block starts, so a snapshot of _close_tasks can be empty.
                    while servicer._beam_tasks or servicer.client._close_tasks:
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_for_handler_cleanup(), timeout=5)
                assert core.input_queue.empty() and core.output_queue.empty()
                assert not core.scheduler.requests and not core.scheduler.waiting
                assert not getattr(core.scheduler, "_gr_v1_sessions", {})
                assert not getattr(core.scheduler, "_gr_v1_request_sessions", {})
                assert (
                    core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
                )
                assert not backend.output_processor.request_states
                assert not backend.output_processor.external_req_ids
                assert not servicer._beam_tasks and not servicer.client._close_tasks

            requests = [message(i, binary=binary) for i in range(4)]
            for request, trie_id in zip(requests, [0, 1, 0, 1], strict=True):
                request.trie_id = trie_id
                request.sampling_params.n = request.beam_search.beam_width = 4
            requests[invalid_index].trie_id = 2
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await stub.GenerateBatch(pb.BatchGenerateRequest(requests=requests), timeout=5)
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "trie_id is outside the loaded constraint table" in caught.value.details()
            core.preprocess_add_request.assert_not_called()
            core.add_request.assert_not_called()
            assert len(delivered) == 1
            owner = submitted[0].items[0].request.request_id
            assert delivered[0].finished_requests == {owner}
            assert [output.request_id for output in delivered[0].outputs] == [owner]
            await assert_clean()
            backend.release_gr_session.assert_awaited_once_with(owner)

            # Reuse external IDs, so leaked frontend registrations cannot hide.
            requests[invalid_index].trie_id = invalid_index % 2
            reply = await stub.GenerateBatch(pb.BatchGenerateRequest(requests=requests), timeout=5)
            assert [item.request_id for item in reply.items] == [f"id-{i}" for i in range(4)]
            assert all(len(item.beams.output_offsets) == 5 for item in reply.items)
            assert core.add_request.call_count == 4
            await assert_clean()
            assert backend.release_gr_session.await_count == 2
            backend.abort_gr_session.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("encoding", ["binary", "repeated", "mixed"])  # type: ignore[misc]
def test_batch_response_preserves_variable_beams_and_releases_session(encoding: str) -> None:
    """Flattened outputs retain offsets, absent scores, stop IDs and item order."""

    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            response = await stub.GenerateBatch(
                pb.BatchGenerateRequest(
                    batch_id="external",
                    requests=[
                        message(
                            i, binary=encoding == "binary" or (encoding == "mixed" and i % 2 == 0)
                        )
                        for i in range(4)
                    ],
                )
            )
            assert response.batch_id == "external" and len(response.items) == 4
            assert [item.request_id for item in response.items] == [f"id-{i}" for i in range(4)]
            assert len(submitted) == 1 and len(submitted[0].items) == 4
            for item in submitted[0].items:
                np.testing.assert_array_equal(item.request.prompt_token_ids, [1, 4, 3])
            beams = response.items[0].beams
            assert not response.items[0].responses
            assert list(beams.cumulative_logprobs) == [-1.0, 0.0]
            assert list(beams.has_cumulative_logprob) == [True, False]
            assert list(beams.output_ids) == [1, 2, 3, 4]
            assert list(beams.output_offsets) == [0, 3, 4]
            assert list(beams.parent_offsets) == [0, 0, 1]
            assert list(beams.parent_beam_indices) == [0]
            assert beams.prompt_tokens == 3
            assert list(beams.finish_reasons) == ["length", "stop"]
            assert list(beams.stop_token_ids) == [-1, 4]
            backend.release_gr_session.assert_awaited_once()
            assert not servicer._beam_tasks

    asyncio.run(run())


@pytest.mark.parametrize(  # type: ignore[misc]
    ("ids", "raw", "error"),
    [
        ([], None, "nonempty"),
        ([128], None, "vocabulary"),
        ([], b"", "divisible by 8"),
        ([], b"bad", "divisible by 8"),
        ([1], np.array([1], dtype="<i8").tobytes(), "mutually exclusive"),
        ([1], b"", "mutually exclusive"),
        ([], np.array([-1], dtype="<i8").tobytes(), "nonnegative"),
        ([], np.array([128], dtype="<i8").tobytes(), "vocabulary"),
    ],
    ids=[
        "empty",
        "repeated-range",
        "empty-binary",
        "unaligned-binary",
        "both",
        "both-empty-binary",
        "negative-binary",
        "binary-range",
    ],
)
def test_invalid_token_input_returns_invalid_argument_without_submission(
    ids: list[int], raw: bytes | None, error: str
) -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            invalid = message()
            invalid.tokenized.CopyFrom(pb.TokenizedInput(input_ids=ids))
            if raw is not None:
                invalid.tokenized.input_ids_i64_le = raw
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await stub.GenerateBatch(pb.BatchGenerateRequest(requests=[invalid]))
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert error in caught.value.details()
            assert not submitted
            backend.release_gr_session.assert_not_called()
            assert not servicer._beam_tasks

    asyncio.run(run())


def test_binary_prompt_length_is_inferred_per_batch_member() -> None:
    servicer = BeamVllmEngineServicer(engine(), 0.0)
    lengths = [1, 2, 3, 4096]
    requests = [message(i) for i in range(4)]
    for request, length in zip(requests, lengths, strict=True):
        tokens = np.arange(length, dtype="<i8") % 128
        request.tokenized.input_ids_i64_le = tokens.tobytes()
    batch, _ = servicer.build_batch(requests)
    for item, length in zip(batch.items, lengths, strict=True):
        assert item.request.prompt_token_ids.shape == (length,)
        np.testing.assert_array_equal(item.request.prompt_token_ids, np.arange(length) % 128)


def test_single_generate_keeps_original_response_shape() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            singles = [value async for value in stub.Generate(message())]
            assert len(singles) == 2
            assert list(singles[1].complete.output_ids) == [4]
            assert singles[1].complete.index == 1
            assert singles[1].complete.matched_token_id == 4
            assert not singles[1].complete.HasField("cumulative_logprob")
            backend.release_gr_session.assert_awaited_once()
            assert not servicer._beam_tasks

    asyncio.run(run())


@pytest.mark.parametrize("width", [2, 5])  # type: ignore[misc]
@pytest.mark.parametrize("code", [ADMISSION_REJECTED_CODE, "model_failure"])  # type: ignore[misc]
def test_terminal_failure_status_and_subsequent_valid_request(width: int, code: str) -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            original_wait = backend.wait_gr_session

            async def rejected(session_id: str, request: Any) -> Any:
                return BeamBatchResult(
                    protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
                    batch_id=session_id,
                    output=None,
                    failure=BeamBatchFailure(code, "request rejected", None),
                )

            backend.wait_gr_session = rejected
            request = message()
            request.sampling_params.n = width
            request.beam_search.beam_width = width
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await stub.GenerateBatch(pb.BatchGenerateRequest(requests=[request]), timeout=5)
            expected = (
                grpc.StatusCode.INVALID_ARGUMENT
                if code == ADMISSION_REJECTED_CODE
                else grpc.StatusCode.INTERNAL
            )
            assert caught.value.code() == expected
            assert caught.value.details() == "request rejected"
            await asyncio.wait_for(
                asyncio.gather(*set(servicer._beam_tasks.values()), return_exceptions=True),
                timeout=5,
            )
            assert not servicer._beam_tasks
            backend.release_gr_session.assert_awaited_once()
            backend.abort_gr_session.assert_not_called()
            backend.release_gr_session.reset_mock()
            backend.wait_gr_session = original_wait
            response = await stub.GenerateBatch(
                pb.BatchGenerateRequest(requests=[message()]), timeout=5
            )
            assert len(response.items[0].beams.output_offsets) == 3
            backend.release_gr_session.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["Generate", "GenerateBatch"])  # type: ignore[misc]
@pytest.mark.parametrize("fault", ["admission", "transport", "engine_dead"])  # type: ignore[misc]
def test_submission_failure_status_cleanup_and_recovery(method: str, fault: str) -> None:
    from vllm.v1.engine.exceptions import EngineDeadError

    from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError

    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            original_submit = backend.submit_gr_session
            failure = {
                "admission": GRAdmissionError("invalid admission"),
                "transport": RuntimeError("transport unavailable"),
                "engine_dead": EngineDeadError(),
            }[fault]
            backend.submit_gr_session = AsyncMock(side_effect=failure)

            async def invoke() -> Any:
                if method == "Generate":
                    return [reply async for reply in stub.Generate(message(), timeout=5)]
                return await stub.GenerateBatch(
                    pb.BatchGenerateRequest(requests=[message()]), timeout=5
                )

            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await invoke()
            assert caught.value.code() == (
                grpc.StatusCode.INVALID_ARGUMENT
                if fault == "admission"
                else grpc.StatusCode.INTERNAL
            )
            assert caught.value.details() == str(failure)
            await asyncio.wait_for(
                asyncio.gather(*set(servicer._beam_tasks.values()), return_exceptions=True),
                timeout=5,
            )
            assert not servicer._beam_tasks
            backend.release_gr_session.assert_not_called()
            if fault == "engine_dead":
                backend.abort_gr_session.assert_not_called()
            else:
                backend.abort_gr_session.assert_awaited_once()
            backend.submit_gr_session = original_submit
            response = await invoke()
            assert len(response if method == "Generate" else response.items) > 0
            backend.release_gr_session.assert_awaited_once()

    asyncio.run(run())


def test_backend_failure_returns_internal_and_aborts_session() -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            backend.wait_gr_session = AsyncMock(side_effect=RuntimeError("backend failed"))
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await stub.GenerateBatch(pb.BatchGenerateRequest(requests=[message()]), timeout=5)
            assert caught.value.code() == grpc.StatusCode.INTERNAL
            await asyncio.wait_for(
                asyncio.gather(*set(servicer._beam_tasks.values()), return_exceptions=True),
                timeout=5,
            )
            backend.abort_gr_session.assert_awaited_once()
            backend.release_gr_session.assert_not_called()
            assert not servicer._beam_tasks

    asyncio.run(run())


@pytest.mark.parametrize("cancel_from", ["client", "abort_rpc"])  # type: ignore[misc]
def test_cancelled_batch_aborts_session_and_clears_member_ids(cancel_from: str) -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            waiting = asyncio.Event()
            aborted = asyncio.Event()

            async def wait(*args: Any) -> Any:
                waiting.set()
                await asyncio.Future()

            async def abort(*args: Any) -> None:
                aborted.set()

            backend.wait_gr_session = wait
            backend.abort_gr_session = AsyncMock(side_effect=abort)
            call = stub.GenerateBatch(
                pb.BatchGenerateRequest(requests=[message(0), message(1)]), timeout=10
            )
            try:
                await asyncio.wait_for(waiting.wait(), timeout=5)
                handler = servicer._beam_tasks["id-0"]
                if cancel_from == "client":
                    call.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await call
                else:
                    await stub.Abort(pb.AbortRequest(request_ids=["id-1"]), timeout=5)
                    with pytest.raises(grpc.aio.AioRpcError) as caught:
                        await call
                    assert caught.value.code() == grpc.StatusCode.CANCELLED
                await asyncio.wait_for(aborted.wait(), timeout=5)
                await asyncio.wait_for(asyncio.gather(handler, return_exceptions=True), timeout=5)
                assert not servicer._beam_tasks
                backend.abort_gr_session.assert_awaited_once()
                backend.release_gr_session.assert_not_called()
            finally:
                call.cancel()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["ordinary_generate", "embed"])  # type: ignore[misc]
def test_non_gr_methods_are_unimplemented(method: str) -> None:
    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                if method == "embed":
                    await stub.Embed(pb.EmbedRequest(), timeout=5)
                else:
                    request = message()
                    request.ClearField("beam_search")
                    _ = [response async for response in stub.Generate(request, timeout=5)]
            assert caught.value.code() == grpc.StatusCode.UNIMPLEMENTED
            assert not submitted

    asyncio.run(run())


def test_inherited_metadata_rpcs_work_with_extended_protocol() -> None:
    """Upstream response classes serialize through our private-pool bindings."""

    async def run() -> None:
        async with connection() as (stub, backend, submitted, servicer):
            model = backend.model_config
            model.model = "random-gr-model"
            model.runner_type = "generate"
            model.is_multimodal_model = False
            model.served_model_name = "gr"
            model.tokenizer = "random-tokenizer"
            model.architectures = ["LlamaForCausalLM"]
            model.hf_config = SimpleNamespace(
                model_type="llama", eos_token_id=[2, 3], pad_token_id=0, bos_token_id=1
            )
            backend.vllm_config = SimpleNamespace(kv_transfer_config=None)
            assert servicer.start_time == 0.0
            for errored in (False, True):
                backend.errored = errored
                health = await stub.HealthCheck(pb.HealthCheckRequest(), timeout=5)
                assert health.healthy is (not errored)
            info = await stub.GetModelInfo(pb.GetModelInfoRequest(), timeout=5)
            assert info.model_path == model.model
            assert info.served_model_name == "gr"
            assert info.vocab_size == 128 and info.max_context_length == 8192
            assert info.is_generation and not info.supports_vision
            assert info.tokenizer_path == model.tokenizer
            assert info.model_type == "llama"
            assert list(info.architectures) == model.architectures
            assert list(info.eos_token_ids) == [2, 3]
            assert info.bos_token_id == 1 and info.pad_token_id == 0
            for config in (None, SimpleNamespace(kv_connector="test", kv_role="kv_both")):
                backend.vllm_config.kv_transfer_config = config
                info = await stub.GetServerInfo(pb.GetServerInfoRequest(), timeout=5)
                assert info.kv_connector == (config.kv_connector if config else "")
                assert info.kv_role == (config.kv_role if config else "")
            assert not submitted

    asyncio.run(run())


def test_upstream_startup_keeps_health_reflection_stats_and_cleanup_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise upstream serve_grpc with real listeners and an observable engine."""
    import signal
    import socket

    pytest.importorskip("smg_grpc_servicer")
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
    from vllm.entrypoints.launchers import grpc_server as upstream

    from vllm_gr.entrypoints.beam_search_v1.client import _ASYNC_CLEANUP_TASKS
    from vllm_gr.entrypoints.grpc.serve import patch_grpc_server

    async def run() -> None:
        backend = engine()
        backend.check_health = AsyncMock()
        logged = asyncio.Event()
        backend.do_log_stats = AsyncMock(side_effect=logged.set)
        cleanup_done = asyncio.Event()
        shutdown_order: list[bool] = []
        backend.shutdown = lambda: shutdown_order.append(cleanup_done.is_set())
        signals = {}
        signals_ready = asyncio.Event()
        listener_stopped = asyncio.Event()
        create_server = grpc.aio.server

        def tracked_server(*args: Any, **kwargs: Any) -> Any:
            server = create_server(*args, **kwargs)
            stop = server.stop

            async def tracked_stop(grace: Any) -> None:
                await stop(grace)
                listener_stopped.set()

            monkeypatch.setattr(server, "stop", tracked_stop)
            return server

        monkeypatch.setattr(grpc.aio, "server", tracked_server)

        def add_signal(sig: int, callback: Callable[[], None]) -> None:
            signals[sig] = callback
            if len(signals) == 2:
                signals_ready.set()

        monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", add_signal)
        monkeypatch.setattr(
            upstream,
            "AsyncEngineArgs",
            SimpleNamespace(
                from_cli_args=lambda args: SimpleNamespace(create_engine_config=lambda **kw: None),
            ),
        )
        monkeypatch.setattr(
            upstream,
            "AsyncLLM",
            SimpleNamespace(
                from_vllm_config=lambda **kw: backend,
            ),
        )
        monkeypatch.setattr(upstream.envs, "VLLM_LOG_STATS_INTERVAL", 0.01)
        original = {
            name: getattr(upstream, name)
            for name in (
                "vllm_engine_pb2",
                "vllm_engine_pb2_grpc",
                "VllmEngineServicer",
                "reflection",
            )
        }
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        args = SimpleNamespace(
            model="test",
            host="127.0.0.1",
            port=port,
            enable_log_requests=False,
            disable_log_stats=False,
        )
        with patch_grpc_server():
            task = asyncio.create_task(upstream.serve_grpc(args))
            try:
                await asyncio.wait_for(signals_ready.wait(), timeout=5)
                async with grpc.aio.insecure_channel(
                    f"127.0.0.1:{port}", options=(("grpc.enable_http_proxy", 0),)
                ) as channel:
                    health = health_pb2_grpc.HealthStub(channel)
                    for service in ("", "vllm.grpc.engine.VllmEngine"):
                        response = await health.Check(
                            health_pb2.HealthCheckRequest(service=service),
                            timeout=5,
                        )
                        assert response.status == health_pb2.HealthCheckResponse.SERVING
                    backend.check_health.side_effect = RuntimeError("unhealthy")
                    response = await health.Check(health_pb2.HealthCheckRequest(), timeout=5)
                    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING

                    reflection = reflection_pb2_grpc.ServerReflectionStub(channel)

                    async def query(**kwargs: Any) -> list[Any]:
                        async def requests() -> AsyncIterator[Any]:
                            yield reflection_pb2.ServerReflectionRequest(**kwargs)

                        return [
                            reply
                            async for reply in reflection.ServerReflectionInfo(
                                requests(),
                                timeout=5,
                            )
                        ]

                    listed = (await query(list_services=""))[0]
                    assert {service.name for service in listed.list_services_response.service} == {
                        "vllm.grpc.engine.VllmEngine",
                        "grpc.health.v1.Health",
                        "grpc.reflection.v1alpha.ServerReflection",
                    }
                    for symbol in (
                        "vllm.grpc.engine.VllmEngine",
                        "grpc.health.v1.Health",
                        "grpc.reflection.v1alpha.ServerReflection",
                    ):
                        reflected = (await query(file_containing_symbol=symbol))[0]
                        assert not reflected.HasField("error_response")
                        descriptors = [
                            descriptor_pb2.FileDescriptorProto.FromString(raw)
                            for raw in reflected.file_descriptor_response.file_descriptor_proto
                        ]
                        if symbol == "vllm.grpc.engine.VllmEngine":
                            engine_service = next(
                                service
                                for file in descriptors
                                for service in file.service
                                if service.name == "VllmEngine"
                            )
                            assert "GenerateBatch" in {
                                method.name for method in engine_service.method
                            }
                    await asyncio.wait_for(logged.wait(), timeout=5)

                async def cleanup() -> None:
                    # Only become runnable once upstream's listener stop returns.
                    # Without GR draining, shutdown would run before this task.
                    await listener_stopped.wait()
                    cleanup_done.set()

                # Simulate detached retry cleanup left by a cancelled request.
                pending = asyncio.create_task(cleanup())
                _ASYNC_CLEANUP_TASKS.add(pending)
                pending.add_done_callback(_ASYNC_CLEANUP_TASKS.discard)
                signals[signal.SIGTERM]()
                await asyncio.wait_for(task, timeout=5)
                assert shutdown_order == [True]
                assert not _ASYNC_CLEANUP_TASKS
            finally:
                if not task.done():
                    signals.get(signal.SIGTERM, task.cancel)()
                await asyncio.gather(task, return_exceptions=True)
        assert all(getattr(upstream, name) is value for name, value in original.items())

    asyncio.run(run())


pytestmark = pytest.mark.cpu_test
