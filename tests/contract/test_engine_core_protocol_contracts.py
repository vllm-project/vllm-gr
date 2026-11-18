# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: ignore-errors
"""Production V1 transport inventory; retired message bytes stay reserved."""

import importlib
from contextlib import contextmanager
from queue import Queue
from threading import Event, Thread
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
import zmq
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_gr.v1.engine.wire import register_engine_core_request_types

UPSTREAM_REQUEST_WIRE_VALUES = {
    "ADD": b"\x00",
    "ABORT": b"\x01",
    "START_DP_WAVE": b"\x02",
    "UTILITY": b"\x03",
    "EXECUTOR_FAILED": b"\x04",
    "WAKEUP": b"\x05",
}


ENGINE_CORE_REQUEST_FIELDS = (
    "request_id",
    "prompt_token_ids",
    "mm_features",
    "sampling_params",
    "pooling_params",
    "arrival_time",
    "lora_request",
    "cache_salt",
    "data_parallel_rank",
    "prompt_embeds",
    "prompt_is_token_ids",
    "client_index",
    "current_wave",
    "priority",
    "trace_headers",
    "resumable",
    "external_req_id",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "abort_immediately",
    "session_id",
)


def _sampling_params() -> SamplingParams:
    return SamplingParams(max_tokens=4, temperature=1.0)


def _native_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="request-0",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=_sampling_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="native-session",
    )


def test_native_request_payload_schema() -> None:
    # Guard against upstream vLLM EngineCoreRequest field schema changes.
    assert EngineCoreRequest.__struct_fields__ == ENGINE_CORE_REQUEST_FIELDS

    request = _native_request()
    encoded = msgspec.msgpack.encode([request])

    raw_payload = msgspec.msgpack.decode(encoded)
    assert isinstance(raw_payload, list)
    assert len(raw_payload) == 1
    assert isinstance(raw_payload[0], list)
    assert raw_payload[0][0] == "request-0"

    decoded = MsgpackDecoder(list[EngineCoreRequest]).decode(encoded)
    assert len(decoded) == 1
    assert isinstance(decoded[0], EngineCoreRequest)
    assert decoded[0].request_id == "request-0"
    assert decoded[0].session_id == "native-session"


def _transport_engine(optional_features: bool):
    from vllm.config.kv_events import KVEventsConfig
    from vllm.v1.engine.core import EngineCoreProc

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=1024, dtype=torch.bfloat16, enable_sleep_mode=optional_features
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=11,
            block_size=32,
            mamba_block_size=64 if optional_features else None,
            kv_cache_size_tokens=352,
            kv_cache_max_concurrency=0.34375,
        ),
        parallel_config=SimpleNamespace(
            world_size=8,
            data_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=17, max_num_batched_tokens=512),
        instance_id="transport-test",
        lora_config=SimpleNamespace(max_loras=3) if optional_features else None,
        weight_transfer_config=SimpleNamespace(backend="test-backend")
        if optional_features
        else None,
    )
    engine = SimpleNamespace(
        vllm_config=config,
        engine_index=1,
        frontend_stats_publish_address="tcp://127.0.0.1:9999",
        tensor_ipc_receiver=None,
        input_queue=Queue(),
        aborts_queue=Queue(),
        output_queue=Queue(),
        scheduler=SimpleNamespace(
            get_kv_event_publisher_config=lambda: KVEventsConfig() if optional_features else None
        ),
        model_executor=SimpleNamespace(supports_draft_weight_updates=lambda: optional_features),
        preprocess_add_request=Mock(side_effect=lambda req: (req, req.current_wave)),
        _handle_request_preproc_error=Mock(),
        ft_commands=Queue(),
    )
    engine.ft_sentinel = SimpleNamespace(
        handle_command=lambda *args: engine.ft_commands.put_nowait(args)
    )
    engine._make_ready_response = Mock(
        wraps=MethodType(EngineCoreProc._make_ready_response, engine)
    )
    engine._handle_mm_cache_miss = MethodType(EngineCoreProc._handle_mm_cache_miss, engine)
    return engine


@contextmanager
def _socket_loop(monkeypatch, implementation="gr", *, coordinator=False, optional_features=False):
    """Real ZMQ transport; only poll termination is supplied by the test."""
    from vllm.v1.engine.core import EngineCoreProc

    import vllm_gr.v1.engine.engine_core_patch as ecp

    ecp.apply_engine_core_request_patches()
    loop = (
        ecp.UPSTREAM_PROCESS_INPUT_SOCKETS
        if implementation == "native"
        else EngineCoreProc.process_input_sockets
    )
    assert loop is not None
    stop = Event()
    ready = Event()
    errors = []
    poller_class = zmq.Poller

    class StopLoop(BaseException):
        pass

    class StoppablePoller:
        def __init__(self):
            self.poller = poller_class()

        def register(self, *args):
            return self.poller.register(*args)

        def poll(self):
            while not stop.is_set():
                events = self.poller.poll(20)
                if events:
                    return events
            raise StopLoop

    monkeypatch.setattr(zmq, "Poller", StoppablePoller)
    engine = _transport_engine(optional_features)
    identity = b"engine-1"
    sockets = []
    with zmq.Context() as context:

        def socket(kind):
            result = context.socket(kind)
            result.setsockopt(zmq.LINGER, 0)
            result.setsockopt(zmq.RCVTIMEO, 5000)
            result.setsockopt(zmq.SNDTIMEO, 5000)
            port = result.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(result)
            return result, f"tcp://127.0.0.1:{port}"

        frontends = [socket(zmq.ROUTER) for _ in range(2)]
        coord, coord_address = socket(zmq.XPUB) if coordinator else (None, None)

        def worker():
            try:
                loop(engine, [address for _, address in frontends], coord_address, identity, ready)
            except StopLoop:
                pass
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        try:
            payloads = []
            for router, _ in frontends:
                sender, payload = router.recv_multipart()
                assert sender == identity
                payloads.append(payload)
            if coord is not None:
                assert not ready.is_set()
                assert coord.recv() == b"\x01"
                coord.send(b"READY")
            assert ready.wait(5)

            def send(kind, payload, *, frontend=0, via_coordinator=False):
                frames = [kind.value, *MsgpackEncoder(size_threshold=0).encode(payload)]
                if via_coordinator:
                    coord.send_multipart(frames)
                else:
                    frontends[frontend][0].send_multipart([identity, *frames])

            yield SimpleNamespace(
                engine=engine, ready_payloads=payloads, send=send, coordinator=coord
            )
        finally:
            stop.set()
            if coord is not None:
                coord.send(b"READY")  # release a coordinator handshake on a failing test
            thread.join(5)
            for item in sockets:
                item.close()
            assert not thread.is_alive(), "input thread did not stop"
            assert not errors, errors


@pytest.mark.parametrize("implementation", ["native", "gr"])
@pytest.mark.parametrize("optional_features", [False, True])
def test_process_input_sockets_adapter_preserves_upstream_handshake(
    monkeypatch, implementation, optional_features
):
    from vllm.v1.engine import EngineCoreReadyResponse
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.core_client import MPClient

    with _socket_loop(
        monkeypatch, implementation, optional_features=optional_features
    ) as transport:
        expected = EngineCoreProc._make_ready_response(transport.engine)
        transport.engine._make_ready_response.assert_called_once_with()
        for payload in transport.ready_payloads:
            response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
            assert response == expected  # all native fields, including optional capabilities
            client = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(max_model_len=4096),
                    cache_config=SimpleNamespace(
                        num_gpu_blocks=5,
                        block_size=16,
                        mamba_block_size=None,
                        kv_cache_size_tokens=None,
                        kv_cache_max_concurrency=None,
                    ),
                ),
                stats_update_address=None,
            )
            MPClient._apply_ready_response(client, payload)
            assert client.vllm_config.model_config.max_model_len == 1024
            cache = client.vllm_config.cache_config
            assert cache.num_gpu_blocks == 16
            assert cache.block_size == 32
            assert cache.mamba_block_size == expected.mamba_block_size
            assert cache.kv_cache_size_tokens == 352
            assert cache.kv_cache_max_concurrency == 0.34375
            assert client.stats_update_address == expected.dp_stats_address


@pytest.mark.parametrize("implementation", ["native", "gr"])
@pytest.mark.parametrize("coordinator", [False, True])
def test_input_socket_control_and_error_routes(monkeypatch, implementation, coordinator):
    from vllm.multimodal.cache import MultiModalCacheMissError
    from vllm.v1.engine import FinishReason
    from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD

    with _socket_loop(monkeypatch, implementation, coordinator=coordinator) as transport:
        engine = transport.engine
        request = _native_request()
        request.current_wave = 7
        request.client_index = 1
        transport.send(EngineCoreRequestType.ADD, request, frontend=1)
        kind, (decoded, wave) = engine.input_queue.get(timeout=5)
        assert kind == EngineCoreRequestType.ADD
        assert decoded == request and wave == 7

        misses = ["image-a", "image-b"]
        engine.preprocess_add_request.side_effect = MultiModalCacheMissError(misses)
        transport.send(EngineCoreRequestType.ADD, request)
        client_idx, output = engine.output_queue.get(timeout=5)
        assert client_idx == 1
        assert output.engine_index == 1
        assert output.outputs[0].mm_cache_miss_hashes == misses
        assert output.outputs[0].finish_reason == FinishReason.ERROR
        assert output.finished_requests == {request.request_id}
        engine._handle_request_preproc_error.assert_not_called()

        engine.preprocess_add_request.side_effect = ValueError("invalid input")
        transport.send(EngineCoreRequestType.ADD, request)
        command = {"command": "test"}
        transport.send(EngineCoreRequestType.UTILITY, (1, 31, FT_UTILITY_METHOD, [command]))
        assert engine.ft_commands.get(timeout=5) == (1, 31, command)
        engine._handle_request_preproc_error.assert_called_once_with(request)
        assert engine.input_queue.empty()  # failed ADD and sentinel command never enter busy loop

        utility = (1, 32, "ordinary_utility", ["argument"])
        transport.send(EngineCoreRequestType.UTILITY, utility)
        kind, decoded = engine.input_queue.get(timeout=5)
        assert kind == EngineCoreRequestType.UTILITY
        assert tuple(decoded) == utility

        ids = ["request-0", "request-1"]
        transport.send(EngineCoreRequestType.ABORT, ids)
        assert engine.aborts_queue.get(timeout=5) == ids
        assert engine.input_queue.get(timeout=5) == (EngineCoreRequestType.ABORT, ids)
        assert engine.aborts_queue.empty()
        if coordinator:
            transport.coordinator.send(b"READY")  # repeated DP notification is ignored
            transport.send(EngineCoreRequestType.START_DP_WAVE, 9, via_coordinator=True)
            assert engine.input_queue.get(timeout=5) == (EngineCoreRequestType.START_DP_WAVE, 9)
        assert engine.input_queue.empty()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_gr_socket_queues_whole_batch_with_tensor_frames(monkeypatch, batch_size):
    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.beam.batch_contracts import BeamBatchItem, BeamBatchRequest

    source = make_batch_request()
    embeds = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    items = tuple(
        BeamBatchItem(
            i,
            msgspec.structs.replace(
                source.items[0].request,
                request_id=f"member-{i}",
                session_id=f"native-session-{i}",
                prompt_embeds=embeds + i,
            ),
        )
        for i in range(batch_size)
    )
    source = msgspec.structs.replace(source, items=items)
    with _socket_loop(monkeypatch) as transport:
        transport.send(EngineCoreRequestType.SUBMIT_GR_SESSION, source)
        kind, decoded = transport.engine.input_queue.get(timeout=5)
        assert kind == EngineCoreRequestType.SUBMIT_GR_SESSION
        assert isinstance(decoded, BeamBatchRequest)
        assert decoded.batch_id == source.batch_id
        assert decoded.beam_params == source.beam_params
        assert decoded.output_options == source.output_options
        assert len(decoded.items) == batch_size
        for i, item in enumerate(decoded.items):
            assert item.item_index == i
            assert isinstance(item.request, EngineCoreRequest)
            assert item.request.request_id == f"member-{i}"
            assert item.request.session_id == f"native-session-{i}"
            torch.testing.assert_close(item.request.prompt_embeds, embeds + i)
        transport.engine.preprocess_add_request.assert_not_called()
        assert transport.engine.input_queue.empty()


def test_only_v1_extends_native_wire_types() -> None:
    register_engine_core_request_types()
    register_engine_core_request_types()
    actual = {name: member.value for name, member in EngineCoreRequestType.__members__.items()}
    expected = UPSTREAM_REQUEST_WIRE_VALUES | {"SUBMIT_GR_SESSION": b"\x0a"}
    assert actual == expected
    for name, value in expected.items():
        assert EngineCoreRequestType(value) is getattr(EngineCoreRequestType, name)


@pytest.mark.parametrize("value", [b"\x06", b"\x07", b"\x08", b"\x09"])
def test_retired_wire_bytes_rejected(value: bytes) -> None:
    register_engine_core_request_types()
    with pytest.raises(ValueError):
        EngineCoreRequestType(value)


def test_gr_dispatch_only_admits_v1(monkeypatch) -> None:
    from vllm_gr.v1.engine import core

    register_engine_core_request_types()
    handled = []
    engine = SimpleNamespace()
    batch = object()
    monkeypatch.setattr(core, "_admit_gr_session", lambda *args: handled.append(args))
    for kind in UPSTREAM_REQUEST_WIRE_VALUES:
        assert not core.dispatch_gr_engine_core_input(
            engine, getattr(EngineCoreRequestType, kind), batch
        )
    assert core.dispatch_gr_engine_core_input(
        engine, EngineCoreRequestType.SUBMIT_GR_SESSION, batch
    )
    assert handled == [(engine, batch)]


@pytest.mark.l1
@pytest.mark.parametrize(
    ("old", "owner"),
    [
        ("engine.arg_utils_gr", "engine.arg_utils"),
        ("v1.engine.async_llm", "v1.engine.output_processor"),
        ("v1.engine.npu_admission", "platforms.npu_admission"),
    ],
)
def test_legacy_engine_import_uses_same_module(old, owner):
    assert importlib.import_module(f"vllm_gr.{old}") is importlib.import_module(f"vllm_gr.{owner}")


@pytest.mark.l1
def test_legacy_engine_symbols_resolve_to_runtime_owners():
    from vllm_gr.v1.engine import core, core_client, core_client_patch, core_socket
    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    assert core.process_input_sockets is core_socket.process_input_sockets
    assert core_client_patch.inproc_engine_failure is core_client.inproc_engine_failure
    assert run_engine_core.__module__ == "vllm_gr.v1.engine.engine_core_patch"


@pytest.mark.l1
@pytest.mark.parametrize(
    ("legacy_name", "owner", "installer"),
    [
        ("patch_batch_and_fork", "vllm_gr.patch", "install_engine_session_hooks"),
        (
            "patch_engine_core_process_entrypoint",
            "vllm_gr.v1.engine.engine_core_patch",
            "patch_engine_core_process_entrypoint",
        ),
    ],
)
def test_legacy_openai_installers_forward_to_runtime_owners(
    monkeypatch, legacy_name, owner, installer
):
    from vllm_gr.entrypoints.openai import beam_search_patch

    install = Mock()
    monkeypatch.setattr(importlib.import_module(owner), installer, install)
    getattr(beam_search_patch, legacy_name)()
    install.assert_called_once_with()


pytestmark = pytest.mark.cpu_test
