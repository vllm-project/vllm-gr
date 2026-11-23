# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""NPU request refusals survive the real multiprocess preprocessing handler."""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCoreProc

from tests.helpers.gr_config import make_gr_config
from vllm_gr.config import (
    ConstraintTableBackendEnum,
    ConstraintTableSpec,
    write_canonical_gr_config,
)
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BEAM_BATCH_PROTOCOL_VERSION,
    GR_BATCH_RESULT_PARAM,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
    GRAdmissionError,
    admission_rejection,
)
from vllm_gr.v1.engine.core import (
    _admit_gr_session,
    _handle_submit_gr_session,
    dispatch_gr_engine_core_input,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types

REASON = "NPU unconstrained beam_search requires candidate_width to match each stage's beam width"


@pytest.fixture(scope="module", autouse=True)
def _register_gr_request_types() -> None:
    # These tests bypass normal EngineCore startup, so initialize their wire ABI explicitly.
    register_engine_core_request_types()


def _config(
    *,
    device: str = "npu",
    sid: bool = False,
    configured_width: int | None = None,
    catalog: bool = False,
    capacity: int = 4,
) -> Any:
    config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device=device),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        use_v2_model_runner=False,
    )
    table = ConstraintTableSpec(
        max_top_k=capacity,
        candidate_width=configured_width,
        backend=ConstraintTableBackendEnum(device),
        enabled=catalog,
        path=str(Path(__file__).resolve().with_name("catalog.npz")) if catalog else None,
        artifact_digest="sha256:" + "a" * 64 if catalog else None,
        tokenizer_digest="sha256:" + "b" * 64 if catalog else None,
    )
    write_canonical_gr_config(
        config,
        make_gr_config(
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_worker_decision=True,
            constrained_generation=catalog,
            beam_sid_stage_vocab=sid,
            constraint_table=table,
        ),
    )
    return config


def _request(
    candidate_width: int | None = 4,
    widths: tuple[int, ...] | None = (2, 4, 4),
    temperature: float = 0.0,
    per_parent: tuple[int, ...] | None = None,
) -> BeamBatchRequest:
    native = EngineCoreRequest(
        request_id="npu-admission",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=3,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=native.request_id,
        items=(BeamBatchItem(0, native),),
        beam_params=BeamBatchParams(
            4,
            3,
            temperature,
            (),
            True,
            candidate_width=candidate_width,
            dynamic_beam_widths=widths,
            per_parent_beam_widths=per_parent,
        ),
        output_options=BeamOutputOptions(4 if widths is None else widths[-1], True, True, False),
    )


def _core(config: Any) -> Any:
    # Retain upstream's actual preprocessing ERROR implementation and its
    # output encoder. No model, native request or Worker allocation is needed.
    core: Any = object.__new__(EngineCoreProc)
    core.vllm_config = config
    core.batch_queue = []
    core.batch_queue_size = 2
    core.input_queue = Queue()
    core.output_queue = Queue()
    core.scheduler = SimpleNamespace(
        requests={}, _gr_v1_sessions={}, _gr_v1_request_sessions={}, _gr_execution_owner=None
    )
    core.preprocess_add_request = Mock(
        side_effect=lambda request: (
            SimpleNamespace(request_id=request.request_id, sampling_params=request.sampling_params),
        )
    )
    core.add_request = Mock()
    core._handle_request_preproc_error = Mock(wraps=core._handle_request_preproc_error)
    return core


def _dispatch(core: Any, request: BeamBatchRequest, path: str) -> None:
    if path == "submit":
        _handle_submit_gr_session(core, request)
        request_type, payload = core.input_queue.get_nowait()
    else:
        # The input-socket loop queues this decoded batch unchanged. Exercise
        # the production busy-loop dispatch that consumes that queue item.
        request_type, payload = EngineCoreRequestType.SUBMIT_GR_SESSION, request
    assert dispatch_gr_engine_core_input(core, request_type, payload)
    assert core.input_queue.empty()


def _output(core: Any) -> Any:
    client_index, outputs = core.output_queue.get_nowait()
    assert client_index == 3
    assert outputs.finished_requests == ["npu-admission"]
    assert len(outputs.outputs) == 1
    output = outputs.outputs[0]
    assert output.request_id == "npu-admission"
    assert output.finish_reason is FinishReason.ERROR
    assert not output.new_token_ids
    assert core.output_queue.empty()
    return output


def _assert_no_allocation(core: Any) -> None:
    core.preprocess_add_request.assert_not_called()
    core.add_request.assert_not_called()
    assert not core.scheduler.requests
    assert not core.scheduler._gr_v1_sessions
    assert not core.scheduler._gr_v1_request_sessions
    assert core.scheduler._gr_execution_owner is None


@pytest.mark.parametrize("path", ["submit", "socket"])
@pytest.mark.parametrize("sid", [False, True])
@pytest.mark.parametrize(
    "candidate_width,widths,configured_width",
    [(2, None, None), (4, (2, 4, 4), None), (None, (2, 4, 4), 4)],
)
def test_npu_width_refusal_preserves_reason_without_preprocessing(
    path: str,
    sid: bool,
    candidate_width: int | None,
    widths: tuple[int, ...] | None,
    configured_width: int | None,
) -> None:
    core = _core(_config(sid=sid, configured_width=configured_width))
    _dispatch(core, _request(candidate_width, widths), path)

    output = _output(core)
    assert admission_rejection(output) == REASON
    failure = output.kv_transfer_params[GR_BATCH_RESULT_PARAM].failure
    assert failure.code == ADMISSION_REJECTED_CODE
    assert failure.message == REASON
    core._handle_request_preproc_error.assert_not_called()
    _assert_no_allocation(core)

    # A refused session leaves the core usable for the next supported request.
    _dispatch(core, _request(4, (4, 4, 4)), path)
    core.preprocess_add_request.assert_called_once()
    core.add_request.assert_called_once()
    assert core.output_queue.empty()


@pytest.mark.parametrize("path", ["submit", "socket"])
def test_inproc_npu_width_refusal_raises_admission_error(path: str) -> None:
    core = _core(_config())
    del core.output_queue
    request = _request()
    with pytest.raises(GRAdmissionError, match="candidate_width.*beam width") as caught:
        if path == "submit":
            _handle_submit_gr_session(core, request)
            request_type, payload = core.input_queue.get_nowait()
            dispatch_gr_engine_core_input(core, request_type, payload)
        else:
            _admit_gr_session(core, request)
    assert str(caught.value) == REASON
    core._handle_request_preproc_error.assert_not_called()
    _assert_no_allocation(core)


@pytest.mark.l1
@pytest.mark.parametrize("path", ["submit", "socket", "inproc"])
@pytest.mark.parametrize("sid", [False, True])
def test_catalog_free_per_parent_refusal_preserves_reason_and_owner(path: str, sid: bool) -> None:
    core = _core(_config(sid=sid))
    request = _request(None, None, per_parent=(4, 2, 1))
    reason = "NPU per_parent_beam_widths requires an enabled canonical NPU constraint table"
    if path == "inproc":
        del core.output_queue
        with pytest.raises(
            GRAdmissionError, match="per_parent_beam_widths.*canonical NPU"
        ) as caught:
            _admit_gr_session(core, request)
        assert str(caught.value) == reason
    else:
        _dispatch(core, request, path)
        output = _output(core)
        assert admission_rejection(output) == reason
        assert (
            output.kv_transfer_params[GR_BATCH_RESULT_PARAM].failure.code == ADMISSION_REJECTED_CODE
        )
    core._handle_request_preproc_error.assert_not_called()
    _assert_no_allocation(core)
    if path != "inproc":
        _dispatch(core, _request(4, None), path)
        core.preprocess_add_request.assert_called_once()
        core.add_request.assert_called_once()
        assert core.output_queue.empty()


@pytest.mark.l1
@pytest.mark.parametrize("path", ["submit", "socket", "inproc"])
@pytest.mark.parametrize("capacity,overflow", [(4, 8), (8, 9)])
@pytest.mark.parametrize("overflow_stages", ["middle", "last", "multiple"])
def test_constrained_per_parent_capacity_refusal_precedes_allocation_and_recovers(
    path: str, capacity: int, overflow: int, overflow_stages: str
) -> None:
    core = _core(_config(catalog=True, capacity=capacity))
    # The first step must produce the initial retained width. Keep that
    # independent shared invariant valid while exceeding the table later.
    schedules = {
        "middle": (4, overflow, 1),
        "last": (4, 1, overflow),
        "multiple": (4, overflow, overflow),
    }
    request = _request(None, None, per_parent=schedules[overflow_stages])
    reason = f"NPU per_parent_beam_widths exceeds constraint_table.max_top_k={capacity}"
    if path == "inproc":
        del core.output_queue
        with pytest.raises(GRAdmissionError, match="per_parent_beam_widths.*max_top_k") as caught:
            _admit_gr_session(core, request)
        assert str(caught.value) == reason
    else:
        _dispatch(core, request, path)
        output = _output(core)
        assert admission_rejection(output) == reason
        failure = output.kv_transfer_params[GR_BATCH_RESULT_PARAM].failure
        assert failure.code == ADMISSION_REJECTED_CODE and failure.message == reason
    core._handle_request_preproc_error.assert_not_called()
    _assert_no_allocation(core)
    assert core.input_queue.empty() and core.batch_queue == []

    # Reuse the same Core and external request ID after refusal. Equality
    # with the configured capacity is supported, including K != W at cap 8.
    supported = _request(None, None, per_parent=(4, capacity, 1))
    if path == "inproc":
        _admit_gr_session(core, supported)
    else:
        _dispatch(core, supported, path)
        assert core.output_queue.empty()
    core.preprocess_add_request.assert_called_once()
    core.add_request.assert_called_once()
    core._handle_request_preproc_error.assert_not_called()
    assert core.input_queue.empty()


@pytest.mark.l1
def test_npu_per_parent_capacity_refusal_does_not_change_cuda_admission() -> None:
    core = _core(_config(device="cuda", catalog=True, capacity=4))
    _dispatch(core, _request(None, None, per_parent=(4, 8, 1)), "socket")
    core.preprocess_add_request.assert_called_once()
    core.add_request.assert_called_once()
    core._handle_request_preproc_error.assert_not_called()
    assert core.input_queue.empty() and core.output_queue.empty()


@pytest.mark.parametrize("path", ["submit", "socket"])
@pytest.mark.parametrize("fault", ["runner-v2", "startup-width", "runtime", "preprocessing"])
def test_deployment_and_preprocessing_faults_keep_the_real_error_path(
    path: str, fault: str
) -> None:
    core = _core(_config(configured_width=2 if fault == "startup-width" else None))
    request = _request(4, (2, 2, 2))
    if fault == "runner-v2":
        core.vllm_config.use_v2_model_runner = True
    elif fault == "runtime":
        core.vllm_config.scheduler_config.async_scheduling = False
    elif fault == "preprocessing":
        request = _request(4, (4, 4, 4))
        core.preprocess_add_request.side_effect = ValueError("native preprocessing failed")

    _dispatch(core, request, path)

    # These outputs come from upstream EngineCoreProc's real handler. A plain
    # ERROR remains a server fault even when validation raised ValueError.
    core._handle_request_preproc_error.assert_called_once_with(request.items[0].request)
    assert admission_rejection(_output(core)) is None
    core.add_request.assert_not_called()
    if fault != "preprocessing":
        _assert_no_allocation(core)


@pytest.mark.parametrize("path", ["submit", "socket"])
@pytest.mark.parametrize("mode", ["cuda", "npu-catalog", "npu-unconstrained"])
def test_supported_admission_paths_are_preserved(path: str, mode: str) -> None:
    core = _core(_config(device="cuda" if mode == "cuda" else "npu", catalog=mode == "npu-catalog"))
    request = _request(4, (4, 4, 4) if mode == "npu-unconstrained" else (2, 4, 4))

    _dispatch(core, request, path)

    core.preprocess_add_request.assert_called_once()
    core.add_request.assert_called_once()
    core._handle_request_preproc_error.assert_not_called()
    assert core.output_queue.empty()


@pytest.mark.parametrize("path", ["submit", "socket", "inproc"])
@pytest.mark.parametrize("temperature", [1e-50, 1e40, 10**40])
def test_unrepresentable_npu_temperature_is_an_admission_rejection(
    path: str, temperature: float
) -> None:
    core = _core(_config())
    request = _request(4, (4, 4, 4), temperature)
    reason = (
        "NPU unconstrained beam_search requires positive temperature "
        "to be representable as a finite nonzero float32 value"
    )
    if path == "inproc":
        del core.output_queue
        with pytest.raises(GRAdmissionError, match="temperature.*float32") as caught:
            _admit_gr_session(core, request)
        assert str(caught.value) == reason
    else:
        _dispatch(core, request, path)
        assert admission_rejection(_output(core)) == reason
    core._handle_request_preproc_error.assert_not_called()
    _assert_no_allocation(core)


@pytest.mark.parametrize("mode", ["cuda", "npu-catalog", "npu-unconstrained"])
@pytest.mark.parametrize("temperature", [0.0, 1e-40, 1.0, 2])
def test_representable_temperatures_preserve_supported_admission(
    mode: str, temperature: float
) -> None:
    core = _core(_config(device="cuda" if mode == "cuda" else "npu", catalog=mode == "npu-catalog"))
    _dispatch(core, _request(4, (4, 4, 4), temperature), "socket")
    core.preprocess_add_request.assert_called_once()
    core.add_request.assert_called_once()
    core._handle_request_preproc_error.assert_not_called()
    assert core.output_queue.empty()


pytestmark = pytest.mark.cpu_test
