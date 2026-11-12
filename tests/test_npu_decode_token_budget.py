# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,misc,var-annotated"
"""NPU physical-token admission through real frontend and Core contracts."""

import asyncio
from copy import copy
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore

from tests.test_async_scheduling_scheduler import receipt
from tests.test_beam_search_v1_input_processor import _canonical_gr_config, _make_engine
from tests.test_gr_admission_queue import (
    assert_one_error,
    bare_core,
    batch,
    complete,
)
from tests.test_gr_admission_queue import core as _core_fixture
from tests.test_gr_admission_queue import scheduler as _scheduler_fixture
from tests.test_gr_async_scheduler import _batch_request
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.input_processor import BeamSearchInputProcessor
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam.batch_contracts import BeamBatchItem
from vllm_gr.v1.engine.beam_search_v1 import validate_gr_request
from vllm_gr.v1.engine.core import _handle_submit_gr_session
from vllm_gr.v1.engine.gr_async_scheduler import (
    GRAdmissionError,
    _prepare_gr_session,
)

core = _core_fixture
scheduler = _scheduler_fixture


def _config(*, device="npu", budget=512, width=128, constrained=True):
    canonical = _canonical_gr_config(enabled=constrained, backend=device)
    if not constrained:
        canonical["constrained_generation"] = False
        canonical["constraint_table"] = {"enabled": False, "backend": device}
    canonical["beam"]["max_width"] = width
    canonical["constraint_table"]["max_top_k"] = width
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device=device)
    engine.vllm_config.model_config.max_model_len = 512
    engine.vllm_config.scheduler_config.max_num_seqs = 128
    engine.vllm_config.scheduler_config.max_num_batched_tokens = budget
    return engine


def _request(*, size=8, width=128, steps=3, widths=None, prompt_len=16):
    template = _batch_request(steps)
    items = []
    for index in range(size):
        native = copy(template.items[0].request)
        native.request_id = template.batch_id if index == 0 else f"{template.batch_id}:{index}"
        native.prompt_token_ids = np.full(prompt_len, 1, dtype=np.int64)
        items.append(BeamBatchItem(index, native))
    params = msgspec.structs.replace(
        template.beam_params,
        beam_width=width,
        dynamic_beam_widths=widths,
    )
    return msgspec.structs.replace(
        template,
        items=tuple(items),
        beam_params=params,
        output_options=msgspec.structs.replace(
            template.output_options,
            num_return_sequences=width if widths is None else widths[-1],
        ),
    )


# These expected results follow the actual Decode input schedule, independently
# of the final output width or the session's persistent storage capacity.
CASES = [
    pytest.param(8, 3, None, 512, False, id="fixed-over-budget"),
    pytest.param(4, 3, None, 512, True, id="fixed-exact-boundary"),
    pytest.param(4, 3, None, 511, False, id="fixed-one-token-short"),
    pytest.param(8, 3, None, 1024, True, id="fixed-larger-budget"),
    pytest.param(8, 1, None, 512, True, id="prefill-only"),
    pytest.param(8, 1, (128,), 512, True, id="dynamic-prefill-only"),
    pytest.param(8, 2, None, 512, False, id="two-steps-fixed-over-budget"),
    pytest.param(8, 2, (32, 128), 256, True, id="two-steps-wide-final-output"),
    pytest.param(8, 2, (128, 32), 512, False, id="two-steps-wide-input"),
    pytest.param(8, 3, (32, 64, 128), 512, True, id="wide-final-output"),
    pytest.param(8, 3, (64, 128, 32), 512, False, id="wide-middle-input"),
    pytest.param(8, 3, (128, 64, 32), 512, False, id="wide-first-input"),
    pytest.param(8, 3, (64, 32, 16), 512, True, id="narrowing-boundary"),
]


@pytest.mark.parametrize("size,steps,widths,budget,accepted", CASES)
def test_shared_npu_request_and_scheduler_admission(size, steps, widths, budget, accepted):
    # Short requests exercise physical-token admission without a three-layer table.
    config = _config(budget=budget, constrained=steps == 3).vllm_config
    request = _request(size=size, steps=steps, widths=widths)
    scheduler = SimpleNamespace(
        scheduler_config=config.scheduler_config,
        vllm_config=config,
        requests={},
        # This is a logical Scheduler limit, not the ModelRunner buffer size.
        max_num_scheduled_tokens=2048,
    )
    if accepted:
        validate_gr_request(request, config=config)
        session = _prepare_gr_session(scheduler, request)
        assert session.request is request
    else:
        with pytest.raises(ValueError, match="NPU Beam Decode requires .*physical tokens"):
            validate_gr_request(request, config=config)
        with pytest.raises(GRAdmissionError, match="max_num_batched_tokens"):
            _prepare_gr_session(scheduler, request)
        assert not hasattr(scheduler, "_gr_v1_request_sessions")
    assert not getattr(scheduler, "_gr_v1_sessions", {})
    assert not getattr(scheduler, "_gr_owner", None)
    assert not getattr(scheduler, "_gr_execution_capacity", None)
    assert not scheduler.requests


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("size,steps,widths,budget,accepted", CASES)
def test_npu_budget_reaches_native_offline_and_online_frontends(
    online, size, steps, widths, budget, accepted
):
    engine = _config(budget=budget, constrained=steps == 3)
    processor = BeamSearchInputProcessor(engine)
    params = BeamSearchParams(
        beam_width=128,
        max_tokens=steps,
        dynamic_beam_widths=None if widths is None else list(widths),
        ignore_eos=True,
    )
    prompts = [[1] * 16 for _ in range(size)]
    tokenizer = SimpleNamespace(eos_token_id=2)

    def prepare():
        if online:
            serving = SimpleNamespace(input_processor=SimpleNamespace(tokenizer=tokenizer))
            return asyncio.run(
                processor.process_online_inputs(
                    serving,
                    OnlineBeamSearchRequest(tuple(prompts), "budget-test", params),
                )
            )
        engine.get_supported_tasks = lambda: ("generate",)
        llm = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            _preprocess_cmpl=lambda prompts: prompts,
            _lora_request_to_seq=lambda lora, size: [lora] * size,
        )
        return processor.process_offline_inputs(llm, OfflineBeamSearchRequest(prompts, params))

    if accepted:
        assert len(prepare().items) == size
    else:
        with pytest.raises(BeamSearchRequestError, match="max_num_batched_tokens"):
            prepare()
    engine.submit_gr_session.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_npu_physical_budget_does_not_change_other_devices(device):
    config = _config(device="cuda").vllm_config
    config.device_config.device = device
    # CPU is not a supported V1 executor, but shared request semantics must
    # remain device-independent; validate_gr_device_config owns that rejection.
    validate_gr_request(_request(), config=config)


def _configure_core_budget(core):
    # Keep the real native Scheduler/KV manager. A tiny equivalent geometry
    # makes rejected B2*W4 and accepted B1*W4 fit this CPU fixture's KV pool.
    core.vllm_config.device_config = SimpleNamespace(device="npu")
    core.vllm_config.scheduler_config.max_num_batched_tokens = 4


def _oversized_core_request():
    return _request(size=2, width=4, prompt_len=1)


@pytest.mark.parametrize("via_socket", [False, True])
def test_npu_budget_rejection_before_native_add_then_valid_request_completes(core, via_socket):
    _configure_core_budget(core)
    scheduler = core.scheduler
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    bad = _oversized_core_request()
    if via_socket:
        core._handle_submit_gr_session(bad)
        assert core.input_queue.empty()
    else:
        core._handle_client_request(EngineCoreRequestType.SUBMIT_GR_SESSION, bad)
    assert_one_error(core, bad.batch_id, client=bad.items[0].request.client_index)
    assert core.is_running()
    assert not scheduler.requests
    assert not scheduler.waiting
    assert not getattr(scheduler, "_gr_v1_sessions", {})
    assert not getattr(scheduler, "_gr_v1_request_sessions", {})
    assert getattr(scheduler, "_gr_owner_generation", 0) == 0
    assert getattr(scheduler, "_gr_execution_capacity", None) is None
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    core._handle_submit_gr_session(batch("good", prompt=1))
    core._process_input_queue()
    complete(core, "good")
    assert core.output_queue.empty()
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_npu_budget_inprocess_rejection_then_valid_request_completes(scheduler):
    core = bare_core(scheduler, EngineCore)
    _configure_core_budget(core)
    with pytest.raises(GRAdmissionError, match="max_num_batched_tokens=4"):
        _handle_submit_gr_session(core, _oversized_core_request())
    assert not scheduler.requests
    assert not getattr(scheduler, "_gr_v1_sessions", {})
    assert getattr(scheduler, "_gr_execution_capacity", None) is None
    _handle_submit_gr_session(core, batch("good", prompt=1))
    complete(core, "good")


def test_npu_budget_rejection_preserves_inflight_owner(core):
    _configure_core_budget(core)
    core._handle_submit_gr_session(batch("active", prompt=1))
    core._process_input_queue()
    scheduler = core.scheduler
    session = scheduler._gr_v1_sessions["active"]
    native = scheduler.requests["active"]
    first = scheduler.schedule()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    generation = scheduler._gr_owner_generation
    pending = set(session._in_flight_dispatches)
    bad = _oversized_core_request()
    core._handle_submit_gr_session(bad)
    assert core.input_queue.empty()
    assert_one_error(core, bad.batch_id, client=bad.items[0].request.client_index)
    assert scheduler._gr_v1_sessions == {"active": session}
    assert scheduler.requests == {"active": native}
    assert scheduler._gr_owner == "active"
    assert scheduler._gr_owner_generation == generation
    assert session._in_flight_dispatches == pending
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.update_from_output(first, receipt(first))
    complete(core, "active")


pytestmark = pytest.mark.cpu_test
