# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for NPU decode graph capture and FULL dispatch."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests._npu_model_runner_patch_helpers import (
    _stub_beam_graph_install,
    _stub_gr_config,
)
from vllm_gr.v1.worker import npu_model_runner_capture as capture_mod


def test_npu_capture_disabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(monkeypatch, SimpleNamespace(beam_graph_active=False))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert capture_mod.capture_beam_decode_graph(runner) is False


def test_npu_capture_cascade_mode_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the graph-safe xllm execution mode, capture must refuse."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: False)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert capture_mod.capture_beam_decode_graph(runner) is False


def test_npu_capture_tp_gt1_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, use_ubatching=False),
    )
    assert capture_mod.capture_beam_decode_graph(runner) is False


def test_npu_capture_ubatching_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=True),
    )
    assert capture_mod.capture_beam_decode_graph(runner) is False


@pytest.mark.parametrize("graph_widths", [None, (4,)])  # type: ignore[misc]
def test_npu_capture_reentry_returns_true(
    monkeypatch: pytest.MonkeyPatch, graph_widths: tuple[int, ...] | None
) -> None:
    """If a graph for the width is already recorded, capture short-circuits True."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(
            beam_graph_active=True,
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_graph_widths=graph_widths,
        ),
    )
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    installs = _stub_beam_graph_install(monkeypatch, ready_widths=(4,))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_graph_keys={4},
        _gr_npu_beam_graph_shapes={(1, 4)},
    )
    assert capture_mod.capture_beam_decode_graph(runner) is True
    assert installs == [runner]
    assert runner.model is runner._gr_npu_graph_model


@pytest.mark.parametrize("graph_widths", [None, (4,)])  # type: ignore[misc]
def test_npu_capture_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch, graph_widths: tuple[int, ...] | None
) -> None:
    """Warmup + capture forwards must run inside torch.inference_mode().

    The ACL graph pool holds inference tensors created by the native
    ``_dummy_run`` (decorated with ``@torch.inference_mode()``).  Running the
    capture forward outside inference mode makes ``capture_begin``'s in-place
    pool updates raise "Inplace update to inference tensor outside
    InferenceMode is not allowed".
    """
    import contextlib

    from vllm.compilation import monitor

    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(
            beam_graph_active=True,
            beam_graph_batch_size=2,
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_graph_widths=graph_widths,
        ),
    )
    monkeypatch.setattr(capture_mod, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(monitor, "set_cudagraph_capturing_enabled", lambda *_: None)

    side_stream = SimpleNamespace(wait_stream=lambda *_: None)
    fake_npu = SimpleNamespace(
        Stream=lambda: side_stream,
        current_stream=lambda: SimpleNamespace(wait_stream=lambda *_: None),
        stream=lambda _side: contextlib.nullcontext(),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    dummy_sessions = tuple(SimpleNamespace(session_id=f"dummy-{index}") for index in range(2))
    attention_batch = SimpleNamespace(
        sessions=dummy_sessions,
        end_session=lambda: None,
    )
    create_args: list[tuple[Any, ...]] = []

    def fake_create(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        create_args.append(args)
        return object(), attention_batch, object(), object()

    monkeypatch.setattr(
        capture_mod,
        "_capture_create_dummy_context_npu",
        fake_create,
    )

    installs = _stub_beam_graph_install(monkeypatch)
    inference_flags: list[bool] = []

    def fake_forward(*args: Any, **kwargs: Any) -> None:
        inference_flags.append(torch.is_inference_mode_enabled())

    monkeypatch.setattr(capture_mod, "_dummy_beam_forward_npu", fake_forward)
    monkeypatch.setattr(
        capture_mod, "_verify_and_record_captured_graph_npu", lambda r, width: width == 8
    )

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _current_beam_info=None,
    )
    assert capture_mod.capture_beam_decode_graph(runner) is True
    assert installs == [runner]
    assert create_args[0][2] == 2
    assert runner._gr_npu_beam_graph_shapes == {(2, 4)}
    assert len(inference_flags) == capture_mod._BEAM_CAPTURE_NUM_WARMUPS + 1
    assert all(inference_flags), f"capture forwards ran outside inference_mode: {inference_flags}"


def test_npu_verify_and_record_records_on_match() -> None:
    beam_model = SimpleNamespace(graph_ready=lambda width: width == 4)
    runner = SimpleNamespace(model=beam_model, _gr_npu_graph_model=beam_model)
    assert capture_mod._verify_and_record_captured_graph_npu(runner, width=4) is True
    assert runner._beam_graph_keys == {4}


@pytest.mark.parametrize("owner_state", ["absent", "replaced", "uncaptured"])  # type: ignore[misc]
def test_npu_verify_and_record_rejects_unowned_or_missing_graph(owner_state: str) -> None:
    # A matching descriptor elsewhere must never establish Beam ownership.
    ordinary = SimpleNamespace(concrete_aclgraph_entries={4: object()})
    runner = SimpleNamespace(model=ordinary)
    if owner_state != "absent":
        beam_model = SimpleNamespace(graph_ready=lambda width: owner_state != "uncaptured")
        runner._gr_npu_graph_model = beam_model
        if owner_state == "uncaptured":
            runner.model = beam_model
    assert capture_mod._verify_and_record_captured_graph_npu(runner, width=4) is False
    assert not hasattr(runner, "_beam_graph_keys")


def test_npu_verify_and_record_appends_to_existing_keys() -> None:
    beam_model = SimpleNamespace(graph_ready=lambda width: width == 8)
    runner = SimpleNamespace(model=beam_model, _gr_npu_graph_model=beam_model, _beam_graph_keys={4})
    assert capture_mod._verify_and_record_captured_graph_npu(runner, width=8) is True
    assert runner._beam_graph_keys == {4, 8}


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.parametrize("first_path", ["ordinary", "beam"])  # type: ignore[misc]
def test_npu_beam_and_ordinary_graphs_isolate_matching_descriptors(
    monkeypatch: pytest.MonkeyPatch, first_path: str
) -> None:
    acl_graph = pytest.importorskip(
        "vllm_ascend.compilation.acl_graph", reason="vllm-ascend is not installed"
    )
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.beam.context import BEAM_DATA_VAR
    from vllm_gr.v1.worker.npu_beam_decode_graph import NPUBeamGraphModel

    descriptor = BatchDescriptor(num_tokens=4, uniform=True)

    def native_forward(wrapper: Any) -> Any:
        entries = wrapper.concrete_aclgraph_entries
        if descriptor not in entries:
            entries[descriptor] = SimpleNamespace(aclgraph=object())
        return entries[descriptor].aclgraph

    monkeypatch.setattr(acl_graph.ACLGraphWrapper, "__call__", native_forward)
    ordinary = object.__new__(acl_graph.ACLGraphWrapper)
    ordinary.concrete_aclgraph_entries = {}
    beam = object.__new__(NPUBeamGraphModel)
    beam.original_model = ordinary
    beam.concrete_aclgraph_entries = {}
    runner = SimpleNamespace(model=beam, _gr_npu_graph_model=beam)

    def forward(path: str) -> Any:
        token = BEAM_DATA_VAR.set({"beam_session": object()} if path == "beam" else None)
        try:
            return beam()
        finally:
            BEAM_DATA_VAR.reset(token)

    first_graph = forward(first_path)
    if first_path == "ordinary":
        assert not capture_mod._verify_and_record_captured_graph_npu(runner, 4)
    other_path = "beam" if first_path == "ordinary" else "ordinary"
    other_graph = forward(other_path)
    assert other_graph is not first_graph
    assert forward(first_path) is first_graph
    assert forward(other_path) is other_graph
    assert beam.concrete_aclgraph_entries is not ordinary.concrete_aclgraph_entries
    assert capture_mod._verify_and_record_captured_graph_npu(runner, 4)
    assert not beam.graph_ready(8)


def test_npu_suffix_bank_storage_count_deduplicates_packed_views() -> None:
    packed = torch.empty((2, 2, 1, 4, 3), dtype=torch.float32)
    pools = [
        SimpleNamespace(unshared_key=packed[0, layer], unshared_value=packed[1, layer])
        for layer in range(2)
    ]
    context = SimpleNamespace(pools=pools, _npu_beam_width_pools={})
    worker = SimpleNamespace(
        _gr_npu_graph_context=context, _gr_npu_graph_contexts={(1, 4): context}
    )
    assert (
        capture_mod._retained_npu_suffix_kv_bytes(worker) == packed.numel() * packed.element_size()
    )


def test_npu_capture_logs_requested_and_retained_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_graph_widths=(2,)),
    )
    worker = SimpleNamespace(_gr_npu_beam_graph_shapes={(1, 4)})
    capture = Mock(side_effect=[True, False])
    logger = Mock()
    monkeypatch.setattr(capture_mod, "_capture_beam_decode_graph_width", capture)
    monkeypatch.setattr(capture_mod, "logger", logger)
    assert capture_mod.capture_beam_decode_graph(worker)
    assert [call.args[1] for call in capture.call_args_list] == [4, 2]
    messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
    assert "widths=[4, 2] (count=2, B=1)" in messages[0]
    assert "captured (B, W)=[(1, 4)]" in messages[-1]
    assert "excludes graph and scratch memory" in messages[-1]


pytestmark = pytest.mark.cpu_test
