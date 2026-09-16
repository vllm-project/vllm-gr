# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import (
    BeamExecutionMode,
    ConstraintTableConfig,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
)
from vllm_gr.config.execution import require_beam_execution_mode
from vllm_gr.v1.engine.beam_search_v1 import validate_gr_execution_config


@pytest.mark.parametrize("mode", list(BeamExecutionMode))  # type: ignore[misc]
def test_execution_mode_survives_both_config_encodings(mode: BeamExecutionMode) -> None:
    config = GRConfig(beam_execution_mode=mode)
    engine = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(engine)
    assert GRConfig.from_vllm_config(engine).beam_execution_mode is mode
    canonical = GRConfig(
        beam_execution_mode=mode, constraint_table=ConstraintTableConfig(max_top_k=4)
    )
    assert gr_config_from_mapping(gr_config_to_mapping(canonical)) == canonical


@pytest.mark.parametrize("invalid", [None, True, "async", "auto", "", 1])  # type: ignore[misc]
def test_execution_mode_rejects_ambiguous_values(invalid: Any) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        GRConfig(beam_execution_mode=invalid)


def test_native_async_does_not_select_v1() -> None:
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    require_beam_execution_mode(config, BeamExecutionMode.LEGACY)
    with pytest.raises(ValueError, match="execution_mode='v1'"):
        validate_gr_execution_config(config)


def test_v1_requires_native_async() -> None:
    config = SimpleNamespace(
        additional_config={"vllm_gr_beam_execution_mode": "v1"},
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    with pytest.raises(ValueError, match="async_scheduling=True"):
        validate_gr_execution_config(config)
    config.scheduler_config.async_scheduling = True
    validate_gr_execution_config(config)
    with pytest.raises(ValueError, match="execution_mode='legacy'"):
        require_beam_execution_mode(config, BeamExecutionMode.LEGACY)


def test_legacy_async_startup_does_not_allocate_v1_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker import gpu_beam_stage_runner, gr_startup

    allocate = Mock(side_effect=AssertionError("legacy startup allocated V1 state"))
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", allocate)
    worker = SimpleNamespace(
        device="cuda",
        use_async_scheduling=True,
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(
            beam_execution_mode=BeamExecutionMode.LEGACY,
            beam_graph_enabled=True,
            beam_max_width=4,
            beam_worker_decision=True,
            constraint_table=SimpleNamespace(enabled=True),
        ),
    )
    assert not gr_startup.capture_async_forward(worker)
    allocate.assert_not_called()


def test_wrong_mode_rejected_before_core_preprocessing() -> None:
    from tests.test_gr_async_scheduler import _batch_request
    from vllm_gr.v1.engine.core import _handle_submit_gr_session

    preprocess = Mock()
    error = Mock()
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True)),
        preprocess_add_request=preprocess,
        _handle_request_preproc_error=error,
    )
    request = _batch_request(3)
    _handle_submit_gr_session(core, request)
    preprocess.assert_not_called()
    error.assert_called_once_with(request.items[0].request)


def test_mode_override_with_canonical_engine_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.engine import arg_utils_gr

    config = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    raw = gr_config_to_mapping(GRConfig(constraint_table=ConstraintTableConfig(max_top_k=4)))
    args = SimpleNamespace(_vllm_gr_config_input=raw, beam_execution_mode="v1")
    assert arg_utils_gr.create_engine_config(args) is config
    assert GRConfig.from_vllm_config(config).beam_execution_mode is BeamExecutionMode.V1


def test_mode_override_rejects_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.engine import arg_utils_gr

    monkeypatch.setattr(
        arg_utils_gr,
        "origin_create_engine_config",
        lambda *a: SimpleNamespace(additional_config={}),
    )
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    with pytest.raises(ValueError, match="execution_mode"):
        arg_utils_gr.create_engine_config(SimpleNamespace(beam_execution_mode=""))
