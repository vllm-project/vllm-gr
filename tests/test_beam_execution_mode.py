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


# Exercise each admission branch; non-V2 and non-Beam paths ignore V2 policy.
@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,use_v2,with_beam,mode",
    [
        ("cuda", True, True, None),
        ("cuda", True, True, "legacy"),
        ("cuda", True, True, "v1"),
        ("cuda", False, True, None),
        ("cuda", True, False, None),
        ("npu", True, True, "legacy"),
    ],
)
@pytest.mark.parametrize("encoding", ["canonical", "legacy"])  # type: ignore[misc]
def test_engine_validates_v2_beam_protocol_without_changing_defaults(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    use_v2: bool,
    with_beam: bool,
    mode: str | None,
    encoding: str,
) -> None:
    from vllm_gr.engine import arg_utils_gr

    config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(async_scheduling=True),
        use_v2_model_runner=use_v2,
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: device_type == "cuda")
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: device_type == "npu")
    gr_config = GRConfig(
        beam_max_width=4 if with_beam else None,
        constraint_table=ConstraintTableConfig(max_top_k=4) if encoding == "canonical" else None,
    )
    if encoding == "canonical":
        raw = gr_config_to_mapping(gr_config)
    else:
        gr_config.write_to_vllm_config(config)
        raw = None
    args = SimpleNamespace(_vllm_gr_config_input=raw, beam_execution_mode=mode)
    original_additional_config = config.additional_config.copy()
    if device_type == "cuda" and use_v2 and with_beam and mode != "v1":
        with pytest.raises(ValueError, match="ModelRunner V2 Beam requires.*execution_mode='v1'"):
            arg_utils_gr.create_engine_config(args)
        assert config.additional_config == original_additional_config
    else:
        assert arg_utils_gr.create_engine_config(args) is config
        assert GRConfig.from_vllm_config(config).beam_execution_mode is (
            BeamExecutionMode.V1 if mode == "v1" else BeamExecutionMode.LEGACY
        )
