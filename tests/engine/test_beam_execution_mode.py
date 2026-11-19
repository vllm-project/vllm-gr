# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
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
from vllm_gr.v1.engine.beam_search_v1 import validate_gr_execution_config


@pytest.mark.parametrize("mode", [None, BeamExecutionMode.V1])  # type: ignore[misc]
def test_execution_mode_survives_both_config_encodings(mode: BeamExecutionMode | None) -> None:
    config = GRConfig(beam_execution_mode=mode)
    engine = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(engine)
    assert GRConfig.from_vllm_config(engine).beam_execution_mode is mode
    canonical = GRConfig(
        beam_execution_mode=mode, constraint_table=ConstraintTableConfig(max_top_k=4)
    )
    assert gr_config_from_mapping(gr_config_to_mapping(canonical)) == canonical


@pytest.mark.parametrize("invalid", ["legacy", True, "async", "auto", "", 1])  # type: ignore[misc]
def test_execution_mode_rejects_ambiguous_values(invalid: Any) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        GRConfig(beam_execution_mode=invalid)


def test_native_async_does_not_select_v1() -> None:
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    assert GRConfig.from_vllm_config(config).beam_execution_mode is None
    assert not GRConfig.from_vllm_config(config).enabled
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
    config.device_config = SimpleNamespace(device="npu")
    validate_gr_execution_config(config)


def test_v1_npu_startup_routes_to_acl_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gr_startup
    from vllm_gr.v1.worker.npu import npu_model_runner_patch

    capture = Mock(return_value=True)
    monkeypatch.setattr(
        npu_model_runner_patch,
        "capture_beam_decode_graph",
        capture,
    )
    worker = SimpleNamespace(
        device="npu",
        model_config=SimpleNamespace(enforce_eager=False),
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr_beam_execution_mode": "v1"},
            scheduler_config=SimpleNamespace(async_scheduling=True),
        ),
        _vllm_gr_config=SimpleNamespace(
            beam_execution_mode=BeamExecutionMode.V1,
            beam_graph_enabled=True,
            beam_max_width=4,
            beam_max_decode_steps=3,
            beam_worker_decision=True,
            constraint_table=SimpleNamespace(enabled=True),
        ),
    )

    from vllm_gr.platforms.ascend.stage import create_stage

    worker._gr_unplanned_stage = create_stage(plan=None, runner=worker)
    assert gr_startup.capture_async_forward(worker)
    capture.assert_called_once_with(worker)


def test_wrong_mode_rejected_before_core_preprocessing() -> None:
    from tests.helpers.beam_requests import make_batch_request
    from vllm_gr.v1.engine.core import _handle_submit_gr_session

    preprocess = Mock()
    error = Mock()
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True)),
        preprocess_add_request=preprocess,
        _handle_request_preproc_error=error,
    )
    request = make_batch_request(3)
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


@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,use_v2,mode,backend,expected_default",
    [
        ("cuda", False, "v1", "inductor", False),
        ("cuda", True, "v1", "inductor", None),
        ("cuda", False, None, "inductor", None),
        ("cuda", False, "v1", "eager", None),
        ("npu", False, "v1", "inductor", None),
        ("cpu", False, "v1", "inductor", None),
    ],
)
@pytest.mark.parametrize("override", [None, True, False])  # type: ignore[misc]
def test_v1_reduction_policy_scope_overrides_and_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    use_v2: bool,
    mode: str | None,
    backend: str,
    expected_default: bool | None,
    override: bool | None,
) -> None:
    from vllm.config import CompilationConfig

    from vllm_gr.engine import arg_utils_gr

    settings = {"combo_kernels": True}
    if override is not None:
        settings["dynamic_scale_rblock"] = override
    compilation = CompilationConfig(backend=backend, inductor_compile_config=settings)
    previous = dict(compilation.inductor_compile_config)
    original_hash = compilation.compute_hash()
    config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(async_scheduling=True),
        use_v2_model_runner=use_v2,
        compilation_config=compilation,
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: device_type == "cuda")
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: device_type == "npu")
    args = SimpleNamespace(beam_execution_mode=mode)
    assert arg_utils_gr.create_engine_config(args) is config
    expected = override if override is not None else expected_default
    assert compilation.inductor_compile_config == (
        previous if expected is None else previous | {"dynamic_scale_rblock": expected}
    )
    changed = override is None and expected_default is not None
    assert (compilation.compute_hash() != original_hash) is changed
    # Public JSON compilation settings and repeated setup retain the cache key.
    restored = CompilationConfig(
        backend=backend,
        inductor_compile_config=json.loads(json.dumps(compilation.inductor_compile_config)),
    )
    assert restored.compute_hash() == compilation.compute_hash()
    arg_utils_gr.create_engine_config(args)
    assert compilation.compute_hash() == restored.compute_hash()


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
    if mode == "legacy":
        with pytest.raises(ValueError, match="removed.*execution_mode='v1'"):
            arg_utils_gr.create_engine_config(args)
        assert config.additional_config == original_additional_config
    elif device_type == "cuda" and use_v2 and with_beam and mode != "v1":
        with pytest.raises(ValueError, match="ModelRunner V2 Beam requires.*execution_mode='v1'"):
            arg_utils_gr.create_engine_config(args)
        assert config.additional_config == original_additional_config
    else:
        assert arg_utils_gr.create_engine_config(args) is config
        assert GRConfig.from_vllm_config(config).beam_execution_mode is (
            BeamExecutionMode.V1 if mode == "v1" else None
        )


@pytest.mark.parametrize("canonical", [False, True])  # type: ignore[misc]
def test_legacy_mode_rejected_during_config_read(canonical: bool) -> None:
    if canonical:
        raw = gr_config_to_mapping(GRConfig(constraint_table=ConstraintTableConfig(max_top_k=4)))
        beam = raw["beam"]
        assert isinstance(beam, dict)
        beam["execution_mode"] = "legacy"
        additional = {"vllm_gr": raw}
    else:
        additional = {"vllm_gr_beam_execution_mode": "legacy"}
    with pytest.raises(ValueError, match="removed.*execution_mode='v1'"):
        GRConfig.from_vllm_config(SimpleNamespace(additional_config=additional))


pytestmark = pytest.mark.cpu_test
