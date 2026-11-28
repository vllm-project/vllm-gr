# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.helpers.gr_config import test_platform
from vllm_gr.config import (
    ConstraintConfig,
    GRConfig,
)
from vllm_gr.config.execution import validate_gr_execution_config


def test_native_async_does_not_select_v1() -> None:
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    assert GRConfig.from_vllm_config(config).beam_max_width is None
    assert not GRConfig.from_vllm_config(config).enabled
    with pytest.raises(ValueError, match="gr.beam_width"):
        validate_gr_execution_config(config)


def test_v1_requires_native_async() -> None:
    config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    GRConfig(
        beam_max_width=4, beam_worker_decision=True, constraint=ConstraintConfig()
    ).write_to_vllm_config(config)
    with pytest.raises(ValueError, match="async_scheduling=True"):
        validate_gr_execution_config(config)
    config.scheduler_config.async_scheduling = True
    validate_gr_execution_config(config)
    config.device_config = SimpleNamespace(device="npu")
    validate_gr_execution_config(config)


def test_v1_npu_startup_routes_to_acl_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.npu import npu_model_runner_patch

    capture = Mock(return_value=True)
    monkeypatch.setattr(
        npu_model_runner_patch,
        "capture_beam_decode_graph",
        capture,
    )
    GRConfig(
        beam_max_width=4, beam_worker_decision=True, constraint=ConstraintConfig()
    ).write_to_vllm_config(
        worker_vllm_config := SimpleNamespace(
            additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
        )
    )
    worker = SimpleNamespace(
        device="npu",
        model_config=SimpleNamespace(enforce_eager=False),
        vllm_config=worker_vllm_config,
    )

    from vllm_gr.v1.worker.npu.npu_beam_stage_runner import capture_ascend_forward

    assert capture_ascend_forward(worker, False)
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


@pytest.mark.parametrize(  # type: ignore[misc]
    "device_type,use_v2,with_beam,backend,expected_default",
    [
        ("cuda", False, True, "inductor", False),
        ("cuda", True, True, "inductor", None),
        ("cuda", False, False, "inductor", None),
        ("cuda", False, True, "eager", None),
        ("npu", False, True, "inductor", None),
        ("cpu", False, True, "inductor", None),
    ],
)
@pytest.mark.parametrize("override", [None, True, False])  # type: ignore[misc]
def test_v1_reduction_policy_scope_overrides_and_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    use_v2: bool,
    with_beam: bool,
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
        scheduler_config=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192),
        cache_config=SimpleNamespace(block_size=16, user_specified_block_size=False),
        model_config=SimpleNamespace(is_hybrid=False),
        use_v2_model_runner=use_v2,
        compilation_config=compilation,
    )
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "get_current_gr_platform", lambda: test_platform(device_type))
    args = SimpleNamespace(
        additional_config={
            "vllm_gr": {
                "gr": {
                    "beam_width": [4] * 3 if with_beam else None,
                    "candidate_width": [4] * 3 if with_beam else None,
                }
            }
        }
    )
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
    config.additional_config = {}  # upstream returns a fresh transport on each setup
    arg_utils_gr.create_engine_config(args)
    assert compilation.compute_hash() == restored.compute_hash()


# Exercise each admission branch; non-V2 and non-Beam paths ignore V2 policy.


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
