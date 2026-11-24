# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reject the unimplemented NPU V2 adapter before starting catalog-free Beam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.helpers.gr_config import engine_args_from_file
from vllm_gr.config import (
    ConstraintBackendKind,
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    GRConfig,
    gr_config_to_mapping,
)
from vllm_gr.platforms.legacy import legacy_config_service
from vllm_gr.v1.engine.beam_admission import validate_gr_device_config

_V1_REQUIRED = "NPU unconstrained.*ModelRunner V1.*VLLM_USE_V2_MODEL_RUNNER=0"


def _gr_config(mode: str) -> GRConfig:
    if mode == "ordinary":
        return GRConfig()
    if mode == "ordinary_unconstrained":
        return GRConfig(constraint=ConstraintConfig())
    common: dict[str, Any] = {
        "beam_max_width": 4,
        "beam_max_decode_steps": 3,
        "beam_worker_decision": True,
    }
    if mode == "catalog":
        table = ConstraintTableSpec(
            max_top_k=4,
            enabled=True,
            backend=ConstraintBackendKind.NPU,
            path=str(Path(__file__).resolve().with_name("catalog.npz")),
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
        )
        return GRConfig(
            **common,
            constraint=ConstraintConfig(ConstraintMode.TABLE, table),
        )
    assert mode in ("generic", "sid")
    return GRConfig(**common, constraint=ConstraintConfig(), beam_sid_stage_vocab=mode == "sid")


def _host(device: str, use_v2: bool) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device=device),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        use_v2_model_runner=use_v2,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_npu_admission_rejects_v2_unconstrained_candidate_width_mismatch() -> None:
    config = _host("npu", False)
    gr_config = GRConfig(
        beam_max_width=4,
        beam_max_decode_steps=3,
        beam_worker_decision=True,
        constraint=ConstraintConfig(),
        beam_candidate_width=8,
    )
    gr_config.write_to_vllm_config(config)

    with pytest.raises(ValueError, match="candidate_width.*beam width"):
        validate_gr_device_config(config)


@pytest.mark.parametrize("mode", ["generic", "sid"])  # type: ignore[misc]
@pytest.mark.parametrize("encoding", ["file", "python"])  # type: ignore[misc]
def test_npu_v2_rejected_during_engine_config_before_runtime_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, encoding: str
) -> None:
    from vllm_gr.config import execution
    from vllm_gr.engine import arg_utils_gr

    config = _host("npu", True)
    options = {
        "beam_width": [4] * 3,
        "constraint_mode": "none",
        "sid_stage_vocab": mode == "sid",
        "worker_decision": True,
        "candidate_width": [4] * 3,
    }
    args = (
        engine_args_from_file(tmp_path, {"gr": options})
        if encoding == "file"
        else SimpleNamespace(gr=options)
    )
    original_additional = config.additional_config.copy()
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "engine_config_service", lambda: legacy_config_service("npu"))
    runtime_validation = Mock(side_effect=AssertionError("runtime initialization reached"))
    canonical_write = Mock(side_effect=AssertionError("worker configuration published"))
    monkeypatch.setattr(execution, "validate_gr_execution_config", runtime_validation)
    monkeypatch.setattr(arg_utils_gr, "write_canonical_gr_config", canonical_write)

    with pytest.raises(ValueError, match=_V1_REQUIRED):
        arg_utils_gr.create_engine_config(args)

    # EngineArgs only constructs configuration. The failure precedes returning
    # it to native engine/model allocation and publishing it to spawned workers.
    assert config.additional_config == original_additional
    runtime_validation.assert_not_called()
    canonical_write.assert_not_called()


def _request(
    *, candidate_width: int | None = None, dynamic_beam_widths: tuple[int, ...] | None = None
) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    from vllm_gr.v1.beam.batch_contracts import (
        BEAM_BATCH_PROTOCOL_VERSION,
        BeamBatchItem,
        BeamBatchParams,
        BeamBatchRequest,
        BeamOutputOptions,
    )

    native = EngineCoreRequest(
        request_id="npu-v2-rejected",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=native.request_id,
        items=(BeamBatchItem(0, native),),
        beam_params=BeamBatchParams(
            4,
            3,
            0.0,
            (),
            True,
            candidate_width=candidate_width,
            dynamic_beam_widths=dynamic_beam_widths,
        ),
        output_options=BeamOutputOptions(4, True, True, False),
    )


@pytest.mark.parametrize("mode", ["generic", "sid"])  # type: ignore[misc]
def test_npu_v2_request_rejected_before_native_preprocessing(mode: str) -> None:
    from vllm_gr.v1.engine.core import _admit_gr_session

    config = _host("npu", True)
    _gr_config(mode).write_to_vllm_config(config)
    request = _request()
    with pytest.raises(ValueError, match=_V1_REQUIRED):
        validate_gr_device_config(config, request)
    core = SimpleNamespace(
        vllm_config=config,
        batch_queue=object(),
        batch_queue_size=2,
        preprocess_add_request=Mock(),
        add_request=Mock(),
    )
    with pytest.raises(ValueError, match=_V1_REQUIRED):
        _admit_gr_session(core, request)
    core.preprocess_add_request.assert_not_called()
    core.add_request.assert_not_called()


@pytest.mark.parametrize("mode", ["generic", "sid"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "candidate_width,dynamic_beam_widths", [(2, None), (4, (2, 4, 4))]
)
def test_native_npu_admission_rejects_candidate_width_before_preprocessing(
    mode: str, candidate_width: int, dynamic_beam_widths: tuple[int, ...] | None
) -> None:
    from vllm_gr.v1.engine.beam_admission import validate_gr_request
    from vllm_gr.v1.engine.core import _admit_gr_session

    config = _host("npu", False)
    _gr_config(mode).write_to_vllm_config(config)
    request = _request(candidate_width=candidate_width, dynamic_beam_widths=dynamic_beam_widths)
    # These are structurally valid wire requests. The native admission path
    # must also apply the NPU-specific K/W guard before allocating requests.
    validate_gr_request(request, config=config)
    core = SimpleNamespace(
        vllm_config=config,
        batch_queue=object(),
        batch_queue_size=2,
        preprocess_add_request=Mock(),
        add_request=Mock(),
    )

    with pytest.raises(ValueError, match="candidate_width.*beam width"):
        _admit_gr_session(core, request)

    core.preprocess_add_request.assert_not_called()
    core.add_request.assert_not_called()


@pytest.mark.parametrize(  # type: ignore[misc]
    "device,use_v2,mode",
    [
        ("npu", False, "generic"),
        ("npu", False, "sid"),
        ("cuda", False, "generic"),
        ("cuda", True, "generic"),
        ("npu", True, "catalog"),
        ("npu", True, "ordinary"),
    ],
)
def test_device_admission_preserves_supported_and_existing_paths(
    device: str, use_v2: bool, mode: str
) -> None:
    config = _host(device, use_v2)
    _gr_config(mode).write_to_vllm_config(config)
    validate_gr_device_config(config)


@pytest.mark.parametrize(  # type: ignore[misc]
    "device,use_v2,mode",
    [
        ("npu", False, "generic"),
        ("npu", False, "sid"),
        ("cuda", False, "generic"),
        ("cuda", True, "generic"),
        ("npu", True, "catalog"),
        ("npu", True, "ordinary"),
        ("npu", True, "ordinary_unconstrained"),
    ],
)
def test_engine_config_preserves_supported_and_existing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, use_v2: bool, mode: str
) -> None:
    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.platforms.npu import config_patch

    config = _host(device, use_v2)
    gr_config = _gr_config(mode)
    args = engine_args_from_file(tmp_path, gr_config_to_mapping(gr_config))
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(
        arg_utils_gr, "engine_config_service", lambda: legacy_config_service(device)
    )
    monkeypatch.setattr(config_patch, "apply_engine_core_config_sanitizer", Mock())

    assert arg_utils_gr.create_engine_config(args) is config
    effective = GRConfig.from_vllm_config(config)
    assert effective.constraint_config.mode is gr_config.constraint_config.mode
    assert effective.beam_sid_stage_vocab == gr_config.beam_sid_stage_vocab
    assert config.use_v2_model_runner is use_v2


pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _isolate_backend_plan_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configuration policy independently of a native backend install."""
    from vllm_gr.platforms import planning

    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
