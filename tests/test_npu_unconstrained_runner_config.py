# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reject the unimplemented NPU V2 adapter before starting catalog-free Beam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import (
    ConstraintTableBackendEnum,
    ConstraintTableConfig,
    GRConfig,
    gr_config_to_mapping,
    write_canonical_gr_config,
)
from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

_V1_REQUIRED = "NPU unconstrained.*ModelRunner V1.*VLLM_USE_V2_MODEL_RUNNER=0"


def _gr_config(mode: str, *, canonical: bool = False, device: str = "npu") -> GRConfig:
    if mode == "ordinary":
        return GRConfig()
    if mode == "ordinary_unconstrained":
        return GRConfig(constrained_generation=False)
    common: dict[str, Any] = {
        "beam_max_width": 4,
        "beam_max_decode_steps": 3,
        "beam_worker_decision": True,
        "beam_execution_mode": "v1",
    }
    if canonical:
        common["constraint_table"] = ConstraintTableConfig(
            max_top_k=4, backend=ConstraintTableBackendEnum(device)
        )
    if mode == "catalog":
        common["constraint_table"] = ConstraintTableConfig(
            max_top_k=4,
            enabled=True,
            backend=ConstraintTableBackendEnum.NPU,
            path=str(Path(__file__).resolve().with_name("catalog.npz")),
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
        )
        return GRConfig(
            **common,
            constrained_generation=True,
        )
    assert mode in ("generic", "sid")
    return GRConfig(**common, constrained_generation=False, beam_sid_stage_vocab=mode == "sid")


def _host(device: str, use_v2: bool) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device=device),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        use_v2_model_runner=use_v2,
    )


def _write_config(host: SimpleNamespace, gr_config: GRConfig) -> None:
    if gr_config.constraint_table is None:
        gr_config.write_to_vllm_config(host)
    else:
        write_canonical_gr_config(host, gr_config)


@pytest.mark.parametrize("mode", ["generic", "sid"])  # type: ignore[misc]
@pytest.mark.parametrize("encoding", ["canonical", "legacy"])  # type: ignore[misc]
def test_npu_v2_rejected_during_engine_config_before_runtime_hooks(
    monkeypatch: pytest.MonkeyPatch, mode: str, encoding: str
) -> None:
    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.v1.engine import beam_search_v1

    config = _host("npu", True)
    gr_config = _gr_config(mode, canonical=encoding == "canonical")
    if encoding == "legacy":
        gr_config.write_to_vllm_config(config)
    args = SimpleNamespace(
        _vllm_gr_config_input=gr_config_to_mapping(gr_config) if encoding == "canonical" else None
    )
    original_additional = config.additional_config.copy()
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: True)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: False)
    runtime_validation = Mock(side_effect=AssertionError("runtime initialization reached"))
    canonical_write = Mock(side_effect=AssertionError("worker configuration published"))
    monkeypatch.setattr(beam_search_v1, "validate_gr_execution_config", runtime_validation)
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
    _write_config(config, _gr_config(mode))
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
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_request
    from vllm_gr.v1.engine.core import _admit_gr_session

    config = _host("npu", False)
    _write_config(config, _gr_config(mode))
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
    _write_config(config, _gr_config(mode))
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
    monkeypatch: pytest.MonkeyPatch, device: str, use_v2: bool, mode: str
) -> None:
    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.v1 import ascend_compat

    config = _host(device, use_v2)
    gr_config = _gr_config(mode, canonical=True, device=device)
    if gr_config.constraint_table is None:
        # Ordinary engines keep their legacy/no-table form. Canonical Beam
        # inputs explicitly carry a disabled or enabled table configuration.
        _write_config(config, gr_config)
        args = SimpleNamespace(_vllm_gr_config_input=None)
    else:
        args = SimpleNamespace(_vllm_gr_config_input=gr_config_to_mapping(gr_config))
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *a: config)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: device == "npu")
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: device == "cuda")
    monkeypatch.setattr(ascend_compat, "apply_engine_core_config_sanitizer", Mock())

    assert arg_utils_gr.create_engine_config(args) is config
    effective = GRConfig.from_vllm_config(config)
    assert effective.constrained_generation == gr_config.constrained_generation
    assert effective.beam_sid_stage_vocab == gr_config.beam_sid_stage_vocab
    assert effective.beam_execution_mode is gr_config.beam_execution_mode
    assert config.use_v2_model_runner is use_v2


pytestmark = pytest.mark.cpu_test
