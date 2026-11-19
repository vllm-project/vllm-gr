# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Opt-in SID stage domains survive transport and reject unsupported admission."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Protocol, TypeVar, cast

import msgspec
import pytest

from vllm_gr.config import (
    ConstraintTableBackendEnum,
    ConstraintTableConfig,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import (
    BEAM_SID_STAGE_VOCAB_KEY,
    GR_CONFIG_KEY,
    parse_gr_config,
)
from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

_TestFunction = TypeVar("_TestFunction", bound=Callable[..., None])


class _Parametrize(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[_TestFunction], _TestFunction]: ...


_parametrize = cast(_Parametrize, pytest.mark.parametrize)


def _config(**overrides: Any) -> GRConfig:
    values: dict[str, Any] = {
        "constrained_generation": False,
        "beam_execution_mode": "v1",
        "beam_worker_decision": True,
        "beam_max_width": 128,
        "beam_max_decode_steps": 3,
        "beam_sid_stage_vocab": True,
    }
    values.update(overrides)
    return GRConfig(**values)


def _mapping(*, enabled: bool = True) -> dict[str, Any]:
    config = _config(
        beam_sid_stage_vocab=enabled,
        constraint_table=ConstraintTableConfig(
            max_top_k=128, backend=ConstraintTableBackendEnum.NPU
        ),
    )
    return cast(dict[str, Any], gr_config_to_mapping(config))


def _device_config(device: str | None, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={GR_CONFIG_KEY: _mapping(enabled=enabled)},
        device_config=SimpleNamespace(device=device),
    )


def test_sid_stage_vocab_defaults_off_without_activating_gr() -> None:
    config = GRConfig()
    assert config.beam_sid_stage_vocab is False
    assert config.enabled is False
    assert parse_gr_config(SimpleNamespace(additional_config={})).beam_sid_stage_vocab is False


@_parametrize("value", [None, 0, 1, "true", "false", "1", [], {}])
@_parametrize("input_kind", ["constructor", "canonical", "legacy"])
def test_sid_stage_vocab_rejects_non_boolean(value: Any, input_kind: str) -> None:
    with pytest.raises(TypeError, match=r"beam\.sid_stage_vocab must be a boolean"):
        if input_kind == "constructor":
            _config(beam_sid_stage_vocab=value)
        elif input_kind == "canonical":
            raw = _mapping()
            raw["beam"]["sid_stage_vocab"] = value
            gr_config_from_mapping(raw)
        else:
            host = SimpleNamespace(additional_config={})
            _config().write_to_vllm_config(host)
            host.additional_config[BEAM_SID_STAGE_VOCAB_KEY] = value
            parse_gr_config(host)


@_parametrize(
    "override",
    [
        {"constrained_generation": None},
        {"constrained_generation": True},
        {"beam_execution_mode": None},
        {"beam_worker_decision": None},
        {"beam_worker_decision": False},
        {"beam_max_decode_steps": 4},
    ],
)
def test_sid_stage_vocab_requires_explicit_supported_beam_semantics(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"beam\.sid_stage_vocab requires"):
        _config(**override)


@_parametrize("steps", [1, 2, 3])
def test_sid_stage_vocab_accepts_supported_generation_budget(steps: int) -> None:
    assert _config(beam_max_decode_steps=steps).beam_sid_stage_vocab is True


@_parametrize("enabled", [False, True])
def test_sid_stage_vocab_canonical_transport_preserves_opt_in(enabled: bool) -> None:
    raw = _mapping(enabled=enabled)
    config = gr_config_from_mapping(raw)
    assert config.beam_sid_stage_vocab is enabled
    assert gr_config_to_mapping(config) == raw
    assert ("sid_stage_vocab" in raw["beam"]) is enabled
    host = SimpleNamespace(additional_config={"unrelated": {"keep": True}})
    write_canonical_gr_config(host, config)
    transported = SimpleNamespace(
        additional_config=msgspec.msgpack.decode(msgspec.msgpack.encode(host.additional_config))
    )
    assert GRConfig.from_vllm_config(transported) == config
    assert transported.additional_config["unrelated"] == {"keep": True}


def test_explicit_default_does_not_change_existing_canonical_mapping() -> None:
    raw = _mapping(enabled=False)
    original = msgspec.msgpack.decode(msgspec.msgpack.encode(raw))
    assert "sid_stage_vocab" not in raw["beam"]
    assert gr_config_to_mapping(gr_config_from_mapping(raw)) == original
    raw["beam"]["sid_stage_vocab"] = False
    assert gr_config_to_mapping(gr_config_from_mapping(raw)) == original


def test_sid_stage_vocab_legacy_transport_and_disable_clear_stale_flag() -> None:
    host = SimpleNamespace(additional_config={"unrelated": 17})
    _config().write_to_vllm_config(host)
    assert host.additional_config[BEAM_SID_STAGE_VOCAB_KEY] is True
    assert GRConfig.from_vllm_config(host).beam_sid_stage_vocab is True
    _config(beam_sid_stage_vocab=False).write_to_vllm_config(host)
    assert BEAM_SID_STAGE_VOCAB_KEY not in host.additional_config
    assert GRConfig.from_vllm_config(host).beam_sid_stage_vocab is False
    assert host.additional_config["unrelated"] == 17


def test_sid_stage_vocab_canonical_update_invalidates_parsed_cache() -> None:
    host = SimpleNamespace(additional_config={GR_CONFIG_KEY: _mapping(enabled=False)})
    assert GRConfig.from_vllm_config(host).beam_sid_stage_vocab is False
    write_canonical_gr_config(host, gr_config_from_mapping(_mapping()))
    assert GRConfig.from_vllm_config(host).beam_sid_stage_vocab is True


@_parametrize("legacy_value", [False, True])
def test_sid_stage_vocab_rejects_mixed_canonical_and_legacy_inputs(
    legacy_value: bool,
) -> None:
    host = SimpleNamespace(
        additional_config={
            GR_CONFIG_KEY: _mapping(),
            BEAM_SID_STAGE_VOCAB_KEY: legacy_value,
        }
    )
    with pytest.raises(ValueError, match="cannot be combined with legacy"):
        parse_gr_config(host)


@_parametrize("device", ["npu", "npu:0", "npu:7"])
def test_sid_stage_vocab_admits_npu_without_importing_device_runtime(device: str) -> None:
    validate_gr_device_config(_device_config(device))


@_parametrize("device", ["cuda", "cuda:0", None])
def test_sid_stage_vocab_rejects_other_or_unknown_device(device: str | None) -> None:
    with pytest.raises(ValueError, match=r"sid_stage_vocab requires a provider"):
        validate_gr_device_config(_device_config(device))


@_parametrize("device", ["cuda", "cuda:0", "npu", "npu:0", None])
def test_full_vocabulary_default_keeps_device_admission(device: str | None) -> None:
    validate_gr_device_config(_device_config(device, enabled=False))


pytestmark = pytest.mark.cpu_test
