# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Configuration declaration checks that do not load the inference runtime."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from types import SimpleNamespace

import pytest

from vllm_gr.config import (
    ConstraintBackend,
    ConstraintTableConfig,
    GRConfig,
    gr_config_from_mapping,
    validation,
)
from vllm_gr.config.schema import ConfigFieldSpec, config_default, config_fields

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("source", ["python", "legacy", "canonical"])  # type: ignore[misc]
def test_field_validation_runs_once_at_each_ingress(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    calls: list[object] = []
    original: Callable[[object], int | None] = validation.validate_beam_max_width

    def record(value: object) -> int | None:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(validation, "validate_beam_max_width", record)
    if source == "python":
        config = GRConfig(beam_max_width=32)
    elif source == "legacy":
        config = GRConfig.from_vllm_config(
            SimpleNamespace(additional_config={"vllm_gr_beam_max_width": 32})
        )
    else:
        config = gr_config_from_mapping(
            {"schema_version": 1, "beam": {"max_width": 32}, "constraint_table": {"max_top_k": 32}}
        )
    assert config.beam_max_width == 32
    assert calls == [32]


def test_existing_positional_constructor_and_frozen_slots_contract() -> None:
    config = GRConfig("catalog.json", ConstraintBackend.TRIE, None, None, None, False, 16, 2)
    assert config.catalog_path == "catalog.json"
    assert config.beam_max_width == 16
    assert config.beam_max_decode_steps == 2
    assert not config.beam_graph_active
    with pytest.raises(FrozenInstanceError):
        setattr(config, "beam_max_width", 32)
    table = ConstraintTableConfig(32)
    assert table.max_top_k == 32
    assert not hasattr(table, "__dict__")


@pytest.mark.parametrize(  # type: ignore[misc]
    ("first", "second", "message"),
    [
        (
            ConfigFieldSpec("validate_catalog_path", legacy_key="duplicate"),
            ConfigFieldSpec("validate_catalog_path", legacy_key="duplicate"),
            "duplicate legacy",
        ),
        (
            ConfigFieldSpec("validate_catalog_path", canonical_path=("beam", "path")),
            ConfigFieldSpec("validate_catalog_path", canonical_path=("beam", "path")),
            "overlapping canonical",
        ),
        (
            ConfigFieldSpec("validate_catalog_path", canonical_path=("beam",)),
            ConfigFieldSpec("validate_catalog_path", canonical_path=("beam", "path")),
            "overlapping canonical",
        ),
    ],
)
def test_conflicting_declarations_fail_before_parsing(
    first: ConfigFieldSpec, second: ConfigFieldSpec, message: str
) -> None:
    @dataclass
    class InvalidConfig:
        one: str | None = field(default=None, metadata={"gr": first})
        two: str | None = field(default=None, metadata={"gr": second})

    with pytest.raises(ValueError, match=message):
        config_fields(InvalidConfig)


def test_unknown_validator_is_not_silently_ignored() -> None:
    @dataclass
    class InvalidConfig:
        value: int = field(default=1, metadata={"gr": ConfigFieldSpec("unknown_validator")})

    with pytest.raises(ValueError, match="unknown config validator"):
        validation.validate_config_fields(InvalidConfig())


def test_default_factory_values_are_not_shared() -> None:
    @dataclass
    class FactoryConfig:
        sizes: list[int] = field(
            default_factory=list,
            metadata={"gr": ConfigFieldSpec("validate_beam_online_batch_sizes")},
        )

    first = config_default(FactoryConfig, "sizes")
    second = config_default(FactoryConfig, "sizes")
    assert first == second == []
    assert first is not second


def test_config_import_keeps_device_runtime_unloaded() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from vllm_gr.config import GRConfig; GRConfig(); "
            "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
        ],
        check=True,
    )
