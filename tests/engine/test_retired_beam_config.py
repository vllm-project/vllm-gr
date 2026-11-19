# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Retired scheduling controls are consumed once, never sent to workers."""

import json
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from vllm_gr.config import (
    ConstraintTableConfig,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
)
from vllm_gr.config.parser import BEAM_ENGINE_DRIVEN_KEY
from vllm_gr.platforms.legacy import legacy_config_service


def parse_retired(source: str, value: object) -> GRConfig:
    if source == "python":
        return GRConfig(beam_engine_driven=cast(bool | None, value))
    if source == "legacy":
        return GRConfig.from_vllm_config(
            SimpleNamespace(additional_config={BEAM_ENGINE_DRIVEN_KEY: value})
        )
    return gr_config_from_mapping(
        {
            "schema_version": 1,
            "beam": {"engine_driven": value},
            "constraint_table": {"max_top_k": 32},
        }
    )


@pytest.mark.parametrize("source", ["python", "legacy", "canonical"])  # type: ignore[misc]
@pytest.mark.parametrize("value", [True, "true"])  # type: ignore[misc]
def test_true_warns_once_and_is_not_serialized(source: str, value: object) -> None:
    with pytest.warns(FutureWarning, match="retired and ignored") as warnings:
        config = parse_retired(source, value)
    assert len(warnings) == 1
    assert config.beam_engine_driven is None
    config = replace(config, constraint_table=ConstraintTableConfig(max_top_k=32))
    beam = gr_config_to_mapping(config)["beam"]
    assert isinstance(beam, dict)
    assert "engine_driven" not in beam
    host = SimpleNamespace(additional_config={BEAM_ENGINE_DRIVEN_KEY: True})
    replace(config, constraint_table=None).write_to_vllm_config(host)
    assert BEAM_ENGINE_DRIVEN_KEY not in host.additional_config
    assert GRConfig.from_vllm_config(host).beam_engine_driven is None


@pytest.mark.parametrize("source", ["python", "legacy", "canonical"])  # type: ignore[misc]
@pytest.mark.parametrize("value", [False, "false", 0, 1, [], "invalid"])  # type: ignore[misc]
def test_false_and_invalid_inputs_fail_at_ingress(source: str, value: object) -> None:
    with pytest.raises(ValueError, match="beam_engine_driven"):
        parse_retired(source, value)


@pytest.mark.parametrize("source", ["python", "legacy", "canonical"])  # type: ignore[misc]
def test_null_does_not_enable_beam(source: str) -> None:
    config = parse_retired(source, None)
    assert config.beam_engine_driven is None
    assert not config.enabled


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_retired_cli_option_reaches_engine_validation(enabled: bool) -> None:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine import arg_utils_gr

    parser = FlexibleArgumentParser()
    with mock.patch.object(arg_utils_gr, "origin_add_cli_args", return_value=parser):
        arg_utils_gr.add_cli_args(parser)
    args = parser.parse_args(["--beam-engine-driven" if enabled else "--no-beam-engine-driven"])
    with mock.patch.object(arg_utils_gr, "origin_from_cli_args", return_value=SimpleNamespace()):
        engine_args = arg_utils_gr.from_cli_args(SimpleNamespace, args)
    config = SimpleNamespace(additional_config={})
    with mock.patch.object(arg_utils_gr, "origin_create_engine_config", return_value=config):
        if enabled:
            with pytest.warns(FutureWarning, match="retired and ignored"):
                result = arg_utils_gr.create_engine_config(engine_args)
            assert BEAM_ENGINE_DRIVEN_KEY not in result.additional_config
        else:
            with pytest.raises(ValueError, match="frontend step loop was removed"):
                arg_utils_gr.create_engine_config(engine_args)


@pytest.mark.parametrize("source", ["legacy", "canonical"])  # type: ignore[misc]
@pytest.mark.parametrize("value", [True, "true", None])  # type: ignore[misc]
def test_engine_creation_removes_raw_retired_input_without_other_changes(
    source: str, value: object
) -> None:
    from vllm_gr.engine import arg_utils_gr

    raw: dict[str, object] = {"unrelated": {"enabled": True}}
    if source == "legacy":
        raw[BEAM_ENGINE_DRIVEN_KEY] = value
    else:
        raw["vllm_gr"] = {
            "schema_version": 1,
            "beam": {"engine_driven": value},
            "constraint_table": {"max_top_k": 32},
        }
    host = SimpleNamespace(additional_config=raw)
    with (
        mock.patch.object(arg_utils_gr, "origin_create_engine_config", return_value=host),
        mock.patch.object(
            arg_utils_gr, "engine_config_service", return_value=legacy_config_service(None)
        ),
        warnings.catch_warnings(record=True) as driver_warnings,
    ):
        warnings.simplefilter("always")
        result = arg_utils_gr.create_engine_config(SimpleNamespace())
    assert result is host
    assert len(driver_warnings) == (0 if value is None else 1)
    if driver_warnings:
        assert issubclass(driver_warnings[0].category, FutureWarning)
        assert "retired and ignored" in str(driver_warnings[0].message)
    assert result.additional_config["unrelated"] == {"enabled": True}
    if source == "legacy":
        assert BEAM_ENGINE_DRIVEN_KEY not in result.additional_config
    else:
        assert "engine_driven" not in result.additional_config["vllm_gr"]["beam"]

    # Worker deserialization must not see the raw field or warn a second time.
    with warnings.catch_warnings(record=True) as worker_warnings:
        warnings.simplefilter("always")
        worker_config = GRConfig.from_vllm_config(
            SimpleNamespace(additional_config=json.loads(json.dumps(result.additional_config)))
        )
    assert not worker_warnings
    assert worker_config.beam_engine_driven is None
    assert not worker_config.enabled


pytestmark = pytest.mark.cpu_test
