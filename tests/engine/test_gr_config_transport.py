# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Publication, transport and caching of GR worker configuration snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers.gr_config import (
    OpaqueAdditionalConfig,
    config_host,
    fixed_beam_mapping,
    stepped_table_mapping,
)
from vllm_gr.config import (
    ConstraintMode,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import COMMON_ENGINE_CONFIG, GR_CONFIG_KEY, parse_gr_config

CATALOG_PATH_KEY = "vllm_gr_catalog_path"

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_empty_vllm_config_uses_defaults() -> None:
    config = GRConfig.from_vllm_config(config_host())
    assert config == GRConfig()
    assert config.enabled is False


def test_none_vllm_config_uses_defaults() -> None:
    assert GRConfig.from_vllm_config(None) == GRConfig()


@pytest.mark.parametrize("value", [1, True, [], "FLASH_ATTN"])  # type: ignore[misc]
def test_rejects_invalid_attention_backend(value: object) -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig.from_vllm_config(
            config_host({"vllm_gr": {"gr": {}, "attention-config": {"backend": value}}})
        )


def test_parse_preserves_opaque_additional_config() -> None:
    additional_config = OpaqueAdditionalConfig()
    vllm_config = config_host(additional_config)

    assert GRConfig.from_vllm_config(vllm_config) == GRConfig()
    assert vllm_config.additional_config is additional_config


def test_write_rejects_opaque_additional_config_when_gr_values_are_stored() -> None:
    with pytest.raises(ValueError, match="must be a mutable mapping"):
        GRConfig(beam_max_width=4).write_to_vllm_config(config_host(OpaqueAdditionalConfig()))


def test_rejects_unknown_prefixed_key_but_preserves_unrelated_keys() -> None:
    with pytest.raises(ValueError, match="vllm_gr_attenton_backend"):
        GRConfig.from_vllm_config(
            config_host(
                {
                    "vllm_gr_attenton_backend": "CUSTOM",
                    "unrelated": "preserved",
                }
            )
        )


@pytest.mark.parametrize(
    "key", ["vllm_gr_unknown_option", "vllm_gr_beam_max_width", CATALOG_PATH_KEY]
)  # type: ignore[misc]
def test_write_rejects_unknown_prefixed_key(key: str) -> None:
    host = config_host({"unrelated": 1})
    original = GRConfig(beam_max_width=4)
    original.write_to_vllm_config(host)
    host.additional_config[key] = True
    snapshot = dict(host.additional_config)

    with pytest.raises(ValueError, match=key):
        GRConfig().write_to_vllm_config(host)

    assert host.additional_config == snapshot
    assert host._vllm_gr_parsed_config is original


@pytest.mark.parametrize("sizes", [[1], [1, 3, 7], list(range(1, 9))])  # type: ignore[misc]
def test_beam_batch_sizes_round_trip(sizes: list[int], tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    raw["gr"].update({"beam_batch_sizes": sizes})
    config = gr_config_from_mapping(raw)
    assert config.deployment.beam_batch_sizes == tuple(sizes)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    write_canonical_gr_config(host, config)
    assert GRConfig.from_vllm_config(host).deployment.beam_batch_sizes == tuple(sizes)


def test_graph_widths_and_beam_batch_sizes_survive_transport(tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["decode_graph_widths"] = [8, 16, 32]
    raw["gr"].update({"beam_batch_sizes": [1, 3, 7]})
    config = gr_config_from_mapping(raw)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config

    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    config.write_to_vllm_config(host)
    restored = GRConfig.from_vllm_config(
        SimpleNamespace(additional_config=json.loads(json.dumps(host.additional_config)))
    )
    assert restored.beam_graph_widths == (8, 16, 32)
    assert restored.deployment.beam_batch_sizes == (1, 3, 7)


def test_canonical_write_invalidates_cached_defaults_for_uniproc(tmp_path: Path) -> None:
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    assert GRConfig.from_vllm_config(host).beam_max_width is None
    config = gr_config_from_mapping(fixed_beam_mapping(str(tmp_path / "table.safetensors")))
    write_canonical_gr_config(host, config)
    assert GRConfig.from_vllm_config(host) == config


def test_canonical_writer_uses_primitive_mapping(tmp_path: Path) -> None:
    config = gr_config_from_mapping(
        fixed_beam_mapping(str(tmp_path / "table.safetensors"), enabled=False)
    )
    vllm_config = SimpleNamespace(additional_config={"unrelated": 1})
    stale = GRConfig.from_vllm_config(vllm_config)
    assert stale.beam_max_width is None

    vllm_config.additional_config["vllm_gr_backend_options"] = {"provider": "example"}
    write_canonical_gr_config(vllm_config, config)

    payload = vllm_config.additional_config[GR_CONFIG_KEY]
    assert json.loads(json.dumps(payload)) == payload
    assert GRConfig.from_vllm_config(vllm_config) == config
    assert vllm_config.additional_config["unrelated"] == 1
    assert vllm_config.additional_config["vllm_gr_backend_options"] == {"provider": "example"}


def test_canonical_and_legacy_additional_config_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    vllm_config = SimpleNamespace(
        additional_config={
            GR_CONFIG_KEY: fixed_beam_mapping(str(tmp_path / "table.safetensors")),
            CATALOG_PATH_KEY: "/legacy/catalog.json",
        }
    )
    with pytest.raises(ValueError, match="Removed vLLM-GR"):
        parse_gr_config(vllm_config)


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_adapter_cache_switch_round_trip(enabled: bool, tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    if enabled:
        beam["adapter_cache_enabled"] = True
    config = gr_config_from_mapping(raw)
    assert config.beam_adapter_cache_enabled is enabled
    snapshot = gr_config_to_mapping(config)
    assert snapshot["gr"]["adapter_cache_enabled"] is enabled
    assert gr_config_from_mapping(snapshot) == config
    host = SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    config = GRConfig(beam_adapter_cache_enabled=enabled)
    config.write_to_vllm_config(host)
    assert GRConfig.from_vllm_config(host).beam_adapter_cache_enabled is enabled


def test_file_snapshot_survives_driver_file_removal(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(stepped_table_mapping()))
    driver = SimpleNamespace(additional_config={"other_plugin": 7})
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        config = load_gr_config(path)
        write_canonical_gr_config(driver, config)
        path.unlink()
        assert GRConfig.from_vllm_config(driver) is config
        worker = SimpleNamespace(additional_config=json.loads(json.dumps(driver.additional_config)))
        first = GRConfig.from_vllm_config(worker)
        assert first == config
        assert GRConfig.from_vllm_config(worker) is first
        other = SimpleNamespace(additional_config={})
        write_canonical_gr_config(other, GRConfig(beam_max_width=4))
        assert GRConfig.from_vllm_config(other).beam_max_width == 4
        assert GRConfig.from_vllm_config(driver) is config
        assert sum(call.args[0] == path for call in opened.call_args_list) == 1
        assert all(call.args[0] in (path, COMMON_ENGINE_CONFIG) for call in opened.call_args_list)
    assert driver.additional_config["other_plugin"] == 7
    assert load_gr_config().constraint.mode is ConstraintMode.NONE
