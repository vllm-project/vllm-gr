# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.config import GRAttentionBackend, GRConfig
from vllm_gr.config.parser import ATTENTION_BACKEND_KEY, CATALOG_PATH_KEY


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _activate_cuda_provider_for_engine_config_tests() -> Iterator[None]:
    from vllm_gr.platforms.cuda import CUDA_BACKEND
    from vllm_gr.platforms.registry import (
        _reset_backend_registry_for_tests,
        activate_backend,
        register_backend,
    )

    _reset_backend_registry_for_tests()
    register_backend(CUDA_BACKEND, source="test:cuda")
    activate_backend("cuda")
    yield
    _reset_backend_registry_for_tests()


def _vllm_config(additional_config: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(additional_config={} if additional_config is None else additional_config)


class _OpaqueAdditionalConfig:
    """Upstream-compatible non-mapping additional configuration."""

    def compute_hash(self) -> str:
        return "opaque-config"


def test_empty_vllm_config_uses_defaults() -> None:
    config = GRConfig.from_vllm_config(_vllm_config())
    assert config == GRConfig()
    assert config.enabled is False


def test_none_vllm_config_uses_defaults() -> None:
    assert GRConfig.from_vllm_config(None) == GRConfig()


def test_parses_known_keys_and_normalizes_backend() -> None:
    config = GRConfig.from_vllm_config(
        _vllm_config(
            {
                CATALOG_PATH_KEY: "/models/catalog.json",
                ATTENTION_BACKEND_KEY: "custom",
            }
        )
    )
    assert config.catalog_path == "/models/catalog.json"
    assert config.attention_backend is GRAttentionBackend.CUSTOM
    assert config.enabled is True


def test_write_preserves_unrelated_additional_config() -> None:
    vllm_config = _vllm_config({"unrelated": {"value": 1}})
    GRConfig(
        catalog_path="/models/catalog.json",
        attention_backend=GRAttentionBackend.CUSTOM,
    ).write_to_vllm_config(vllm_config)

    assert vllm_config.additional_config == {
        "unrelated": {"value": 1},
        CATALOG_PATH_KEY: "/models/catalog.json",
        ATTENTION_BACKEND_KEY: "CUSTOM",
    }


def test_write_removes_stale_gr_keys() -> None:
    vllm_config = _vllm_config(
        {
            CATALOG_PATH_KEY: "/old/catalog.json",
            ATTENTION_BACKEND_KEY: "CUSTOM",
            "unrelated": True,
        }
    )
    GRConfig().write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config == {"unrelated": True}


@pytest.mark.parametrize("value", [1, True, [], {}])  # type: ignore[misc]
def test_rejects_non_string_catalog_path(value: object) -> None:
    with pytest.raises(ValueError, match="catalog_path"):
        GRConfig.from_vllm_config(_vllm_config({CATALOG_PATH_KEY: value}))


@pytest.mark.parametrize("value", ["", "   "])  # type: ignore[misc]
def test_rejects_empty_catalog_path(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GRConfig.from_vllm_config(_vllm_config({CATALOG_PATH_KEY: value}))


@pytest.mark.parametrize("value", [1, True, [], "FLASH_ATTN"])  # type: ignore[misc]
def test_rejects_invalid_attention_backend(value: object) -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig.from_vllm_config(_vllm_config({ATTENTION_BACKEND_KEY: value}))


def test_parse_preserves_opaque_additional_config() -> None:
    additional_config = _OpaqueAdditionalConfig()
    vllm_config = _vllm_config(additional_config)

    assert GRConfig.from_vllm_config(vllm_config) == GRConfig()
    assert vllm_config.additional_config is additional_config


def test_write_rejects_opaque_additional_config_when_gr_values_are_stored() -> None:
    with pytest.raises(ValueError, match="must be a mutable mapping"):
        GRConfig(catalog_path="/models/catalog.json").write_to_vllm_config(
            _vllm_config(_OpaqueAdditionalConfig())
        )


def test_rejects_unknown_prefixed_key_but_preserves_unrelated_keys() -> None:
    with pytest.raises(ValueError, match="vllm_gr_attenton_backend"):
        GRConfig.from_vllm_config(
            _vllm_config(
                {
                    "vllm_gr_attenton_backend": "CUSTOM",
                    "unrelated": "preserved",
                }
            )
        )


def test_write_rejects_unknown_prefixed_key() -> None:
    with pytest.raises(ValueError, match="unknown vLLM-GR"):
        GRConfig().write_to_vllm_config(_vllm_config({"vllm_gr_unknown_option": True}))


def test_direct_construction_validates_and_normalizes() -> None:
    config = GRConfig(attention_backend="custom")
    assert config.attention_backend is GRAttentionBackend.CUSTOM

    with pytest.raises(ValueError, match="catalog_path"):
        GRConfig(catalog_path=123)
    with pytest.raises(ValueError, match="must not be empty"):
        GRConfig(catalog_path="   ")
    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig(attention_backend="FLASH_ATTN")


def test_serialized_wire_config_round_trip() -> None:
    vllm_config = _vllm_config({"unrelated": "preserved"})
    expected = GRConfig(
        catalog_path="/models/catalog.json",
        attention_backend=GRAttentionBackend.CUSTOM,
    )
    expected.write_to_vllm_config(vllm_config)

    json_round_trip = json.loads(json.dumps(vllm_config.additional_config))
    restored = _vllm_config(json_round_trip)

    assert GRConfig.from_vllm_config(restored) == expected
    assert restored.additional_config["unrelated"] == "preserved"


def test_engine_config_entrypoint_writes_typed_catalog_config() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config({"unrelated": True})
    engine_args = SimpleNamespace(
        catalog_path="/models/catalog.json",
        attention_backend=None,
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == GRConfig(catalog_path="/models/catalog.json")
    assert result.additional_config["unrelated"] is True


def test_engine_config_delegates_platform_configuration() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    backend = mock.Mock()
    engine_args = SimpleNamespace(
        catalog_path=None,
        attention_backend=None,
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch("vllm_gr.platforms.get_active_backend", return_value=backend),
    ):
        result = create_engine_config(engine_args)

    assert result is vllm_config
    backend.configure_engine.assert_called_once_with(vllm_config)


def test_engine_config_merges_existing_backend_with_cli_catalog() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config({ATTENTION_BACKEND_KEY: "CUSTOM"})
    engine_args = SimpleNamespace(
        catalog_path="/models/catalog.json",
        attention_backend=None,
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == GRConfig(
        catalog_path="/models/catalog.json",
        attention_backend=GRAttentionBackend.CUSTOM,
    )


def test_engine_config_merges_existing_catalog_with_cli_backend() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config({CATALOG_PATH_KEY: "/models/catalog.json"})
    engine_args = SimpleNamespace(
        catalog_path=None,
        attention_backend="CUSTOM",
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == GRConfig(
        catalog_path="/models/catalog.json",
        attention_backend=GRAttentionBackend.CUSTOM,
    )


def test_explicit_native_backend_clears_existing_gr_backend() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config({ATTENTION_BACKEND_KEY: "CUSTOM"})
    engine_args = SimpleNamespace(
        catalog_path=None,
        attention_backend="FLASH_ATTN",
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).attention_backend is None


@pytest.mark.parametrize("raw_backend", ["auto", None])  # type: ignore[misc]
def test_explicit_nested_auto_or_null_clears_existing_gr_backend(
    raw_backend: object,
) -> None:
    """Explicit nested auto and JSON null both request native auto-selection."""
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine.arg_utils_gr import (
        add_cli_args,
        create_engine_config,
        from_cli_args,
    )

    def add_upstream_attention_config(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        def normalize_attention_config(value: str) -> SimpleNamespace:
            backend = json.loads(value)["backend"]
            return SimpleNamespace(backend=None if backend in ("auto", None) else backend)

        parser.add_argument(
            "--attention-config",
            type=normalize_attention_config,
            default=SimpleNamespace(backend=None),
        )
        return parser

    parser = FlexibleArgumentParser()
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_add_cli_args",
        side_effect=add_upstream_attention_config,
    ):
        add_cli_args(parser)
    cli_args = parser.parse_args(["--attention-config", json.dumps({"backend": raw_backend})])
    normalized_engine_args = SimpleNamespace(
        attention_backend=None,
        attention_config=SimpleNamespace(backend=None),
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_from_cli_args",
        return_value=normalized_engine_args,
    ):
        engine_args = from_cli_args(SimpleNamespace, cli_args)

    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=_vllm_config({ATTENTION_BACKEND_KEY: "CUSTOM"}),
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).attention_backend is None


def test_engine_config_preserves_opaque_additional_config_without_gr_overrides() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    additional_config = _OpaqueAdditionalConfig()
    vllm_config = _vllm_config(additional_config)
    engine_args = SimpleNamespace(
        catalog_path=None,
        attention_backend=None,
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == GRConfig()
    assert result.additional_config is additional_config


def test_engine_config_rejects_invalid_raw_gr_value_without_cli_override() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        catalog_path=None,
        attention_backend=None,
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=_vllm_config({ATTENTION_BACKEND_KEY: "INVALID"}),
        ),
        pytest.raises(ValueError, match="unsupported.*attention_backend"),
    ):
        create_engine_config(engine_args)
