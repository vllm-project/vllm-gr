# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gr.config import (
    ConstraintBackend,
    GRAttentionBackend,
    GRConfig,
    NPUDecodeExecution,
)
from vllm_gr.config.parser import (
    ATTENTION_BACKEND_KEY,
    BEAM_MAX_DECODE_STEPS_KEY,
    CATALOG_PATH_KEY,
    CONSTRAINT_BACKEND_KEY,
    NPU_DECODE_EXECUTION_KEY,
    PERSISTENT_REQUEST_KEY,
    PERSISTENT_RUNNER_STATE_KEY,
)


def _vllm_config(additional_config: object | None = None) -> SimpleNamespace:
    normalized = {} if additional_config is None else additional_config
    return SimpleNamespace(additional_config=normalized)


class _OpaqueAdditionalConfig:
    """Upstream-compatible non-mapping additional configuration."""

    def compute_hash(self) -> str:
        return "opaque-config"


def test_empty_vllm_config_uses_defaults() -> None:
    config = GRConfig.from_vllm_config(_vllm_config())
    assert config == GRConfig()
    assert config.enabled is False
    assert config.persistent_request is False
    assert config.persistent_runner_state is False


def test_persistent_flags_round_trip_and_default_elision() -> None:
    vllm_config = _vllm_config()
    expected = GRConfig(persistent_request=True, persistent_runner_state=True)
    expected.write_to_vllm_config(vllm_config)

    assert vllm_config.additional_config[PERSISTENT_REQUEST_KEY] is True
    assert vllm_config.additional_config[PERSISTENT_RUNNER_STATE_KEY] is True
    assert GRConfig.from_vllm_config(vllm_config) == expected

    GRConfig().write_to_vllm_config(vllm_config)
    assert PERSISTENT_REQUEST_KEY not in vllm_config.additional_config
    assert PERSISTENT_RUNNER_STATE_KEY not in vllm_config.additional_config


def test_persistent_runner_state_requires_persistent_request() -> None:
    with pytest.raises(ValueError, match="requires persistent_request"):
        GRConfig(persistent_runner_state=True)


def test_engine_config_writes_persistent_cli_flags() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    vllm_config.scheduler_config = SimpleNamespace(
        async_scheduling=True,
        scheduler_cls="vllm.v1.core.sched.async_scheduler.AsyncScheduler",
    )
    engine_args = SimpleNamespace(
        _vllm_gr_config_input=None,
        catalog_path=None,
        constraint_backend=None,
        attention_backend=None,
        attention_config=None,
        persistent_request=True,
        persistent_runner_state=True,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_cuda_platform", return_value=False),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_npu_platform", return_value=False),
    ):
        create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(vllm_config) == GRConfig(
        persistent_request=True,
        persistent_runner_state=True,
    )


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


def test_constraint_backend_round_trip_and_default_elision() -> None:
    config = GRConfig.from_vllm_config(
        _vllm_config(
            {
                CATALOG_PATH_KEY: "/models/catalog.json",
                CONSTRAINT_BACKEND_KEY: "constraint_table",
            }
        )
    )
    assert config.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
    assert config.enabled is True

    vllm_config = _vllm_config()
    config.write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config[CONSTRAINT_BACKEND_KEY] == "constraint_table"

    GRConfig().write_to_vllm_config(vllm_config)
    assert CONSTRAINT_BACKEND_KEY not in vllm_config.additional_config


@pytest.mark.parametrize("value", [1, True, [], "unknown"])  # type: ignore[misc]
def test_rejects_invalid_constraint_backend(value: object) -> None:
    with pytest.raises(ValueError, match="constraint_backend"):
        GRConfig.from_vllm_config(_vllm_config({CONSTRAINT_BACKEND_KEY: value}))


def test_constraint_table_requires_catalog_path() -> None:
    with pytest.raises(ValueError, match="requires catalog_path"):
        GRConfig(constraint_backend=ConstraintBackend.CONSTRAINT_TABLE)


def test_npu_decode_execution_defaults_to_unset() -> None:
    # Unset (None): the runtime resolves per platform (xllm on NPU with the
    # operator library, cascade otherwise).
    assert GRConfig().npu_decode_execution is None
    assert GRConfig.from_vllm_config(_vllm_config()).npu_decode_execution is None
    # Unset is not an explicit GR configuration; explicit cascade is.
    assert GRConfig().enabled is False
    assert GRConfig(npu_decode_execution=NPUDecodeExecution.CASCADE).enabled is True


def test_npu_decode_execution_parses_and_normalizes() -> None:
    config = GRConfig.from_vllm_config(_vllm_config({NPU_DECODE_EXECUTION_KEY: "XLLM"}))
    assert config.npu_decode_execution is NPUDecodeExecution.XLLM
    assert config.enabled is True


@pytest.mark.parametrize("value", [1, True, [], {}, "flash"])  # type: ignore[misc]
def test_rejects_invalid_npu_decode_execution(value: object) -> None:
    with pytest.raises(ValueError, match="npu_decode_execution"):
        GRConfig.from_vllm_config(_vllm_config({NPU_DECODE_EXECUTION_KEY: value}))


def test_npu_decode_execution_round_trip_and_default_elision() -> None:
    vllm_config = _vllm_config()
    GRConfig(npu_decode_execution=NPUDecodeExecution.XLLM).write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config[NPU_DECODE_EXECUTION_KEY] == "xllm"
    assert GRConfig.from_vllm_config(vllm_config).npu_decode_execution is NPUDecodeExecution.XLLM

    # Explicit cascade is preserved through a round trip so a later write
    # cannot silently re-enable the platform default.
    GRConfig(npu_decode_execution=NPUDecodeExecution.CASCADE).write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config[NPU_DECODE_EXECUTION_KEY] == "cascade"

    # Writing the unset default removes the key entirely.
    GRConfig().write_to_vllm_config(vllm_config)
    assert NPU_DECODE_EXECUTION_KEY not in vllm_config.additional_config
    assert GRConfig.from_vllm_config(vllm_config).npu_decode_execution is None


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


def test_engine_config_writes_constraint_backend_override() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    engine_args = SimpleNamespace(
        catalog_path="/models/catalog.json",
        constraint_backend="constraint_table",
        attention_backend=None,
        attention_config=None,
        model=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result) == GRConfig(
        catalog_path="/models/catalog.json",
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
    )


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


def test_attention_selector_config_reader_propagates_validation_errors() -> None:
    from vllm_gr.v1.ascend_compat import _get_current_gr_attention_backend

    with (
        mock.patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=_vllm_config({ATTENTION_BACKEND_KEY: "INVALID"}),
        ),
        pytest.raises(ValueError, match="unsupported.*attention_backend"),
    ):
        _get_current_gr_attention_backend()


def test_beam_max_decode_steps_defaults_to_onerec_sid_length() -> None:
    """OneRec SID decode is max_tokens=5 minus begin/end → 3 steps."""
    config = GRConfig.from_vllm_config(_vllm_config())
    assert config.beam_max_decode_steps == 3

    vllm_config = _vllm_config()
    config.write_to_vllm_config(vllm_config)
    assert BEAM_MAX_DECODE_STEPS_KEY not in vllm_config.additional_config

    overridden = GRConfig(beam_max_decode_steps=8)
    overridden.write_to_vllm_config(vllm_config)
    assert vllm_config.additional_config[BEAM_MAX_DECODE_STEPS_KEY] == 8
    assert GRConfig.from_vllm_config(vllm_config).beam_max_decode_steps == 8


def _npic_engine_args(beam_max_width: int, catalog_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        catalog_path=catalog_path,
        constraint_backend=None,
        attention_backend=None,
        attention_config=None,
        model=None,
        beam_max_width=beam_max_width,
        beam_worker_decision=None,
        no_beam_graph=False,
        _vllm_gr_config_input=None,
    )


@pytest.mark.parametrize(
    "beam_max_width",
    [1, 128, 256],
)  # type: ignore[misc]
def test_npu_constraint_table_auto_promotion_respects_capacity(
    beam_max_width: int,
) -> None:
    """An NPU beam deployment with a catalog promotes, capacity-limited at 256.

    With a constraint data source (a Triples document) available, auto-promotion
    to Constraint Tables fires on any beam_max_width within the fused
    ``rec_constrained_topk`` top_k cap of 256. Wider deployments are rejected
    by the shared configuration validator. The user's catalog is kept as the
    data source — never replaced by the tokenizer auto-build sentinel.
    """
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    catalog_path = "generated_data/video_constraint_triples.json"
    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=True,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._catalog_is_triples_document",
            return_value=True,
        ),
    ):
        result = create_engine_config(_npic_engine_args(beam_max_width, catalog_path))

    config = GRConfig.from_vllm_config(result)
    assert config.catalog_path == catalog_path
    assert config.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
    assert config.beam_worker_decision is True
    assert config.beam_engine_driven is True


def test_beam_max_width_rejects_values_above_global_limit() -> None:
    with pytest.raises(ValueError, match=r"beam_max_width must be <= 256"):
        GRConfig(beam_max_width=257)


def test_npu_beam_without_catalog_stays_unconstrained() -> None:
    """A catalog-free NPU GRLLM must not auto-promote to Constraint Tables.

    Auto-promotion needs a constraint data source.  Without a catalog the
    tokenizer auto-build sentinel path would fail fast on models whose
    s_a/s_b/s_c groups exceed ``max_valid_items`` (e.g. OneRec 8192 > 500),
    regressing catalog-free single-prompt beam search — so no catalog means
    the deployment stays unconstrained (legacy Trie default, no catalog).
    """
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=True,
        ),
    ):
        result = create_engine_config(_npic_engine_args(128))

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_backend is not ConstraintBackend.CONSTRAINT_TABLE
    assert config.catalog_path is None


def test_npu_trie_list_catalog_is_not_promoted() -> None:
    """A Trie-list catalog (JSON array) stays Trie on NPU beam deployments.

    ``_catalog_is_triples_document`` gates the auto-promotion: only a
    Constraint Tables triples document (JSON object) promotes to
    ``CONSTRAINT_TABLE`` on NPU.  A Trie frontend catalog is a JSON list, and
    forced promotion would feed a list to ``compile_constraint_table_resource``
    (which requires a dict) and fail startup — exactly the failure the serving
    catalog dead-end test hits on NPU.
    """
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=True,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._catalog_is_triples_document",
            return_value=False,
        ),
    ):
        result = create_engine_config(_npic_engine_args(128, "generated_data/video_catalog.json"))

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_backend is ConstraintBackend.TRIE
    assert config.catalog_path == "generated_data/video_catalog.json"


@pytest.mark.parametrize(
    ("constraint_backend", "expected_backend"),
    [
        (None, ConstraintBackend.CONSTRAINT_TABLE),
        (ConstraintBackend.TRIE, ConstraintBackend.TRIE),
        (ConstraintBackend.CONSTRAINT_TABLE, ConstraintBackend.CONSTRAINT_TABLE),
    ],
)  # type: ignore[misc]
def test_npu_catalog_path_without_backend_promotes_to_constraint_table(
    constraint_backend: object, expected_backend: ConstraintBackend
) -> None:
    """On NPU beam deployments a catalog path alone selects the data source.

    When the backend is not explicitly declared, the auto-promotion to
    Constraint Tables keeps the user's catalog path as the constraint data
    source (a Triples document) instead of replacing it with the tokenizer
    auto-build sentinel.  An explicit backend — including a ``trie``
    opt-out — is respected unchanged.
    """
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        catalog_path="generated_data/video_constraint_triples.json",
        constraint_backend=constraint_backend,
        attention_backend=None,
        attention_config=None,
        model=None,
        beam_max_width=128,
        beam_worker_decision=None,
        no_beam_graph=False,
        _vllm_gr_config_input=None,
    )

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=True,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._catalog_is_triples_document",
            return_value=True,
        ),
        mock.patch(
            "vllm_gr.utils.catalog.compute_catalog_slice_config",
            side_effect=AssertionError("vocab slice computation must not run"),
        ),
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_backend is expected_backend
    assert config.catalog_path == "generated_data/video_constraint_triples.json"


def test_gpu_catalog_path_without_backend_keeps_trie_default() -> None:
    """Off NPU the auto-promotion never fires: Trie default + user catalog."""
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        catalog_path="generated_data/video_catalog.json",
        constraint_backend=None,
        attention_backend=None,
        attention_config=None,
        model=None,
        beam_max_width=128,
        beam_worker_decision=None,
        no_beam_graph=False,
        _vllm_gr_config_input=None,
    )

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=False,
        ),
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_backend is ConstraintBackend.TRIE
    assert config.catalog_path == "generated_data/video_catalog.json"


def test_cuda_beam_enables_worker_decision_by_default() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = _npic_engine_args(beam_max_width=128)
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=_vllm_config(),
        ),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_npu_platform", return_value=False),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_cuda_platform", return_value=True),
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.beam_worker_decision is True
    # The unified flag auto-enables for accelerator beam with worker
    # decisions regardless of backend; actual chaining is lazy-gated by
    # data availability (trie-catalog deployments never produce worker
    # decisions, unconstrained deployments chain decode from step 1).
    assert config.beam_engine_driven is True
    assert config.constraint_backend is ConstraintBackend.TRIE


@pytest.mark.parametrize("catalog_path", ["triples.json", "AUTO"])  # type: ignore[misc]
def test_non_cuda_constraint_table_backend_skips_vocab_slice_computation(
    catalog_path: str,
) -> None:
    """NPU/unknown Constraint Tables keep the full-vocabulary logits ABI."""
    from vllm_gr.config.models import AUTO_CONSTRAINT_TABLE_PATH
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    effective_path = AUTO_CONSTRAINT_TABLE_PATH if catalog_path == "AUTO" else catalog_path
    engine_args = SimpleNamespace(
        catalog_path=effective_path,
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        attention_backend=None,
        attention_config=None,
        model="test-model",
        beam_max_width=32,
        beam_worker_decision=None,
        no_beam_graph=False,
        _vllm_gr_config_input=None,
    )

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_npu_platform",
            return_value=False,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr._is_cuda_platform",
            return_value=False,
        ),
        mock.patch(
            "vllm_gr.utils.catalog.compute_catalog_slice_config",
            side_effect=AssertionError("vocab slice computation must not run"),
        ) as slice_spy,
        mock.patch("vllm_gr.ops.sliced_logits.set_vocab_slice_config") as set_slice,
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
    assert config.catalog_path == effective_path
    assert config.vocab_offset is None
    assert config.vocab_slice_size is None
    slice_spy.assert_not_called()
    set_slice.assert_not_called()


@pytest.mark.parametrize(
    ("engine_driven", "async_scheduling", "expected_engine_driven"),
    [(None, False, True), (False, False, False), (None, True, True)],
)  # type: ignore[misc]
def test_cuda_constraint_table_backend_restores_vocab_slice_computation(
    engine_driven: bool | None,
    async_scheduling: bool,
    expected_engine_driven: bool | None,
) -> None:
    """CUDA triples use the sliced lm_head path supported by the CUDA backend."""
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        catalog_path="triples.json",
        constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        attention_backend=None,
        attention_config=None,
        model="test-model",
        trust_remote_code=False,
        beam_max_width=32,
        beam_worker_decision=None,
        beam_engine_driven=engine_driven,
        async_scheduling=async_scheduling,
        no_beam_graph=False,
        _vllm_gr_config_input=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=_vllm_config(),
        ),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_npu_platform", return_value=False),
        mock.patch("vllm_gr.engine.arg_utils_gr._is_cuda_platform", return_value=True),
        mock.patch(
            "vllm_gr.utils.catalog.compute_catalog_slice_config",
            return_value=(151669, 24576),
        ) as slice_spy,
        mock.patch("vllm_gr.ops.sliced_logits.set_vocab_slice_config") as set_slice,
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.vocab_offset == 151669
    assert config.vocab_slice_size == 24576
    assert config.beam_worker_decision is True
    assert config.beam_engine_driven is expected_engine_driven
    slice_spy.assert_called_once()
    set_slice.assert_called_once_with(151669, 24576)
