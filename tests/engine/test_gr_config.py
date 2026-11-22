# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.helpers.gr_config import engine_args_from_file, table_constraint
from vllm_gr.config import (
    MAX_BEAM_WIDTH,
    ConstraintMode,
    GRAttentionBackend,
    GRConfig,
    NPUDecodeExecution,
    gr_config_to_mapping,
    resolve_candidate_width,
)
from vllm_gr.platforms.legacy import legacy_config_service

ATTENTION_BACKEND_KEY = "vllm_gr_attention_backend"
CATALOG_PATH_KEY = "vllm_gr_catalog_path"


def _vllm_config(additional_config: object | None = None) -> SimpleNamespace:
    normalized = {} if additional_config is None else additional_config
    return SimpleNamespace(
        additional_config=normalized, scheduler_config=SimpleNamespace(async_scheduling=True)
    )


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


def test_npu_decode_execution_defaults_to_unset() -> None:
    # Unset (None): the runtime resolves per platform (xllm on NPU with the
    # operator library, cascade otherwise).
    assert GRConfig().npu_decode_execution is None
    assert GRConfig.from_vllm_config(_vllm_config()).npu_decode_execution is None
    # Unset is not an explicit GR configuration; explicit cascade is.
    assert GRConfig().enabled is False
    assert GRConfig(npu_decode_execution=NPUDecodeExecution.CASCADE).enabled is True


@pytest.mark.parametrize("value", [1, True, [], {}, "flash"])  # type: ignore[misc]
def test_rejects_invalid_npu_decode_execution(value: object) -> None:
    with pytest.raises(ValueError, match="npu_decode_execution"):
        GRConfig.from_vllm_config(
            _vllm_config({"vllm_gr": {"gr": {"npu_decode_execution": value}}})
        )


@pytest.mark.parametrize("value", [1, True, [], "FLASH_ATTN"])  # type: ignore[misc]
def test_rejects_invalid_attention_backend(value: object) -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig.from_vllm_config(
            _vllm_config({"vllm_gr": {"gr": {}, "attention-config": {"backend": value}}})
        )


def test_parse_preserves_opaque_additional_config() -> None:
    additional_config = _OpaqueAdditionalConfig()
    vllm_config = _vllm_config(additional_config)

    assert GRConfig.from_vllm_config(vllm_config) == GRConfig()
    assert vllm_config.additional_config is additional_config


def test_write_rejects_opaque_additional_config_when_gr_values_are_stored() -> None:
    with pytest.raises(ValueError, match="must be a mutable mapping"):
        GRConfig(beam_max_width=4).write_to_vllm_config(_vllm_config(_OpaqueAdditionalConfig()))


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


@pytest.mark.parametrize(
    "key", ["vllm_gr_unknown_option", "vllm_gr_beam_max_width", CATALOG_PATH_KEY]
)  # type: ignore[misc]
def test_write_rejects_unknown_prefixed_key(key: str) -> None:
    host = _vllm_config({"unrelated": 1})
    original = GRConfig(beam_max_width=4)
    original.write_to_vllm_config(host)
    host.additional_config[key] = True
    snapshot = dict(host.additional_config)

    with pytest.raises(ValueError, match=key):
        GRConfig().write_to_vllm_config(host)

    assert host.additional_config == snapshot
    assert host._vllm_gr_parsed_config is original


def test_direct_construction_validates_and_normalizes() -> None:
    config = GRConfig(attention_backend="custom")
    assert config.attention_backend is GRAttentionBackend.CUSTOM

    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig(attention_backend="FLASH_ATTN")


@pytest.mark.l1  # type: ignore[misc]
def test_engine_config_entrypoint_preserves_unrelated_config(tmp_path: Path) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config({"unrelated": True})
    engine_args = engine_args_from_file(
        tmp_path, {"gr": {"beam_width": None, "candidate_width": None}}
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).beam_max_width is None
    assert result.additional_config["unrelated"] is True


def test_cli_parser_uses_vllm_gr_engine_defaults() -> None:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine.arg_utils_gr import add_cli_args

    def add_upstream_engine_args(
        parser: FlexibleArgumentParser,
    ) -> FlexibleArgumentParser:
        for name in ("max_model_len", "max_num_batched_tokens"):
            parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=None)
        for name in (
            "enable_chunked_prefill",
            "enable_prefix_caching",
        ):
            parser.add_argument(f"--{name.replace('_', '-')}", action="store_true", default=None)
        return parser

    parser = FlexibleArgumentParser()
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_add_cli_args",
        side_effect=add_upstream_engine_args,
    ):
        add_cli_args(parser)

    defaults = parser.parse_args([])
    # GR defaults are resolved at EngineArgs construction after mode selection.
    assert defaults.max_model_len is None
    assert defaults.max_num_batched_tokens is None
    assert defaults.enable_chunked_prefill is None
    assert defaults.enable_prefix_caching is None

    overridden = parser.parse_args(
        [
            "--max-model-len",
            "4096",
            "--max-num-batched-tokens",
            "4096",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
        ]
    )
    assert overridden.max_model_len == 4096
    assert overridden.max_num_batched_tokens == 4096
    assert overridden.enable_chunked_prefill is True
    assert overridden.enable_prefix_caching is True


def test_engine_config_preserves_opaque_additional_config_without_gr_overrides() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    additional_config = _OpaqueAdditionalConfig()
    vllm_config = _vllm_config(additional_config)
    engine_args = SimpleNamespace(
        attention_backend=None,
        attention_config=None,
    )
    with mock.patch(
        "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
        return_value=vllm_config,
    ):
        result = create_engine_config(engine_args)

    assert GRConfig.from_vllm_config(result).beam_max_width is None
    assert result.additional_config is additional_config


def test_engine_config_rejects_invalid_raw_gr_value_without_cli_override() -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = SimpleNamespace(
        attention_backend=None,
        attention_config=None,
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=_vllm_config({ATTENTION_BACKEND_KEY: "INVALID"}),
        ),
        pytest.raises(ValueError, match="Supply GR settings"),
    ):
        create_engine_config(engine_args)


def test_attention_selector_config_reader_propagates_validation_errors() -> None:
    from vllm_gr.v1.attention.selector_patch import _get_current_gr_attention_backend

    with (
        mock.patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=_vllm_config(
                {"vllm_gr": {"attention-config": {"backend": "INVALID"}, "gr": {}}}
            ),
        ),
        pytest.raises(ValueError, match="unsupported.*attention_backend"),
    ):
        _get_current_gr_attention_backend()


def test_beam_max_width_rejects_values_above_global_limit() -> None:
    with pytest.raises(ValueError, match=rf"beam_max_width must be <= {MAX_BEAM_WIDTH}"):
        GRConfig(beam_max_width=MAX_BEAM_WIDTH + 1)


def test_candidate_width_follows_the_beam_width_until_it_cannot() -> None:
    """An omitted candidate width follows the configured beam width."""
    config = GRConfig(beam_max_width=128, constraint=table_constraint())
    assert resolve_candidate_width(config) == 128
    assert resolve_candidate_width(replace(config, beam_candidate_width=128)) == 128


def test_unconstrained_candidate_width_follows_wide_beam() -> None:
    for width in (257, 512, MAX_BEAM_WIDTH):
        assert resolve_candidate_width(GRConfig(beam_max_width=width)) == width


def test_wide_beam_is_servable_once_the_candidate_width_is_stated() -> None:
    config = GRConfig(
        beam_max_width=MAX_BEAM_WIDTH,
        beam_candidate_width=256,
        constraint=table_constraint(max_top_k=256),
    )
    assert resolve_candidate_width(config) == 256


@pytest.mark.parametrize("candidate_width", [64, 256])  # type: ignore[misc]
def test_explicit_candidate_width_may_differ_from_the_beam_width(candidate_width: int) -> None:
    config = GRConfig(beam_max_width=128, beam_candidate_width=candidate_width)
    assert resolve_candidate_width(config) == candidate_width


@pytest.mark.parametrize("widths", [tuple(range(1, 8)), (*range(1, 8), 128)])  # type: ignore[misc]
def test_beam_graph_capture_limit_counts_implicit_capacity_once(
    widths: tuple[int, ...],
) -> None:
    config = GRConfig(beam_max_width=128, beam_graph_widths=widths)
    assert config.beam_graph_active
    assert config.beam_graph_widths == widths


@pytest.mark.parametrize("widths", [tuple(range(1, 9)), (*range(1, 9), 128)])  # type: ignore[misc]
def test_beam_graph_capture_limit_rejects_more_than_eight_effective_widths(
    widths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="at most 8 captured widths including max_width"):
        GRConfig(beam_max_width=128, beam_graph_widths=widths)


def test_beam_graph_capture_limit_preserves_default_and_disabled_graphs() -> None:
    assert GRConfig().beam_graph_widths is None
    default = GRConfig(beam_max_width=128)
    assert default.beam_graph_widths is None
    assert default.beam_graph_active
    disabled = GRConfig(
        beam_graph_enabled=False,
        beam_max_width=128,
        beam_graph_widths=tuple(range(1, 129)),
    )
    assert not disabled.beam_graph_active
    assert disabled.beam_graph_widths == tuple(range(1, 129))


def test_npu_beam_without_catalog_stays_unconstrained(tmp_path: Path) -> None:
    """A catalog-free NPU deployment uses full-vocabulary generation."""
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    vllm_config = _vllm_config()
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=vllm_config,
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service("npu"),
        ),
    ):
        result = create_engine_config(
            engine_args_from_file(
                tmp_path, {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            )
        )

    config = GRConfig.from_vllm_config(result)
    assert config.constraint_config.mode is ConstraintMode.NONE
    assert config.constraint_config.table is None


def test_cuda_beam_enables_worker_decision_by_default(tmp_path: Path) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    engine_args = engine_args_from_file(
        tmp_path, {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
    )
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config",
            return_value=_vllm_config(),
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service("cuda"),
        ),
    ):
        result = create_engine_config(engine_args)

    config = GRConfig.from_vllm_config(result)
    assert config.beam_worker_decision is True


@pytest.mark.parametrize("platform", ["cpu", "cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("worker_decision", [None, False, True])  # type: ignore[misc]
def test_engine_config_preserves_explicit_decision_controls(
    tmp_path: Path,
    platform: str,
    worker_decision: bool | None,
) -> None:
    from vllm_gr.engine.arg_utils_gr import create_engine_config

    initial = GRConfig(
        beam_max_width=32,
        beam_worker_decision=worker_decision,
    )
    vllm_config = _vllm_config()
    engine_args = engine_args_from_file(tmp_path, gr_config_to_mapping(initial))
    with (
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.origin_create_engine_config", return_value=vllm_config
        ),
        mock.patch(
            "vllm_gr.engine.arg_utils_gr.engine_config_service",
            return_value=legacy_config_service(platform),
        ),
    ):
        result = GRConfig.from_vllm_config(create_engine_config(engine_args))
    expected_worker = worker_decision
    if platform != "cpu":
        if expected_worker is None:
            expected_worker = True
    assert result.beam_worker_decision is expected_worker


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("execution", list(NPUDecodeExecution))  # type: ignore[misc]
def test_engine_config_merge_keeps_npu_execution_choice(
    tmp_path: Path, execution: str | None
) -> None:
    from vllm_gr.config.parser import load_engine_gr_config

    args = engine_args_from_file(
        tmp_path,
        gr_config_to_mapping(GRConfig(npu_decode_execution=execution, beam_graph_batch_size=4)),
    )
    result = load_engine_gr_config(
        args.config, {}, {"beam_width": [32, 32, 32], "candidate_width": [32, 32, 32]}
    )
    assert result.npu_decode_execution is execution
    assert result.beam_graph_batch_size == 4
    assert result.beam_max_width == 32


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _isolate_backend_plan_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configuration policy independently of a native backend install."""
    from vllm_gr.platforms import planning

    monkeypatch.setattr(planning, "admit_backend_plan", lambda config: None)
