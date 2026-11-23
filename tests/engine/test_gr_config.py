# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.helpers.gr_config import engine_args_from_file
from vllm_gr.config import (
    MAX_BEAM_WIDTH,
    MAX_CANDIDATE_WIDTH,
    ConstraintTableSpec,
    GRAttentionBackend,
    GRConfig,
    NPUDecodeExecution,
    gr_config_to_mapping,
    resolve_candidate_width,
)
from vllm_gr.platforms.legacy import legacy_config_service

ATTENTION_BACKEND_KEY = "vllm_gr_attention_backend"
CATALOG_PATH_KEY = "vllm_gr_catalog_path"
CONSTRAINT_BACKEND_KEY = "vllm_gr_constraint_backend"
NPU_DECODE_EXECUTION_KEY = "vllm_gr_npu_decode_execution"


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


@pytest.mark.parametrize("value", [1, True, [], "unknown"])  # type: ignore[misc]
def test_rejects_invalid_constraint_backend(value: object) -> None:
    with pytest.raises(ValueError, match="constraint_backend"):
        GRConfig.from_vllm_config(_vllm_config({CONSTRAINT_BACKEND_KEY: value}))


def test_constraint_table_requires_catalog_path() -> None:
    with pytest.raises(ValueError, match="requires catalog_path"):
        GRConfig(constraint_backend="constraint_table")


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
        GRConfig.from_vllm_config(_vllm_config({NPU_DECODE_EXECUTION_KEY: value}))


@pytest.mark.parametrize("value", [1, True, [], {}])  # type: ignore[misc]
def test_rejects_non_string_catalog_path(value: object) -> None:
    with pytest.raises(ValueError, match="catalog_path"):
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
        GRConfig(catalog_path=str(Path("/models/catalog.json").resolve())).write_to_vllm_config(
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
        catalog_path=None,
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
        catalog_path=None,
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
    """K follows W up to the supported width; invalid derived K still fails."""
    table = ConstraintTableSpec(
        max_top_k=MAX_CANDIDATE_WIDTH,
        enabled=True,
        path=str(Path("/tmp/catalog.json").resolve()),
        artifact_digest="sha256:" + "0" * 64,
        tokenizer_digest="sha256:" + "0" * 64,
    )
    # Unset K follows the Beam width.
    follows = GRConfig(beam_max_width=128, constraint_table=table)
    assert resolve_candidate_width(follows) == 128
    # An explicit K equal to W is still accepted; it states today's behaviour.
    same = GRConfig(beam_max_width=128, constraint_table=replace(table, candidate_width=128))
    assert resolve_candidate_width(same) == 128

    # The derivation refuses a Beam wider than the candidate ceiling unless the
    # config states K itself.  Build the object directly: construction rejects
    # this width before the derivation is ever reached.
    wide = SimpleNamespace(
        beam_max_width=MAX_CANDIDATE_WIDTH * 2,
        constraint_table=replace(table, enabled=False),
    )
    with pytest.raises(ValueError, match="exceeds the candidate width ceiling"):
        resolve_candidate_width(wide)
    wide.constraint_table = replace(table, enabled=False, candidate_width=MAX_CANDIDATE_WIDTH)
    assert resolve_candidate_width(wide) == MAX_CANDIDATE_WIDTH


@pytest.mark.parametrize(  # type: ignore[misc]
    "constraint_table",
    [
        None,
        ConstraintTableSpec(
            max_top_k=MAX_CANDIDATE_WIDTH,
            enabled=False,
            path=str(Path("/tmp/catalog.json").resolve()),
            artifact_digest="sha256:" + "0" * 64,
            tokenizer_digest="sha256:" + "0" * 64,
        ),
    ],
    ids=["no-table", "disabled-table"],
)
def test_unconstrained_candidate_width_follows_wide_beam(
    constraint_table: ConstraintTableSpec | None,
) -> None:
    """Unconstrained K can follow W above 256 without an explicit override."""
    for width in (257, 512, MAX_BEAM_WIDTH):
        accepted = GRConfig(beam_max_width=width, constraint_table=constraint_table)
        assert resolve_candidate_width(accepted) == width


def test_wide_beam_is_servable_once_the_candidate_width_is_stated() -> None:
    """Beam capacity and an explicit table candidate width resolve independently."""

    table = ConstraintTableSpec(
        max_top_k=MAX_CANDIDATE_WIDTH,
        enabled=True,
        path=str(Path("/tmp/catalog.json").resolve()),
        artifact_digest="sha256:" + "0" * 64,
        tokenizer_digest="sha256:" + "0" * 64,
        candidate_width=MAX_CANDIDATE_WIDTH,
    )
    wide = GRConfig(beam_max_width=MAX_BEAM_WIDTH, constraint_table=table)
    assert wide.beam_max_width == MAX_BEAM_WIDTH
    assert resolve_candidate_width(wide) == MAX_CANDIDATE_WIDTH


@pytest.mark.parametrize("candidate_width", [64, 256])  # type: ignore[misc]
def test_explicit_candidate_width_may_differ_from_the_beam_width(
    candidate_width: int,
) -> None:
    """The engine-wiring layer makes K live, so K != W is now accepted.

    The workspace bucket, the constrained Top-K and the decision state are all
    keyed by ``resolve_candidate_width``, so a configured K that differs from W
    is carried through instead of being executed as W.
    """
    table = ConstraintTableSpec(
        max_top_k=MAX_CANDIDATE_WIDTH,
        enabled=True,
        path=str(Path("/tmp/catalog.json").resolve()),
        artifact_digest="sha256:" + "0" * 64,
        tokenizer_digest="sha256:" + "0" * 64,
    )
    decoupled = GRConfig(
        beam_max_width=128,
        constraint_table=replace(table, candidate_width=candidate_width),
    )
    assert resolve_candidate_width(decoupled) == candidate_width
    assert decoupled.beam_max_width == 128

    # K == W still resolves, and unset K still follows the Beam width.
    matching = GRConfig(
        beam_max_width=candidate_width,
        constraint_table=replace(table, candidate_width=candidate_width),
    )
    assert resolve_candidate_width(matching) == candidate_width
    follows = GRConfig(beam_max_width=128, constraint_table=table)
    assert resolve_candidate_width(follows) == 128


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
    assert config.constraint_backend != "constraint_table"
    assert config.catalog_path is None


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
    # The unified flag auto-enables for accelerator beam with worker
    # decisions regardless of backend; actual chaining is lazy-gated by
    # data availability (trie-catalog deployments never produce worker
    # decisions, unconstrained deployments chain decode from step 1).
    assert config.beam_engine_driven is None
    assert config.constraint_backend == "trie"


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
    assert result.beam_engine_driven is None


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
