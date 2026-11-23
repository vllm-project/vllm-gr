# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""GR configuration values, mapping validation and YAML overrides."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.helpers.gr_config import (
    config_host,
    fixed_beam_mapping,
    stepped_table_mapping,
    table_constraint,
)
from vllm_gr.config import (
    MAX_BEAM_WIDTH,
    ConstraintBackendKind,
    ConstraintMode,
    ConstraintTableSpec,
    GRAttentionBackend,
    GRConfig,
    NPUDecodeExecution,
    gr_config_from_mapping,
    gr_config_to_mapping,
    load_gr_config,
    resolve_candidate_width,
)

_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_TOKENIZER_DIGEST = "sha256:" + "b" * 64

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_npu_decode_execution_defaults_to_unset() -> None:
    # Unset (None): the runtime resolves per platform (xllm on NPU with the
    # operator library, cascade otherwise).
    assert GRConfig().npu_decode_execution is None
    assert GRConfig.from_vllm_config(config_host()).npu_decode_execution is None
    # Unset is not an explicit GR configuration; explicit cascade is.
    assert GRConfig().enabled is False
    assert GRConfig(npu_decode_execution=NPUDecodeExecution.CASCADE).enabled is True


@pytest.mark.parametrize("value", [1, True, [], {}, "flash"])  # type: ignore[misc]
def test_rejects_invalid_npu_decode_execution(value: object) -> None:
    with pytest.raises(ValueError, match="npu_decode_execution"):
        GRConfig.from_vllm_config(config_host({"vllm_gr": {"gr": {"npu_decode_execution": value}}}))


def test_direct_construction_validates_and_normalizes() -> None:
    config = GRConfig(attention_backend="custom")
    assert config.attention_backend is GRAttentionBackend.CUSTOM

    with pytest.raises(ValueError, match="attention_backend"):
        GRConfig(attention_backend="FLASH_ATTN")


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


@pytest.mark.parametrize("execution", list(NPUDecodeExecution))  # type: ignore[misc]
def test_file_overrides_keep_npu_execution_choice(tmp_path: Path, execution: str | None) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(
        json.dumps(
            gr_config_to_mapping(GRConfig(npu_decode_execution=execution, beam_graph_batch_size=4))
        )
    )
    result = load_gr_config(
        path, overrides={"beam_width": [32, 32, 32], "candidate_width": [32, 32, 32]}
    )
    assert result.npu_decode_execution is execution
    assert result.beam_graph_batch_size == 4
    assert result.beam_max_width == 32


def test_canonical_mapping_round_trip_is_exact(tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    config = gr_config_from_mapping(raw)

    assert config.beam_max_width == 32
    table = config.constraint_config.table
    assert table is not None
    assert table.backend is ConstraintBackendKind.CPU
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


def test_canonical_graph_batch_shape_round_trip(tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["decode_graph_batch_size"] = 2

    config = gr_config_from_mapping(raw)

    assert config.beam_graph_batch_size == 2
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


def test_default_beam_batch_sizes() -> None:
    assert GRConfig().deployment.beam_batch_sizes == (1, 2, 4)


@pytest.mark.parametrize(  # type: ignore[misc]
    "sizes",
    [[], [7, 3], [1, 1], [True, 3], [1, False], [1, 3.0], [1, "3"], [1, 0], [1, 9], "1,3,7"],
)
def test_invalid_beam_batch_sizes_fail_during_config_parsing(sizes: object, tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    raw["gr"].update({"beam_batch_sizes": sizes})
    with pytest.raises(ValueError, match="beam.batch.sizes"):
        gr_config_from_mapping(raw)


@pytest.mark.parametrize("worker_decision", [True, False])  # type: ignore[misc]
def test_canonical_beam_runtime_options_round_trip(
    worker_decision: bool,
    tmp_path: Path,
) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["worker_decision"] = worker_decision

    config = gr_config_from_mapping(raw)

    assert config.beam_worker_decision is worker_decision
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config


def test_canonical_mapping_defaults_constraint_backend_to_auto(tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    table_mapping = raw["gr"]["constraint_table"]
    assert isinstance(table_mapping, dict)
    del table_mapping["backend"]

    table = gr_config_from_mapping(raw).constraint_config.table

    assert table is not None
    assert table.backend is ConstraintBackendKind.AUTO


@pytest.mark.parametrize("version", [True, 1.0, "1", None])  # type: ignore[misc]
def test_mapping_rejects_schema_version_key(version: object, tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    raw["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        gr_config_from_mapping(raw)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("layer", "key"),
    [("top", "unknown"), ("beam", "width"), ("table", "digest")],
)
def test_unknown_keys_are_rejected(layer: str, key: str, tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    if layer == "top":
        raw[key] = 1
    elif layer == "beam":
        raw["gr"][key] = 1
    else:
        raw["gr"]["constraint_table"][key] = 1
    with pytest.raises(ValueError, match="(?i)unknown"):
        gr_config_from_mapping(raw)


def test_yaml_path_is_resolved_relative_to_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "gr.yaml"
    config_path.write_text(
        """gr:
  beam_graph_enabled: true
  beam_width:
  - 32
  - 32
  - 32
  max_decode_steps: 3
  constraint_mode: table
  constraint_table:
    backend: cpu
    path: ../artifacts/table.safetensors
    format: constraint_table_v1
    digests:
      artifact: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      tokenizer: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    max_top_k: 128
  candidate_width:
  - 32
  - 32
  - 32
""",
        encoding="utf-8",
    )

    config = load_gr_config(config_path)

    table = config.constraint_config.table
    assert table is not None
    assert table.path == str((tmp_path / "artifacts/table.safetensors").resolve())


def test_mapping_rejects_relative_artifact_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        gr_config_from_mapping(fixed_beam_mapping("table.safetensors"))


def test_mapping_base_dir_resolves_relative_artifact_path(tmp_path: Path) -> None:
    config = gr_config_from_mapping(fixed_beam_mapping("table.safetensors"), base_dir=tmp_path)
    table = config.constraint_config.table
    assert table is not None
    assert table.path == str((tmp_path / "table.safetensors").resolve())


def test_constraint_table_requires_complete_enabled_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_digest"):
        ConstraintTableSpec(
            max_top_k=128,
            enabled=True,
            path=str(tmp_path / "table.safetensors"),
        )


def test_canonical_mapping_requires_explicit_max_top_k(tmp_path: Path) -> None:
    raw = fixed_beam_mapping(str(tmp_path / "table.safetensors"))
    table = raw["gr"]["constraint_table"]
    assert isinstance(table, dict)
    del table["max_top_k"]

    with pytest.raises(ValueError, match=r"gr\.constraint_table\.max_top_k"):
        gr_config_from_mapping(raw)


def test_constraint_max_top_k_has_no_npu_upper_bound(tmp_path: Path) -> None:
    config = ConstraintTableSpec(
        enabled=True,
        backend=ConstraintBackendKind.NPU,
        path=str(tmp_path / "table.safetensors"),
        artifact_digest=_ARTIFACT_DIGEST,
        tokenizer_digest=_TOKENIZER_DIGEST,
        max_top_k=257,
    )
    assert config.max_top_k == 257


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])  # type: ignore[misc]
@pytest.mark.parametrize("source", ["python", "canonical"])  # type: ignore[misc]
def test_adapter_cache_switch_requires_boolean(value: object, source: str) -> None:
    with pytest.raises(ValueError, match="adapter_cache_enabled must be a boolean"):
        if source == "python":
            GRConfig(beam_adapter_cache_enabled=value)
        else:
            gr_config_from_mapping({"gr": {"adapter_cache_enabled": value}})


def test_table_mode_preserves_auto_placement_and_candidate_width() -> None:
    config = gr_config_from_mapping(stepped_table_mapping())

    assert config.constraint is not None
    assert config.constraint.mode is ConstraintMode.TABLE
    assert config.constraint.table is not None and config.constraint.table.enabled
    assert config.constraint.table.backend is ConstraintBackendKind.AUTO
    assert config.beam_candidate_width == 64


def test_none_mode_has_no_table() -> None:
    config = gr_config_from_mapping(
        {
            "gr": {
                "beam_width": [8] * 3,
                "constraint_mode": "none",
                "constraint_table": None,
                "candidate_width": [8] * 3,
            }
        }
    )

    assert config.constraint is not None
    assert config.constraint.mode is ConstraintMode.NONE
    assert config.constraint.table is None


def test_table_mapping_round_trip_preserves_constraint_and_candidate_width() -> None:
    config = gr_config_from_mapping(stepped_table_mapping())

    mapping = gr_config_to_mapping(config)

    assert mapping["gr"]["constraint_mode"] == "table"
    assert mapping["gr"]["candidate_width"] == [64, 64, 64]
    round_tripped = gr_config_from_mapping(mapping)
    assert round_tripped.constraint == config.constraint
    assert round_tripped.beam_candidate_width == 64


def test_rejects_removed_top_level_keys() -> None:
    mapping = stepped_table_mapping()
    mapping["constraint_table"] = {"enabled": True}

    with pytest.raises(ValueError, match="configuration"):
        gr_config_from_mapping(mapping)


def test_table_requires_digests() -> None:
    raw = stepped_table_mapping()
    del raw["gr"]["constraint_table"]["digests"]
    with pytest.raises(ValueError, match="digests"):
        gr_config_from_mapping(raw)


@pytest.mark.parametrize("width", [257, 512, 1024])  # type: ignore[misc]
def test_unconstrained_widths_above_256(width: int) -> None:
    config = gr_config_from_mapping(
        {"gr": {"beam_width": [width] * 3, "candidate_width": [width] * 3}}
    )
    assert config.beam_max_width == width
    assert resolve_candidate_width(config) == width


def test_none_keeps_explicit_candidate_width_above_beam_ceiling() -> None:
    config = gr_config_from_mapping(
        {
            "gr": {
                "beam_width": [128, 1024, 1024],
                "candidate_width": [128] * 3,
                "constraint_mode": "none",
                "constraint_table": None,
            }
        }
    )

    assert config.constraint_config.mode is ConstraintMode.NONE
    assert config.beam_candidate_width == 128
    assert resolve_candidate_width(config) == 128


def test_none_rejects_table_block() -> None:
    with pytest.raises(ValueError, match="forbids gr.constraint_table"):
        gr_config_from_mapping(
            {
                "gr": {
                    "beam_width": [8] * 3,
                    "constraint_mode": "none",
                    "constraint_table": {},
                    "candidate_width": [8] * 3,
                }
            }
        )


def test_common_prefill_grid_and_head_defaults_round_trip() -> None:
    config = load_gr_config(overrides={"constraint_mode": "none"})
    assert config.deployment.prefill_token_buckets == tuple(range(128, 8193, 128))
    assert config.deployment.head_slice == "auto"
    assert config.deployment.unconstrained_head_layout == "transposed"
    for overrides in (
        {"prefill_token_buckets": [64, 320], "head_slice": 16384},
        {"prefill_token_buckets": None, "head_slice": 0, "unconstrained_head_layout": None},
    ):
        custom = load_gr_config(overrides={"constraint_mode": "none", **overrides})
        assert gr_config_from_mapping(gr_config_to_mapping(custom)) == custom
        buckets = cast(list[int] | None, overrides["prefill_token_buckets"])
        assert custom.deployment.prefill_token_buckets == (tuple(buckets) if buckets else None)
        assert custom.deployment.head_slice == overrides["head_slice"]


def test_driver_rejects_mapping_in_place_of_file() -> None:
    with pytest.raises(TypeError, match="file path"):
        load_gr_config(stepped_table_mapping())


def test_python_override_can_disable_a_file_constraint(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(stepped_table_mapping()))
    config = load_gr_config(path, overrides={"constraint_mode": "none"})
    assert config.constraint.mode is ConstraintMode.NONE
    assert config.beam_max_width == 128


def test_python_artifact_override_is_relative_to_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    path = scenario / "scenario.yaml"
    path.write_text(json.dumps(stepped_table_mapping()))
    monkeypatch.chdir(tmp_path)
    config = load_gr_config(path, overrides={"constraint_table": {"path": "new.safetensors"}})
    assert config.constraint.table.path == str(tmp_path / "new.safetensors")


@pytest.mark.parametrize("overrides", [False, [], "invalid"])  # type: ignore[misc]
def test_python_gr_requires_a_mapping(overrides: Any) -> None:
    with pytest.raises(ValueError, match="gr must be a mapping"):
        load_gr_config(overrides=overrides)


@pytest.mark.parametrize("mode", ["table", "none"])  # type: ignore[misc]
@pytest.mark.parametrize("execution", [None, "cascade", "xllm"])  # type: ignore[misc]
def test_python_decode_execution_survives_worker_transport(
    mode: str, execution: str | None
) -> None:
    raw = (
        stepped_table_mapping()
        if mode == "table"
        else {"gr": {"constraint_mode": "none", "constraint_table": None}}
    )
    config = replace(gr_config_from_mapping(raw), npu_decode_execution=execution)
    worker = gr_config_from_mapping(gr_config_to_mapping(config))
    assert worker.npu_decode_execution == config.npu_decode_execution


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_auto_backend_remains_unbound_until_driver_selects_device(device: str) -> None:
    from vllm_gr.config.models import resolve_constraint_backend

    config = gr_config_from_mapping(stepped_table_mapping())
    assert config.constraint.table.enabled and config.constraint.table.backend.value == "auto"
    worker = gr_config_from_mapping(
        gr_config_to_mapping(resolve_constraint_backend(config, device))
    )
    assert worker.constraint.table.enabled and worker.constraint.table.backend.value == device


def test_first_candidate_width_must_match_first_beam_width() -> None:
    with pytest.raises(ValueError, match="candidate_width"):
        gr_config_from_mapping({"gr": {"beam_width": [4] * 3, "candidate_width": [8] * 3}})


def test_public_beam_steps_and_batch_inheritance(tmp_path: Path) -> None:
    table = stepped_table_mapping()["gr"]["constraint_table"]
    defaults = load_gr_config()
    assert defaults.constraint.mode is ConstraintMode.NONE
    assert defaults.constraint.table is None
    assert (
        load_gr_config(overrides={"constraint_table": table}).constraint.mode
        is ConstraintMode.TABLE
    )
    with pytest.raises(ValueError, match="table requires"):
        load_gr_config(overrides={"constraint_mode": "table"})
    assert defaults.beam_max_width == resolve_candidate_width(defaults) == 128
    assert defaults.beam_max_decode_steps == 3
    assert defaults.deployment.candidate_width == (128, 128, 128)
    path = tmp_path / "base.yaml"
    path.write_text(json.dumps({"gr": {"constraint_table": table}}))
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("import: ./base.yaml\ngr:\n  beam_batch_sizes: [1, 3]\n")
    opts = {"constraint_mode": "none", "beam_width": [8, 16, 4], "candidate_width": [8, 2, 1]}
    cfg = load_gr_config(scenario, overrides=opts)
    assert cfg.constraint.mode is ConstraintMode.NONE
    assert cfg.deployment.beam_batch_sizes == (1, 3)
    assert cfg.beam_max_width == 16 and resolve_candidate_width(cfg) == 8
    assert cfg.deployment.beam_width == (8, 16, 4)
    assert cfg.deployment.candidate_width == (8, 2, 1)
    assert gr_config_from_mapping(gr_config_to_mapping(cfg)) == cfg
    with pytest.raises(ValueError, match="Unknown gr settings"):
        load_gr_config(scenario, overrides={**opts, "online_batch_sizes": [1, 2]})


@pytest.mark.parametrize("expanded", [[4], [4, 0], [2, 4], [4, 1]])  # type: ignore[misc]
def test_public_beam_steps_reject_invalid_expansion(expanded: list[int]) -> None:
    with pytest.raises(ValueError, match="candidate"):
        load_gr_config(
            overrides={"constraint_mode": "none", "beam_width": [4, 8], "candidate_width": expanded}
        )


def test_beam_batch_sizes_can_omit_singleton() -> None:
    config = load_gr_config(overrides={"constraint_mode": "none", "beam_batch_sizes": [2, 4]})
    assert config.deployment.beam_batch_sizes == (2, 4)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config
    with pytest.raises(ValueError, match="strictly increasing"):
        load_gr_config(overrides={"constraint_mode": "none", "beam_batch_sizes": [4, 1]})
