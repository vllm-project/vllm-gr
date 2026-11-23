# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project


from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from vllm_gr.config import (
    ConstraintBackendKind,
    ConstraintMode,
    gr_config_from_mapping,
    gr_config_to_mapping,
    resolve_candidate_width,
)

_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _v2_table(*, digests: bool = True, max_top_k: int | None = 128) -> dict[str, Any]:
    table: dict[str, object] = {
        "path": "/tmp/constraint_table.safetensors",
        "backend": "auto",
    }
    if max_top_k is not None:
        table["max_top_k"] = max_top_k
    if digests:
        table["digests"] = {
            "artifact": _ARTIFACT_DIGEST,
            "tokenizer": _TOKENIZER_DIGEST,
        }
    return {
        "gr": {
            "beam_width": [64, 128, 128],
            "candidate_width": [64] * 3,
            "constraint_mode": "table",
            "constraint_table": table,
        }
    }


def test_table_mode_preserves_auto_placement_and_candidate_width() -> None:
    config = gr_config_from_mapping(_v2_table())

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


def test_v2_round_trip() -> None:
    config = gr_config_from_mapping(_v2_table())

    mapping = gr_config_to_mapping(config)

    assert mapping["gr"]["constraint_mode"] == "table"
    assert mapping["gr"]["candidate_width"] == [64, 64, 64]
    round_tripped = gr_config_from_mapping(mapping)
    assert round_tripped.constraint == config.constraint
    assert round_tripped.beam_candidate_width == 64


def test_v2_rejects_removed_top_level_keys() -> None:
    mapping = _v2_table()
    mapping["constraint_table"] = {"enabled": True}

    with pytest.raises(ValueError, match="configuration"):
        gr_config_from_mapping(mapping)


def test_v2_table_requires_digests() -> None:
    with pytest.raises(ValueError, match="digests"):
        gr_config_from_mapping(_v2_table(digests=False))


def test_v2_table_requires_max_top_k() -> None:
    with pytest.raises(ValueError, match="max_top_k"):
        gr_config_from_mapping(_v2_table(max_top_k=None))


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("width", [257, 512, 1024])  # type: ignore[misc]
def test_v2_unconstrained_widths_above_256(width: int) -> None:
    config = gr_config_from_mapping(
        {"gr": {"beam_width": [width] * 3, "candidate_width": [width] * 3}}
    )
    assert config.beam_max_width == width
    assert resolve_candidate_width(config) == width


def test_v2_none_keeps_explicit_candidate_width_above_beam_ceiling() -> None:
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


def test_v2_adapter_cache_opt_in_round_trips() -> None:
    raw = _v2_table()
    beam = raw["gr"]
    assert isinstance(beam, dict)
    beam["adapter_cache_enabled"] = True

    config = gr_config_from_mapping(raw)

    assert config.beam_adapter_cache_enabled is True
    assert gr_config_to_mapping(config)["gr"]["adapter_cache_enabled"] is True


def test_v2_none_rejects_table_block() -> None:
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


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_common_prefill_grid_and_head_defaults_round_trip() -> None:
    from vllm_gr.config.parser import load_gr_config

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


def test_file_snapshot_survives_driver_file_removal(tmp_path: Path) -> None:
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm_gr.config import GRConfig, write_canonical_gr_config
    from vllm_gr.config.parser import COMMON_ENGINE_CONFIG, load_gr_config

    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(_v2_table()))
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


@pytest.mark.parametrize("extra", [{"vllm_gr": {}}, {"vllm_gr_beam_max_width": 8}])  # type: ignore[misc]
def test_driver_rejects_inline_settings(extra: Any) -> None:
    from vllm_gr.engine.startup import prepare_engine_settings

    with pytest.raises(ValueError, match="--config"):
        prepare_engine_settings(None, {"additional_config": extra}, {})


def test_driver_rejects_mapping_in_place_of_file() -> None:
    from vllm_gr.config.parser import load_gr_config

    with pytest.raises(TypeError, match="file path"):
        load_gr_config(_v2_table())


def test_python_override_can_disable_a_file_constraint(tmp_path: Path) -> None:
    import json

    from vllm_gr.config.parser import load_gr_config

    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(_v2_table()))
    config = load_gr_config(path, overrides={"constraint_mode": "none"})
    assert config.constraint.mode is ConstraintMode.NONE
    assert config.beam_max_width == 128


def test_python_artifact_override_is_relative_to_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from vllm_gr.config.parser import load_gr_config

    scenario = tmp_path / "scenario"
    scenario.mkdir()
    path = scenario / "scenario.yaml"
    path.write_text(json.dumps(_v2_table()))
    monkeypatch.chdir(tmp_path)
    config = load_gr_config(path, overrides={"constraint_table": {"path": "new.safetensors"}})
    assert config.constraint.table.path == str(tmp_path / "new.safetensors")


@pytest.mark.parametrize("overrides", [False, [], "invalid"])  # type: ignore[misc]
def test_python_gr_requires_a_mapping(overrides: Any) -> None:
    from vllm_gr.config.parser import load_gr_config

    with pytest.raises(ValueError, match="gr must be a mapping"):
        load_gr_config(overrides=overrides)


@pytest.mark.parametrize("mode", ["table", "none"])  # type: ignore[misc]
@pytest.mark.parametrize("execution", [None, "cascade", "xllm"])  # type: ignore[misc]
def test_python_decode_execution_survives_worker_transport(
    mode: str, execution: str | None
) -> None:
    from dataclasses import replace

    raw = (
        _v2_table()
        if mode == "table"
        else {"gr": {"constraint_mode": "none", "constraint_table": None}}
    )
    config = replace(gr_config_from_mapping(raw), npu_decode_execution=execution)
    worker = gr_config_from_mapping(gr_config_to_mapping(config))
    assert worker.npu_decode_execution == config.npu_decode_execution


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_auto_backend_remains_unbound_until_driver_selects_device(device: str) -> None:
    from vllm_gr.config.models import resolve_constraint_backend

    config = gr_config_from_mapping(_v2_table())
    assert config.constraint.table.enabled and config.constraint.table.backend.value == "auto"
    worker = gr_config_from_mapping(
        gr_config_to_mapping(resolve_constraint_backend(config, device))
    )
    assert worker.constraint.table.enabled and worker.constraint.table.backend.value == device


def test_v1_candidate_width_cannot_be_silently_ignored() -> None:
    with pytest.raises(ValueError, match="candidate_width"):
        gr_config_from_mapping({"gr": {"beam_width": [4] * 3, "candidate_width": [8] * 3}})


def test_public_beam_steps_and_batch_inheritance(tmp_path: Path) -> None:
    import json

    from vllm_gr.config.parser import load_gr_config

    table = _v2_table()["gr"]["constraint_table"]
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
    from vllm_gr.config.parser import load_gr_config

    with pytest.raises(ValueError, match="candidate"):
        load_gr_config(
            overrides={"constraint_mode": "none", "beam_width": [4, 8], "candidate_width": expanded}
        )


def test_beam_batch_sizes_can_omit_singleton() -> None:
    from vllm_gr.config.parser import load_gr_config

    config = load_gr_config(overrides={"constraint_mode": "none", "beam_batch_sizes": [2, 4]})
    assert config.deployment.beam_batch_sizes == (2, 4)
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config
    with pytest.raises(ValueError, match="strictly increasing"):
        load_gr_config(overrides={"constraint_mode": "none", "beam_batch_sizes": [4, 1]})
