# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_gr.config import (
    ConstraintBackend,
    ConstraintMode,
    ConstraintTableBackendEnum,
    gr_config_from_mapping,
    gr_config_to_mapping,
)

_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _v2_table(*, digests: bool = True, max_top_k: int | None = 128) -> dict[str, object]:
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
        "schema_version": 2,
        "constraint": {"mode": "table", "table": table},
        "beam": {
            "execution_mode": "v1",
            "max_width": 128,
            "candidate_width": 64,
        },
    }


def test_v2_table_parses_and_derives_legacy_view() -> None:
    config = gr_config_from_mapping(_v2_table())

    assert config.schema_version == 2
    assert config.constraint is not None
    assert config.constraint.mode is ConstraintMode.TABLE
    assert config.constraint_config.mode is ConstraintMode.TABLE
    assert config.constraint_table is not None
    assert config.constraint_table.enabled is True
    # "auto" is resolved later against the active platform; the derived legacy
    # view keeps the historical CUDA default until then.
    assert config.constraint_table.backend is ConstraintTableBackendEnum.CUDA
    assert config.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
    assert config.constrained_generation is None
    assert config.beam_candidate_width == 64
    assert config.constraint_table.candidate_width == 64


def test_v2_none_derives_unconstrained_view() -> None:
    config = gr_config_from_mapping(
        {
            "schema_version": 2,
            "constraint": {"mode": "none"},
            "beam": {"max_width": 8},
        }
    )

    assert config.constraint is not None
    assert config.constraint_config.mode is ConstraintMode.NONE
    assert config.constraint_backend is ConstraintBackend.TRIE
    assert config.constrained_generation is False
    assert config.constraint_table is None


def test_v2_round_trip() -> None:
    config = gr_config_from_mapping(_v2_table())

    mapping = gr_config_to_mapping(config)

    assert mapping["schema_version"] == 2
    assert mapping["constraint"]["mode"] == "table"  # type: ignore[index]
    assert mapping["beam"]["candidate_width"] == 64  # type: ignore[index]
    round_tripped = gr_config_from_mapping(mapping)
    assert round_tripped.constraint == config.constraint
    assert round_tripped.beam_candidate_width == 64


def test_v2_rejects_removed_top_level_keys() -> None:
    mapping = _v2_table()
    mapping["constraint_table"] = {"enabled": True}

    with pytest.raises(ValueError, match="constraint.mode/constraint.table"):
        gr_config_from_mapping(mapping)


def test_v2_table_requires_digests() -> None:
    with pytest.raises(ValueError, match="digests"):
        gr_config_from_mapping(_v2_table(digests=False))


def test_v2_table_requires_max_top_k() -> None:
    with pytest.raises(ValueError, match="max_top_k"):
        gr_config_from_mapping(_v2_table(max_top_k=None))


def test_v2_none_rejects_table_block() -> None:
    with pytest.raises(ValueError, match="forbids constraint.table"):
        gr_config_from_mapping(
            {
                "schema_version": 2,
                "constraint": {"mode": "none", "table": {}},
                "beam": {"max_width": 8},
            }
        )


pytestmark = pytest.mark.cpu_test
