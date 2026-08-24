# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_gr.config import BeamConfig, ConstraintTableConfig, GRConfig
from vllm_gr.entrypoints.gr import (
    _validate_offline_constraint_capacity,
    _validate_offline_constraint_options,
    _validate_offline_constraint_protocol_available,
)
from vllm_gr.entrypoints.openai.serving_engine import (
    _validate_serving_constraint_capacity,
    _validate_serving_constraint_options,
)
from vllm_gr.entrypoints.openai.serving_models import (
    _validate_constraint_protocol_available,
)

DIGEST = "sha256:" + "1" * 64


def _config(*, max_width: int = 64, max_decode_steps: int = 3) -> GRConfig:
    return GRConfig(
        beam=BeamConfig(
            max_width=max_width,
            max_decode_steps=max_decode_steps,
        ),
        constraint_table=ConstraintTableConfig(
            enabled=True,
            path="/constraint_table.safetensors",
            artifact_digest=DIGEST,
            tokenizer_digest=DIGEST,
            max_top_k=max_width,
        ),
    )


@pytest.mark.parametrize(
    "validator",
    [_validate_offline_constraint_capacity, _validate_serving_constraint_capacity],
)
def test_request_capacity_uses_gr_config_and_protocol_limit(validator) -> None:
    validator(_config(), beam_width=64, max_generate_steps=3)

    with pytest.raises(ValueError, match=r"beam_width.*beam.max_width 64.*65"):
        validator(_config(), beam_width=65, max_generate_steps=3)

    with pytest.raises(ValueError, match=r"beam.max_decode_steps 2.*3"):
        validator(
            _config(max_decode_steps=2),
            beam_width=1,
            max_generate_steps=3,
        )

    with pytest.raises(ValueError, match="protocol supports at most 3"):
        validator(
            _config(max_decode_steps=10),
            beam_width=1,
            max_generate_steps=4,
        )


@pytest.mark.parametrize(
    "validator",
    [_validate_offline_constraint_options, _validate_serving_constraint_options],
)
def test_request_options_are_owned_by_frontend(validator) -> None:
    assert validator(_config(), temperature=0.0, catalog_strategy=None) == (0.0, None)

    for value in (True, -0.1, float("inf"), float("nan"), "1.0"):
        with pytest.raises(ValueError, match="finite non-negative"):
            validator(_config(), temperature=value, catalog_strategy=None)

    with pytest.raises(ValueError, match="not applicable to constraint_table"):
        validator(_config(), temperature=1.0, catalog_strategy="null")


def test_protocol_capability_is_checked_by_frontend_owner() -> None:
    config = _config()
    custom = SimpleNamespace(attention_config=SimpleNamespace(backend="CUSTOM"))
    assert _validate_offline_constraint_protocol_available(custom, config)

    with pytest.raises(RuntimeError, match="CUSTOM attention backend"):
        _validate_offline_constraint_protocol_available(SimpleNamespace(), config)

    client = SimpleNamespace(beam_request_step_update=lambda update: None)
    _validate_constraint_protocol_available(client, config)
    with pytest.raises(RuntimeError, match="beam_request_step_update"):
        _validate_constraint_protocol_available(SimpleNamespace(), config)
