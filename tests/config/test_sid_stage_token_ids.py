# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The public startup configuration carries explicit stage domains to workers."""

import json
from types import SimpleNamespace

import pytest

from vllm_gr.config import GRConfig, gr_config_from_mapping, gr_config_to_mapping
from vllm_gr.platforms.beam_requests import validate_gr_device_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_sid_stage_token_ids_survive_worker_transport() -> None:
    ids = [[9, 3, 12], [3, 8], [17]]
    config = gr_config_from_mapping(
        {
            "gr": {
                "sid_stage_token_ids": ids,
                "constraint_mode": "none",
                "worker_decision": True,
                "max_decode_steps": 3,
            }
        }
    )
    snapshot = json.loads(json.dumps(gr_config_to_mapping(config)))
    assert snapshot["gr"]["sid_stage_token_ids"] == ids
    worker = SimpleNamespace(additional_config={"vllm_gr": snapshot})
    assert GRConfig.from_vllm_config(worker) == config


@pytest.mark.parametrize(  # type: ignore[misc]
    "token_ids",
    [[], [[], [5], [7]], [[3, 3], [5], [7]], [[-1], [5], [7]], [[True], [5], [7]]],
)
def test_sid_stage_token_ids_reject_invalid_domains(token_ids: list[list[int]]) -> None:
    with pytest.raises((TypeError, ValueError), match="sid_stage_token_ids"):
        gr_config_from_mapping(
            {
                "gr": {
                    "sid_stage_token_ids": token_ids,
                    "constraint_mode": "none",
                    "worker_decision": True,
                    "max_decode_steps": 3,
                }
            }
        )


@pytest.mark.parametrize("token_id", [7, 8])  # type: ignore[misc]
def test_sid_stage_token_ids_respect_model_vocab_size(token_id: int) -> None:
    host = SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        model_config=SimpleNamespace(get_vocab_size=lambda: 8),
        additional_config={
            "vllm_gr": {
                "gr": {
                    "sid_stage_token_ids": [[token_id], [3], [4]],
                    "constraint_mode": "none",
                    "worker_decision": True,
                    "max_decode_steps": 3,
                }
            }
        },
    )
    if token_id == 7:
        validate_gr_device_config(host)
    else:
        with pytest.raises(ValueError, match="sid_stage_token_ids"):
            validate_gr_device_config(host)
