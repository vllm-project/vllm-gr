# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""EngineArgs integration; requires the normal Torch/vLLM test runtime."""

from types import SimpleNamespace

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.engine import arg_utils_gr
from vllm_gr.platforms import loader
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.plan import read_backend_plan

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_engine_args_round_trip_keeps_the_admitted_backend_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = SimpleNamespace(additional_config={})
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: native)
    monkeypatch.setattr(arg_utils_gr, "_is_cuda_platform", lambda: True)
    monkeypatch.setattr(arg_utils_gr, "_is_npu_platform", lambda: False)
    monkeypatch.setattr(arg_utils_gr, "_apply_npu_default_block_size", lambda config: None)
    monkeypatch.setattr(loader, "activate_current_backend", lambda: CUDA_BACKEND)
    args = SimpleNamespace(beam_max_width=32, constrained_generation=False)
    first = arg_utils_gr.create_engine_config(args)
    plan = read_backend_plan(first)
    gr = GRConfig.from_vllm_config(first)
    assert plan is not None
    assert arg_utils_gr.create_engine_config(args) is first
    assert read_backend_plan(first) == plan
    assert GRConfig.from_vllm_config(first) == gr
    assert plan.request.config.to_dict()["beam_max_width"] == 32
