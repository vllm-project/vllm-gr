# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import argparse
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from benchmarks.streaming_head import (
    add_streaming_head_argument,
    read_inference_cuda_peak,
    reset_inference_cuda_peak,
    streaming_head_config,
)
from vllm_gr.config import GRConfig, gr_config_from_mapping, load_gr_config
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.v1.worker.gr_context import get_beam_graph_runtime_stats

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


@pytest.mark.parametrize(  # type: ignore[misc]
    "flag,enabled",
    [
        (None, False),
        ("--no-streaming-constrained-head", False),
        ("--streaming-constrained-head", True),
    ],
)
def test_streaming_flag_roundtrips_through_worker_config(flag: str | None, enabled: bool) -> None:
    parser = argparse.ArgumentParser()
    add_streaming_head_argument(parser)
    args = parser.parse_args([] if flag is None else [flag])
    overrides = streaming_head_config(args)
    assert ("enable_streaming_constrained_head" in overrides) is enabled
    config = gr_config_from_mapping({"gr": overrides})
    worker = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(worker)
    assert load_gr_config().deployment.enable_streaming_constrained_head is False
    assert GRConfig.from_vllm_config(worker).deployment.enable_streaming_constrained_head is enabled


@pytest.mark.parametrize("wrong", [None, 0, 1, "true"])  # type: ignore[misc]
def test_invalid_streaming_flag_is_rejected(wrong: Any) -> None:
    with pytest.raises(ValueError, match="enable_streaming_constrained_head requires a boolean"):
        DeploymentOptions(enable_streaming_constrained_head=wrong)


def test_memory_reporting_polls_named_inference_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    workers = [
        SimpleNamespace(device=torch.device("cuda", index), model_runner=SimpleNamespace())
        for index in (1, 3)
    ]
    sync = Mock()
    reset = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda device: 10 * device.index)
    monkeypatch.setattr(torch.accelerator, "max_memory_reserved", lambda device: 20 * device.index)

    def rpc(method: str, *, kwargs: dict[str, str]) -> list[Any]:
        assert method == "get_gr_beam_graph_runtime_stats"
        return [get_beam_graph_runtime_stats(worker, **kwargs) for worker in workers]

    llm = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    assert reset_inference_cuda_peak(llm)
    memory = read_inference_cuda_peak(llm)
    assert [call.args[0] for call in reset.call_args_list] == [worker.device for worker in workers]
    assert [call.args[0] for call in sync.call_args_list] == [
        worker.device for worker in workers
    ] * 2
    assert memory["peak_allocated_bytes"] == 30
    assert memory["peak_reserved_bytes"] == 60
    assert [row["peak_allocated_bytes"] for row in memory["workers"]] == [10, 30]
