# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm_gr.config import ConstraintTableConfig, GRConfig
from vllm_gr.config.benchmark_options import (
    add_recogem_arguments,
    beam_optimization_config,
    configure_benchmark_precision,
)
from vllm_gr.config.parser import (
    gr_config_from_mapping,
    gr_config_to_mapping,
    parse_gr_config,
    write_gr_config,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_full_step_config_roundtrip(enabled: bool) -> None:
    config = GRConfig(
        enable_full_step_graph=enabled,
        constraint_table=ConstraintTableConfig(max_top_k=128),
    )
    assert gr_config_from_mapping(gr_config_to_mapping(config)).enable_full_step_graph is enabled
    engine = SimpleNamespace(additional_config={})
    write_gr_config(engine, GRConfig(enable_full_step_graph=enabled))
    assert parse_gr_config(engine).enable_full_step_graph is enabled
    parser = argparse.ArgumentParser()
    add_recogem_arguments(parser)
    args = parser.parse_args(["--full-step-graph" if enabled else "--no-full-step-graph"])
    assert beam_optimization_config(args).get("enable_full_step_graph", False) is enabled


@pytest.mark.parametrize("value", [None, 0, 1, "true"])  # type: ignore[misc]
def test_full_step_rejects_non_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="enable_full_step_graph must be a boolean"):
        GRConfig(enable_full_step_graph=cast(Any, value))


def test_strict_precision_requires_in_process_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    with pytest.raises(ValueError, match="requires VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        configure_benchmark_precision(argparse.Namespace(strict_bf16=True))


def test_precision_is_only_changed_by_explicit_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    request = pytest.MonkeyPatch()
    request.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    assert configure_benchmark_precision(argparse.Namespace(strict_bf16=False)) == {
        "strict_bf16": False,
        "bf16_reduced_precision_reduction": True,
    }
    try:
        assert configure_benchmark_precision(argparse.Namespace(strict_bf16=True)) == {
            "strict_bf16": True,
            "bf16_reduced_precision_reduction": False,
        }
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        request.undo()


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_streaming_head_config_roundtrip(enabled: bool) -> None:
    config = GRConfig(
        enable_streaming_constrained_head=enabled,
        constraint_table=ConstraintTableConfig(max_top_k=128),
    )
    assert (
        gr_config_from_mapping(gr_config_to_mapping(config)).enable_streaming_constrained_head
        is enabled
    )
    engine = SimpleNamespace(additional_config={})
    write_gr_config(engine, GRConfig(enable_streaming_constrained_head=enabled))
    assert parse_gr_config(engine).enable_streaming_constrained_head is enabled
    parser = argparse.ArgumentParser()
    add_recogem_arguments(parser)
    args = parser.parse_args(
        ["--streaming-constrained-head" if enabled else "--no-streaming-constrained-head"]
    )
    assert beam_optimization_config(args).get("enable_streaming_constrained_head", False) is enabled


@pytest.mark.parametrize("value", [None, 0, 1, "true"])  # type: ignore[misc]
def test_streaming_head_rejects_non_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="enable_streaming_constrained_head must be a boolean"):
        GRConfig(enable_streaming_constrained_head=cast(Any, value))
