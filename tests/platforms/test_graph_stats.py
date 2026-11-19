# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check native and third-party graph diagnostics through the shared Worker RPC."""

from enum import Enum
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from vllm_gr.platforms.ascend.graph import AscendGraph
from vllm_gr.platforms.cuda.graph import CudaGraph
from vllm_gr.v1.worker.gr_context import get_beam_graph_runtime_stats

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class Mode(Enum):
    FULL = 1
    PIECEWISE = 2


class Descriptor(NamedTuple):
    cg_mode: Mode
    num_tokens: int
    num_reqs: int


def test_shared_prefill_rejects_inherited_capture_stubs() -> None:
    from vllm_gr.platforms.graph import shared_prefill_runner_class
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    class IncompletePrefill(BasePrefillGraphRunner):
        def capture_graphs(self, runner: object) -> None:
            pass

    graph = SimpleNamespace(prefill_runner_class=lambda: IncompletePrefill)
    with pytest.raises(RuntimeError, match="concrete runner class"):
        shared_prefill_runner_class(graph)


def test_cuda_graph_stats_describe_native_captures_and_dispatches() -> None:
    full = Descriptor(Mode.FULL, 128, 2)
    piecewise = Descriptor(Mode.PIECEWISE, 256, 1)
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_max_width=4),
        cudagraph_manager=SimpleNamespace(
            _successful=[piecewise, full],
            dispatch_counts={"FULL": 5, "PIECEWISE": 3, "NONE": 1},
            full_capture_failures={Descriptor(Mode.FULL, 512, 2): "shape_miss"},
        ),
    )
    runner._gr_native_graph = CudaGraph(runner, None)
    stats = get_beam_graph_runtime_stats(SimpleNamespace(model_runner=runner))
    assert stats["graph_service"]["implementation"] == "cuda.graph"
    assert stats["prefill_graph_captured_buckets"] == [128]
    assert stats["prefill_graph_captured_shapes"] == [(2, 128)]
    assert stats["prefill_graph_piecewise_buckets"] == [256]
    assert stats["prefill_graph_full_capture_failures"] == {"FULL:T=512:B=2": "shape_miss"}
    assert stats["prefill_graph_full_dispatches"] == 5
    assert stats["prefill_graph_piecewise_dispatches"] == 3
    assert stats["prefill_graph_none_dispatches"] == 1
    assert "captured_shapes" not in stats


def test_ascend_graph_stats_describe_shape_misses() -> None:
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(
            beam_graph_enabled=True, beam_max_width=4, beam_graph_batch_size=2
        ),
        _gr_npu_beam_graph_shapes={(2, 4), (1, 4)},
        _beam_graph_fallback_count=2,
        _beam_graph_fallback_reasons={"shape_miss": 2},
    )
    runner._gr_native_graph = AscendGraph(runner, None)
    stats = get_beam_graph_runtime_stats(SimpleNamespace(model_runner=runner))
    assert stats["graph_service"]["implementation"] == "ascend.graph"
    assert stats["beam_graph_batch_size"] == 2
    assert stats["captured_shapes"] == [(1, 4), (2, 4)]
    assert stats["fallbacks"] == 2
    assert stats["fallback_reasons"] == {"shape_miss": 2}
    assert "prefill_graph_full_dispatches" not in stats


def test_graph_stats_use_the_bound_provider_without_reading_native_internals() -> None:
    class Runner(SimpleNamespace):
        _vllm_gr_config = SimpleNamespace(beam_graph_enabled=True, beam_max_width=4)

        @property
        def cudagraph_manager(self) -> object:
            pytest.fail("shared RPC inspected an unselected backend")

        @property
        def _gr_npu_beam_graph_shapes(self) -> object:
            pytest.fail("shared RPC inspected an unselected backend")

        @property
        def _beam_graph_keys(self) -> object:
            pytest.fail("shared RPC inspected a native worker instead of the provider")

    snapshot = {"implementation": "example.graph", "runtime_fields": {"vendor_replays": 11}}
    runner = Runner()
    runner._gr_backend_binding = SimpleNamespace(graph=SimpleNamespace(stats=lambda: snapshot))
    stats = get_beam_graph_runtime_stats(SimpleNamespace(model_runner=runner))
    assert stats["vendor_replays"] == 11
    assert stats["graph_service"] == snapshot
