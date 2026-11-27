# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared streaming head options and inference-worker CUDA allocator probes."""

from __future__ import annotations

import argparse
from importlib import import_module
from inspect import signature
from typing import Any


def add_streaming_head_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--streaming-constrained-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream constrained CUDA Decode logits by vocabulary tile when eligible.",
    )


def streaming_head_config(args: argparse.Namespace) -> dict[str, bool]:
    # The disabled setting is also compatible with older A/B runtimes.
    enabled = getattr(args, "streaming_constrained_head", False)
    return {"enable_streaming_constrained_head": True} if enabled else {}


def reset_inference_cuda_peak(llm: Any) -> bool:
    """Return whether the selected runtime can report inference memory."""
    module_name = "vllm_gr.v1.worker.gr_context"
    try:
        context = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return False
    # The common benchmark driver can also import an older runtime. Do not
    # silently read the frontend counters when that runtime lacks the RPC option.
    if "cuda_allocator" not in signature(context.get_beam_graph_runtime_stats).parameters:
        return False
    llm.llm_engine.collective_rpc(
        "get_gr_beam_graph_runtime_stats", kwargs={"cuda_allocator": "reset"}
    )
    return True


def read_inference_cuda_peak(llm: Any) -> dict[str, Any]:
    """Measure in actual inference workers, including a separate EngineCore."""
    replies = llm.llm_engine.collective_rpc(
        "get_gr_beam_graph_runtime_stats", kwargs={"cuda_allocator": "read"}
    )
    if not replies:
        raise RuntimeError("No workers reported CUDA allocator statistics")
    workers = [reply["cuda_allocator"] for reply in replies]
    return {
        "scope": "inference worker CUDA allocator; reset after warmup, read before profiling",
        "aggregation": "maximum across workers",
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in workers),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in workers),
        "workers": workers,
    }
