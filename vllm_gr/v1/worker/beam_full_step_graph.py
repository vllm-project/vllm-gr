# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the projection/decision tail of a native graph-backed Decode step."""

from collections.abc import Callable, Sequence

import torch
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.model_executor.offloader.base import get_offloader


class FullBeamStepGraph:
    """Retain one step's executable and its private allocator pool.

    Warmup restores all mutable Beam/KV tensors after each execution. Capture
    preserves the admitted state; its first replay executes the step. Replays use updated
    device inputs at the same addresses and require no checkpoint copies.
    """

    @torch.inference_mode()  # type: ignore[misc]
    def __init__(
        self,
        operation: Callable[[], None],
        tensors: Sequence[torch.Tensor],
        *,
        pool: tuple[int, int] | None = None,
    ) -> None:
        self.graph = torch.cuda.CUDAGraph()
        self.bindings: tuple[object, ...] = ()
        self.logits: torch.Tensor | None = None
        pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.tensors = tuple(tensors)
        snapshots = tuple(tensor.clone() for tensor in tensors)
        current = torch.cuda.current_stream(tensors[0].device)
        stream = torch.cuda.Stream(device=tensors[0].device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    operation()
                    for tensor, snapshot in zip(tensors, snapshots):
                        tensor.copy_(snapshot)
                set_graph_pool_id(pool)
                get_offloader().sync_prev_onload()
                with torch.cuda.graph(self.graph, pool=pool, stream=stream):
                    operation()
                    get_offloader().join_after_forward()
        except Exception:
            current.wait_stream(stream)
            for tensor, snapshot in zip(tensors, snapshots):
                tensor.copy_(snapshot)
            raise
        current.wait_stream(stream)

    def replay(self) -> None:
        self.graph.replay()
