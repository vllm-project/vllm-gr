# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Beam Forward capture and replay for both execution interfaces."""

from collections.abc import Callable
from typing import Any, cast

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)


class BeamDecodeGraph:
    """One engine's fixed-width graph, with a Beam-only native graph registry.

    Each interface prepares its own stable input/metadata buffers. The graph
    retains their capture-time views and Beam context; replay updates values,
    never addresses. LM head remains with the caller, as in legacy decode.
    """

    def __init__(self, worker: Any, width: int) -> None:
        self.worker = worker
        self.width = width
        self.descriptor = BatchDescriptor(num_tokens=width, uniform=True)
        self.context: Any = None
        self.capture_inputs: Any = None
        self.capture_attention: Any = None
        self.capturing = False
        self.original_model = worker.model
        runnable = worker.model
        # Native FULL graphs may already contain the same descriptor for
        # ordinary attention. Beam captures must never replay those entries.
        while isinstance(runnable, CUDAGraphWrapper):
            runnable = runnable.unwrap()
        self.wrapper = _BeamDecodeModel(self, runnable)
        worker.model = self.wrapper

    @property
    def ready(self) -> bool:
        entry = self.wrapper.concrete_cudagraph_entries.get(self.descriptor)
        return entry is not None and entry.cudagraph is not None

    def _check_context(self) -> None:
        if self.context is not self.worker._beam_context:
            raise RuntimeError("Beam graph context changed after capture")

    @torch.inference_mode()
    def capture(self, forward: Callable[[bool], Any]) -> None:
        """Run the caller's input adapter twice eager, then capture Forward."""
        if self.ready:
            self._check_context()
            return
        self.context = self.worker._beam_context
        stream = torch.cuda.current_stream(self.worker.device)
        side = torch.cuda.Stream(device=self.worker.device)  # type: ignore[no-untyped-call]
        side.wait_stream(stream)
        set_cudagraph_capturing_enabled(True)
        self.capturing = True
        try:
            with torch.cuda.stream(side):
                forward(False)
                forward(False)
                forward(True)
            stream.wait_stream(side)
            stream.synchronize()
            if not self.ready:
                raise RuntimeError("Beam Forward did not register its startup CUDA graph")
            self.worker._beam_graph_keys = {self.width}
        except Exception:
            self.wrapper.clear_graphs()
            self.capture_inputs = self.capture_attention = None
            raise
        finally:
            self.capturing = False
            set_cudagraph_capturing_enabled(False)

    def replay(self) -> torch.Tensor:
        self._check_context()
        if not self.ready:
            raise RuntimeError("Beam graph replay requires successful startup capture")
        entry = self.wrapper.concrete_cudagraph_entries[self.descriptor]
        assert entry.cudagraph is not None
        entry.cudagraph.replay()
        return cast(torch.Tensor, entry.output)

    def model_forward(self, *args: Any, **kwargs: Any) -> Any:
        context = get_forward_context()
        if context.batch_descriptor != self.descriptor:
            raise RuntimeError("Beam FULL dispatch does not match the startup graph width")
        self._check_context()
        if self.capturing:
            self.capture_inputs = (args, kwargs)
            self.capture_attention = context.attn_metadata
        elif not self.ready:
            raise RuntimeError("Beam FULL dispatch requires successful startup capture")
        return CUDAGraphWrapper.__call__(self.wrapper, *args, **kwargs)


class _BeamDecodeModel(CUDAGraphWrapper):
    """Route only Beam FULL forwards into the shared component.

    Other forwards keep the original native model wrapper and its registry.
    This also permits Beam graphs when native compilation/cudagraphs are off.
    """

    def __init__(self, graph: BeamDecodeGraph, runnable: Any) -> None:
        self.graph = graph
        # Preserve native get_model()/unwrap() and capability discovery.
        super().__init__(
            runnable,
            graph.worker.vllm_config,
            CUDAGraphMode.FULL,
            CUDAGraphOptions(weak_ref_output=False),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph.original_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if is_forward_context_available():
            context = get_forward_context()
            metadata = context.attn_metadata
            if (
                context.cudagraph_runtime_mode is CUDAGraphMode.FULL
                and isinstance(metadata, dict)
                and any(getattr(md, "is_beam_decode", False) for md in metadata.values())
            ):
                return self.graph.model_forward(*args, **kwargs)
        return self.graph.original_model(*args, **kwargs)


def get_beam_decode_graph(worker: Any) -> BeamDecodeGraph:
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    graph = getattr(worker, "_beam_decode_graph", None)
    if graph is None:
        graph = BeamDecodeGraph(worker, int(_get_gr_config(worker).beam_max_width))
        worker._beam_decode_graph = graph
    return graph
