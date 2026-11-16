# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared Beam Forward capture and replay for both execution interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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

from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error


class BeamDecodeCaptureUnsupported(RuntimeError):
    """Capture was refused; startup must still verify cleanup and device health."""


@dataclass(frozen=True, slots=True)
class BeamDecodeGraphSignature:
    batch_size: int
    beam_width: int
    num_model_rows: int
    prefix_bucket: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.prefix_bucket) is not int
            or self.prefix_bucket < 0
            or type(self.batch_size) is not int
            or type(self.beam_width) is not int
            or self.batch_size < 1
            or self.beam_width < 1
            or self.num_model_rows != self.batch_size * self.beam_width
        ):
            raise ValueError("invalid Beam Decode graph signature")


class BeamDecodeGraph:
    """One engine's fixed-width graph, with a Beam-only native graph registry.

    Each interface prepares its own stable input/metadata buffers. The graph
    retains their capture-time views and Beam context; replay updates values,
    never addresses. LM head remains with the caller, as in legacy decode.
    """

    def __init__(self, worker: Any, width: int) -> None:
        self.worker = worker
        self.signature = BeamDecodeGraphSignature(1, width, width)
        self.descriptor = BatchDescriptor(num_tokens=width, uniform=True)
        self.context: Any = None
        self.capture_inputs: Any = None
        self.capture_attention: Any = None
        self._bindings: dict[BeamDecodeGraphSignature, tuple[Any, Any, Any]] = {}
        self.capturing = False
        self.original_model = worker.model
        runnable = worker.model
        # Native FULL graphs may already contain the same descriptor for
        # ordinary attention. Beam captures must never replay those entries.
        while isinstance(runnable, CUDAGraphWrapper):
            runnable = runnable.unwrap()
        self.wrapper = _BeamDecodeModel(self, runnable)
        self._registries: dict[BeamDecodeGraphSignature, dict[Any, Any]] = {
            self.signature: self.wrapper.concrete_cudagraph_entries
        }
        # The caller chooses whether this wrapper remains the native model
        # entry (V1) or is installed only for the capture window (V2).
        self.installed = False

    @property
    def ready(self) -> bool:
        entry = self.wrapper.concrete_cudagraph_entries.get(self.descriptor)
        return entry is not None and entry.cudagraph is not None

    def select(self, batch_size: int, beam_width: int, prefix_bucket: int = 0) -> BeamDecodeGraph:
        signature = BeamDecodeGraphSignature(
            batch_size, beam_width, batch_size * beam_width, prefix_bucket
        )
        self.signature = signature
        self.wrapper.concrete_cudagraph_entries = self._registries.setdefault(signature, {})
        self.descriptor = BatchDescriptor(num_tokens=signature.num_model_rows, uniform=True)
        return self

    def discard_geometry(self, batch_size: int, beam_width: int) -> None:
        """New input storage invalidates every length bucket for this geometry."""
        for signature in tuple(self._registries):
            if (signature.batch_size, signature.beam_width) == (batch_size, beam_width):
                self.select(batch_size, beam_width, signature.prefix_bucket).discard_selected()
        self.select(batch_size, beam_width)

    def discard_selected(self) -> None:
        """Drop a capture whose fixed input addresses cannot be reused."""
        self.wrapper.concrete_cudagraph_entries.pop(self.descriptor, None)
        self._bindings.pop(self.signature, None)
        self.capture_inputs = self.capture_attention = None
        self.worker._beam_graph_keys = {signature.num_model_rows for signature in self._bindings}

    def _check_context(self) -> None:
        if self.context is not self.worker._beam_context:
            raise RuntimeError("Beam graph context changed after capture")

    def install(self) -> None:
        """Take over ``worker.model`` so Beam FULL forwards route here."""
        if self.installed:
            return
        if self.worker.model is not self.original_model:
            raise RuntimeError("Beam graph install conflicts with a foreign model owner")
        self.worker.model = self.wrapper
        self.installed = True

    def uninstall(self) -> None:
        """Restore the native model; replay/eager decode no longer need it."""
        if not self.installed:
            return
        if self.worker.model is not self.wrapper:
            raise RuntimeError("Beam graph uninstall found an unexpected model owner")
        self.worker.model = self.original_model
        self.installed = False

    def clear(self) -> None:
        """Discard graph entries and captured views after an aborted startup."""
        for entries in self._registries.values():
            entries.clear()
        self._registries.clear()
        self.wrapper.clear_graphs()
        self._bindings.clear()
        self.capture_inputs = self.capture_attention = None
        self.worker._beam_graph_keys = set()

    @torch.inference_mode()
    def capture(self, forward: Callable[[bool], Any], *, persistent_install: bool) -> None:
        """Run the caller's input adapter twice eager, then capture Forward."""
        if self.ready:
            self._check_context()
            return
        # Keep Beam captures separate from vLLM's shared pool. A pool whose
        # last graph was destroyed can remain in the allocator while captured
        # outputs are still alive; attempting to reuse it then fails. Reuse
        # our pool only while another Beam graph still owns it.
        if not any(
            getattr(entry, "cudagraph", None) is not None
            for entries in self._registries.values()
            for entry in entries.values()
        ):
            self.wrapper.graph_pool = torch.cuda.graph_pool_handle()
        # V1 replays beam FULL graphs through the native execute_model FULL
        # branch calling ``worker.model`` — the wrapper must STAY installed
        # for the engine's lifetime (uninstalled only by shutdown).  V2
        # (beam_search_v1 stage path) replays explicitly through
        # ``BeamDecodeGraph.replay()`` and never routes via ``worker.model``,
        # so the wrapper is scoped to the capture window there.
        if persistent_install:
            self.install()
        self.context = self.worker._beam_context
        stream = torch.cuda.current_stream(self.worker.device)
        side = torch.cuda.Stream(device=self.worker.device)  # type: ignore[no-untyped-call]
        side.wait_stream(stream)
        set_cudagraph_capturing_enabled(True)
        self.capturing = True
        try:
            with torch.cuda.stream(side):
                if not persistent_install:
                    self.install()
                try:
                    forward(False)
                    forward(False)
                    try:
                        forward(True)
                    except RuntimeError as error:
                        # Only the capture call can request V2's optional
                        # fallback. Warmup, synchronization and restoration
                        # failures must propagate, regardless of their text.
                        if not persistent_install and is_recoverable_capture_error(error):
                            raise BeamDecodeCaptureUnsupported(str(error)) from error
                        raise
                finally:
                    if not persistent_install:
                        self.uninstall()
            stream.wait_stream(side)
            stream.synchronize()
            if not self.ready:
                raise RuntimeError("Beam Forward did not register its startup CUDA graph")
            self._bindings[self.signature] = (
                self.context,
                self.capture_inputs,
                self.capture_attention,
            )
            self.worker._beam_graph_keys = {
                signature.num_model_rows for signature in self._bindings
            }
        except Exception:
            if self._bindings:
                # A failed capture of another batch size must not evict graphs
                # already bound to a successfully captured signature.
                self.wrapper.concrete_cudagraph_entries.pop(self.descriptor, None)
                self.capture_inputs = self.capture_attention = None
            else:
                self.clear()
            raise
        finally:
            self.capturing = False
            set_cudagraph_capturing_enabled(False)

    def replay(self) -> torch.Tensor:
        binding = self._bindings.get(self.signature)
        if binding is None:
            raise RuntimeError("Beam graph miss for the exact batch signature")
        self.context, self.capture_inputs, self.capture_attention = binding
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
        if context.batch_descriptor.num_tokens != self.signature.num_model_rows:
            raise RuntimeError("Beam FULL dispatch changed physical row geometry")
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


def get_beam_decode_graph(
    worker: Any,
    *,
    batch_size: int = 1,
    beam_width: int | None = None,
    graph_type: type[BeamDecodeGraph] = BeamDecodeGraph,
) -> BeamDecodeGraph:
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    graph = getattr(worker, "_beam_decode_graph", None)
    if graph is None:
        graph = graph_type(worker, int(_get_gr_config(worker).beam_max_width))
        worker._beam_decode_graph = graph
    width = int(_get_gr_config(worker).beam_max_width) if beam_width is None else beam_width
    return graph.select(batch_size, width)
