# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep variable-width NPU Beam graphs separate from ordinary FULL graphs."""

from __future__ import annotations

from typing import Any

from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper


class NPUBeamGraphModel(ACLGraphWrapper):  # type: ignore[misc]
    """Route Beam forwards through a private native ACL graph registry.

    Ordinary attention can have the same token bucket as a narrow Beam
    batch. A separate wrapper prevents either path from replaying the other
    path's graph. Native model attributes remain accessible through the
    original model, including its LM head and logits processor.
    """

    def __init__(self, model: Any, vllm_config: Any) -> None:
        self.original_model = model
        runnable = model
        while isinstance(runnable, ACLGraphWrapper) and runnable.runtime_mode == CUDAGraphMode.FULL:
            runnable = runnable.unwrap()
        # Native get_model() recognizes ACLGraphWrapper and unwraps it for
        # generation capability discovery and model attribute inspection.
        super().__init__(runnable, vllm_config, CUDAGraphMode.FULL)

    @property
    def beam_wrapper(self) -> NPUBeamGraphModel:
        return self

    def unwrap(self) -> Any:
        if isinstance(self.original_model, ACLGraphWrapper):
            return self.original_model.unwrap()
        return self.original_model

    @staticmethod
    def _is_beam_forward() -> bool:
        from vllm_gr.v1.beam.context import BEAM_DATA_VAR

        info = BEAM_DATA_VAR.get()
        return isinstance(info, dict) and info.get("beam_session") is not None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_model, name)

    @property
    def cudagraph_wrapper(self) -> Any:
        if self._is_beam_forward():
            return self.beam_wrapper
        return self.original_model.cudagraph_wrapper

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_beam_forward():
            return super().__call__(*args, **kwargs)
        return self.original_model(*args, **kwargs)

    def graph_ready(self, num_tokens: int) -> bool:
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor

        descriptor = BatchDescriptor(num_tokens=num_tokens, uniform=True)
        entry = self.beam_wrapper.concrete_aclgraph_entries.get(descriptor)
        return entry is not None and entry.aclgraph is not None


def install_npu_beam_graph_model(runner: Any) -> NPUBeamGraphModel:
    """Install once, retaining the native wrapper for ordinary requests."""
    current = getattr(runner, "_gr_npu_graph_model", None)
    if current is not None:
        if not isinstance(current, NPUBeamGraphModel):
            raise RuntimeError("invalid NPU Beam graph model installation")
        if runner.model is not current:
            raise RuntimeError("NPU Beam graph model changed after installation")
        return current
    model = NPUBeamGraphModel(runner.model, runner.vllm_config)
    runner._gr_npu_graph_model = model
    runner.model = model
    return model
