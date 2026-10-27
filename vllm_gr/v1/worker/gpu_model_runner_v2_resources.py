# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned input storage and graph bindings for ModelRunner V2."""

from collections.abc import Callable
from typing import Any

import torch

from vllm_gr.v1.worker.beam_decode_graph import (
    BeamDecodeGraph,
    close_beam_decode_runtime,
)
from vllm_gr.v1.worker.gr_inputs import GRInputBuffers


class V2InputBuffers(GRInputBuffers):
    """Keep model inputs at fixed addresses across V2 batch-size changes."""

    def __init__(
        self,
        width: int,
        max_model_len: int,
        block_size: int,
        device: torch.device | str,
        *,
        max_batch: int,
    ) -> None:
        super().__init__(width, max_model_len, block_size, device, max_batch=max_batch)
        rows = max_batch * width
        self.input_ids = torch.empty(rows, dtype=torch.int32, device=device)
        self.positions = torch.empty(rows, dtype=torch.int64, device=device)
        self.slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
        self.execution_mask = torch.empty(rows, dtype=torch.bool, device=device)


class V2BeamDecodeGraph(BeamDecodeGraph):
    """Reject replay after replacement of V2's captured storage or model."""

    def __init__(self, worker: Any, width: int) -> None:
        super().__init__(worker, width)
        self.input_buffers: Any = None

    def _check_context(self) -> None:
        super()._check_context()
        if self.input_buffers is not getattr(self.worker, "_gr_input_buffers", None):
            raise RuntimeError("Beam graph input storage changed after capture")

    def clear(self) -> None:
        super().clear()
        self.input_buffers = None

    def capture(self, forward: Callable[[bool], Any], *, persistent_install: bool) -> None:
        if not self.ready:
            self.input_buffers = getattr(self.worker, "_gr_input_buffers", None)
        super().capture(forward, persistent_install=persistent_install)


def close_v2_runtime(runner: Any) -> None:
    """Invalidate V2 captures before replacing or shutting down the model."""
    from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
        uninstall_prefill_graph_manager_v2,
    )

    uninstall_prefill_graph_manager_v2(runner)
    close_beam_decode_runtime(runner)
