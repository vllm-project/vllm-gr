# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Static regressions checked with the complete production CUDA boundary.

Each narrow ignore is an expected diagnostic: --warn-unused-ignores makes the
check fail if a weakened contract stops detecting that error. Pytest does not
import or execute these deliberately incompatible implementations.
"""

from typing import Any, ClassVar

from vllm_gr.v1.worker.gpu.stage_hooks import CudaStageHooks, CudaStageHooksType
from vllm_gr.v1.worker.gpu.v1.beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.gpu.v2.beam_stage_runner_v2 import CudaStageHooksV2

v1: CudaStageHooksType = CudaStageHooksV1
v2: CudaStageHooksType = CudaStageHooksV2


class WrongReturn(CudaStageHooks):
    def decode_block_ids(self, session_id: str) -> str:  # type: ignore[override]
        return "not block IDs"


class WrongFinishReturn(CudaStageHooks):
    def finish_native_output(self) -> int:  # type: ignore[override]
        return 1


class WrongConstructor(CudaStageHooksV1):
    def __init__(self, worker: Any, required_argument: int) -> None:
        super().__init__(worker)


invalid_constructor: CudaStageHooksType = WrongConstructor  # type: ignore[assignment]


class WrongPolicy(CudaStageHooks):
    persistent_graph_install: ClassVar[str] = "not a bool"  # type: ignore[assignment]
