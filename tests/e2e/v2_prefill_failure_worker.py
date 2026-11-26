# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Test-only worker import injects a clean FULL capture refusal at startup.

No CUDA stream is invalidated here. Real invalidation is covered in an isolated
kernel test, since torch 2.11 cannot safely recover its allocator/RNG state.
"""

from typing import Any

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

from tests.e2e.test_modelrunner_v2_stage_parity_st import OneRecV2ParityWorkerExtension
from vllm_gr.v1.worker.cuda.v2.gpu_prefill_graph_runner_v2 import (
    GRPrefillCudaGraphManager,
)

_capture = ModelCudaGraphManager.capture


def _refuse_full(self: ModelCudaGraphManager, *args: Any, **kwargs: Any) -> Any:
    if isinstance(self, GRPrefillCudaGraphManager) and CUDAGraphMode.FULL in self._capture_descs:
        raise RuntimeError("operation not supported when stream is capturing (ST injection)")
    return _capture(self, *args, **kwargs)


ModelCudaGraphManager.capture = _refuse_full


class FailedFullWorkerExtension(OneRecV2ParityWorkerExtension):
    pass
