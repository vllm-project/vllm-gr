# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only startup refusal; real CUDA invalidation has an isolated kernel test."""

from typing import Any

from tests.st.test_modelrunner_v2_stage_parity_st import OneRecV2ParityWorkerExtension
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks

_forward_hidden = V2StageExecutionHooks.forward_hidden


def _refuse_capture(
    self: Any, input_ids: Any, positions: Any, attention: Any, slot_mappings: Any, capture: bool
) -> Any:
    if capture:
        self.worker._test_decode_capture_refusals = (
            getattr(self.worker, "_test_decode_capture_refusals", 0) + 1
        )
        raise RuntimeError("operation not supported when stream is capturing (ST injection)")
    return _forward_hidden(self, input_ids, positions, attention, slot_mappings, capture)


V2StageExecutionHooks.forward_hidden = _refuse_capture


class FailedDecodeWorkerExtension(OneRecV2ParityWorkerExtension):  # type: ignore[misc]
    def onerec_v2_parity_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = super().onerec_v2_parity_snapshot()
        if snapshot.get("retired"):
            # One refusal in temporary profiling and one in real startup.
            # Serving requests must never retry capture after either fallback.
            refusals = self.model_runner._test_decode_capture_refusals
            assert refusals == 2
            snapshot["decode_capture_refusals"] = refusals
        return snapshot
