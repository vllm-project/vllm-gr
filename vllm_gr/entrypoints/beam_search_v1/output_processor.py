# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert terminal beam results to frontend output objects."""

from typing import Any

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchEngineError
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.v1.beam.batch_contracts import BeamBatchResult


def _raise_failure(result: BeamBatchResult) -> None:
    if result.failure is not None:
        raise BeamSearchEngineError(result.failure)


class BeamSearchOutputProcessor:
    """A5 extension point for offline and online terminal output conversion."""

    def process_outputs(self, call: OfflineBeamSearchRequest, result: BeamBatchResult) -> list[Any]:
        _raise_failure(result)
        raise NotImplementedError("A5 must convert BeamBatchResult to BeamSearchOutput")

    def process_online_outputs(
        self, call: OnlineBeamSearchRequest, result: BeamBatchResult
    ) -> list[Any]:
        _raise_failure(result)
        raise NotImplementedError("A5 must convert BeamBatchResult to RequestOutput")
