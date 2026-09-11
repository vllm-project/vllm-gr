# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request admission and EngineCore errors exposed to beam-search frontends."""

from vllm_gr.v1.beam.batch_contracts import BeamBatchFailure


class BeamSearchRequestError(ValueError):
    """A request that cannot be admitted as a B=1 GR session."""


class BeamSearchEngineError(RuntimeError):
    """A typed Frontend view of an EngineCore session failure."""

    def __init__(self, failure: BeamBatchFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure
