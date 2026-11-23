# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Request admission and EngineCore errors exposed to beam-search frontends."""

from vllm.exceptions import VLLMValidationError

from vllm_gr.v1.beam.batch_contracts import ADMISSION_REJECTED_CODE, BeamBatchFailure


class BeamSearchRequestError(ValueError):
    """A request that cannot be admitted as a GR batch session."""


class BeamSearchEngineError(RuntimeError):
    """A typed Frontend view of an EngineCore session failure."""

    def __init__(self, failure: BeamBatchFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def failure_exception(failure: BeamBatchFailure) -> Exception:
    """The exception a frontend raises for one terminal Beam failure.

    Most codes mean the EngineCore or its Worker broke, which is a server
    fault. An admission rejection is the engine answering the request: the
    session never started, the message names what the caller has to change,
    and resending the same request fails the same way. Reporting it as an
    engine error would file a client mistake under server faults.
    """
    if failure.code == ADMISSION_REJECTED_CODE:
        return VLLMValidationError(failure.message)
    return BeamSearchEngineError(failure)
