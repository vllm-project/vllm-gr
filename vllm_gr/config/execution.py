# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam interface selection shared by request admission and Workers."""

from vllm_gr.config.models import BeamExecutionMode, GRConfig


def require_beam_execution_mode(config: object, expected: BeamExecutionMode) -> None:
    actual = GRConfig.from_vllm_config(config).beam_execution_mode
    if actual is not expected:
        raise ValueError(
            f"Beam interface requires beam.execution_mode='{expected.value}'; "
            f"this engine was initialized with '{actual.value}'. "
            "Select the execution mode before engine initialization."
        )
