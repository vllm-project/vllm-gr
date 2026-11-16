# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam interface selection shared by request admission and Workers."""

from vllm_gr.config.models import BeamExecutionMode, GRConfig


def require_beam_execution_mode(config: object, expected: BeamExecutionMode) -> None:
    actual = GRConfig.from_vllm_config(config).beam_execution_mode
    if actual is not expected:
        actual_name = actual.value if actual is not None else "unset"
        raise ValueError(
            f"Beam interface requires beam.execution_mode='{expected.value}'; "
            f"this engine was initialized with '{actual_name}'. "
            "Select the execution mode before engine initialization."
        )
