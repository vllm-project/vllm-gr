# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU-specific prompt type.

HSTUPrompt extends GRPromptBase with HSTU's three-segment input fields
(user_history, candidates, context) plus the optional full_uih_keys.
"""

from typing_extensions import NotRequired

from vllm_gr.inputs.base import GRPromptBase


class HSTUPrompt(GRPromptBase):
    """HSTU model prompt with optional full-UIH feature specification.

    Extends GRPromptBase with fields specific to HSTU's feature processing
    pipeline.

    NOTE: This type is NOT used as a Pydantic model field.  It serves as
    a structural contract for the HSTU adapter internals.  The user-facing
    API continues to use GRPrompt from inputs/data.py.
    """

    user_history: list[dict[str, int]]
    candidates: list[dict[str, int]]
    context: NotRequired[dict[str, int]]
    full_uih_keys: NotRequired[list[str]]
