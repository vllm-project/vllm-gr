# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Engine-level data structures for generative recommendation.

Defines GREngineCoreRequest for transporting GR features through
vLLM's EngineCore pipeline.  Model-specific payload types live in
their respective adapter packages (e.g. vllm_gr.adapters.hstu.payload).

Serialization note:
    EngineCoreRequest uses array_like=True (positional msgspec encoding).
    GREngineCoreRequest appends gr_features as the last field.  The
    monkey-patch in patch.py replaces EngineCoreRequest in both source
    and consumer modules so that:
      - The client-side MsgpackEncoder serializes gr_features.
      - The subprocess-side MsgpackDecoder(EngineCoreRequest) — which
        now resolves to GREngineCoreRequest — deserializes it correctly.
"""

import logging

from vllm.v1.engine import EngineCoreRequest

logger = logging.getLogger("vllm_gr.engine")


class GREngineCoreRequest(EngineCoreRequest):
    """EngineCoreRequest extended with a type-erased GR features field.

    Appended as the last positional element in the array_like serialization.
    The monkey patch in patch.py replaces EngineCoreRequest globally so
    MsgpackDecoder uses this subclass schema when decoding.

    ``gr_features`` holds msgpack-encoded bytes; each adapter family chooses
    its own payload struct at encode/decode time (see ``gr_features_codec``).
    """

    gr_features: bytes | None = None


__all__ = [
    "GREngineCoreRequest",
]
