# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Engine-level data structures for generative recommendation.

Defines GRFeaturePayload, HSTUFeaturesPayload (aliased as
GRFeaturesPayload), and GREngineCoreRequest for transporting
recommendation features through vLLM's EngineCore pipeline.

Serialization note:
    EngineCoreRequest uses array_like=True (positional msgspec encoding).
    GREngineCoreRequest appends gr_features as the last field.  The
    monkey-patch in patch.py replaces EngineCoreRequest in both source
    and consumer modules so that:
      - The client-side MsgpackEncoder serializes gr_features.
      - The subprocess-side MsgpackDecoder(EngineCoreRequest) — which
        now resolves to GREngineCoreRequest — deserializes it correctly.
"""

from typing import Any
import logging

import msgspec

from vllm.v1.engine import EngineCoreRequest

logger = logging.getLogger("vllm_gr.engine")


class GRFeaturePayload(msgspec.Struct):
    """Variable-length feature group — keys/values/lengths triple.

    A reusable structural unit for representing a single feature group
    (e.g. UIH, candidates, context).  Higher-level payload types compose
    one or more GRFeaturePayload triples.
    """

    keys: list[str]
    values_per_key: list[list[int]]
    lengths_per_key: list[int]


class HSTUFeaturesPayload(msgspec.Struct):
    """HSTU model payload — UIH + candidates + context feature groups.

    Uses GRFeaturePayload internally for feature group encapsulation.
    Field order and types are identical to the former GRFeaturesPayload
    for msgspec serialization compatibility.
    """

    uih_keys: list[str]
    uih_values_per_key: list[list[int]]
    uih_lengths_per_key: list[int]

    cand_keys: list[str]
    cand_values_per_key: list[list[int]]
    cand_lengths_per_key: list[int]

    ctx_keys: list[str] = []
    ctx_values_per_key: list[list[int]] = []
    full_uih_keys: list[str] = []

    user_id: str = ""
    original_candidates: list[dict[str, Any]] = []


# Backward-compatible alias — all existing code uses GRFeaturesPayload.
GRFeaturesPayload = HSTUFeaturesPayload


class GREngineCoreRequest(EngineCoreRequest):
    """EngineCoreRequest extended with a GRFeaturesPayload field.

    Appended as the last positional element in the array_like serialization.
    The monkey patch in patch.py replaces EngineCoreRequest globally so
    MsgpackDecoder uses this subclass schema when decoding.
    """

    gr_features: GRFeaturesPayload | None = None


__all__ = [
    "GRFeaturePayload",
    "HSTUFeaturesPayload",
    "GRFeaturesPayload",
    "GREngineCoreRequest",
]
