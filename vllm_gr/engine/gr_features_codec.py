# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Encode/decode GR feature payloads for cross-process transport.

``GREngineCoreRequest.gr_features`` is stored as ``bytes`` so each model
family can use its own msgspec Struct without sharing a single static type.
"""

from __future__ import annotations

import msgspec


def encode_gr_features_payload(payload: object) -> bytes:
    """Serialize a family-specific payload struct to msgpack bytes."""
    return msgspec.msgpack.encode(payload)


def decode_gr_features_payload(encoded: bytes, payload_type: type) -> object:
    """Deserialize msgpack bytes into the adapter's payload struct."""
    return msgspec.msgpack.decode(encoded, type=payload_type)
