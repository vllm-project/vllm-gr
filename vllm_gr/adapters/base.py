# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRModelAdapter — base class for model-specific recommendation adapters.

Each model family (HSTU, SASRec, etc.) subclasses ``GRModelAdapter`` to
provide:

- ``parse_prompt``      — extract features from a GRPrompt
- ``build_payload``     — serialize features into a msgspec-safe payload
- ``decode_payload``    — deserialize payload back into feature tensors
- ``build_model_kwargs``— build model-specific kwargs (e.g. KJT for HSTU)
- ``process_output``    — convert raw pooling output into GRRequestOutput (directly, not via GRModelOutput)

Optional methods provide prefix-cache support, validation, and metadata
extraction.  The default implementations are no-ops so adapters only
override what they need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_gr.config.gr_config import GRConfig
    from vllm_gr.inputs.base import GRParsedInput


class GRModelAdapter:
    """Base adapter for GR model inference.

    Subclasses must set ``model_type`` and ``payload_type`` class attributes
    and implement the five core methods.
    """

    model_type: str = ""
    """Unique model family identifier (e.g. 'hstu', 'sasrec')."""

    payload_type: type = object
    """The msgspec-serializable payload class used by this adapter."""

    def __init__(self, vllm_config: "VllmConfig", gr_config: "GRConfig"):
        self._vllm_config = vllm_config
        self._hf_config = vllm_config.model_config.hf_config
        self._gr_config = gr_config

    # ------------------------------------------------------------------
    # Core methods (subclasses must implement)
    # ------------------------------------------------------------------

    def parse_prompt(self, prompt: dict) -> "GRParsedInput":
        """Extract feature data from a GRPrompt dict.

        Returns a GRParsedInput instance whose ``raw_features`` dict
        is consumed by ``build_payload`` and the optional helpers.
        """
        raise NotImplementedError

    def build_payload(self, parsed: "GRParsedInput") -> Any:
        """Build a msgspec-serializable payload from parsed features."""
        raise NotImplementedError

    def decode_payload(self, payload: Any) -> dict[str, Any]:
        """Decode a serialized payload into per-feature tensors/metadata."""
        raise NotImplementedError

    def build_model_kwargs(
        self, batch_features: list[tuple[str, dict[str, Any]]], device
    ) -> dict[str, Any]:
        """Build model-specific kwargs for the forward pass.

        ``batch_features`` is a list of ``(req_id, decoded_features)`` pairs
        for all requests in the current batch.
        """
        raise NotImplementedError

    def process_output(self, raw_output: Any, request_meta: dict) -> Any:
        """Convert raw pooling output into a GRRequestOutput.

        Currently returns GRRequestOutput directly.  In a future
        "output contract unification" stage, adapters will return
        GRModelOutput subclasses and the engine will wrap them.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional methods (default implementations)
    # ------------------------------------------------------------------

    def validate_prompt(self, prompt: dict) -> None:
        """Validate a prompt dict.  Raise ``ValueError`` / ``TypeError`` on
        invalid input.  Default is a no-op."""

    def build_request_lengths(self, features: list[dict]) -> tuple | None:
        """Return ``(num_targets, history_lengths, contextual_seq_len)`` or
        ``None`` if the model does not use request-length metadata."""
        return None

    def build_prompt_token_ids(self, parsed: "GRParsedInput") -> list[int] | None:
        """Build synthetic prompt token IDs for prefix-cache hashing.

        Return ``None`` to use a default single-token placeholder.
        """
        return None

    def build_cache_salt(self, parsed: "GRParsedInput") -> str | None:
        """Build a cache-salt string for prefix-cache identity.

        Return ``None`` to skip prefix-cache salting.
        """
        return None

    def extract_request_meta(self, payload: Any) -> dict:
        """Extract per-request metadata from a payload for output processing.

        The returned dict is stored in ``GRRequestState`` and passed to
        ``process_output`` when the request completes.
        """
        return {}
