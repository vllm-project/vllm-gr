# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compatibility fixes for the official openPangu Embedded checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger("vllm_gr.config.pangu_patch")

_PATCH_MARKER = "_vllm_gr_pangu_tied_embeddings_patch"
_PANGU_CAUSAL_LM_ARCHITECTURE = "PanguEmbeddedForCausalLM"


def _uses_pangu_tied_embeddings(model_config: Any) -> bool:
    """Return whether the model uses Pangu's intentional tied-weight layout."""
    hf_config = model_config.hf_config
    architectures = getattr(hf_config, "architectures", ()) or ()
    return bool(
        getattr(hf_config, "tie_word_embeddings", False)
        and _PANGU_CAUSAL_LM_ARCHITECTURE in architectures
    )


def _make_patched_maybe_untie(
    original: Callable[[Any], None],
) -> Callable[[Any], None]:
    """Keep the official Pangu zero-filled ``lm_head`` as a tied alias.

    openPangu Embedded checkpoints declare tied embeddings and carry a
    zero-filled ``lm_head.weight`` placeholder. vLLM 0.30 treats the presence
    of that placeholder as evidence of an untied head and consequently loads
    all-zero output logits. Keeping the model tied lets vLLM's weights loader
    load ``model.embed_tokens.weight`` once and skip its ``lm_head`` alias.
    """

    @wraps(original)
    def patched(model_config: Any) -> None:
        if _uses_pangu_tied_embeddings(model_config):
            logger.debug(
                "Keeping tied word embeddings for %s despite the checkpoint lm_head placeholder",
                _PANGU_CAUSAL_LM_ARCHITECTURE,
            )
            return
        original(model_config)

    setattr(patched, _PATCH_MARKER, True)
    return patched


def apply_pangu_tied_embeddings_patch() -> bool:
    """Install the Pangu loader compatibility wrapper once per process."""
    from vllm.config import ModelConfig

    current = ModelConfig.maybe_untie_word_embeddings
    if getattr(current, _PATCH_MARKER, False):
        return False

    ModelConfig.maybe_untie_word_embeddings = _make_patched_maybe_untie(current)
    return True
