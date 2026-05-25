# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Core data types for generative recommendation.

GRPrompt is the current user-facing HSTU-oriented input format for the
GRLLM API. ``is_gr_prompt()`` preserves backward compatibility with that
shape while also allowing future model-specific prompt shapes to opt into
the GR pipeline by setting a registered ``model_type``.
"""

import importlib
from typing import Any, Union, List, Dict

# Pydantic v2 requires typing_extensions.TypedDict on Python < 3.12.
# Keep using typing_extensions to stay compatible with the supported
# Python range (>=3.10,<3.14).
from typing_extensions import NotRequired, TypedDict


class GRPrompt(TypedDict):
    """Input prompt for generative recommendation.

    Feature keys in ``user_history`` and ``candidates`` are
    dataset-specific and must match the model's configured feature
    names.  For UIH the keys correspond to the left side of
    ``merge_uih_candidate_feature_mapping``; for candidates the keys
    correspond to the right side (the model-internal names).

    Example (MovieLens-1M)::

        prompt: GRPrompt = {
            "user_id": 123,
            "user_history": [
                {"movie_id": 1193, "movie_rating": 5,
                 "action_timestamp": 978300760, "item_weights": 1,
                 "dummy_watch_time": 0},
            ],
            "candidates": [
                {"item_movie_id": 1},
                {"item_movie_id": 48},
            ],
            "context": {"user_id": 1, "sex": 0, "age_group": 3},
        }
    """

    user_id: Union[str, int]

    user_history: List[Dict[str, int]]
    """UIH feature dicts keyed by model UIH feature names."""

    candidates: List[Dict[str, int]]
    """Candidate feature dicts keyed by model candidate feature names."""

    context: NotRequired[Dict[str, int]]
    """Contextual features (user-level attributes, one value each)."""


def is_gr_prompt(prompt: Any) -> bool:
    """Return True if *prompt* looks like a GR prompt.

    Matching follows a two-stage strategy:

    1. Model-type path: if ``model_type`` is present and corresponds to a
       registered adapter family, accept based on ``user_id + model_type``
       alone.  This allows future prompt shapes (e.g. SASRec) to enter the
       GR pipeline without reusing HSTU-specific fields.
    2. Generic path: if ``model_type`` is absent, accept any dict that
       contains ``user_id``.  Model-specific field validation (e.g.
       ``user_history`` / ``candidates`` for HSTU) is deferred to the
       adapter's ``validate_prompt()`` so that ``is_gr_prompt()`` itself
       stays model-family-agnostic.
    """
    if not isinstance(prompt, dict) or "user_id" not in prompt:
        return False

    # "model_type" present → registry-based dispatch only.
    # An invalid value (empty string, non-string, whitespace-only)
    # means the caller explicitly intended registry matching but gave
    # a bad value — reject rather than silently falling through to the
    # generic path.
    if "model_type" in prompt:
        model_type = prompt["model_type"]
        if not isinstance(model_type, str):
            return False
        normalized_model_type = model_type.strip().lower()
        if not normalized_model_type:
            return False
        _ensure_registered_model_family(normalized_model_type)
        from vllm_gr.adapters.registry import get_registered_families

        return normalized_model_type in get_registered_families()

    # No model_type → generic path: user_id alone is sufficient.
    # Adapter validate_prompt() handles stricter checks downstream.
    return True


def _ensure_registered_model_family(model_type: str) -> None:
    """Best-effort Layer-1 registration for prompt-type dispatch.

    Importing ``vllm_gr.adapters.<model_type>`` triggers the family
    package's ``register_family()`` call when such a package exists.

    Only suppresses ``ModuleNotFoundError`` when the missing module is
    the adapter package itself.  A broken transitive dependency (e.g.
    torchrec) propagates so the caller sees a real error, not a silent
    "unregistered family" failure.
    """
    from vllm_gr.adapters.registry import get_registered_families

    if model_type in get_registered_families():
        return

    target = f"vllm_gr.adapters.{model_type}"
    try:
        importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name == target:
            pass  # adapter package not installed — family stays unregistered
        else:
            raise
