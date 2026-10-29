# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks at the boundaries where GR consumes native runner return tuples."""

from typing import Any


def validate_return_tuple(
    value: object,
    *,
    method: str,
    branch: str,
    expected_items: int,
    patch_id: str,
    allow_tail: bool = False,
) -> tuple[Any, ...]:
    """Keep positional consumers strict; allow tails only for opaque passthrough."""
    if isinstance(value, tuple):
        count = len(value)
        if count == expected_items or (allow_tail and count > expected_items):
            return value
        actual = f"tuple with {count} items"
    else:
        actual = type(value).__name__
    comparison = "at least" if allow_tail else "exactly"
    raise RuntimeError(
        f"{method} [{branch}] returned {actual}; expected {comparison} "
        f"{expected_items} items in a tuple with the validated positional layout. "
        f"See docs/runtime_patch_inventory.md ({patch_id})."
    )
