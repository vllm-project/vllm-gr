# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from typing import Any


def completion_prompt(
    prompt: str,
    sample: dict[str, Any],
    tokenizer: Any,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Keep the rendered workload, applying explicit template overrides locally."""
    if not overrides:
        return prompt
    messages = sample.get("chat_messages")
    if messages is None:
        raise ValueError(
            "Template overrides require source messages; set thinking before prompt normalization"
        )
    if tokenizer is None:
        raise ValueError("Template overrides require the model tokenizer")
    options = {**(sample.get("chat_template_kwargs") or {}), **overrides}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **options
    )
