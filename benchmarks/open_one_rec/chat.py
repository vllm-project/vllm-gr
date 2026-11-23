# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from typing import Any

import regex as re

_CHAT_TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)


def messages_for_prompt(
    prompt: Any, chat_messages: list[dict[str, Any]] | None = None, *, parse_transcript: bool = True
) -> list[dict[str, Any]]:
    """Use the source chat or wrap an explicitly prepared workload as user content."""
    if chat_messages is not None:
        return chat_messages
    if parse_transcript and isinstance(prompt, str):
        # Compatibility for accuracy callers that supply only a rendered transcript.
        turns = _CHAT_TURN_RE.findall(prompt)
        parsed_chars = sum(len(role) + len(content) for role, content in turns)
        if turns and parsed_chars >= 0.5 * len(prompt):
            return [{"role": role, "content": content.strip("\n")} for role, content in turns]
    return [{"role": "user", "content": prompt}]


def text_from_message(message: dict[str, Any]) -> str:
    """Select one reasoning field and append the answer text once."""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return reasoning + (message.get("content") or "")
