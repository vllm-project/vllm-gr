# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""LLM-as-Judge prompts used by the OpenOneRec evaluators.

The prompt texts are evaluation material from OpenOneRec and are kept as
data files next to this module; see README.md in this directory.
"""

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Return the prompt template ``<name>.txt`` exactly as upstream wrote it.

    Each upstream prompt ends with a single space after its final colon. The
    file stores the text with a trailing newline instead, so that whitespace
    hooks leave it alone, and the space is restored here.
    """
    text = (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return text.removesuffix("\n") + " "
