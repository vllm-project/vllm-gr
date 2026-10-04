# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sid-aware prompt-tokenization fast path for recommender chat prompts.

The OneRec-style prompt is thousands of ``<s_a_N><s_b_N><s_c_N>`` sid tokens
interleaved with spaces.  Those sids are added tokens, so the Rust backend's
added-token matching dominates the prompt encode (measured ~7-13 ms for a
~3.9K-token prompt, most of the API handler's input-processing time).
``fast_sid_encode`` splits the text on the sid regex and batch-encodes only
the distinct plain-text runs (typically the label and one space), mapping the
sids arithmetically through their contiguous added-token id ranges.  Output
ids are byte-identical to the plain encode, and every non-applicable shape
(no sids, non-contiguous ids, other added tokens, real truncation) falls back
to the original ``tokenizer.encode``.
"""


def patch_fast_sid_encode() -> None:
    """Install the sid fast path on the renderer's tokenize entrypoints."""
    from functools import wraps

    from vllm.renderers.base import BaseRenderer

    from vllm_gr.tokenizer_fast import fast_sid_encode

    if getattr(BaseRenderer, "_vllm_gr_sid_encode_patch", False) is True:
        return

    _orig_sync = BaseRenderer._tokenize_prompt
    _orig_async = BaseRenderer._tokenize_prompt_async

    def _try_fast(self, prompt, params):
        text = prompt.get("prompt") if isinstance(prompt, dict) else None
        if not isinstance(text, str) or "<" not in text:
            return None
        kwargs = params.get_encode_kwargs()
        return fast_sid_encode(
            self.get_tokenizer(),
            text,
            add_special_tokens=kwargs.get("add_special_tokens", False),
            truncation=kwargs.get("truncation", False),
            max_length=kwargs.get("max_length"),
        )

    @wraps(_orig_sync)
    def _patched_sync(self, prompt, params):
        ids = _try_fast(self, prompt, params)
        if ids is not None:
            from vllm.inputs.llm import TokensPrompt

            return TokensPrompt(prompt_token_ids=ids, **prompt)
        return _orig_sync(self, prompt, params)

    @wraps(_orig_async)
    async def _patched_async(self, prompt, params):
        ids = _try_fast(self, prompt, params)
        if ids is not None:
            from vllm.inputs.llm import TokensPrompt

            return TokensPrompt(prompt_token_ids=ids, **prompt)
        return await _orig_async(self, prompt, params)

    BaseRenderer._tokenize_prompt = _patched_sync
    BaseRenderer._tokenize_prompt_async = _patched_async
    BaseRenderer._vllm_gr_sid_encode_patch = True
