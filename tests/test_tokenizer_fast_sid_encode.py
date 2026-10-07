# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sid-aware prompt encode fast path."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from vllm_gr.tokenizer_fast import fast_sid_encode


class _FakeBackend:
    """Deterministic stub: each distinct text run encodes to one id."""

    def __init__(self, base_id: int = 1000) -> None:
        self.base_id = base_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> SimpleNamespace:
        prefix = [999999] if add_special_tokens else []
        return SimpleNamespace(ids=[*prefix, self.base_id + len(text)])

    def encode_batch(
        self,
        texts: Sequence[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[SimpleNamespace]:
        assert not is_pretokenized
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]


class _FakeTokenizer:
    def __init__(
        self, sid_counts: tuple[int, int, int] = (4, 4, 4), include_markers: bool = True
    ) -> None:
        self.added_tokens_encoder: dict[str, int] = {}
        bases = [5000, 6000, 7000]
        for group, (prefix, count) in enumerate(zip(("<s_a_", "<s_b_", "<s_c_"), sid_counts)):
            for i in range(count):
                self.added_tokens_encoder[f"{prefix}{i}>"] = bases[group] + i
        if include_markers:
            self.added_tokens_encoder["<|sid_begin|>"] = 176245
            self.added_tokens_encoder["<|sid_end|>"] = 176246
            self.added_tokens_encoder["<|im_start|>"] = 151644
            self.added_tokens_encoder["<|im_end|>"] = 151645
        self._tokenizer: _FakeBackend | None = _FakeBackend()


def test_happy_path_interleaves_runs_and_sids() -> None:
    tok = _FakeTokenizer()
    text = "ab<s_a_0>c<s_b_2><s_c_1>"
    ids = fast_sid_encode(tok, text)
    assert ids == [1002, 5000, 1001, 6002, 7001]


def test_adjacent_sids_and_trailing_text() -> None:
    tok = _FakeTokenizer()
    text = "<s_a_0><s_a_1>x"
    ids = fast_sid_encode(tok, text)
    assert ids == [5000, 5001, 1001]


def test_template_markers_are_mapped_via_dict() -> None:
    tok = _FakeTokenizer()
    text = "<|im_start|>system\n<s_a_3><|im_end|>\nuser\n"
    ids = fast_sid_encode(tok, text)
    assert ids is not None
    assert ids[0] == 151644
    assert 5003 in ids
    assert 151645 in ids
    assert ids[-1] == 1006


def test_non_contiguous_ids_use_dict_values() -> None:
    tok = _FakeTokenizer()
    tok.added_tokens_encoder["<s_a_3>"] = 99999
    ids = fast_sid_encode(tok, "<s_a_3>")
    assert ids == [99999]


def test_sparse_and_arbitrary_non_contiguous_lut() -> None:
    tok = _FakeTokenizer(sid_counts=(0, 0, 0))
    tok.added_tokens_encoder["<s_a_10>"] = 12345
    tok.added_tokens_encoder["<s_b_2>"] = 67890
    tok.added_tokens_encoder["<s_c_100>"] = 42
    ids = fast_sid_encode(tok, "<s_a_10> <s_b_2> <s_c_100>")
    assert ids == [12345, 1001, 67890, 1001, 42]


def test_plain_text_returns_none() -> None:
    tok = _FakeTokenizer()
    assert fast_sid_encode(tok, "no sids here") is None
    assert fast_sid_encode(tok, "") is None


def test_add_special_tokens_returns_none() -> None:
    tok = _FakeTokenizer()
    assert fast_sid_encode(tok, "<s_a_0>", add_special_tokens=True) is None


def test_angle_bracket_outside_added_tokens_returns_none() -> None:
    tok = _FakeTokenizer()
    assert fast_sid_encode(tok, "<s_a_0> <unknown_tag> <s_b_1>") is None


def test_added_token_without_angle_bracket_disables_path() -> None:
    tok = _FakeTokenizer()
    tok.added_tokens_encoder["plainmarker"] = 88888
    assert fast_sid_encode(tok, "<s_a_0>") is None


def test_no_added_tokens_returns_none() -> None:
    tok = _FakeTokenizer(sid_counts=(0, 0, 0))
    assert fast_sid_encode(tok, "<s_a_0>") is None


def test_truncation_beyond_max_length_returns_none() -> None:
    tok = _FakeTokenizer()
    text = "<s_a_0><s_a_1><s_a_2><s_a_3>"
    ids = fast_sid_encode(tok, text, truncation=True, max_length=4)
    assert ids == [5000, 5001, 5002, 5003]
    assert fast_sid_encode(tok, text, truncation=True, max_length=3) is None


def test_too_many_distinct_runs_returns_none() -> None:
    tok = _FakeTokenizer()
    text = "".join(f"t{i}<s_a_0>" for i in range(520))
    assert fast_sid_encode(tok, text) is None


def test_missing_rust_backend_returns_none() -> None:
    tok = _FakeTokenizer()
    tok._tokenizer = None
    assert fast_sid_encode(tok, "<s_a_0>") is None


def test_sid_digit_overflow_and_leading_zero_return_none() -> None:
    tok = _FakeTokenizer()
    # 4294967300 = 2^32 + 4. Must not overflow into index 4 (5004)
    assert fast_sid_encode(tok, "<s_a_4294967300>") is None
    # Leading-zero SID tag must not alias <s_a_1>
    assert fast_sid_encode(tok, "<s_a_01>") is None


def test_double_space_parity() -> None:
    tok = _FakeTokenizer()
    # "x  " is kept as a single text run rather than emitting two space_ids
    ids = fast_sid_encode(tok, "x  <s_a_0>")
    # Fake backend encodes text length: "x  " has length 3 -> base_id (1000) + 3 = 1003
    assert ids == [1003, 5000]


def test_rust_bpe_tokenizer_parity() -> None:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {
        " ": 0,
        "\u0120": 1,
        "H": 2,
        "i": 3,
        "s": 4,
        ":": 5,
        "x": 6,
        "\n": 7,
        "\u0120x": 8,
        ">": 9,
        ">\u0120": 10,
    }
    merges = [("\u0120", "x"), (">", "\u0120")]
    rust_tok = Tokenizer(BPE(vocab=vocab, merges=merges))
    rust_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    added_tokens = [
        AddedToken("<|im_start|>", special=True),
        AddedToken("<|im_end|>", special=True),
        AddedToken("<|sid_begin|>", special=False),
        AddedToken("<|sid_end|>", special=False),
        AddedToken("<s_a_0>", special=False),
        AddedToken("<s_a_1>", special=False),
        AddedToken("<s_b_2>", special=False),
        AddedToken("<s_c_3>", special=False),
    ]
    rust_tok.add_tokens(added_tokens)

    class _WrappedRustTokenizer:
        def __init__(self, backend: Tokenizer) -> None:
            self._tokenizer = backend
            self.added_tokens_encoder = {
                t.content: backend.token_to_id(t.content) for t in added_tokens
            }

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    tok = _WrappedRustTokenizer(rust_tok)
    samples = [
        "<|im_start|>His: <|sid_begin|><s_a_0><s_b_2><s_c_3><|sid_end|> <|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|><|im_end|>\n",
        "x <s_a_0>",
        "x  <s_a_0>",
        "xx> <s_a_0>",
        " <s_a_0>",
        "  <s_a_0>",
        "<s_a_0> <s_a_1>",
    ]
    for s in samples:
        fast_ids = fast_sid_encode(tok, s)
        plain_ids = tok.encode(s, add_special_tokens=False)
        assert fast_ids == plain_ids, f"Mismatch on {s!r}: {fast_ids} vs {plain_ids}"


def test_real_tokenizer_parity_if_available() -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("OpenOneRec/OneRec-1.7B", local_files_only=True)
    except Exception:  # noqa: BLE001
        return
    samples = [
        "<|im_start|>system\nYou are a recommender.<|im_end|>\n<|im_start|>user\nHistory: <|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> <|sid_begin|><s_a_10><s_b_20><s_c_30><|sid_end|><|im_end|>\n",
        "x  <s_a_0>",
        "xx> <s_a_0>",
        " <s_a_0>",
        "  <s_a_0>",
    ]
    for s in samples:
        fast_ids = fast_sid_encode(tok, s)
        plain_ids = tok.encode(s, add_special_tokens=False)
        assert fast_ids == plain_ids, f"Mismatch on {s}: {fast_ids} vs {plain_ids}"


def test_c_extension_is_built() -> None:
    from vllm_gr import tokenizer_fast

    assert tokenizer_fast._fast_sid_c is not None


def test_c_and_python_fallback_differential_parity(monkeypatch: object) -> None:
    import pytest

    from vllm_gr import tokenizer_fast

    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    assert tokenizer_fast._fast_sid_c is not None

    tok = _FakeTokenizer()
    samples = [
        "ab<s_a_0>c<s_b_2><s_c_1>",
        "<s_a_0><s_a_1>x",
        "<|im_start|>system\n<s_a_3><|im_end|>\nuser\n",
        "History: <|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> <|sid_begin|><s_a_0><s_b_1><s_c_2><|sid_end|>",
        "x <s_a_0>",
        "x  <s_a_0>",
        "xx> <s_a_0>",
        " <s_a_0>",
        "  <s_a_0>",
        "no sids here",
        "<s_a_0> <unknown_tag> <s_b_1>",
        "<s_a_4294967300>",
        "<s_a_01>",
    ]
    c_results = [fast_sid_encode(tok, s) for s in samples]

    monkeypatch.setattr(tokenizer_fast, "_fast_sid_c", None)
    py_results = [fast_sid_encode(tok, s) for s in samples]

    assert c_results == py_results


def test_renderer_patch_sync_and_async(monkeypatch: object) -> None:
    import asyncio

    import pytest
    from vllm.renderers.base import BaseRenderer

    from vllm_gr.entrypoints.openai.tokenizer_patch import patch_fast_sid_encode

    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    orig_sync = BaseRenderer._tokenize_prompt
    orig_async = BaseRenderer._tokenize_prompt_async
    orig_flag = getattr(BaseRenderer, "_vllm_gr_sid_encode_patch", False)
    try:
        monkeypatch.setattr(BaseRenderer, "_vllm_gr_sid_encode_patch", False, raising=False)
        patch_fast_sid_encode()
        assert getattr(BaseRenderer, "_vllm_gr_sid_encode_patch", False) is True

        tok = _FakeTokenizer()
        fallback_calls: list[str] = []

        class _DummyRenderer:
            def get_tokenizer(self) -> _FakeTokenizer:
                return tok

        def fake_orig_sync(
            self: object, prompt: dict[str, object], params: object
        ) -> dict[str, object]:
            fallback_calls.append("sync")
            return {"prompt_token_ids": [42], **prompt}

        async def fake_orig_async(
            self: object, prompt: dict[str, object], params: object
        ) -> dict[str, object]:
            fallback_calls.append("async")
            return {"prompt_token_ids": [42], **prompt}

        monkeypatch.setattr(BaseRenderer, "_tokenize_prompt", fake_orig_sync)
        monkeypatch.setattr(BaseRenderer, "_tokenize_prompt_async", fake_orig_async)
        monkeypatch.setattr(BaseRenderer, "_vllm_gr_sid_encode_patch", False, raising=False)
        patch_fast_sid_encode()

        params = SimpleNamespace(
            get_encode_kwargs=lambda: {
                "add_special_tokens": False,
                "truncation": False,
                "max_length": None,
            }
        )
        renderer = _DummyRenderer()

        sid_prompt = {"prompt": "<s_a_0> <s_b_1>"}
        sync_res = BaseRenderer._tokenize_prompt(renderer, sid_prompt, params)
        async_res = asyncio.run(BaseRenderer._tokenize_prompt_async(renderer, sid_prompt, params))
        assert sync_res["prompt_token_ids"] == [5000, 1001, 6001]
        assert async_res["prompt_token_ids"] == [5000, 1001, 6001]
        assert fallback_calls == []

        plain_prompt = {"prompt": "plain text without sids"}
        sync_fb = BaseRenderer._tokenize_prompt(renderer, plain_prompt, params)
        async_fb = asyncio.run(BaseRenderer._tokenize_prompt_async(renderer, plain_prompt, params))
        assert sync_fb["prompt_token_ids"] == [42]
        assert async_fb["prompt_token_ids"] == [42]
        assert fallback_calls == ["sync", "async"]
    finally:
        monkeypatch.setattr(BaseRenderer, "_tokenize_prompt", orig_sync)
        monkeypatch.setattr(BaseRenderer, "_tokenize_prompt_async", orig_async)
        monkeypatch.setattr(BaseRenderer, "_vllm_gr_sid_encode_patch", orig_flag, raising=False)
