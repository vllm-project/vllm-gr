# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast detokenization helpers that bypass the HuggingFace Python wrapper.

``transformers`` ``batch_decode`` / batched ``decode`` first recursively
converts every token id through the pure-Python ``to_py_obj`` helper and then
loops over sequences in Python, calling the single-sequence Rust ``decode``
once per beam.  The underlying ``tokenizers.Tokenizer`` (``tokenizer._tokenizer``)
already exposes a natively batched ``decode_batch`` (and a single ``decode``)
implemented in Rust.  Routing straight to those methods removes the
``to_py_obj`` traversal and the per-sequence Python call frames.

Both ``decode`` and ``decode_batch`` on the Rust tokenizer are ``&self``
(PyO3) methods, so they are safe to call concurrently without the
``TokenizerPool`` deep-copy borrow.

Correctness guard (behavior probe): merely having a ``_tokenizer`` backend is
not proof that bypassing the HuggingFace wrapper is byte-safe.  Since
transformers v5 even legacy tokenizers (e.g. ``Qwen2Tokenizer``) subclass the
tokenizers backend and report ``is_fast=True``, an ``isinstance`` check cannot
distinguish them.  Such a legacy tokenizer's wrapper still augments
added-token decoding: special ids such as ``<|im_start|>`` decode to their
marker text through the wrapper but to an **empty string** through the raw
Rust backend.  Skipping the wrapper there drops ChatML markers from the
prompt prefix and silently corrupts downstream text slicing (evaluators slice
``text[len(prompt):]``).  ``_rust_decode_verified`` therefore probes every
*special* added token (plus a sample of regular ones) once per tokenizer and
enables the Rust path only when both decoders agree.  ``skip_special_tokens``
defaults to ``False``, matching the no-kwarg call sites (and transformers v5's
default), so ChatML markers in the prompt prefix are preserved and the prefix
round-trips exactly.
"""

from __future__ import annotations

import array
import weakref
from collections.abc import Sequence

import regex as re

__all__ = [
    "fast_batch_decode",
    "fast_decode",
    "fast_sid_encode",
]

# tokenizer -> bool: whether its raw Rust backend decodes added tokens
# identically to the HuggingFace wrapper.
_RUST_DECODE_VERIFIED: weakref.WeakKeyDictionary[object, bool] = weakref.WeakKeyDictionary()


def _rust_backend(tokenizer):
    """Return the native ``tokenizers.Tokenizer`` if available, else None."""
    return getattr(tokenizer, "_tokenizer", None)


def _rust_decode_verified(tokenizer, backend) -> bool:
    """True iff the raw Rust backend decodes added tokens like the HF wrapper.

    Probed once per tokenizer instance (cached weakly).  Every *special*
    added token is compared (these are exactly the ids whose handling can
    diverge), plus a small sample of regular added tokens, under both
    ``skip_special_tokens`` values.
    """
    try:
        cached = _RUST_DECODE_VERIFIED.get(tokenizer)
    except TypeError:
        cached = None  # non-weakrefable tokenizer: do not cache, probe per call
    if cached is not None:
        return cached

    ok = False
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if decoder:
        special_ids: list[int] = []
        regular_ids: list[int] = []
        for token_id, added_token in decoder.items():
            if getattr(added_token, "special", False):
                special_ids.append(token_id)
            else:
                regular_ids.append(token_id)
        probe_ids = special_ids + regular_ids[:32]
        ok = True
        for token_id in probe_ids:
            ids = [token_id]
            if backend.decode(ids) != tokenizer.decode(ids, skip_special_tokens=False):
                ok = False
                break
            if backend.decode(ids, skip_special_tokens=True) != tokenizer.decode(
                ids, skip_special_tokens=True
            ):
                ok = False
                break

    try:
        _RUST_DECODE_VERIFIED[tokenizer] = ok
    except TypeError:
        pass
    return ok


def _fast_tokenizer(tokenizer):
    """True iff ``tokenizer`` is a HuggingFace *fast* tokenizer (or a
    thread-safe pool wrapper around one, as produced by vLLM's
    ``maybe_make_thread_pool`` via in-place ``__class__`` reassignment).

    Only fast tokenizers carry the ``_decode`` fast path used by the
    fallbacks below; slow tokenizers define ``_decode`` as an abstract
    ``NotImplementedError`` stub, so they must keep going through
    ``decode`` / ``batch_decode``.
    """
    return bool(getattr(tokenizer, "is_fast", False))


def fast_decode(tokenizer, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
    """Decode a single sequence, preferring the native Rust ``Tokenizer.decode``.

    ``skip_special_tokens`` defaults to ``False`` to match the no-kwarg call
    sites (and transformers v5's default).  The Rust path is used only after a
    behavior probe proves it matches the wrapper; otherwise this delegates to
    ``tokenizer.decode``.  The keyword is forwarded only when ``True`` on the
    fallback so minimal tokenizer stubs implementing just ``decode(ids)``
    keep working.
    """
    ids = list(token_ids)
    backend = _rust_backend(tokenizer)
    if (
        backend is not None
        and hasattr(backend, "decode")
        and _rust_decode_verified(tokenizer, backend)
    ):
        if skip_special_tokens:
            return backend.decode(ids, skip_special_tokens=True)
        return backend.decode(ids)
    # Fast-tokenizer shortcut: ``PreTrainedTokenizerFast._decode`` is the
    # terminal implementation that ``decode`` dispatches to after its
    # ``to_py_obj`` conversion.  The inputs here are already plain Python
    # lists, so calling ``_decode`` directly skips the per-element recursive
    # ``to_py_obj`` traversal (measured ~2x on 128-beam batches) while
    # producing byte-identical output (same Rust decode + same
    # ``clean_up_tokenization_spaces`` default: ``_decode`` is invoked with
    # its ``clean_up_tokenization_spaces`` default of ``None``, matching the
    # single-sequence branch of ``decode``).
    if _fast_tokenizer(tokenizer):
        _decode = getattr(tokenizer, "_decode", None)
        if callable(_decode):
            return _decode(ids, skip_special_tokens=skip_special_tokens)
    if skip_special_tokens:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer.decode(ids)


def fast_batch_decode(
    tokenizer,
    sequences: Sequence[Sequence[int]],
    skip_special_tokens: bool = False,
) -> list[str]:
    """Decode a batch of sequences, preferring the native Rust ``decode_batch``.

    ``skip_special_tokens`` defaults to ``False`` to match the no-kwarg call
    sites (and transformers v5's default).  The Rust path is used only after a
    behavior probe proves it matches the wrapper; otherwise this falls back to
    ``tokenizer.batch_decode`` and then to a per-sequence ``decode`` loop.
    The keyword is only forwarded when ``True`` to stay compatible with
    minimal tokenizer stubs.
    """
    seqs = [list(seq) for seq in sequences]
    backend = _rust_backend(tokenizer)
    if (
        backend is not None
        and hasattr(backend, "decode_batch")
        and _rust_decode_verified(tokenizer, backend)
    ):
        if skip_special_tokens:
            return backend.decode_batch(seqs, skip_special_tokens=True)
        return backend.decode_batch(seqs)
    # Fast-tokenizer shortcut: ``batch_decode`` forwards to ``decode``'s
    # batched branch, which runs the recursive ``to_py_obj`` conversion over
    # the already-plain-``int`` sequences and then loops ``_decode`` per
    # sequence, forwarding ``clean_up_tokenization_spaces=None`` (its own
    # default).  Calling ``_decode`` directly over the same loop skips the
    # traversal and the nested dispatch (measured ~2x on 128-beam batches)
    # while staying byte-identical: omitting the cleanup kwarg leaves it at
    # ``None`` in ``_decode``'s signature, which resolves from
    # ``self.clean_up_tokenization_spaces`` — hard-coding ``False`` would
    # silently skip cleanup for tokenizers explicitly configured with
    # ``clean_up_tokenization_spaces=True``.
    if _fast_tokenizer(tokenizer):
        _decode = getattr(tokenizer, "_decode", None)
        if callable(_decode):
            return [_decode(seq, skip_special_tokens=skip_special_tokens) for seq in seqs]
    if hasattr(tokenizer, "batch_decode"):
        if skip_special_tokens:
            return tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return tokenizer.batch_decode(seqs)
    if skip_special_tokens:
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
    return [tokenizer.decode(seq) for seq in seqs]


_SID_ENCODE_CTX_MISSING = object()
# Sid-heavy recommender prompts carry tens of thousands of added tokens
# (<s_a_N>/<s_b_N>/<s_c_N> plus a handful of chat-template markers).  The
# Rust backend's added-token matching makes a single encode of such text
# several times slower than plain-text encodes.  When every added token
# contains ``<``, the encode can be split into one regex pass (three sid
# families + the few non-sid added tokens) plus one batch encode of the
# distinct plain-text runs (typically the label, the single inter-sid space
# and the template framing).
_MAX_DISTINCT_TEXT_RUNS = 512
_SID_FAMILY_PATTERN = r"<s_a_\d+>|<s_b_\d+>|<s_c_\d+>"
_SID_ENCODE_CONTEXT: weakref.WeakKeyDictionary[object, tuple] = weakref.WeakKeyDictionary()
_SID_ENCODE_VERIFIED: weakref.WeakKeyDictionary[object, bool] = weakref.WeakKeyDictionary()


def _sid_encode_verified(tokenizer) -> bool:
    """True iff fast_sid_encode produces identical ids to the plain encode on probe samples."""
    try:
        cached = _SID_ENCODE_VERIFIED.get(tokenizer)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    if not hasattr(tokenizer, "encode") or not callable(tokenizer.encode):
        return True

    ok = True
    added = getattr(tokenizer, "added_tokens_encoder", {})
    first_sid = next((k for k in added if k.startswith(("<s_a_", "<s_b_", "<s_c_"))), None)
    if first_sid:
        test_samples = [
            f"x {first_sid}",
            f"x  {first_sid}",
            f"{first_sid} {first_sid}",
            f"<s_a_4294967300> {first_sid}",
        ]
        for sample in test_samples:
            fast_ids = fast_sid_encode(tokenizer, sample, _skip_verification_probe=True)
            if fast_ids is not None:
                plain_ids = tokenizer.encode(sample, add_special_tokens=False)
                if fast_ids != plain_ids:
                    ok = False
                    break

    try:
        _SID_ENCODE_VERIFIED[tokenizer] = ok
    except TypeError:
        pass
    return ok


try:
    from vllm_gr import _fast_sid_c
except ImportError:
    _fast_sid_c = None


def _build_sid_encode_context(tokenizer):
    """One-time probe: a compiled alternation over the added tokens.

    Returns the context tuple when the tokenizer has added tokens, every
    one of them contains ``<`` (the fallback guard counts ``<`` characters),
    and the Rust backend exposes ``encode_batch``, else None.
    """
    added = getattr(tokenizer, "added_tokens_encoder", None)
    if not added or not all("<" in token for token in added):
        return None
    backend = _rust_backend(tokenizer)
    if backend is None or not hasattr(backend, "encode_batch"):
        return None
    non_sid = {
        token: tid
        for token, tid in added.items()
        if not token.startswith(("<s_a_", "<s_b_", "<s_c_"))
    }
    pattern = _SID_FAMILY_PATTERN
    if non_sid:
        pattern += "|" + "|".join(
            re.escape(token) for token in sorted(non_sid, key=len, reverse=True)
        )

    def _build_lut(lvl: str) -> bytes:
        prefix = f"<s_{lvl}_"
        max_idx = -1
        for k in added:
            if k.startswith(prefix) and k.endswith(">"):
                sub = k[len(prefix) : -1]
                if sub.isdigit():
                    idx = int(sub)
                    max_idx = max(max_idx, idx)
        if max_idx < 0 or max_idx > 100000:
            return b""
        lut = array.array("i", [-1] * (max_idx + 1))
        for k, tid in added.items():
            if k.startswith(prefix) and k.endswith(">"):
                sub = k[len(prefix) : -1]
                if sub.isdigit():
                    idx = int(sub)
                    if idx <= max_idx:
                        lut[idx] = tid
        return lut.tobytes()

    lut_a = _build_lut("a")
    lut_b = _build_lut("b")
    lut_c = _build_lut("c")

    space_id = -1
    try:
        space_enc = backend.encode(" ", add_special_tokens=False)
        if hasattr(space_enc, "ids") and len(space_enc.ids) == 1:
            space_id = space_enc.ids[0]
    except Exception:  # noqa: BLE001
        space_id = -1

    sid_begin_id = added.get("<|sid_begin|>", -1)
    sid_end_id = added.get("<|sid_end|>", -1)

    return (
        re.compile(pattern),
        dict(added),
        non_sid,
        lut_a,
        lut_b,
        lut_c,
        space_id,
        sid_begin_id,
        sid_end_id,
        backend,
    )


def _sid_encode_context(tokenizer):
    try:
        cached = _SID_ENCODE_CONTEXT.get(tokenizer, _SID_ENCODE_CTX_MISSING)
    except TypeError:
        cached = _SID_ENCODE_CTX_MISSING  # non-weakrefable tokenizer: probe per call
    if cached is not _SID_ENCODE_CTX_MISSING:
        return cached
    ctx = _build_sid_encode_context(tokenizer)
    try:
        _SID_ENCODE_CONTEXT[tokenizer] = ctx
    except TypeError:
        pass
    return ctx


def fast_sid_encode(
    tokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
    truncation: bool = False,
    max_length: int | None = None,
    _skip_verification_probe: bool = False,
) -> list[int] | None:
    """Added-token-aware fast encode; returns ``None`` when the plain encode must run.

    Replaces the Rust added-token matching over the full text with an ultra-fast C
    scanner (or regex pass fallback) plus one batch encode of the distinct
    plain-text runs. Ids are byte-identical to
    ``tokenizer.encode(text, add_special_tokens=False)`` because added-token
    matches are exact boundaries in the original encode as well, so no BPE
    merge can cross a run boundary.

    Returns ``None`` (caller falls back to the original encode) whenever:
    * the tokenizer shape does not satisfy the probe above;
    * ``add_special_tokens`` or actual truncation is requested;
    * the text contains any ``<`` outside added-token matches;
    * the text runs are too varied for the batch encode to stay profitable.
    """
    if add_special_tokens or not text or "<" not in text:
        return None
    if not _skip_verification_probe and not _sid_encode_verified(tokenizer):
        return None
    ctx = _sid_encode_context(tokenizer)
    if ctx is None:
        return None

    (
        regex,
        added_ids,
        non_sid,
        lut_a,
        lut_b,
        lut_c,
        space_id,
        sid_begin_id,
        sid_end_id,
        backend,
    ) = ctx

    if _fast_sid_c is not None:
        ids = _fast_sid_c.fast_sid_encode(
            text,
            lut_a,
            lut_b,
            lut_c,
            space_id,
            sid_begin_id,
            sid_end_id,
            added_ids,
            non_sid,
            backend.encode_batch,
        )
        if ids is None:
            return None
        if truncation and max_length is not None and len(ids) > max_length:
            return None
        return ids

    # Python fallback path
    matches = list(regex.finditer(text))
    if not matches or text.count("<") != len(matches):
        return None

    runs: list[tuple] = []
    pos = 0
    for match in matches:
        if match.start() > pos:
            runs.append(("t", text[pos : match.start()]))
        token = match.group(0)
        token_id = added_ids.get(token)
        if token_id is None:
            return None
        runs.append(("a", token, token_id))
        pos = match.end()
    if pos < len(text):
        runs.append(("t", text[pos:]))

    text_runs = [run[1] for run in runs if run[0] == "t"]
    distinct_runs = list(dict.fromkeys(text_runs))
    if len(distinct_runs) > _MAX_DISTINCT_TEXT_RUNS:
        return None
    encoded_runs = backend.encode_batch(distinct_runs, add_special_tokens=False)
    run_ids = {run: enc.ids for run, enc in zip(distinct_runs, encoded_runs)}

    ids: list[int] = []
    for run in runs:
        if run[0] == "t":
            ids.extend(run_ids[run[1]])
        else:
            ids.append(run[2])

    if truncation and max_length is not None and len(ids) > max_length:
        return None  # let the original encode apply the real truncation
    return ids
