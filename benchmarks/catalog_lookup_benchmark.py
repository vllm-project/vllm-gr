# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A/B micro-benchmark for the catalog valid-token lookup hot path.

Compares the pre- and post-optimization serving paths for resolving a beam's
catalog-valid next tokens and applying the catalog constraint to its
candidate logprobs:

- **old**: ``catalog.valid(beam.tokens[prompt_len:])`` (full O(t) trie walk)
  + ``apply_catalog_filter(...)`` / ``mask_candidate_logprobs(...)``.
- **new**: ``resolve_catalog_node(beam, catalog, prompt_len)`` (O(1) read of
  the cached ``_catalog_node``, with a ``catalog.valid()`` fallback) +
  ``apply_catalog_filter(...)``.

The benchmark runs in two modes:

- **production mode** (preferred): imports the real functions
  (``Catalog``, ``resolve_catalog_node``, ``apply_catalog_filter``) and drives
  them exactly as the serving loop does.  This needs a vLLM install.
- **replica mode** (fallback, no vLLM/torch): re-implements the same logic with
  pure standard-library data structures, so the script still runs locally and
  gives a directional answer.  The replica mirrors the production code paths
  line-for-line (see ``_replica_*`` helpers).

Reported per configuration (``beam_width`` x ``catalog_strategy``):
median / min / p95 per-step latency, per-request total, and the number of
``catalog.valid()`` calls (expected to drop from ``beam_width * steps`` to
only the fallback cases).  ``--repeats`` controls the number of timed runs;
each run is timed and the min / median / p95 are reported across runs.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Any

# ---------------------------------------------------------------------------
# Production-mode imports (fall back to replica mode when vLLM is absent).
# ---------------------------------------------------------------------------
_PROD = False
try:  # pragma: no cover - depends on local vLLM availability
    from vllm_gr.beam_search_decision_utils import resolve_catalog_node
    from vllm_gr.utils.catalog import Catalog

    _PROD = True
except Exception:  # pragma: no cover - no vLLM/torch on this machine
    _PROD = False


# ---------------------------------------------------------------------------
# Shared trie builder (used by both modes).
# ---------------------------------------------------------------------------
_TERMINAL = "__catalog_terminal__"


def build_trie(
    num_sequences: int, seq_len: int, rng: random.Random
) -> tuple[dict[int, Any], list[int]]:
    """Build a trie of ``num_sequences`` sequences of length ``seq_len``.

    Mirrors ``Catalog.load``: node maps token_id -> child dict, end of every
    sequence carries the terminal sentinel string key.  All sequences share a
    fixed prefix so intermediate nodes have non-empty child sets — the
    beam-search case where every step offers several valid next tokens.
    Returns ``(root, shared_prefix)``.
    """
    vocab_size = max(8, int(num_sequences**0.5) + 1)
    shared_prefix_len = max(2, seq_len // 2)
    shared = [rng.randrange(vocab_size) for _ in range(shared_prefix_len)]
    root: dict[int, Any] = {}
    for _ in range(num_sequences):
        node = root
        for tid in shared:
            child = node.get(tid)
            if child is None:
                child = {}
                node[tid] = child
            node = child
        for _ in range(seq_len - shared_prefix_len):
            tid = rng.randrange(vocab_size)
            child = node.get(tid)
            if child is None:
                child = {}
                node[tid] = child
            node = child
        node[_TERMINAL] = True
    return root, shared


def _valid_keys(node: dict[int, Any]) -> set[int]:
    """Child token ids of a trie node, excluding the terminal sentinel."""
    return {k for k in node if not isinstance(k, str)}


# ---------------------------------------------------------------------------
# Replica mode (no vLLM): mirrors the production functions exactly.
# ---------------------------------------------------------------------------
def _replica_valid(trie: dict[int, Any], generated: list[int]) -> set[int]:
    """Replicates ``Catalog.valid``: O(t) walk from the root."""
    node: dict[int, Any] = trie
    for tid in generated:
        if tid in node:
            node = node[tid]
        else:
            return set()
    return _valid_keys(node)


def _replica_resolve(beam: Any, trie: dict[int, Any], prompt_len: int) -> dict[int, Any] | set[int]:
    """Replicates ``resolve_catalog_node``: cached node or full re-walk."""
    node = getattr(beam, "_catalog_node", None)
    if node is not None:
        return node
    return _replica_valid(trie, beam.tokens[prompt_len:])


def _replica_apply_filter(
    token_ids_pos: list[int],
    logprobs_pos: list[float],
    ranks_pos: list[int | None] | None,
    decoded_pos: list[str | None] | None,
    valid_node: dict[int, Any] | set[int] | None,
    strategy: str,
) -> tuple[list[int], list[float], list[int | None] | None, list[str | None] | None]:
    """Replicates ``Catalog.apply_catalog_filter`` (fill / null)."""
    if not valid_node:
        return (
            [],
            [],
            ([] if ranks_pos is not None else None),
            ([] if decoded_pos is not None else None),
        )
    if strategy == "null":
        n = len(logprobs_pos)
        out = logprobs_pos[:]
        all_inf = True
        for i in range(n):
            if token_ids_pos[i] not in valid_node:
                out[i] = -float("inf")
            else:
                all_inf = False
        if all_inf:
            return (
                [],
                [],
                ([] if ranks_pos is not None else None),
                ([] if decoded_pos is not None else None),
            )
        return token_ids_pos, out, ranks_pos, decoded_pos
    if isinstance(valid_node, dict):
        valid_list = sorted(k for k in valid_node if not isinstance(k, str))
    else:
        valid_list = sorted(valid_node)
    n_valid = len(valid_list)
    if n_valid == 0:
        return (
            [],
            [],
            ([] if ranks_pos is not None else None),
            ([] if decoded_pos is not None else None),
        )
    n = len(token_ids_pos)
    new_tids = token_ids_pos[:]
    new_lps = logprobs_pos[:]
    new_ranks: list[int | None] | None = None
    new_decoded: list[str | None] | None = None
    if ranks_pos is not None:
        new_ranks = [ranks_pos[i] if i < len(ranks_pos) else None for i in range(n)]
    if decoded_pos is not None:
        new_decoded = [decoded_pos[i] if i < len(decoded_pos) else None for i in range(n)]
    fill_used = 0
    for j, tid in enumerate(token_ids_pos):
        if tid not in valid_node:
            ft = valid_list[fill_used % n_valid]
            new_tids[j] = ft
            new_lps[j] = -10000.0 - (fill_used % n_valid) * 0.1
            fill_used += 1
            if new_ranks is not None:
                new_ranks[j] = None
            if new_decoded is not None:
                new_decoded[j] = None
    return new_tids, new_lps, new_ranks, new_decoded


# ---------------------------------------------------------------------------
# Beam object (works for both modes).
# ---------------------------------------------------------------------------
class _Beam:
    def __init__(self, tokens: list[int], cached_node: Any | None) -> None:
        self.tokens = tokens
        self._catalog_node = cached_node


# ---------------------------------------------------------------------------
# Workload driver.
# ---------------------------------------------------------------------------
def _make_beams(
    root: dict[int, Any],
    shared: list[int],
    beam_width: int,
    prompt_len: int,
    steps: int,
    rng: random.Random,
) -> list[_Beam]:
    """Build ``beam_width`` beams sharing a real trie prefix.

    Every beam carries a cached ``_catalog_node`` positioned at the trie node
    after its generated prefix — the post-optimization state.
    """
    node: dict[int, Any] = root
    for tid in shared[:prompt_len]:
        node = node[tid]
    beams: list[_Beam] = []
    for _ in range(beam_width):
        b = _Beam(shared[:prompt_len], node)
        beams.append(b)
    return beams


def _drive(
    beams: list[_Beam],
    trie: dict[int, Any],
    beam_width: int,
    steps: int,
    prompt_len: int,
    strategy: str,
    use_new: bool,
    rng: random.Random,
    catalog: Any = None,
) -> tuple[float, int]:
    """One full request's catalog step loop.

    ``use_new`` selects old (valid() re-walk) or new (resolve_catalog_node)
    path.  ``catalog`` is only used in production mode.  Returns (wall time in
    ms, number of catalog.valid() calls).
    """
    vocab = 100
    # candidate windows: beam_width beams x logprobs_num candidates
    logprobs_num = 16
    total_ms = 0.0
    valid_calls = 0
    for _ in range(steps):
        candidates = [rng.randrange(vocab) for _ in range(beam_width * logprobs_num)]
        cand_lps = [-(i + 1) * 0.01 for i in range(beam_width * logprobs_num)]
        ranks = list(range(beam_width * logprobs_num))
        decoded = [str(i) for i in range(beam_width * logprobs_num)]

        t0 = time.perf_counter()
        for beam_idx, b in enumerate(beams):
            gen = b.tokens[prompt_len:]
            if use_new:
                if catalog is not None:
                    node = resolve_catalog_node(b, catalog, prompt_len)
                else:
                    node = _replica_resolve(b, trie, prompt_len)
            else:
                if catalog is not None:
                    node = catalog.valid(gen)
                else:
                    node = _replica_valid(trie, gen)
                valid_calls += 1
            if node is None:
                node = set()
            # apply filter over this beam's candidate slice (beam-stride layout)
            start = beam_idx * logprobs_num
            end = start + logprobs_num
            seg_tids = candidates[start:end]
            seg_lps = cand_lps[start:end]
            seg_ranks = ranks[start:end]
            seg_dec = decoded[start:end]
            if catalog is not None:
                catalog.apply_catalog_filter(seg_tids, seg_lps, seg_ranks, seg_dec, node, strategy)
            else:
                _replica_apply_filter(seg_tids, seg_lps, seg_ranks, seg_dec, node, strategy)
        total_ms += (time.perf_counter() - t0) * 1e3
    return total_ms, valid_calls


def _format_stats(name: str, ms_list: list[float]) -> str:
    if not ms_list:
        return ""
    med = statistics.median(ms_list)
    mn = min(ms_list)
    p95 = sorted(ms_list)[int(len(ms_list) * 0.95) - 1]
    return f"{name}: med={med:.3f}ms min={mn:.3f}ms p95={p95:.3f}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--max-tokens", type=int, default=5, help="decode steps per request")
    parser.add_argument(
        "--prompt-len", type=int, default=50, help="tokens before the catalog segment"
    )
    parser.add_argument("--num-sequences", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10, help="timed runs per config")
    args = parser.parse_args()

    rng = random.Random(0)
    trie, shared = build_trie(args.num_sequences, args.seq_len, rng)
    mode = "production" if _PROD else "replica"

    catalog: Any = None
    if _PROD:  # pragma: no cover - vLLM available
        tokenizer = type("T", (), {"convert_tokens_to_ids": lambda self, s: [int(x) for x in s]})()
        catalog = Catalog(tokenizer)  # type: ignore[name-defined]
        catalog.trie = trie
        catalog._TERMINAL_MARKER = _TERMINAL

    print(f"mode: {mode} (production import={'ok' if _PROD else 'unavailable, using replica'})")
    print(f"trie: {args.num_sequences} seq x len {args.seq_len}, shared prefix {len(shared)}")
    print(f"workload: beam_width x {args.max_tokens} steps, prompt_len {args.prompt_len}")
    print()

    for beam_width in args.beam_widths:
        beams = _make_beams(trie, shared, beam_width, args.prompt_len, args.max_tokens, rng)
        for strategy in ("fill", "null"):
            old_ms: list[float] = []
            new_ms: list[float] = []
            old_calls: int = 0
            new_calls: int = 0
            for _ in range(args.repeats):
                ms_old, calls_old = _drive(
                    beams,
                    trie,
                    beam_width,
                    args.max_tokens,
                    args.prompt_len,
                    strategy,
                    False,
                    rng,
                    catalog,
                )
                ms_new, calls_new = _drive(
                    beams,
                    trie,
                    beam_width,
                    args.max_tokens,
                    args.prompt_len,
                    strategy,
                    True,
                    rng,
                    catalog,
                )
                old_ms.append(ms_old)
                new_ms.append(ms_new)
                old_calls += calls_old
                new_calls += calls_new

            med_old = statistics.median(old_ms)
            med_new = statistics.median(new_ms)
            speedup = med_old / med_new if med_new > 0 else float("inf")
            print(f"beam_width={beam_width} strategy={strategy}")
            print(f"  old: {_format_stats('total', old_ms)}  valid() calls={old_calls}")
            print(f"  new: {_format_stats('total', new_ms)}  valid() calls={new_calls}")
            print(f"  speedup (median): {speedup:.1f}x   valid() calls: {old_calls} -> {new_calls}")
            print()


if __name__ == "__main__":
    main()
