# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-stage timer breakdown of the catalog beam-search hot path.

Runs in replica mode (pure standard library, no vLLM/torch required) and
reports where wall time actually goes in one decode step over ``beam_width``
beams, following the reviewer request to *add timers and find the hotspots*
before deciding on further optimization.

Stages timed per step (all over ``beam_width`` beams):

- **old lookup**: ``Catalog.valid(beam.tokens[prompt_len:])`` — full trie walk.
- **new lookup**: ``resolve_catalog_node`` reading the cached ``_catalog_node``.
- **apply filter (null)**: whole ``apply_catalog_filter`` under ``"null"``.
- **apply filter (fill)**: whole ``apply_catalog_filter`` under ``"fill"``.
- **fill inner -- sorted valid list**: the ``sorted(k for k in node if
  not isinstance(k, str))`` step (dict node).  This is the suspected hotspot.
- **fill inner -- membership/filter loop**: iterating the candidate window,
  checking each token against the valid node, filling invalid slots.
- **fill inner -- list copy**: the ``list[:]`` copies of tids/lps/ranks/decoded.

A synthetic comparison of ``sorted(dict)`` vs reading a precomputed
``tuple`` of valid tokens is included to quantify the ceiling of caching the
sorted view.

Reported per config: median over ``--repeats`` runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import time
from pathlib import Path
from typing import Any

# Load the sibling benchmark module without requiring a package context.
_benchmark_path = Path(__file__).resolve().parent / "catalog_lookup_benchmark.py"
_spec = importlib.util.spec_from_file_location("catalog_lookup_benchmark", _benchmark_path)
_benchmark = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_benchmark)

_Beam = _benchmark._Beam
_replica_apply_filter = _benchmark._replica_apply_filter
_replica_resolve = _benchmark._replica_resolve
_replica_valid = _benchmark._replica_valid
build_trie = _benchmark.build_trie

_TERMINAL = "__catalog_terminal__"


def _timed(fn: Any, repeats: int) -> list[float]:
    """Run fn ``repeats`` times, return per-call wall times in us."""
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def _fmt(name: str, times_us: list[float]) -> str:
    med = statistics.median(times_us)
    mn = min(times_us)
    p95 = sorted(times_us)[int(len(times_us) * 0.95) - 1]
    return f"{name:>34}: med={med:9.2f}us  min={mn:8.2f}us  p95={p95:9.2f}us"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prompt-len", type=int, default=50)
    parser.add_argument("--num-sequences", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    rng = random.Random(0)
    trie, shared = build_trie(args.num_sequences, args.seq_len, rng)
    # Position every beam at the shared prefix (real valid next tokens).
    node: dict[int, Any] = trie
    for tid in shared:
        node = node[tid]
    sorted_valid = tuple(sorted(k for k in node if not isinstance(k, str)))
    vocab = 100
    logprobs_num = 16

    print(f"mode: replica (no vLLM); trie {args.num_sequences} seq x len {args.seq_len}")
    print(f"prompt_len={args.prompt_len} steps={args.steps} repeats={args.repeats}")
    print()

    for beam_width in args.beam_widths:
        beams = [_Beam(shared[: args.prompt_len], node) for _ in range(beam_width)]
        # Candidate window per beam (16 logprobs), deterministic.
        cands = [rng.randrange(vocab) for _ in range(beam_width * logprobs_num)]
        lps = [-(i + 1) * 0.01 for i in range(beam_width * logprobs_num)]
        ranks = list(range(beam_width * logprobs_num))
        decoded = [str(i) for i in range(beam_width * logprobs_num)]

        def seg(i: int) -> tuple[list[int], list[float], list[int], list[str]]:
            return (
                cands[i : i + logprobs_num],
                lps[i : i + logprobs_num],
                ranks[i : i + logprobs_num],
                decoded[i : i + logprobs_num],
            )

        print(f"===== beam_width={beam_width} =====")

        # 1. lookup paths (per beam, single step)
        def _old_lookup() -> None:
            for b in beams:
                _replica_valid(trie, b.tokens)

        def _new_lookup() -> None:
            for b in beams:
                _replica_resolve(b, trie, args.prompt_len)

        print(_fmt("old lookup: valid() full walk", _timed(_old_lookup, args.repeats)))
        print(_fmt("new lookup: resolve cached node", _timed(_new_lookup, args.repeats)))

        # 2. apply_catalog_filter under both strategies
        def _null_apply() -> None:
            for i in range(beam_width):
                vn = _replica_resolve(beams[i], trie, args.prompt_len)
                _replica_apply_filter(*seg(i), vn, "null")

        def _fill_apply() -> None:
            for i in range(beam_width):
                vn = _replica_resolve(beams[i], trie, args.prompt_len)
                _replica_apply_filter(*seg(i), vn, "fill")

        print(_fmt("apply_catalog_filter (null)", _timed(_null_apply, args.repeats)))
        print(_fmt("apply_catalog_filter (fill)", _timed(_fill_apply, args.repeats)))

        # 3. fill inner breakdown
        def _fill_sorted() -> None:
            for _ in range(beam_width):
                _ = sorted(k for k in node if not isinstance(k, str))

        def _fill_membership() -> None:
            # per beam: check each candidate against node, mask invalid
            for _ in range(beam_width):
                for t in cands[:logprobs_num]:
                    _ = t in node

        def _fill_copy() -> None:
            for _ in range(beam_width):
                _ = (
                    cands[:logprobs_num],
                    lps[:logprobs_num],
                    ranks[:logprobs_num],
                    decoded[:logprobs_num],
                )

        print(_fmt("  fill inner: sorted valid list", _timed(_fill_sorted, args.repeats)))
        print(_fmt("  fill inner: membership/filter loop", _timed(_fill_membership, args.repeats)))
        print(_fmt("  fill inner: list copy", _timed(_fill_copy, args.repeats)))

        # 4. synthetic: per-step sorted vs precomputed read
        def _per_step_sorted() -> None:
            for _ in range(beam_width):
                _ = sorted(k for k in node if not isinstance(k, str))

        def _precomputed_read() -> None:
            for _ in range(beam_width):
                _ = sorted_valid

        print(_fmt("synthetic: sorted(dict) per beam", _timed(_per_step_sorted, args.repeats)))
        print(_fmt("synthetic: read precomputed tuple", _timed(_precomputed_read, args.repeats)))
        print()


if __name__ == "__main__":
    main()
