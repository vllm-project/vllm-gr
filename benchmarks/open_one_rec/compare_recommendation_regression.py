#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare per-sample OneRec baseline and candidate JSONL results."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.open_one_rec.regression import (
    compare_jsonl,
    comparison_passed,
    validate_golden_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--policy",
        choices=["exact", "no-regression"],
        default="exact",
        help="exact compares top-1/per-sample metrics; no-regression gates aggregate metrics.",
    )
    parser.add_argument(
        "--allow-differences",
        action="store_true",
        help="Report differences without returning a failing exit status.",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=0.0,
        help="Allowed absolute aggregate drop under no-regression.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="Allowed relative aggregate drop under no-regression (0..1).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Aggregate metric to gate under no-regression; repeat as needed.",
    )
    args = parser.parse_args()
    # A reused golden baseline must share the candidate's model, data digest,
    # and beam settings — a weaker baseline (e.g. different beam width) would
    # otherwise let a candidate pass the no-regression gate vacuously.
    identity = validate_golden_identity(args.baseline, args.candidate)
    result = compare_jsonl(
        args.baseline,
        args.candidate,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if identity["mismatches"]:
        print(
            "Golden baseline run_identity does not match the candidate; "
            "regenerate the golden or align the run settings:",
            file=sys.stderr,
        )
        for mismatch in identity["mismatches"]:
            print(f"  {mismatch}", file=sys.stderr)
        return 2
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    passed = comparison_passed(
        result,
        policy=args.policy,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        metric_names=args.metrics,
    )
    return 0 if args.allow_differences or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
