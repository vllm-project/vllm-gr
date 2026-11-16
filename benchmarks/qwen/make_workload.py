# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Generate reproducible, fixed-length input IDs from an explicit token interval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_workload(args: argparse.Namespace) -> list[dict]:
    """Sample input IDs with replacement; the interval is [start, end)."""
    config = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("model config.json must define a positive integer vocab_size")
    if not 0 <= args.token_id_start < args.token_id_end <= vocab_size:
        raise ValueError(
            f"require 0 <= token-id-start < token-id-end <= model vocab_size ({vocab_size})"
        )
    if args.context_len <= 0 or args.requests <= 0:
        raise ValueError("context-len and requests must be positive")

    rows = []
    for request in range(args.requests):
        # Independent streams preserve existing requests when their count grows.
        rng = random.Random(args.seed + request)
        ids = [
            rng.randrange(args.token_id_start, args.token_id_end) for _ in range(args.context_len)
        ]
        rows.append(
            {
                "request_id": f"request-{request}",
                "input_ids": ids,
                "context_len": len(ids),
                "metadata": {
                    "generator": "uniform_token_range_v1",
                    "token_id_start": args.token_id_start,
                    "token_id_end_exclusive": args.token_id_end,
                    "seed": args.seed,
                    "request_seed": args.seed + request,
                    "vocab_size": vocab_size,
                },
            }
        )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True, help="Local model directory with config.json."
    )
    parser.add_argument("--token-id-start", type=int, required=True, help="First allowed input ID.")
    parser.add_argument("--token-id-end", type=int, required=True, help="Exclusive upper bound.")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_workload(args)
    write_jsonl(rows, args.output_jsonl)
    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "requests": len(rows),
                "context_len": args.context_len,
                "token_id_range": [args.token_id_start, args.token_id_end],
                "end_exclusive": True,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
