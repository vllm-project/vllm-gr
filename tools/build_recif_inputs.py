#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate RECIF benchmark history files by repeating seed SIDs to target lengths.

Each output file is named "<num_sids>.txt" (e.g. 1024.txt for 1024 sids).
Each line in the file is one history: comma-separated 42-bit RECIF sids.
The token length of a history is 4x its sid count (4 tokens per sid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Classic seed SIDs used in daily testing.
DEFAULT_SEED_SIDS = [598080194427, 628177754964, 755993681678]
# Target sid counts per history (1k/2k/4k/8k/10k sids).
DEFAULT_SID_LENGTHS = [256, 512, 1024, 2048, 2560]


def build_histories(num_sids: int, seed_sids: list[int]) -> list[list[int]]:
    """Repeat each seed sid num_sids times, returning one history per seed sid.

    e.g. num_sids=1024, seed=598080194427 ->
         [598080194427] * 1024 (1024 sids in total)
    """
    return [[sid] * num_sids for sid in seed_sids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RECIF benchmark history files by repeating seed SIDs."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_inputs",
        help="Output directory for generated history files (default: tools/benchmark_inputs)",
    )
    parser.add_argument(
        "--sids",
        type=str,
        default=",".join(map(str, DEFAULT_SEED_SIDS)),
        help="Comma-separated seed sids to repeat (default: 598080194427,628177754964,755993681678)",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=",".join(map(str, DEFAULT_SID_LENGTHS)),
        help="Comma-separated target sid counts per history (default: 1024,2048,4096,8192,10240)",
    )
    args = parser.parse_args()

    seed_sids = [int(x.strip()) for x in args.sids.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if not seed_sids:
        print("Error: --sids must contain at least one sid", file=sys.stderr)
        return 2
    if not lengths:
        print("Error: --lengths must contain at least one length", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for num_sids in lengths:
        if num_sids <= 0:
            print(f"Error: length {num_sids} must be positive", file=sys.stderr)
            return 2
        histories = build_histories(num_sids, seed_sids)
        out_file = args.output_dir / f"{num_sids}.txt"
        lines = [",".join(str(s) for s in h) for h in histories]
        out_file.write_text("\n".join(lines) + "\n")
        print(
            f"[build_recif_inputs] {out_file} : {len(seed_sids)} histories x {num_sids} sids "
            f"(= {num_sids * 4} tokens each)"
        )

    print(f"[build_recif_inputs] Done. Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
