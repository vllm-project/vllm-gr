# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify the stable RECIF top-20 SIDs while tolerating minor rank drift."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_TOP20 = [
    755993681678,
    1574911709347,
    741270118158,
    2146718517115,
    598080194427,
    137439842467,
    1120558817038,
    755937992462,
    572319781027,
    1304132591374,
    899277391378,
    109308989603,
    2140591120142,
    2020301000867,
    1788691554148,
    741270118244,
    2078615559950,
    1835898736734,
    1905735878523,
    547724005539,
]

RESULT_LINE = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<sid>\d+)\s+(?P<score>-?\d+(?:\.\d+)?)\s*$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-rank-drift", type=int, default=2)
    args = parser.parse_args()

    output_text = args.output.read_text(encoding="utf-8")
    ranked_sids: dict[int, int] = {}
    if output_text.lstrip().startswith("{"):
        payload = json.loads(output_text)
        ranked_sids = {
            int(result["rank"]): int(result["sid"])
            for result in payload["results"]
        }
    else:
        for line in output_text.splitlines():
            match = RESULT_LINE.match(line)
            if match:
                ranked_sids[int(match.group("rank"))] = int(match.group("sid"))

    if sorted(ranked_sids) != list(range(64)):
        raise SystemExit(
            "RECIF output must contain exactly 64 ranked result lines (0..63); "
            f"found ranks: {sorted(ranked_sids)}"
        )

    actual_top20 = [ranked_sids[index] for index in range(20)]
    if set(actual_top20) != set(EXPECTED_TOP20):
        missing = sorted(set(EXPECTED_TOP20) - set(actual_top20))
        unexpected = sorted(set(actual_top20) - set(EXPECTED_TOP20))
        raise SystemExit(
            f"RECIF top-20 SID mismatch; missing={missing}, unexpected={unexpected}"
        )

    actual_ranks = {sid: rank for rank, sid in enumerate(actual_top20)}
    excessive_drift = {
        sid: (expected_rank, actual_ranks[sid])
        for expected_rank, sid in enumerate(EXPECTED_TOP20)
        if abs(actual_ranks[sid] - expected_rank) > args.max_rank_drift
    }
    if excessive_drift:
        raise SystemExit(
            "RECIF top-20 rank drift exceeded "
            f"{args.max_rank_drift}: {excessive_drift}"
        )

    print("RECIF beam-search top-20 verification passed.")


if __name__ == "__main__":
    main()
