# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import urllib.request
from pathlib import Path

from examples.offline_inference.beam_search.recif_utils import (
    build_prefix,
    generated_tokens_to_sid,
)

HISTORY_SIDS = [598080194427, 628177754964, 755993681678]
EXPECTED_TOP_20 = [
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
MIN_TOP_20_HITS = 18
MAX_RANK_SHIFT = 2


def verify_sids(actual: list[int], mode: str) -> None:
    if len(actual) != 64:
        raise AssertionError(f"Expected 64 RECIF results, got {len(actual)}")
    top_20 = actual[:20]
    matched_sids = set(top_20) & set(EXPECTED_TOP_20)
    if len(matched_sids) < MIN_TOP_20_HITS:
        raise AssertionError(
            f"RECIF top-20 matched {len(matched_sids)}/{len(EXPECTED_TOP_20)}; "
            f"at least {MIN_TOP_20_HITS} matches are required:\n"
            f"expected={EXPECTED_TOP_20}\nactual={top_20}"
        )
    actual_rank = {sid: rank for rank, sid in enumerate(top_20)}
    expected_rank = {sid: rank for rank, sid in enumerate(EXPECTED_TOP_20)}
    for sid in matched_sids:
        rank_shift = abs(actual_rank[sid] - expected_rank[sid])
        if rank_shift > MAX_RANK_SHIFT:
            raise AssertionError(
                f"SID {sid} moved from rank {expected_rank[sid]} to {actual_rank[sid]}"
            )
    print(f"[RECIF E2E] {mode} result count: {len(actual)}")
    print(f"[RECIF E2E] {mode} Top-20 matches: {len(matched_sids)}/20")
    print(f"[RECIF E2E] {mode} Top-20 SIDs: {top_20}")
    print(f"[RECIF E2E] {mode} result verification passed")


def verify_offline(path: Path) -> None:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    results = [row for row in rows if len(row) == 3 and row[0].isdigit()]
    verify_sids([int(row[1]) for row in results], "offline")


def verify_online(url: str, model: str) -> None:
    prompt = build_prefix(HISTORY_SIDS)
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "use_beam_search": True,
            "n": 64,
            "max_tokens": 3,
            "return_token_ids": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    verify_sids(
        [generated_tokens_to_sid(choice["token_ids"]) for choice in payload["choices"]],
        "online",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-output", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.offline_output:
        verify_offline(args.offline_output)
    elif args.url and args.model:
        verify_online(args.url, args.model)
    else:
        parser.error("provide --offline-output or both --url and --model")


if __name__ == "__main__":
    main()
