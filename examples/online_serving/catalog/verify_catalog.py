#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import argparse
import json
import sys
from pathlib import Path

import regex as re

SID = re.compile(r"<\|sid_begin\|><s_a_[0-9]+><s_b_[0-9]+><s_c_[0-9]+><\|sid_end\|>")


def verify(result_path: Path, catalog_path: Path, *, show_sids: bool = False) -> int:
    body = result_path.read_text(encoding="utf-8").lstrip()
    results, end = json.JSONDecoder().raw_decode(body)
    # Accept result files saved by the previous client, which appended curl timing.
    trailer = body[end:].strip()
    if trailer and not re.fullmatch(r"total time:[0-9]+(?:\.[0-9]+)?s", trailer):
        raise ValueError("Unexpected content after the response JSON")
    if not isinstance(results, dict):
        raise ValueError("Expected a response JSON object")
    if "error" in results:
        raise ValueError(f"Server returned an error: {results['error']}")
    choices = results.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response must contain a non-empty choices list")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("Expected a non-empty catalog list")
    catalog_set = set()
    for item in catalog:
        if (
            not isinstance(item, list)
            or len(item) != 5
            or not all(isinstance(token, str) for token in item)
        ):
            raise ValueError("Expected catalog entries with five SID tokens")
        catalog_set.add("".join(item))

    verified_sids = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise ValueError(f"Choice {index} must be an object")
        message = choice.get("message")
        content = choice.get("text", message.get("content") if isinstance(message, dict) else None)
        if not isinstance(content, str) or not SID.fullmatch(content):
            raise ValueError(f"Choice {index} must contain a complete SID")
        if content not in catalog_set:
            raise ValueError(f"Choice {index} is not in the catalog: {content}")
        verified_sids.append(content)
    if show_sids:
        print(*verified_sids, sep="\n")
    return len(choices)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated SIDs against a catalog.")
    parser.add_argument("result_path", type=Path, help="Path to the result JSON file")
    parser.add_argument("catalog_path", type=Path, help="Path to the source catalog JSON")
    parser.add_argument("--expected-count", type=int, help="Expected number of returned choices")
    parser.add_argument("--show-sids", action="store_true", help="Print the verified SIDs")
    args = parser.parse_args()
    try:
        count = verify(args.result_path, args.catalog_path, show_sids=args.show_sids)
        if args.expected_count is not None and count != args.expected_count:
            raise ValueError(f"Expected {args.expected_count} choices, got {count}")
    except (OSError, ValueError) as exc:
        print(f"FAILURE: {exc}", file=sys.stderr)
        return 1
    print(f"SUCCESS: All {count} generated SIDs exist in the catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
