#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the Constraint Tables triples JSON from a catalog JSON.

Constraint Tables require every business sequence to contain exactly three
semantic tokens ``[s_a, s_b, s_c]``.  This tool validates a catalog and
reduces it to the tokenizer-independent triples document consumed by the
``ConstraintTableCompiler``.

Example catalog with begin/end sentinels::

    [
      ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"],
      ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_4>", "<|sid_end|>"]
    ]

Usage::

    python tools/build_constraint_triples.py \\
        --catalog catalog.json \\
        --output constraint_triples.json \\
        --begin "<|sid_begin|>" --end "<|sid_end|>"

A catalog without sentinels (raw three-token sequences) is also accepted when
``--begin``/``--end`` are omitted.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from vllm_gr.constraints.json_compiler import compile_relations_json

LOGGER = logging.getLogger(__name__)


def process_catalog(
    input_path: str,
    output_path: str,
    *,
    begin_token: str | None = None,
    end_token: str | None = None,
) -> int:
    """Validate a catalog JSON and write the triples JSON.

    Args:
        input_path: path to the catalog JSON.
        output_path: path to write the triples JSON to.
        begin_token: optional begin sentinel stripped from every sequence.
            ``None`` (default) means no begin sentinel.
        end_token: optional end sentinel stripped from every sequence.
            ``None`` (default) means no end sentinel.

    Returns:
        The number of triples written.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    try:
        with input_file.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Input catalog not found: {input_file}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to decode JSON from: {input_file}") from error

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of sequences in {input_file}, got {type(data).__name__}"
        )

    document = compile_relations_json(
        data,
        begin_token=begin_token,
        end_token=end_token,
    )
    triples = document["triples"]

    output_dir = output_file.parent
    if str(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, ensure_ascii=False)

    return len(triples)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Constraint Tables triples JSON from a catalog JSON."
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to the input catalog JSON (list of token-string sequences).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="constraint_triples.json",
        help="Output triples JSON path. Default: constraint_triples.json.",
    )
    parser.add_argument(
        "--begin",
        default=None,
        help="Begin sentinel token stripped from every sequence, e.g. '<|sid_begin|>'.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End sentinel token stripped from every sequence, e.g. '<|sid_end|>'.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        count = process_catalog(
            args.catalog,
            args.output,
            begin_token=args.begin,
            end_token=args.end,
        )
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info("Saved %d triples to %s", count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
