# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the immutable six-table safetensors Artifact used by Workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.artifact import write_constraint_table_artifact
from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.table import ConstraintTable, ConstraintTableInfo
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest


def build_constraint_table(
    *,
    triples_path: str | Path,
    model: str,
    output_path: str | Path,
    trust_remote_code: bool = False,
) -> tuple[str, str, int]:
    """Compile triples offline and return artifact/tokenizer digests and vocab size."""

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    vocab_size = getattr(model_config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"model {model!r} does not expose a positive vocab_size")

    triples_file = Path(triples_path)
    try:
        document: Any = json.loads(triples_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read triples document {str(triples_file)!r}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("constraint triples document must be a JSON object")

    compiled = ConstraintTableCompiler(tokenizer, vocab_size).compile(document)
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    info = ConstraintTableInfo(
        format=CONSTRAINT_TABLE_FORMAT,
        artifact_digest="unpersisted",
        tokenizer_digest=tokenizer_digest,
        vocab_size=vocab_size,
        max_prefix1_degree=compiled.max_prefix1_degree,
        max_prefix2_degree=compiled.max_prefix2_degree,
        total_bytes=compiled.total_bytes,
    )
    table = ConstraintTable(
        first_token_ids=compiled.first_token_ids,
        prefix1_offsets=compiled.prefix1_offsets,
        prefix1_values=compiled.prefix1_values,
        prefix1_pair_keys=compiled.prefix1_pair_keys,
        prefix2_value_offsets=compiled.prefix2_value_offsets,
        prefix2_values=compiled.prefix2_values,
        info=info,
    )
    artifact_digest = write_constraint_table_artifact(table, output_path)
    return artifact_digest, tokenizer_digest, vocab_size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", required=True, help="Constraint triples JSON document")
    parser.add_argument("--model", required=True, help="Tokenizer/model name or local directory")
    parser.add_argument("--output", required=True, help="Output .safetensors Artifact")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifact_digest, tokenizer_digest, vocab_size = build_constraint_table(
        triples_path=args.triples,
        model=args.model,
        output_path=args.output,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"artifact_digest: {artifact_digest}")
    print(f"tokenizer_digest: {tokenizer_digest}")
    print(f"vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
