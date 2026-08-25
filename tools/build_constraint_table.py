#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a canonical constraint-table safetensors artifact offline."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

logger = logging.getLogger(__name__)
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _resolve_model_revision(model: str, revision: str | None) -> str | None:
    """Pin a remote model reference once; local snapshots need no revision.

    A local directory is treated as caller-owned immutable build input.  Every
    other value is a remote model ID (or an invalid local path) and must name a
    revision.  Symbolic branches/tags are resolved to a commit SHA before any
    Transformers loader runs so tokenizer and config cannot observe different
    commits if the ref moves between their two calls.
    """

    local_path = Path(model).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(f"local model path {str(local_path)!r} must be a directory")
        return revision
    if revision is None:
        raise ValueError(
            f"remote model {model!r} requires --revision; use an immutable commit "
            "SHA or a branch/tag that can be resolved to one"
        )
    if _COMMIT_SHA_RE.fullmatch(revision) is not None:
        return revision.lower()

    try:
        from huggingface_hub import HfApi

        resolved_revision = HfApi().model_info(model, revision=revision).sha
    except Exception as exc:
        raise ValueError(
            f"cannot resolve remote model {model!r} revision {revision!r} "
            f"to an immutable commit SHA: {exc}"
        ) from exc
    if (
        not isinstance(resolved_revision, str)
        or _COMMIT_SHA_RE.fullmatch(resolved_revision) is None
    ):
        raise ValueError(
            f"remote model {model!r} revision {revision!r} resolved to invalid "
            f"commit SHA {resolved_revision!r}"
        )
    resolved_revision = resolved_revision.lower()
    logger.info(
        "Resolved remote model %s revision %s to immutable commit %s",
        model,
        revision,
        resolved_revision,
    )
    return resolved_revision


def build_constraint_table(
    *,
    triples_path: str | Path,
    model: str,
    revision: str | None,
    output_path: str | Path,
    trust_remote_code: bool = False,
) -> tuple[str, str, int]:
    """Compile triples and return artifact digest, tokenizer digest, and vocab.

    ``revision`` is required for a remote model ID and is resolved to one
    immutable commit before loading.  It may be omitted only for an existing
    local model directory, which callers must preserve as an immutable build
    input if they require reproducibility.
    """

    resolved_revision = _resolve_model_revision(model, revision)

    from transformers import AutoConfig, AutoTokenizer

    load_kwargs = {
        "revision": resolved_revision,
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
    model_config = AutoConfig.from_pretrained(model, **load_kwargs)
    vocab_size = getattr(model_config, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError(f"model {model!r} does not expose a positive integer vocab_size")

    triples_file = Path(triples_path)
    try:
        with triples_file.open("r", encoding="utf-8") as input_file:
            document: Any = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read triples document {str(triples_file)!r}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("constraint triples document must be a JSON object")

    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        vocab_size,
        tokenizer_digest=tokenizer_digest,
    ).compile(document)
    artifact_digest = save_constraint_table_artifact(table, output_path)
    return artifact_digest, tokenizer_digest, vocab_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", required=True, help="Constraint triples JSON")
    parser.add_argument("--model", required=True, help="Model ID or local path")
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Model/tokenizer revision; required for remote model IDs and "
            "resolved to an immutable commit SHA"
        ),
    )
    parser.add_argument("--output", required=True, help="Output .safetensors artifact path")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact_digest, tokenizer_digest, vocab_size = build_constraint_table(
            triples_path=args.triples,
            model=args.model,
            revision=args.revision,
            output_path=args.output,
            trust_remote_code=args.trust_remote_code,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    print(f"artifact_digest: {artifact_digest}")
    print(f"tokenizer_digest: {tokenizer_digest}")
    print(f"vocab_size: {vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
