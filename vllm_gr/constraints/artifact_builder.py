# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Build a single-trie constraint artifact from an offline triples catalog."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest


def load_triples(path: Path) -> list[list[str]]:
    """Read legacy triples and sort them before rebuilding their relations."""
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} is not a nonempty Constraint Table triples document")
    triples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(key in row for key in ("a", "b", "c")):
            raise ValueError(f"invalid triples row {index}: {row!r}")
        triples.append([str(row["a"]), str(row["b"]), str(row["c"])])
    return sorted(triples)


def build_single_trie_artifact(
    model: Path,
    catalog: Path,
    artifact_path: Path,
) -> tuple[str, str]:
    """Build a single-trie artifact for a trusted model and return both digests.

    Accept legacy offline catalogs with only ``triples``. Rebuild the compiler
    document from those triples, ignoring any supplied ``format_version``,
    ``sentinels``, and ``relations`` fields. Model loading always uses
    ``trust_remote_code=True`` and leaves the revision unpinned.
    """
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        int(model_config.vocab_size),
        tokenizer_digest=tokenizer_digest,
    ).compile(compile_relations_json(load_triples(catalog)))
    artifact_digest = save_constraint_table_artifact(table, artifact_path)
    return artifact_digest, tokenizer_digest
