#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare equivalent Trie and Constraint-Table data for five OneRec tasks.

The four video-ID tasks share ``sid2pid.json``; product uses ``sid2iid.json``.
Each logical universe is materialized once as a Trie catalog and then compiled
to the triples document consumed by decode_graph.  An optional canonical
safetensors artifact avoids runtime compilation on branches that support the
schema-v1 constraint-table configuration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any, Iterable

import regex as re

from benchmarks.open_one_rec.regression import sha256_file
from benchmarks.open_one_rec.tools.convert_catalog import process_json
from tools.build_constraint_table import build_constraint_table
from tools.build_constraint_triples import process_catalog

LOGGER = logging.getLogger(__name__)

DOMAINS = {
    "video": {
        "mapping_file": "sid2pid.json",
        "tasks": ["video", "ad", "label_cond", "interactive"],
    },
    "product": {
        "mapping_file": "sid2iid.json",
        "tasks": ["product"],
    },
}

_BASE = 8192
_SID_TOKEN_RE = re.compile(r"^<s_([abc])_(\d+)>$")


def _json_length(path: Path) -> int:
    with path.open(encoding="utf-8") as file_obj:
        value: Any = json.load(file_obj)
    if not isinstance(value, (dict, list)):
        raise ValueError(f"expected JSON object/list in {path}, got {type(value).__name__}")
    return len(value)


def _digest_sid_values(values: Iterable[int], *, source: Path) -> tuple[int, str]:
    """Return an order-independent digest of one logical SID universe."""
    normalized = sorted(values)
    digest = hashlib.sha256()
    previous: int | None = None
    for value in normalized:
        if value < 0 or value >= _BASE**3:
            raise ValueError(f"SID value {value} in {source} is outside the 3-token space")
        if value == previous:
            raise ValueError(f"duplicate SID value {value} in {source}")
        digest.update(struct.pack(">Q", value))
        previous = value
    return len(normalized), digest.hexdigest()


def _tokens_to_sid_value(tokens: Any, *, source: Path) -> int:
    if not isinstance(tokens, list) or len(tokens) != 3:
        raise ValueError(f"expected one [s_a, s_b, s_c] triple in {source}: {tokens!r}")
    values: list[int] = []
    for expected_level, token in zip("abc", tokens):
        match = _SID_TOKEN_RE.fullmatch(token) if isinstance(token, str) else None
        if match is None or match.group(1) != expected_level:
            raise ValueError(f"invalid s_{expected_level} token in {source}: {token!r}")
        component = int(match.group(2))
        if component >= _BASE:
            raise ValueError(f"SID component is outside [0, {_BASE}) in {source}: {token!r}")
        values.append(component)
    return values[0] * _BASE * _BASE + values[1] * _BASE + values[2]


def _mapping_universe(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8") as file_obj:
        document: Any = json.load(file_obj)
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object in {path}, got {type(document).__name__}")
    try:
        values = (int(key) for key in document)
        return _digest_sid_values(values, source=path)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer SID key in {path}: {error}") from error


def _catalog_universe(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8") as file_obj:
        document: Any = json.load(file_obj)
    if not isinstance(document, list):
        raise ValueError(f"expected JSON list in {path}, got {type(document).__name__}")

    def values() -> Iterable[int]:
        for sequence in document:
            if not isinstance(sequence, list) or len(sequence) != 5:
                raise ValueError(f"invalid catalog sequence in {path}: {sequence!r}")
            if sequence[0] != "<|sid_begin|>" or sequence[-1] != "<|sid_end|>":
                raise ValueError(f"invalid catalog sentinels in {path}: {sequence!r}")
            yield _tokens_to_sid_value(sequence[1:4], source=path)

    return _digest_sid_values(values(), source=path)


def _triples_universe(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8") as file_obj:
        document: Any = json.load(file_obj)
    triples = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(triples, list):
        raise ValueError(f"invalid triples document: {path}")

    def values() -> Iterable[int]:
        for triple in triples:
            if not isinstance(triple, dict) or set(triple) != {"a", "b", "c"}:
                raise ValueError(f"invalid relation triple in {path}: {triple!r}")
            yield _tokens_to_sid_value([triple["a"], triple["b"], triple["c"]], source=path)

    return _digest_sid_values(values(), source=path)


def _reuse_or_build(
    *,
    source: Path,
    catalog: Path,
    triples: Path,
    force: bool,
) -> tuple[int, int, int, str]:
    source_count = _json_length(source)

    if force or not catalog.exists():
        LOGGER.info("Building Trie catalog %s from %s", catalog, source)
        catalog_count = process_json(str(source), str(catalog))
    else:
        LOGGER.info("Reusing Trie catalog %s", catalog)
        catalog_count = _json_length(catalog)
    gc.collect()

    if force or not triples.exists():
        LOGGER.info("Building decode triples %s from %s", triples, catalog)
        triples_count = process_catalog(
            str(catalog),
            str(triples),
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )
    else:
        LOGGER.info("Reusing decode triples %s", triples)
        with triples.open(encoding="utf-8") as file_obj:
            document = json.load(file_obj)
        raw_triples = document.get("triples") if isinstance(document, dict) else None
        if not isinstance(raw_triples, list):
            raise ValueError(f"invalid triples document: {triples}")
        triples_count = len(raw_triples)
        del raw_triples, document
    gc.collect()

    source_universe = _mapping_universe(source)
    catalog_universe = _catalog_universe(catalog)
    triples_universe = _triples_universe(triples)
    if source_count != catalog_count or source_count != triples_count:
        raise ValueError(
            f"logical universe count mismatch for {source.name}: "
            f"source={source_count}, catalog={catalog_count}, triples={triples_count}"
        )
    if source_universe != catalog_universe or source_universe != triples_universe:
        raise ValueError(
            f"logical SID universe mismatch for {source.name}: "
            f"source={source_universe}, catalog={catalog_universe}, "
            f"triples={triples_universe}"
        )
    return source_count, catalog_count, triples_count, source_universe[1]


def prepare_decode_data(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    model: str | None = None,
    revision: str | None = None,
    build_artifacts: bool = False,
    artifact_backend: str = "cuda",
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    previous_manifest: dict[str, Any] = {}
    previous_manifest_path = output_root / "manifest.json"
    if previous_manifest_path.is_file():
        try:
            with previous_manifest_path.open(encoding="utf-8") as file_obj:
                loaded_manifest: Any = json.load(file_obj)
            if isinstance(loaded_manifest, dict):
                previous_manifest = loaded_manifest
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring unreadable previous manifest %s", previous_manifest_path)

    if build_artifacts and not model:
        raise ValueError("--model is required with --build-artifacts")

    # Verify a local model's current tokenizer identity before reusing a
    # tokenizer-bound artifact. Remote references are rebuilt because their
    # revision pinning is resolved inside build_constraint_table.
    current_tokenizer_digest: str | None = None
    current_vocab_size: int | None = None
    if build_artifacts and model and Path(model).expanduser().is_dir():
        from transformers import AutoConfig, AutoTokenizer

        from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

        load_kwargs = {"revision": revision, "trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
        model_config = AutoConfig.from_pretrained(model, **load_kwargs)
        current_tokenizer_digest = compute_tokenizer_digest(tokenizer)
        current_vocab_size = getattr(model_config, "vocab_size", None)
        if type(current_vocab_size) is not int or current_vocab_size <= 0:
            raise ValueError(f"model {model!r} does not expose a positive integer vocab_size")

    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset_path": str(dataset_root),
        "tasks": {},
        "domains": {},
    }

    for domain, config in DOMAINS.items():
        mapping = dataset_root / str(config["mapping_file"])
        if not mapping.is_file():
            raise FileNotFoundError(f"OneRec mapping was not found: {mapping}")
        domain_dir = output_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        catalog = domain_dir / "catalog.json"
        triples = domain_dir / "constraint_triples.json"
        artifact = domain_dir / "constraint_table.safetensors"

        source_count, catalog_count, triples_count, universe_digest = _reuse_or_build(
            source=mapping,
            catalog=catalog,
            triples=triples,
            force=force,
        )

        domain_manifest: dict[str, Any] = {
            "mapping_file": str(mapping),
            "mapping_sha256": sha256_file(mapping),
            "catalog_path": str(catalog),
            "catalog_sha256": sha256_file(catalog),
            "triples_path": str(triples),
            "triples_sha256": sha256_file(triples),
            "source_count": source_count,
            "catalog_count": catalog_count,
            "triples_count": triples_count,
            "logical_universe_sha256": universe_digest,
        }

        previous_domains = previous_manifest.get("domains", {})
        previous_domain = (
            previous_domains.get(domain, {}) if isinstance(previous_domains, dict) else {}
        )
        previous_artifact = (
            previous_domain.get("artifact", {}) if isinstance(previous_domain, dict) else {}
        )
        if build_artifacts:
            artifact_inputs_match = (
                isinstance(previous_artifact, dict)
                and {"artifact_digest", "tokenizer_digest", "vocab_size"} <= set(previous_artifact)
                and previous_domain.get("mapping_sha256") == domain_manifest["mapping_sha256"]
                and previous_domain.get("triples_sha256") == domain_manifest["triples_sha256"]
                and current_tokenizer_digest is not None
                and previous_artifact.get("tokenizer_digest") == current_tokenizer_digest
                and previous_artifact.get("vocab_size") == current_vocab_size
            )
            if force or not artifact.exists() or not artifact_inputs_match:
                LOGGER.info("Building canonical Constraint Table artifact %s", artifact)
                artifact_digest, tokenizer_digest, vocab_size = build_constraint_table(
                    triples_path=triples,
                    model=str(model),
                    revision=revision,
                    output_path=artifact,
                    trust_remote_code=True,
                )
                if current_tokenizer_digest is not None and (
                    tokenizer_digest != current_tokenizer_digest or vocab_size != current_vocab_size
                ):
                    raise ValueError("artifact compiler used a different tokenizer/model identity")
            else:
                from vllm_gr.constraints.artifact import load_constraint_table_artifact

                LOGGER.info("Validating existing canonical Constraint Table artifact %s", artifact)
                table = load_constraint_table_artifact(
                    artifact,
                    expected_digest=previous_artifact["artifact_digest"],
                    expected_tokenizer_digest=previous_artifact["tokenizer_digest"],
                    expected_vocab_size=previous_artifact["vocab_size"],
                )
                artifact_digest = table.metadata.artifact_digest
                tokenizer_digest = table.metadata.tokenizer_digest
                vocab_size = table.metadata.vocab_size
            domain_manifest["artifact"] = {
                "path": str(artifact),
                "sha256": sha256_file(artifact),
                "artifact_digest": artifact_digest,
                "tokenizer_digest": tokenizer_digest,
                "vocab_size": vocab_size,
                "backend": artifact_backend,
                "max_top_k": 256,
                "model": str(model),
                "revision": revision,
            }
        elif (
            isinstance(previous_artifact, dict)
            and {
                "artifact_digest",
                "tokenizer_digest",
                "vocab_size",
                "sha256",
            }
            <= set(previous_artifact)
            and previous_domain.get("mapping_sha256") == domain_manifest["mapping_sha256"]
            and previous_domain.get("triples_sha256") == domain_manifest["triples_sha256"]
            and artifact.is_file()
            and sha256_file(artifact) == previous_artifact["sha256"]
        ):
            # A triples-only validation must not erase an already validated
            # canonical artifact from the manifest. Rebase its path to this
            # output directory (important for restored CI caches); a future
            # artifact build will still revalidate model/tokenizer identity.
            domain_manifest["artifact"] = {
                **previous_artifact,
                "path": str(artifact),
            }

        manifest["domains"][domain] = domain_manifest
        for task in config["tasks"]:
            manifest["tasks"][task] = domain

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    LOGGER.info("Prepared OneRec decode data manifest: %s", manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="OpenOneRec benchmark_data root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None, help="Model path/ID for artifact compilation")
    parser.add_argument("--revision", default=None, help="Pinned revision for a remote model")
    parser.add_argument("--build-artifacts", action="store_true")
    parser.add_argument("--artifact-backend", choices=["cuda", "npu", "cpu"], default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    try:
        prepare_decode_data(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model=args.model,
            revision=args.revision,
            build_artifacts=args.build_artifacts,
            artifact_backend=args.artifact_backend,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
