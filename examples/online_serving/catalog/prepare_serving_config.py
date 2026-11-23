#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Build a OneRec Constraint Table and shared HTTP/gRPC server configuration."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

SAMPLE_CATALOG = Path(__file__).with_name("sample_catalog.json")


def prepare_serving_config(
    *,
    model: Path,
    workdir: Path,
    backend: str,
    raw_catalog: Path | None = None,
    triples: Path | None = None,
) -> tuple[Path, Path]:
    """Return the published YAML and the table it references."""
    if raw_catalog is not None and triples is not None:
        raise ValueError("Choose either --raw-catalog or --triples")
    if backend not in {"cuda", "npu"}:
        raise ValueError("Constraint backend must be cuda or npu")
    model = model.expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"Expected a local model directory: {model}")
    workdir = workdir.expanduser().resolve()
    config_path = workdir / "vllm_gr_config.yaml"
    source = triples if triples is not None else raw_catalog or SAMPLE_CATALOG
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    if source == config_path.resolve():
        raise ValueError(f"Input file conflicts with generated YAML: {source}")

    import yaml

    from tools.constraints.build_constraint_table import build_constraint_table
    from tools.constraints.build_constraint_triples import process_catalog

    workdir.mkdir(parents=True, exist_ok=True)
    # A server using the previous YAML may still need its table. Publish a
    # complete new table under its own path before replacing the YAML.
    artifacts = Path(tempfile.mkdtemp(prefix="constraint-", dir=workdir))
    table_path = artifacts / "constraint_table.safetensors"
    pending_config = artifacts / config_path.name
    publishing = False
    try:
        triples_path = source
        if triples is None:
            triples_path = artifacts / "constraint_triples.json"
            process_catalog(
                str(source),
                str(triples_path),
                begin_token="<|sid_begin|>",
                end_token="<|sid_end|>",
            )
        artifact_digest, tokenizer_digest, _ = build_constraint_table(
            triples_path=triples_path,
            model=str(model),
            revision=None,
            output_path=table_path,
            trust_remote_code=True,
        )
        config = {
            "attention-config": {"backend": "CUSTOM"},
            "gr": {
                "beam_graph_enabled": True,
                "worker_decision": True,
                "beam_width": [128, 128, 128],
                "candidate_width": [128, 128, 128],
                "max_decode_steps": 3,
                "constraint_mode": "table",
                "constraint_table": {
                    "backend": backend,
                    "max_top_k": 128,
                    "path": str(table_path),
                    "format": "constraint_table_v1",
                    "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
                },
            },
        }
        pending_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        publishing = True
        pending_config.replace(config_path)
    except (Exception, KeyboardInterrupt):
        # A successful replace consumes the pending file. Keep its table
        # if the interrupt arrived after publication.
        if not publishing or pending_config.exists():
            shutil.rmtree(artifacts, ignore_errors=True)
        raise
    return config_path, table_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local OneRec model directory")
    parser.add_argument("--workdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--constraint-backend", choices=("cuda", "npu"), default="cuda")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--raw-catalog",
        type=Path,
        help="Raw catalog JSON with SID markers (default: two-SID sample)",
    )
    inputs.add_argument("--triples", type=Path, help="Existing constraint triples JSON")
    args = parser.parse_args(argv)
    try:
        config, table = prepare_serving_config(
            model=args.model,
            workdir=args.workdir,
            backend=args.constraint_backend,
            raw_catalog=args.raw_catalog,
            triples=args.triples,
        )
    except (OSError, ValueError) as exc:
        print(f"Could not prepare serving configuration: {exc}", file=sys.stderr)
        return 1
    print(f"Working directory: {config.parent}")
    print(f"Constraint Table: {table}")
    print(f"Server configuration: {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
