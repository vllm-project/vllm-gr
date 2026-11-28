#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Download and verify a pinned subset of the gated OpenOneRec-RecIF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import string
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

DEFAULT_REPO_ID = "OpenOneRec/OpenOneRec-RecIF"
DEFAULT_REVISION = "8f7cf2ee0b949e955a87a708d02024687be232c8"
DOWNLOAD_SOURCE = "hf-hub-download"
MANIFEST_SCHEMA = "vllm-gr.dataset.v1"
TASK_FILES = {
    "ad": "benchmark_data/ad/ad_test.parquet",
    "interactive": "benchmark_data/interactive/interactive_test.parquet",
    "label_cond": "benchmark_data/label_cond/label_cond_test.parquet",
    "label_pred": "benchmark_data/label_pred/label_pred_test.parquet",
    "product": "benchmark_data/product/product_test.parquet",
    "video": "benchmark_data/video/video_test.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tasks(value: str) -> list[str]:
    tasks = [task.strip() for task in value.split(",") if task.strip()]
    unknown = sorted(set(tasks) - TASK_FILES.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown task(s): {', '.join(unknown)}; available: {', '.join(TASK_FILES)}"
        )
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    return list(dict.fromkeys(tasks))


def destination_for(data_dir: Path, task: str) -> Path:
    return data_dir / task / f"{task}_test.parquet"


def manifest_entry(path: Path, task: str, source_file: str) -> dict[str, object]:
    return {
        "task": task,
        "source_file": source_file,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def install_file(cache_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(cache_path, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid dataset manifest: {path}")
    return manifest


def check_revision(path: Path, manifest: dict, repo_id: str, revision: str) -> None:
    actual_repo = manifest.get("repo_id")
    actual_revision = manifest.get("revision")
    if (actual_repo, actual_revision) != (repo_id, revision):
        raise ValueError(
            f"{path} records {actual_repo}@{actual_revision}; "
            f"expected {repo_id}@{revision}. Use a separate --data-dir."
        )


def verify_manifest(path: Path, args: argparse.Namespace) -> dict:
    manifest = read_manifest(path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("source") != DOWNLOAD_SOURCE
    ):
        raise ValueError(
            f"no download manifest at {path}; run without --verify-only to register data"
        )
    check_revision(path, manifest, args.repo_id, args.revision)
    recorded = {entry["task"]: entry for entry in manifest["files"]}
    verified = []
    for task in args.tasks:
        expected = recorded.get(task)
        if expected is None:
            raise ValueError(f"dataset manifest has no entry for task {task}")
        actual = manifest_entry(destination_for(path.parent, task), task, TASK_FILES[task])
        for field in ("source_file", "bytes", "sha256"):
            if expected.get(field) != actual[field]:
                raise ValueError(f"dataset integrity check failed for {actual['path']}: {field}")
        verified.append(actual)
    return {**manifest, "files": verified}


def download_manifest(path: Path, args: argparse.Namespace) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(".lock")):
        previous = read_manifest(path)
        entries = {}
        if previous.get("source") == DOWNLOAD_SOURCE:
            check_revision(path, previous, args.repo_id, args.revision)
            entries.update((entry["task"], entry) for entry in previous["files"])
        for task in args.tasks:
            source_file = TASK_FILES[task]
            destination = destination_for(path.parent, task)
            cached = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=source_file,
                    repo_type="dataset",
                    revision=args.revision,
                    endpoint=args.endpoint,
                )
            )
            install_file(cached, destination)
            entries[task] = manifest_entry(destination, task, source_file)

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "source": DOWNLOAD_SOURCE,
            "repo_id": args.repo_id,
            "revision": args.revision,
            "endpoint": args.endpoint,
            "generated_at": datetime.now(UTC).isoformat(),
            "files": list(entries.values()),
        }
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tasks", type=parse_tasks, default=parse_tasks("video"))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT") or "https://huggingface.co",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if len(args.revision) != 40 or any(c not in string.hexdigits for c in args.revision):
        parser.error("--revision must be a full commit SHA")

    manifest_path = args.data_dir.resolve() / ".onerec-manifest.json"
    try:
        manifest = (
            verify_manifest(manifest_path, args)
            if args.verify_only
            else download_manifest(manifest_path, args)
        )
    except GatedRepoError:
        print(
            "Dataset access is restricted. Accept the dataset terms on Hugging Face, "
            "then run `hf auth login` in your benchmark environment.",
            file=sys.stderr,
        )
        return 3
    except HfHubHTTPError as exc:
        print(f"dataset download failed: {exc}", file=sys.stderr)
        return 4
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"dataset preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
