# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retry stages inside one uninterrupted device job using exact-snapshot evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

MAX_AGE_SECONDS = 6 * 60 * 60


def content_digest(path: Path) -> str:
    """Read the bytes: restored timestamps cannot prove content is unchanged."""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ):
        raise RuntimeError(f"Asset changed while computing its digest: {path}")
    return digest.hexdigest()


def asset_inventory(stage: str) -> list[tuple[str, str]]:
    """Identify stage inputs by content, ignoring regeneration and restore times."""
    roots = {"OpenOneRec", "generated_data"}
    if stage in {"device-tests", "pangu-offline", "pangu-online"}:
        roots.add("openPangu-Embedded-1B-V1.1")
    if stage == "device-tests":
        # Catalogs are created after this stage, so their presence on a retry
        # must not invalidate a device-suite checkpoint.
        roots.discard("generated_data")
    roots.add(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    roots.update(
        value
        for key, value in os.environ.items()
        if key.endswith(("MODEL_PATH", "DATASET_PATH", "TRIPLES_PATH", "CATALOG_PATH"))
        and (
            not key.startswith("PANGU1B_")
            or stage in {"device-tests", "pangu-offline", "pangu-online"}
        )
    )
    files = set()
    for root in sorted(roots):
        path = Path(root)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
    return [(str(path), content_digest(path)) for path in sorted(files)]


def fingerprint(stage: str, command: list[str]) -> str:
    required = ("CI_HEAD_SHA", "CI_BASE_SHA", "CI_MERGE_SHA", "RUNNER_NAME")
    if not all(os.environ.get(key) for key in required):
        raise ValueError("stage checkpoints require complete snapshot and runner identity")
    mapping = Path(os.environ["CI_CUDA_MAPPING_FILE"]).read_text()
    policy = Path(os.environ["RUNNER_TEMP"], "resolved-benchmark-policy.env").read_text()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        text=True,
        timeout=30,
    )
    versions = sorted(
        (dist.metadata.get("Name", ""), dist.version) for dist in importlib.metadata.distributions()
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("VLLM_", "VLLM_GR_", "V1_", "PANGU1B_", "CUDA_")) or key in required
    }
    payload = [
        stage,
        command,
        environment,
        mapping,
        policy,
        versions,
        driver,
        asset_inventory(stage),
        os.environ.get("CI_RUNTIME_REVISION", "1"),
        platform.platform(),
        sys.version,
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable(record: dict, key: str, now: float) -> bool:
    completed = record.get("completed_at")
    return (
        record.get("fingerprint") == key
        and record.get("returncode") == 0
        and isinstance(completed, (int, float))
        and 0 <= now - completed < MAX_AGE_SECONDS
    )


def execute(stage: str, command: list[str], cache: Path, key: str, reuse: bool) -> int:
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{key}.json"
    record = {}
    try:
        record = json.loads(checkpoint.read_text())
    except (OSError, ValueError):
        pass
    reused = reuse and isinstance(record, dict) and reusable(record, key, time.time())
    if reused:
        print(f"Reuse {stage}: passed in run {record.get('run_id')}, same snapshot and runtime.")
    else:
        started = time.time()
        # Invalidate the old proof before executing; interruption cannot leave
        # behind a reusable success from an earlier attempt.
        checkpoint.unlink(missing_ok=True)
        result = subprocess.run(command, check=False)
        record = {
            "stage": stage,
            "fingerprint": key,
            "returncode": result.returncode,
            "started_at": started,
            "completed_at": time.time(),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        }
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(checkpoint)
    evidence = Path("ci-results")
    evidence.mkdir(exist_ok=True)
    (evidence / f"{stage}.json").write_text(
        json.dumps({**record, "reused": reused}, indent=2) + "\n"
    )
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        elapsed = record.get("completed_at", 0) - record.get("started_at", 0)
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(
                f"- {stage}: {'reused' if reused else 'executed'}, "
                f"exit={record['returncode']}, {elapsed:.1f}s; source run {record.get('run_id')}\n"
            )
    return int(record["returncode"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "device-tests",
            "recall",
            "performance",
            "decode",
            "pangu-offline",
            "pangu-online",
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a stage command is required")
    key = fingerprint(args.stage, command)
    cache = Path(os.environ["CI_CHECKPOINT_ROOT"])
    return execute(args.stage, command, cache, key, os.environ.get("CI_REUSE_SUCCESS") == "true")


if __name__ == "__main__":
    raise SystemExit(main())
