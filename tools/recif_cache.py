# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve and publish a validated, atomic RECIF checkpoint cache entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

try:
    from tools.convert_recif_checkpoint import (
        InsufficientDiskSpaceError,
        convert_checkpoint,
    )
except ImportError:  # pragma: no cover - executed directly from tools/
    from convert_recif_checkpoint import InsufficientDiskSpaceError, convert_checkpoint


CONVERTER_PATH = Path(__file__).with_name("convert_recif_checkpoint.py")
MANIFEST_NAME = "conversion_manifest.json"
MANIFEST_VERSION = 1
REQUIRED_RUNTIME_FILES = ("config.json", "model.safetensors")
_HASH_CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def converter_sha() -> str:
    """Return the content hash of the converter that produces runtime files."""

    return _sha256_file(CONVERTER_PATH)


def checkpoint_identity(source: Path) -> str:
    """Return a content-derived identity for the source checkpoint."""

    required = ("config.json", "_model_rank0.pt", "external_rank0.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid RECIF checkpoint {source}: missing {', '.join(missing)}")

    digest = hashlib.sha256()
    for name in required:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(source / name).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_manifest(runtime_dir: Path) -> dict[str, Any] | None:
    manifest_path = runtime_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _runtime_cache_is_valid(
    runtime_dir: Path,
    source_identity: str,
    converter_sha_value: str,
) -> bool:
    manifest = _read_manifest(runtime_dir)
    if manifest is None:
        return False
    if manifest.get("version") != MANIFEST_VERSION:
        return False
    if manifest.get("source_identity") != source_identity:
        return False
    if manifest.get("converter_sha") != converter_sha_value:
        return False

    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for name in REQUIRED_RUNTIME_FILES:
        runtime_file = runtime_dir / name
        if not runtime_file.is_file():
            return False
        expected_hash = files.get(name)
        if not isinstance(expected_hash, str) or _sha256_file(runtime_file) != expected_hash:
            return False
    return True


def _build_manifest(
    source: Path,
    source_identity: str,
    converter_sha_value: str,
    runtime_dir: Path,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "source_identity": source_identity,
        "converter_sha": converter_sha_value,
        "source_path": str(source),
        "files": {name: _sha256_file(runtime_dir / name) for name in REQUIRED_RUNTIME_FILES},
    }


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is not None:
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return

    # Best-effort fallback for platforms without fcntl.
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        yield
    finally:
        lock_path.rmdir()


def _publish_runtime_dir(temp_dir: Path, output_dir: Path) -> None:
    """Atomically publish a fully written runtime directory."""

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.old-{os.getpid()}-{uuid.uuid4().hex}"
        )
        os.replace(output_dir, backup_dir)

    try:
        os.replace(temp_dir, output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def prepare_runtime_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> Path:
    """Return a validated cached checkpoint, converting it when necessary."""

    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_identity = checkpoint_identity(source)
    converter_sha_value = converter_sha()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")

    with _exclusive_lock(lock_path):
        if _runtime_cache_is_valid(output_dir, source_identity, converter_sha_value):
            return output_dir

        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
        )
        try:
            convert_checkpoint(
                source,
                temp_dir,
                allow_unsafe_pickle=allow_unsafe_pickle,
            )
            manifest = _build_manifest(
                source,
                source_identity,
                converter_sha_value,
                temp_dir,
            )
            (temp_dir / MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _publish_runtime_dir(temp_dir, output_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow a weights_only=False fallback for legacy checkpoints. "
            "This can execute arbitrary code and must only be used with trusted input."
        ),
    )
    args = parser.parse_args()
    source = args.source
    output_dir = args.output if args.output is not None else source.with_name(f"{source.name}_hf")
    try:
        runtime_dir = prepare_runtime_checkpoint(
            source,
            output_dir,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        )
    except InsufficientDiskSpaceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(runtime_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
