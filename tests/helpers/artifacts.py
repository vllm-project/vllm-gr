# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""On-demand, verified inputs for model tests; importing this module does no I/O."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from tests.helpers.paths import REPO_ROOT

_MANIFEST = ".gr-artifact.json"
_SCHEMA = 1


class ArtifactError(RuntimeError):
    """A required test input could not be prepared; never a skip condition."""


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    repo_id: str
    revision: str
    repo_type: Literal["model", "dataset"]
    required_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.repo_id
            or self.repo_type not in ("model", "dataset")
            or len(self.revision) != 40
            or any(char not in "0123456789abcdef" for char in self.revision)
            or not self.required_files
        ):
            raise ValueError("Snapshot specs require a name, repository, full commit SHA and files")
        for name in self.required_files:
            _relative(name)


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or name == _MANIFEST
        or path.parts[0] == ".cache"
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in ("..", ".") for part in name.split("/"))
    ):
        raise ValueError(f"Unsafe artifact-relative path: {name!r}")
    return Path(*path.parts)


def snapshot_spec(name: str) -> SnapshotSpec:
    """Read the reviewed registry only when a selected fixture needs it."""
    registry = json.loads((REPO_ROOT / "tests/resources/artifacts.json").read_text("utf-8"))
    entry = registry[name]
    return SnapshotSpec(name=name, **{**entry, "required_files": tuple(entry["required_files"])})


def cache_root() -> Path:
    value = os.environ.get("VLLM_GR_TEST_CACHE")
    return (
        Path(value).expanduser().resolve()
        if value
        else (Path.home() / ".cache/vllm-gr/test-artifacts")
    )


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _file(root: Path, name: str, *, allow_external: bool = False) -> Path:
    path = root / _relative(name)
    if not allow_external and not path.resolve().is_relative_to(root.resolve()):
        raise ArtifactError(f"Artifact file escapes its directory: {name}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"Missing or empty artifact file: {name}")
    return path


def _files(root: Path) -> dict[str, Any]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache" or relative.as_posix() == _MANIFEST:
            continue
        if path.is_symlink():
            # Local overrides may be HF snapshots with links into their blob
            # cache. Containment is enforced for managed cache files instead.
            if not path.is_file():
                raise ArtifactError(f"Broken artifact link: {relative}")
        if path.is_file():
            result[relative.as_posix()] = file_identity(path)
    if not result:
        raise ArtifactError("Artifact directory has no files")
    return result


def _validate_snapshot(root: Path, spec: SnapshotSpec, *, allow_external: bool = False) -> None:
    # A user checkpoint may use another valid shard layout (e.g. .bin instead
    # of safetensors). Do not require the public snapshot's exact filenames.
    required = (
        ("config.json", "tokenizer_config.json")
        if allow_external and spec.repo_type == "model"
        else spec.required_files
    )
    for name in required:
        path = _file(root, name, allow_external=allow_external)
        if path.suffix == ".json":
            try:
                json.loads(path.read_text("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ArtifactError(f"Invalid JSON artifact: {name}") from exc
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            try:
                pq.read_metadata(path)
            except (OSError, ValueError) as exc:
                raise ArtifactError(f"Invalid Parquet metadata: {name}") from exc
    if spec.repo_type != "model":
        return
    for name in ("config.json", "tokenizer_config.json"):
        if not isinstance(
            json.loads(_file(root, name, allow_external=allow_external).read_text("utf-8")), dict
        ):
            raise ArtifactError(f"Model {name} must be an object")
    tokenizer_files = next(
        (
            names
            for names in (("tokenizer.json",), ("tokenizer.model",), ("vocab.json", "merges.txt"))
            if all((root / name).is_file() for name in names)
        ),
        None,
    )
    if tokenizer_files is None:
        raise ArtifactError("Missing or empty tokenizer files")
    for name in tokenizer_files:
        tokenizer_file = _file(root, name, allow_external=allow_external)
        if tokenizer_file.suffix == ".json":
            try:
                json.loads(tokenizer_file.read_text("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ArtifactError(f"Invalid tokenizer JSON: {name}") from exc
    if allow_external:
        for config_name in ("config.json", "tokenizer_config.json"):
            config = json.loads((root / config_name).read_text("utf-8"))
            auto_map = config.get("auto_map", {})
            if auto_map is None:
                auto_map = {}
            if not isinstance(auto_map, dict):
                raise ArtifactError(f"Invalid auto_map in {config_name}")
            for references in auto_map.values():
                references = references if isinstance(references, list) else [references]
                for reference in references:
                    if reference is None:
                        continue
                    if not isinstance(reference, str) or "." not in reference or "--" in reference:
                        raise ArtifactError(
                            "Local custom-code references must resolve within the model"
                        )
                    module = reference.rsplit(".", 1)[0].replace(".", "/") + ".py"
                    _file(root, module, allow_external=True)
    shards: set[Path] = set()
    indexes = list(root.glob("*.index.json"))
    if indexes:
        for index in indexes:
            document = json.loads(index.read_text("utf-8"))
            weights = document.get("weight_map") if isinstance(document, dict) else None
            if not isinstance(weights, dict) or not weights:
                raise ArtifactError(f"Invalid weight index: {index.name}")
            for shard in weights.values():
                if not isinstance(shard, str):
                    raise ArtifactError(f"Invalid shard name in {index.name}")
                shards.add(_file(root, shard, allow_external=allow_external))
    else:
        weights = [*root.glob("*.safetensors"), *root.glob("*.bin")]
        if not weights:
            raise ArtifactError("Model has no weights")
        for weight in weights:
            shards.add(_file(root, weight.name, allow_external=allow_external))
    for path in shards:
        if path.suffix == ".safetensors":
            from safetensors import SafetensorError, safe_open

            try:
                with safe_open(str(path), framework="np") as weights_file:
                    if not weights_file.keys():
                        raise ArtifactError(f"Empty weight shard: {path.name}")
            except SafetensorError as exc:
                raise ArtifactError(f"Invalid safetensors shard: {path.name}") from exc


def _download(spec: SnapshotSpec, output: Path) -> None:
    from huggingface_hub import snapshot_download

    # The Hub SDK owns HTTP retries, timeouts, authentication and its download cache.
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            revision=spec.revision,
            local_dir=str(output),
            allow_patterns=list(spec.required_files),
            etag_timeout=60,
        )
    except Exception:
        raise ArtifactError(f"Download failed for {spec.repo_id}@{spec.revision}") from None


def _cached(
    identity: dict[str, Any],
    producer: Callable[[Path], None],
    validator: Callable[[Path], None],
    root: Path | None,
    check_inputs: Callable[[], None] | None = None,
) -> Path:
    from filelock import FileLock

    storage = ((root if root is not None else cache_root()) / "objects").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    key = _digest(identity)
    target = storage / key
    with FileLock(str(storage / f"{key}.lock"), timeout=600):
        if check_inputs is not None:
            check_inputs()
        if target.is_symlink() or target.resolve().parent != storage:
            raise ArtifactError("Managed artifact cache must not be a symbolic link")
        if target.is_dir():
            try:
                manifest = json.loads((target / _MANIFEST).read_text("utf-8"))
                validator(target)
                if manifest == {"schema": _SCHEMA, "input": identity, "files": _files(target)}:
                    return target
            except (OSError, ValueError, ArtifactError):
                pass
        # Build elsewhere on the same filesystem. A failed/cancelled producer
        # never leaves a directory that can be mistaken for a complete input.
        with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=storage) as temporary:
            payload = Path(temporary) / "payload"
            payload.mkdir()
            producer(payload)
            validator(payload)
            files = _files(payload)
            for name in files:
                _file(payload, name)
            (payload / _MANIFEST).write_text(
                json.dumps(
                    {"schema": _SCHEMA, "input": identity, "files": files}, sort_keys=True, indent=2
                )
                + "\n",
                "utf-8",
            )
            # target is our digest-named child of storage, checked against links
            # above. Never remove or write to a local model override.
            if target.exists():
                if not target.is_dir():
                    raise ArtifactError("Managed artifact target is not a directory")
                shutil.rmtree(target)
            payload.replace(target)
    return target


def _ensure_snapshot(spec: SnapshotSpec, root: Path | None, local_override: Path | None) -> Path:
    if local_override is not None:
        local = local_override.expanduser().resolve()
        if not local.is_dir():
            raise ArtifactError(f"Explicit local {spec.name} directory does not exist: {local}")
        _validate_snapshot(local, spec, allow_external=True)
        return local
    return _cached(
        {"kind": "hub-snapshot", **json.loads(json.dumps(asdict(spec)))},
        lambda output: _download(spec, output),
        lambda output: _validate_snapshot(output, spec),
        root,
    )


def ensure_model(
    spec: SnapshotSpec, *, cache_dir: Path | None = None, local_override: Path | None = None
) -> Path:
    if spec.repo_type != "model":
        raise ValueError("ensure_model requires a model spec")
    return _ensure_snapshot(spec, cache_dir, local_override)


def ensure_dataset(
    spec: SnapshotSpec, *, cache_dir: Path | None = None, local_override: Path | None = None
) -> Path:
    if spec.repo_type != "dataset":
        raise ValueError("ensure_dataset requires a dataset spec")
    return _ensure_snapshot(spec, cache_dir, local_override)


def ensure_derived(
    *,
    name: str,
    inputs: Mapping[str, Path],
    parameters: Mapping[str, Any],
    builder_version: str,
    required_files: tuple[str, ...],
    builder: Callable[[Path], None],
    cache_dir: Path | None = None,
) -> Path:
    """Cache derived outputs by input bytes, parameters and builder version.

    Include builder source files in inputs when calling existing CLI tools.
    A builder receives a private output directory and may not mutate inputs.
    """
    if not name or not inputs or not builder_version or not required_files:
        raise ValueError("Derived artifacts require name, inputs, builder version and output files")
    for filename in required_files:
        _relative(filename)
    sources = {key: path.resolve() for key, path in inputs.items()}
    fingerprints = {key: file_identity(path) for key, path in sources.items()}
    identity = json.loads(
        json.dumps(
            {
                "kind": "derived",
                "name": name,
                "inputs": fingerprints,
                "parameters": dict(parameters),
                "builder_version": builder_version,
                "required_files": required_files,
            }
        )
    )

    def check_inputs() -> None:
        if fingerprints != {key: file_identity(path) for key, path in sources.items()}:
            raise ArtifactError("Derived input changed while the builder was running or waiting")

    def validate(output: Path) -> None:
        check_inputs()
        for filename in required_files:
            path = _file(output, filename)
            if path.suffix == ".json":
                try:
                    json.loads(path.read_text("utf-8"))
                except (ValueError, UnicodeError) as exc:
                    raise ArtifactError(f"Invalid derived JSON: {filename}") from exc

    return _cached(identity, builder, validate, cache_dir, check_inputs)
