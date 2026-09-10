# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lifecycle helper for the Rust beam-proc sidecar.

The APIServer/main process owns sidecar startup.  When explicitly enabled,
it generates local ZMQ addresses, starts the compiled Rust binary, and exposes
the EngineCore-facing address through the environment for the spawned
EngineCore process.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_ENABLED = "VLLM_GR_BEAM_PROC_ENABLED"
ENV_BINARY = "VLLM_GR_BEAM_PROC_BINARY"
ENV_CONFIG = "VLLM_GR_BEAM_PROC_CONFIG"
ENV_APISERVER_OUTPUT_ADDR = "VLLM_GR_BEAM_PROC_APISERVER_OUTPUT_ADDR"
ENV_ENGINECORE_ADDR = "VLLM_GR_BEAM_PROC_ENGINECORE_ADDR"
ENV_ENGINECORE_TX_ADDR = "VLLM_GR_BEAM_PROC_ENGINECORE_TX_ADDR"
ENV_ENGINECORE_RX_ADDR = "VLLM_GR_BEAM_PROC_ENGINECORE_RX_ADDR"
ENV_STARTED = "VLLM_GR_BEAM_PROC_STARTED"
ENV_NUM_WORKERS = "VLLM_GR_BEAM_PROC_NUM_WORKERS"
ENV_LOG_LEVEL = "VLLM_GR_BEAM_PROC_LOG_LEVEL"
ENV_LOG_FILE = "VLLM_GR_BEAM_PROC_LOG_FILE"
ENV_READY_TIMEOUT_S = "VLLM_GR_BEAM_PROC_READY_TIMEOUT_S"
ENV_QUEUE_CAPACITY = "VLLM_GR_BEAMPROC_QUEUE_CAPACITY"
ENV_ZMQ_HWM = "VLLM_GR_BEAMPROC_ZMQ_HWM"
CONFIG_CATALOG_TOKEN_IDS_PATH = "catalog_token_ids_path"
CONFIG_CATALOG_TOKEN_IDS_FORMAT = "catalog_token_ids_format"
CONFIG_CATALOG_SOURCE_PATH = "catalog_source_path"
_BINARY_NAME = "beam-proc.exe" if os.name == "nt" else "beam-proc"
_PACKAGED_BINARY = Path(__file__).resolve().parents[2] / "bin" / _BINARY_NAME
_SOURCE_BINARY = Path(__file__).resolve().parents[3] / "rust" / "target" / "release" / _BINARY_NAME

_HANDLE: BeamProcHandle | None = None
_PREPARED: BeamProcConfig | None = None
_CATALOG_RESULTS: dict[tuple[Path, str | None], Path | None] = {}


@dataclass
class BeamProcConfig:
    config_path: Path
    enginecore_beamproc_addr: str
    enginecore_beamproc_tx_addr: str
    enginecore_beamproc_rx_addr: str
    apiserver_output_addr: str
    ready_path: Path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None


@dataclass
class BeamProcHandle:
    process: subprocess.Popen
    config_path: Path
    enginecore_beamproc_addr: str
    enginecore_beamproc_tx_addr: str
    enginecore_beamproc_rx_addr: str
    apiserver_output_addr: str
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def shutdown(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def ensure_beam_proc_started() -> BeamProcHandle | None:
    """Start beam-proc once when explicitly enabled."""
    global _HANDLE

    if _HANDLE is not None:
        exit_code = _HANDLE.process.poll()
        if exit_code is None:
            return _HANDLE
        raise RuntimeError(
            "BeamProc sidecar exited unexpectedly "
            f"with status {exit_code}; refusing to reuse its dead endpoint."
        )

    if not _is_enabled():
        return None

    if os.getenv(ENV_STARTED):
        logger.debug(
            "BeamProc already started by parent process; reusing enginecore_addr=%s",
            os.getenv(ENV_ENGINECORE_ADDR),
        )
        return None

    prepared = ensure_beam_proc_config()
    if prepared is None:
        return None

    binary_override = os.getenv(ENV_BINARY)
    if binary_override:
        binary = Path(binary_override)
    elif _PACKAGED_BINARY.is_file():
        binary = _PACKAGED_BINARY
    else:
        binary = _SOURCE_BINARY
    if not binary.exists():
        raise RuntimeError(
            f"BeamProc is enabled but the binary was not found at {binary}. "
            "Install a wheel containing BeamProc, build it with "
            f"`cargo build --release` under {_SOURCE_BINARY.parents[2]}, "
            f"or set {ENV_BINARY}."
        )

    env = os.environ.copy()
    env[ENV_STARTED] = "1"

    prepared.ready_path.unlink(missing_ok=True)
    process = subprocess.Popen(
        [str(binary), str(prepared.config_path)],
        cwd=str(binary.parent),
        env=env,
    )
    try:
        _wait_for_beam_proc_ready(process, prepared.ready_path)
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        raise
    _HANDLE = BeamProcHandle(
        process=process,
        config_path=prepared.config_path,
        enginecore_beamproc_addr=prepared.enginecore_beamproc_addr,
        enginecore_beamproc_tx_addr=prepared.enginecore_beamproc_tx_addr,
        enginecore_beamproc_rx_addr=prepared.enginecore_beamproc_rx_addr,
        apiserver_output_addr=prepared.apiserver_output_addr,
        temp_dir=prepared.temp_dir,
    )
    os.environ[ENV_STARTED] = "1"
    atexit.register(_HANDLE.shutdown)
    logger.info(
        "Started beam-proc pid=%s config=%s enginecore_tx_addr=%s "
        "enginecore_rx_addr=%s apiserver_output_addr=%s",
        process.pid,
        prepared.config_path,
        prepared.enginecore_beamproc_tx_addr,
        prepared.enginecore_beamproc_rx_addr,
        prepared.apiserver_output_addr,
    )
    return _HANDLE


def beam_proc_enabled() -> bool:
    """Return whether the Rust BeamProc sidecar was explicitly enabled."""
    return _is_enabled()


def beam_proc_exit_code() -> int | None:
    """Return the owned sidecar's exit status, if it has terminated."""
    if _HANDLE is None:
        return None
    return _HANDLE.process.poll()


def ensure_beam_proc_config() -> BeamProcConfig | None:
    """Prepare beam-proc config and env without spawning the Rust process."""
    global _PREPARED

    if _PREPARED is not None:
        configured_path = os.getenv(ENV_CONFIG)
        if configured_path is None or Path(configured_path) == _PREPARED.config_path:
            return _PREPARED
        # The caller selected a different explicit config after an earlier
        # generated one. Drop the cached preparation so the requested config
        # becomes authoritative.
        if _PREPARED.temp_dir is not None:
            _PREPARED.temp_dir.cleanup()
        _PREPARED = None

    if not _is_enabled():
        return None

    config_path_env = os.getenv(ENV_CONFIG)
    if config_path_env:
        config_path = Path(config_path_env)
        temp_dir = None
        config = _load_config(config_path)
        enginecore_addr = str(config["enginecore_beamproc_addr"])
        enginecore_tx_addr = str(config.get("enginecore_beamproc_tx_addr") or enginecore_addr)
        enginecore_rx_addr = str(
            config.get("enginecore_beamproc_rx_addr") or _derive_rx_addr(enginecore_tx_addr)
        )
        apiserver_output_addr = str(config["apiserver_output_addr"])
        ready_path = Path(str(config.get("ready_path") or config_path.with_suffix(".ready")))
        config["ready_path"] = str(ready_path)
        _write_json_atomic(config_path, config)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="vllm-gr-beam-proc-")
        config_path = Path(temp_dir.name) / "beam-proc-config.json"
        legacy_enginecore_addr = os.getenv(ENV_ENGINECORE_ADDR)
        enginecore_tx_addr = os.getenv(
            ENV_ENGINECORE_TX_ADDR,
            legacy_enginecore_addr
            or f"ipc://{Path(temp_dir.name) / 'enginecore-beamproc-tx.sock'}",
        )
        enginecore_rx_addr = os.getenv(
            ENV_ENGINECORE_RX_ADDR,
            (
                _derive_rx_addr(enginecore_tx_addr)
                if legacy_enginecore_addr
                else f"ipc://{Path(temp_dir.name) / 'enginecore-beamproc-rx.sock'}"
            ),
        )
        enginecore_addr = os.getenv(
            ENV_ENGINECORE_ADDR,
            enginecore_tx_addr,
        )
        apiserver_output_addr = os.getenv(
            ENV_APISERVER_OUTPUT_ADDR,
            f"ipc://{Path(temp_dir.name) / 'apiserver-output.sock'}",
        )
        ready_path = Path(temp_dir.name) / "beam-proc.ready"
        config = {
            "num_workers": int(os.getenv(ENV_NUM_WORKERS, "1")),
            "queue_capacity": int(os.getenv(ENV_QUEUE_CAPACITY, "64")),
            "zmq_hwm": int(os.getenv(ENV_ZMQ_HWM, "64")),
            "log_level": os.getenv(ENV_LOG_LEVEL, "info"),
            "log_file": os.getenv(
                ENV_LOG_FILE,
                str(Path(temp_dir.name) / "beam-proc.log"),
            ),
            "enginecore_beamproc_addr": enginecore_addr,
            "enginecore_beamproc_tx_addr": enginecore_tx_addr,
            "enginecore_beamproc_rx_addr": enginecore_rx_addr,
            "apiserver_output_addr": apiserver_output_addr,
            "ready_path": str(ready_path),
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    _PREPARED = BeamProcConfig(
        config_path=config_path,
        enginecore_beamproc_addr=enginecore_addr,
        enginecore_beamproc_tx_addr=enginecore_tx_addr,
        enginecore_beamproc_rx_addr=enginecore_rx_addr,
        apiserver_output_addr=apiserver_output_addr,
        ready_path=ready_path,
        temp_dir=temp_dir,
    )

    os.environ[ENV_CONFIG] = str(config_path)
    os.environ[ENV_ENGINECORE_ADDR] = enginecore_addr
    os.environ[ENV_ENGINECORE_TX_ADDR] = enginecore_tx_addr
    os.environ[ENV_ENGINECORE_RX_ADDR] = enginecore_rx_addr
    os.environ[ENV_APISERVER_OUTPUT_ADDR] = apiserver_output_addr

    logger.info(
        "Prepared beam-proc config=%s enginecore_tx_addr=%s enginecore_rx_addr=%s "
        "apiserver_output_addr=%s",
        config_path,
        enginecore_tx_addr,
        enginecore_rx_addr,
        apiserver_output_addr,
    )
    return _PREPARED


def _wait_for_beam_proc_ready(
    process: subprocess.Popen,
    ready_path: Path,
) -> None:
    timeout_s = float(os.getenv(ENV_READY_TIMEOUT_S, "10"))
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"BeamProc exited with status {exit_code} before READY acknowledgment"
            )
        if ready_path.is_file() and ready_path.read_text(encoding="utf-8").strip() == "READY":
            return
        time.sleep(0.01)
    raise RuntimeError(f"BeamProc did not acknowledge READY within {timeout_s:g}s: {ready_path}")


def prepare_beam_proc_catalog(
    model_config,
    tokenizer,
    catalog_path: str | None = None,
) -> Path | None:
    """Export the configured catalog as token ids before Rust BeamProc starts."""
    prepared = ensure_beam_proc_config()
    if prepared is None:
        return None

    catalog_path_value = (
        catalog_path if catalog_path is not None else getattr(model_config, "catalog_path", None)
    )
    catalog_key = (
        prepared.config_path,
        str(catalog_path_value) if catalog_path_value else None,
    )
    if catalog_key in _CATALOG_RESULTS:
        return _CATALOG_RESULTS[catalog_key]

    if not catalog_path_value:
        logger.info("BeamProc catalog export skipped: no catalog_path configured")
        _CATALOG_RESULTS[catalog_key] = None
        return None

    if tokenizer is None:
        raise RuntimeError("BeamProc catalog export requires a tokenizer")

    catalog_path = Path(catalog_path_value)
    if not catalog_path.exists():
        raise RuntimeError(f"BeamProc catalog source file not found: {catalog_path}")

    output_dir = (
        Path(prepared.temp_dir.name)
        if prepared.temp_dir is not None
        else prepared.config_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    token_ids_path = output_dir / "catalog-token-ids.json"

    config = _load_config(prepared.config_path)
    existing_path = config.get(CONFIG_CATALOG_TOKEN_IDS_PATH)
    existing_source = config.get(CONFIG_CATALOG_SOURCE_PATH)
    if existing_path and existing_source == str(catalog_path) and Path(str(existing_path)).exists():
        logger.info(
            "BeamProc catalog token-id export already exists path=%s source=%s",
            existing_path,
            catalog_path,
        )
        result = Path(str(existing_path))
        _CATALOG_RESULTS[catalog_key] = result
        return result

    entry_count = _export_catalog_token_ids(catalog_path, token_ids_path, tokenizer)
    config[CONFIG_CATALOG_TOKEN_IDS_PATH] = str(token_ids_path)
    config[CONFIG_CATALOG_TOKEN_IDS_FORMAT] = "json"
    config[CONFIG_CATALOG_SOURCE_PATH] = str(catalog_path)
    _write_json_atomic(prepared.config_path, config)

    logger.info(
        "Prepared BeamProc token-id catalog source=%s output=%s entries=%s config=%s",
        catalog_path,
        token_ids_path,
        entry_count,
        prepared.config_path,
    )
    _CATALOG_RESULTS[catalog_key] = token_ids_path
    return token_ids_path


def _is_enabled() -> bool:
    return os.getenv(ENV_ENABLED, "").lower() in {"1", "true", "yes", "y", "on"}


def _load_config(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    for key in ("enginecore_beamproc_addr", "apiserver_output_addr"):
        if key not in config:
            raise RuntimeError(f"BeamProc config {config_path} is missing `{key}`.")
    return config


def _derive_rx_addr(tx_addr: str) -> str:
    if tx_addr.startswith("ipc://"):
        path = Path(tx_addr[len("ipc://") :])
        return f"ipc://{path.with_name(path.stem + '-rx' + path.suffix)}"
    if tx_addr.startswith("tcp://"):
        prefix, _, port_text = tx_addr.rpartition(":")
        try:
            return f"{prefix}:{int(port_text) + 1}"
        except ValueError:
            return tx_addr
    return tx_addr


def _export_catalog_token_ids(
    catalog_path: Path,
    output_path: Path,
    tokenizer,
) -> int:
    try:
        with catalog_path.open(encoding="utf-8") as file:
            catalog_data = json.load(file)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to decode BeamProc catalog JSON {catalog_path}") from err

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    entry_count = 0
    with temp_path.open("w", encoding="utf-8") as file:
        file.write("[")
        first_entry = True
        for seq in catalog_data:
            if isinstance(seq, list) and all(type(token_id) is int for token_id in seq):
                normalized_token_ids = list(seq)
            else:
                token_ids = tokenizer.convert_tokens_to_ids(seq)
                normalized_token_ids = []
                for index, token_id in enumerate(token_ids):
                    if token_id is None:
                        raise RuntimeError(
                            "BeamProc catalog contains token without tokenizer id: "
                            f"path={catalog_path} entry={entry_count} "
                            f"index={index} token={seq[index]!r}"
                        )
                    normalized_token_ids.append(int(token_id))
            if first_entry:
                first_entry = False
            else:
                file.write(",")
            json.dump(normalized_token_ids, file, separators=(",", ":"))
            entry_count += 1
        file.write("]")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, output_path)
    _fsync_parent_dir(output_path)
    return entry_count


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
    _fsync_parent_dir(path)


def _fsync_parent_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
