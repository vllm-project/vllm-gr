# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, read-only device selection and NVML queries. Does not import torch."""

import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

MAPPING_ENV = "CI_CUDA_MAPPING_FILE"
QUERY_TIMEOUT = 5.0


class DeviceError(RuntimeError):
    """Device selection or a required GPU query failed."""


def indices(value: str | None) -> list[int]:
    if not value or not value.strip():
        raise DeviceError("CUDA_VISIBLE_DEVICES must be a nonempty numeric device list")
    parts = [part.strip() for part in value.split(",")]
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise DeviceError(f"Unsupported CUDA_VISIBLE_DEVICES={value!r}; expected numeric indices")
    result = [int(part) for part in parts]
    if len(set(result)) != len(result):
        raise DeviceError(f"Duplicate indices in CUDA_VISIBLE_DEVICES={value!r}")
    return result


def normalize_uuid(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceError(f"Missing GPU UUID: {value!r}")
    value = value.strip().lower()
    if value.startswith("gpu-"):
        value = value[4:]
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise DeviceError(f"Invalid GPU UUID: {value!r}") from exc


def command(args: list[str], timeout: float = QUERY_TIMEOUT) -> str:
    program = Path(args[0]).name
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise DeviceError(f"{program} query timed out after {timeout}s") from exc
    except (OSError, UnicodeError) as exc:
        raise DeviceError(f"{program} query failed: {exc}") from exc
    if result.returncode:
        raise DeviceError(
            f"{program} query failed (exit {result.returncode})\n"
            f"{result.stderr.strip()}\n{result.stdout.strip()}"
        )
    return result.stdout


def csv_rows(output: str, columns: int) -> list[list[str]]:
    try:
        rows = [
            [cell.strip() for cell in row] for row in csv.reader(io.StringIO(output), strict=True)
        ]
    except csv.Error as exc:
        raise DeviceError(f"Malformed nvidia-smi CSV: {output!r}") from exc
    if any(len(row) != columns or any(not cell for cell in row) for row in rows):
        raise DeviceError(f"Malformed nvidia-smi CSV (expected {columns} columns): {output!r}")
    return rows


def query_gpu(selector: str) -> dict[str, str]:
    expected = normalize_uuid(selector)
    output = command(
        [
            "nvidia-smi",
            "-i",
            "GPU-" + expected,
            "--query-gpu=uuid,index",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = csv_rows(output, 2)
    if len(rows) != 1:
        raise DeviceError(f"Expected one GPU for {selector!r}, got {len(rows)}: {output!r}")
    uuid, physical_index = rows[0]
    uuid = "GPU-" + normalize_uuid(uuid)
    if expected != normalize_uuid(uuid):
        raise DeviceError(f"nvidia-smi returned the wrong UUID for {selector!r}: {uuid}")
    if not physical_index.isascii() or not physical_index.isdecimal():
        raise DeviceError(f"Invalid nvidia-smi index: {physical_index!r}")
    return dict(uuid=uuid, physical_index=physical_index)


def selection_indices() -> list[int]:
    order = os.environ.get("CUDA_DEVICE_ORDER")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if order != "PCI_BUS_ID":
        raise DeviceError(f"CUDA_DEVICE_ORDER must be PCI_BUS_ID, got {order!r}")
    return indices(visible)


def configured_devices(driver_uuids: list[str]) -> list[dict[str, Any]]:
    """Resolve the runner's CUDA ordinals through independent driver UUIDs."""
    configured = selection_indices()
    if len(driver_uuids) != len(configured):
        raise DeviceError(
            f"Visible device count mismatch: configured={len(configured)} "
            f"CUDA driver={len(driver_uuids)}"
        )
    result = []
    for logical_id, (index, uuid) in enumerate(zip(configured, driver_uuids)):
        gpu = query_gpu("GPU-" + normalize_uuid(uuid))
        result.append({"logical_id": logical_id, "index": index, **gpu})
    if len({gpu["uuid"] for gpu in result}) != len(result):
        raise DeviceError("Configured indices resolved to duplicate GPU UUIDs")
    return result


def verified_devices() -> list[dict[str, Any]]:
    """Read verified identities without CUDA initialization or numeric NVML guesses."""
    configured = selection_indices()
    mapping_path = os.environ.get(MAPPING_ENV)
    if not mapping_path:
        raise DeviceError(f"{MAPPING_ENV} is required; run check_cuda_device_mapping.py first")
    try:
        saved = json.loads(Path(mapping_path).read_text())
        result: list[dict[str, Any]] = saved["devices"]
        if (
            saved.get("selection_source") != "cuda_driver_uuid"
            or saved["cuda_device_order"] != os.environ.get("CUDA_DEVICE_ORDER")
            or indices(saved["cuda_visible_devices"]) != configured
            or [d["index"] for d in result] != configured
            or [d["logical_id"] for d in result] != list(range(len(configured)))
            or len({normalize_uuid(d["uuid"]) for d in result}) != len(configured)
            or any(normalize_uuid(d["actual_uuid"]) != normalize_uuid(d["uuid"]) for d in result)
            or any(
                not str(d["physical_index"]).isascii() or not str(d["physical_index"]).isdecimal()
                for d in result
            )
        ):
            raise DeviceError("Device selection changed since the CUDA mapping check")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DeviceError(f"Missing or invalid verified mapping {mapping_path!r}: {exc}") from exc
    return result


def selected_devices() -> list[dict[str, Any]]:
    """Recheck verified UUIDs; never interpret CUDA ordinals as NVML IDs."""
    current = []
    for verified in verified_devices():
        gpu = query_gpu(verified["uuid"])
        if gpu["physical_index"] != verified["physical_index"]:
            raise DeviceError("NVML device identity changed since the CUDA mapping check")
        current.append({**verified, **gpu})
    return current


def compute_processes(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processes = []
    for gpu in gpus:
        output = command(
            [
                "nvidia-smi",
                "-i",
                gpu["uuid"],
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        for uuid, pid, name, memory in csv_rows(output, 4):
            if normalize_uuid(uuid) != normalize_uuid(gpu["uuid"]):
                raise DeviceError(f"nvidia-smi returned a process outside target GPU {gpu['uuid']}")
            if not pid.isascii() or not pid.isdecimal() or int(pid) <= 0:
                raise DeviceError(f"Invalid compute process PID: {pid!r}")
            processes.append(dict(uuid=gpu["uuid"], pid=int(pid), name=name, memory=memory))
    return processes
