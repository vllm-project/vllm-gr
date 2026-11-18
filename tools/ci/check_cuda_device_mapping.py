# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check numeric CUDA selection against actual PyTorch UUIDs before CI cleanup."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__:
    from . import cuda_devices as devices
else:
    import cuda_devices as devices

# Use the test interpreter in a short-lived subprocess so even a hung CUDA
# initialization has a deadline. Neither this parent nor cleanup imports torch.
TORCH_PROBE = """
import json
import os
import sys
expected = json.loads(sys.argv[1])
def check_environment():
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError('CUDA selection changed in the probe: ' + json.dumps(actual))
check_environment()
import torch
cuda = torch.cuda
check_environment()
uuids = [str(cuda.get_device_properties(i).uuid) for i in range(cuda.device_count())]
check_environment()
print('CUDA_UUIDS=' + json.dumps(uuids), flush=True)
"""


def driver_uuids() -> list[str]:
    """Run only in the bounded child; enumerate selected devices without creating contexts."""
    import ctypes
    from uuid import UUID

    library = ctypes.CDLL("libcuda.so.1")

    def call(name: str, argtypes: list[Any], *args: Any) -> None:
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        code = function(*args)
        if code:
            raise devices.DeviceError(f"CUDA Driver API {name} failed with error {code}")

    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    call("cuInit", [ctypes.c_uint], 0)
    count = integer()
    call("cuDeviceGetCount", [pointer], ctypes.byref(count))
    if count.value < 0:
        raise devices.DeviceError(f"Invalid CUDA device count: {count.value}")
    uuid_api = "cuDeviceGetUuid_v2" if hasattr(library, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    result = []
    for ordinal in range(count.value):
        device = integer()
        call("cuDeviceGet", [pointer, integer], ctypes.byref(device), ordinal)
        uuid = (ctypes.c_ubyte * 16)()
        call(uuid_api, [ctypes.c_void_p, integer], ctypes.byref(uuid), device)
        result.append(str(UUID(bytes=bytes(uuid))))
    return result


def probe_uuids(*, driver: bool, timeout: float) -> list[str]:
    expected = {key: os.environ.get(key) for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")}
    # Disable site hooks in the driver child. PyTorch uses the tests' startup environment.
    args = (
        [sys.executable, "-S", str(Path(__file__).resolve()), "--driver-probe"]
        if driver
        else [sys.executable, "-c", TORCH_PROBE, json.dumps(expected)]
    )
    output = devices.command(args, timeout=timeout)
    lines = [
        line.removeprefix("CUDA_UUIDS=")
        for line in output.splitlines()
        if line.startswith("CUDA_UUIDS=")
    ]
    try:
        values = json.loads(lines[0]) if len(lines) == 1 else None
        if not isinstance(values, list):
            raise ValueError("expected a UUID list")
        return [devices.normalize_uuid(value) for value in values]
    except ValueError as exc:
        raise devices.DeviceError(f"Invalid CUDA UUID probe output: {output!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=float, default=60, help="Timeout per CUDA probe in seconds"
    )
    parser.add_argument(
        "--output", type=Path, help="Write successful mapping for cleanup and checks"
    )
    parser.add_argument("--driver-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if not 0 < args.timeout < float("inf"):
            raise devices.DeviceError("Timeout must be finite and positive")
        if args.driver_probe:
            print("CUDA_UUIDS=" + json.dumps(driver_uuids()))
            return 0
        if args.output:
            # Do not leave an old successful result if this attempt fails.
            args.output.unlink(missing_ok=True)
        devices.selection_indices()
        gpus = devices.configured_devices(probe_uuids(driver=True, timeout=args.timeout))
        actual = probe_uuids(driver=False, timeout=args.timeout)
        if len(actual) != len(gpus):
            raise devices.DeviceError(
                f"Visible device count mismatch: configured={len(gpus)} PyTorch={len(actual)}"
            )
        for gpu, uuid in zip(gpus, actual):
            if devices.normalize_uuid(gpu["uuid"]) != uuid:
                raise devices.DeviceError(
                    f"CUDA UUID mismatch for cuda:{gpu['logical_id']}: "
                    f"expected {gpu['uuid']}, actual {uuid}"
                )
            gpu["actual_uuid"] = uuid
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
                        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "selection_source": "cuda_driver_uuid",
                        "devices": gpus,
                    },
                    indent=2,
                )
                + "\n"
            )
        print("PASS: CUDA device mapping verified.")
        return 0
    except (devices.DeviceError, OSError, AttributeError) as exc:
        print(f"FAIL: CUDA device mapping: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
