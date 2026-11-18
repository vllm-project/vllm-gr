#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Minimal Yuanrong datasystem smoke test for NPU D2H/H2D KV transfer.

This script bypasses the vLLM engine and directly exercises the Yuanrong
datasystem client path used by the Yuanrong KV backend:

1. `exist(key)` before write
2. `mset_d2h(key, device_blob)` write
3. `exist(key)` after write
4. `mget_h2d(key, device_blob)` read back
5. Compare written tensor and loaded tensor

Example:
    python examples/yuanrong_datasystem_smoke.py \
        --expect-device npu \
        --device-id 0 \
        --worker-addr 127.0.0.1:12345
"""

import argparse
import os
import time
from pathlib import Path

import torch
from runtime_utils import add_repo_root_to_path, add_runtime_args, describe_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Yuanrong datasystem smoke test.",
    )
    add_runtime_args(parser, include_model=False)
    parser.add_argument(
        "--worker-addr",
        type=str,
        default=None,
        help=(
            "Override Yuanrong worker address. Defaults to "
            "YUANRONG_DS_WORKER_ADDR or DS_WORKER_ADDR."
        ),
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="NPU device id used for the smoke test (default: 0).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=8,
        help="Number of rows in the test tensor (default: 8).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=16,
        help="Number of cols in the test tensor (default: 16).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32", "int32"],
        default="float16",
        help="Tensor dtype used for the write/readback test (default: float16).",
    )
    parser.add_argument(
        "--key-prefix",
        type=str,
        default="vllm_gr_yuanrong_smoke",
        help="Key prefix used for the Yuanrong object store key.",
    )
    parser.add_argument(
        "--enable-exclusive-connection",
        action="store_true",
        help="Force DS_ENABLE_EXCLUSIVE_CONNECTION=1 for this process.",
    )
    parser.add_argument(
        "--enable-remote-h2d",
        action="store_true",
        help="Force DS_ENABLE_REMOTE_H2D=1 for this process.",
    )
    return parser.parse_args()


def _resolve_worker_addr(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.getenv("YUANRONG_DS_WORKER_ADDR") or os.getenv("DS_WORKER_ADDR", "")


def _split_host_port(worker_addr: str) -> tuple[str, int]:
    if ":" not in worker_addr:
        raise ValueError(f"Invalid worker address {worker_addr!r}. Expected host:port format.")
    host, port = worker_addr.rsplit(":", 1)
    return host, int(port)


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    return mapping[name]


def _print_environment(worker_addr: str) -> None:
    print("\n[Environment]")
    print(f"   worker_addr={worker_addr or '<unset>'}")
    print(f"   YUANRONG_DS_WORKER_ADDR={os.getenv('YUANRONG_DS_WORKER_ADDR', '<unset>')}")
    print(f"   DS_WORKER_ADDR={os.getenv('DS_WORKER_ADDR', '<unset>')}")
    print(f"   DS_ENABLE_EXCLUSIVE_CONNECTION={os.getenv('DS_ENABLE_EXCLUSIVE_CONNECTION', '0')}")
    print(f"   DS_ENABLE_REMOTE_H2D={os.getenv('DS_ENABLE_REMOTE_H2D', '0')}")


def _find_packaged_acl_plugin() -> str:
    try:
        import yr.datasystem  # type: ignore
    except Exception:
        return "<yr.datasystem unavailable>"

    package_dir = Path(getattr(yr.datasystem, "__file__", "")).resolve().parent
    candidate = package_dir / "lib" / "libacl_plugin.so"
    if candidate.exists():
        return str(candidate)
    return f"<not found under {package_dir}>"


def main() -> int:
    args = _parse_args()
    describe_runtime(expected_device=args.expect_device)

    worker_addr = _resolve_worker_addr(args.worker_addr)
    if not worker_addr:
        raise ValueError(
            "Missing worker address. Pass --worker-addr or set "
            "YUANRONG_DS_WORKER_ADDR / DS_WORKER_ADDR."
        )

    if args.enable_exclusive_connection:
        os.environ["DS_ENABLE_EXCLUSIVE_CONNECTION"] = "1"
    if args.enable_remote_h2d:
        os.environ["DS_ENABLE_REMOTE_H2D"] = "1"

    _print_environment(worker_addr)

    import acl  # type: ignore
    from yr.datasystem.hetero_client import (  # type: ignore
        Blob,
        DeviceBlobList,
        HeteroClient,
    )
    from yr.datasystem.kv_client import SetParam  # type: ignore
    from yr.datasystem.object_client import WriteMode  # type: ignore

    host, port = _split_host_port(worker_addr)
    enable_exclusive_connection = bool(int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0")))
    enable_remote_h2d = bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0")))

    print("\n[Client Init]")
    print(f"   host={host} port={port}")
    print(f"   exclusive_connection={enable_exclusive_connection}")
    print(f"   remote_h2d={enable_remote_h2d}")
    print(f"   packaged_acl_plugin={_find_packaged_acl_plugin()}")

    client = HeteroClient(
        host,
        port,
        enable_exclusive_connection=enable_exclusive_connection,
        enable_remote_h2d=enable_remote_h2d,
    )
    client.init()

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    device_id = int(torch.npu.current_device())
    torch_dtype = _torch_dtype(args.dtype)

    print("\n[ACL Init]")
    acl.init()
    acl.rt.set_device(device_id)
    print(f"   acl_device_id={device_id}")

    print("\n[Tensor Setup]")
    print(f"   device={device}")
    print(f"   dtype={torch_dtype}")
    print(f"   shape=({args.rows}, {args.cols})")

    write_tensor = (
        torch.arange(
            args.rows * args.cols,
            device=device,
            dtype=torch.float32,
        )
        .reshape(args.rows, args.cols)
        .to(torch_dtype)
    )
    read_tensor = torch.zeros_like(write_tensor)

    write_blob_list = DeviceBlobList(
        device_id,
        [Blob(write_tensor.data_ptr(), write_tensor.numel() * write_tensor.element_size())],
    )
    read_blob_list = DeviceBlobList(
        device_id,
        [Blob(read_tensor.data_ptr(), read_tensor.numel() * read_tensor.element_size())],
    )

    set_param = SetParam()
    set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

    key = f"{args.key_prefix}@device{args.device_id}@{int(time.time() * 1000)}"
    print(f"   key={key}")

    print("\n[Step 1] exist before write")
    exists_before = client.exist([key])
    print(f"   exists_before={exists_before}")

    print("\n[Step 2] mset_d2h write")
    client.mset_d2h([key], [write_blob_list], set_param)
    torch.npu.synchronize()
    print("   write_ok=True")

    print("\n[Step 3] exist after write")
    exists_after = client.exist([key])
    print(f"   exists_after={exists_after}")

    print("\n[Step 4] mget_h2d read")
    failed_keys = client.mget_h2d([key], [read_blob_list], 0)
    torch.npu.synchronize()
    print(f"   failed_keys={failed_keys}")

    print("\n[Step 5] compare")
    max_abs_diff = (write_tensor.float() - read_tensor.float()).abs().max().item()
    allclose = torch.allclose(
        write_tensor.float(),
        read_tensor.float(),
        atol=1e-3,
        rtol=1e-3,
    )
    print(f"   allclose={allclose}")
    print(f"   max_abs_diff={max_abs_diff}")

    if not allclose:
        raise RuntimeError(
            f"Yuanrong datasystem smoke test failed: readback mismatch. max_abs_diff={max_abs_diff}"
        )

    print("\nSmoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
