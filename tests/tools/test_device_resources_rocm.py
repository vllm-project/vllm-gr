# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ROCm device selection and locking against a fake KFD/sysfs tree."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.devices import lock_gpu

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

GIB = 1 << 30


def _add_gpu(
    root: Path,
    node: int,
    gpu_id: int,
    bus: int,
    *,
    vram_gib: int = 48,
    busy: str = "0",
) -> None:
    node_dir = root / "topology" / str(node)
    node_dir.mkdir(parents=True)
    (node_dir / "gpu_id").write_text(f"{gpu_id}\n")
    (node_dir / "properties").write_text(
        f"cpu_cores_count 0\nsimd_count 192\nlocation_id {bus << 8}\ndomain 0\n"
    )
    pci = root / "pci" / f"0000:{bus:02x}:00.0"
    pci.mkdir(parents=True)
    (pci / "gpu_busy_percent").write_text(f"{busy}\n")
    (pci / "mem_info_vram_total").write_text(f"{vram_gib * GIB}\n")


def _use_vram(root: Path, pid: int, gpu_id: int, mib: int) -> None:
    proc = root / "proc" / str(pid)
    proc.mkdir(parents=True, exist_ok=True)
    (proc / f"vram_{gpu_id}").write_text(f"{mib << 20}\n")


@pytest.fixture  # type: ignore[misc]
def sysfs(tmp_path: Path) -> Path:
    """A CPU node, then GPUs in KFD order: busy 0x75, idle 0x05, integrated 0x0e."""
    cpu = tmp_path / "topology" / "0"
    cpu.mkdir(parents=True)
    (cpu / "gpu_id").write_text("0\n")
    (cpu / "properties").write_text("cpu_cores_count 32\nsimd_count 0\n")
    (tmp_path / "proc").mkdir()
    _add_gpu(tmp_path, 1, 111, 0x75)
    _add_gpu(tmp_path, 2, 222, 0x05)
    _add_gpu(tmp_path, 3, 333, 0x0E, vram_gib=2)
    _use_vram(tmp_path, 4242, 111, 1024)
    return tmp_path


def run_rocm(sysfs: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("bash") is None:
        pytest.skip("Bash required")
    prelude = f"""
        export VLLM_GR_KFD_TOPOLOGY={sysfs}/topology VLLM_GR_KFD_PROC={sysfs}/proc
        export VLLM_GR_PCI_DEVICES={sysfs}/pci
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
        rocm_state=true
        source tools/devices/resource.sh
        sleep() {{ :; }}
    """
    return subprocess.run(
        ["bash", "-c", prelude + script, "device-test", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_indices_follow_kfd_order_and_skip_busy_and_small_gpus(sysfs: Path) -> None:
    result = run_rocm(
        sysfs,
        """
        rocm_gpu_count
        rocm_gpu_pci_address 0
        rocm_gpu_pci_address 2
        get_gpu 18 1
        get_gpus 2 18 1 || echo none
        """,
    )
    assert result.stdout.splitlines() == ["3", "0000:75:00.0", "0000:0e:00.0", "1", "none"]


def test_lock_sized_usage_makes_an_idle_gpu_busy(sysfs: Path) -> None:
    _use_vram(sysfs, 99, 222, lock_gpu.ROCM_MIN_LOCK_MB)
    result = run_rocm(sysfs, "get_gpu 18 1 || echo none")
    assert result.stdout.splitlines() == ["none"]


def test_hip_mask_takes_precedence_and_rocr_mask_is_rejected(sysfs: Path) -> None:
    _use_vram(sysfs, 4242, 111, 0)
    result = run_rocm(
        sysfs,
        """
        HIP_VISIBLE_DEVICES=1,0 CUDA_VISIBLE_DEVICES=0 get_gpu 18 1
        CUDA_VISIBLE_DEVICES=1,0 get_gpu 18 1
        HIP_VISIBLE_DEVICES= get_gpu 18 1 || echo empty
        ROCR_VISIBLE_DEVICES=0 get_gpu 18 1 || echo rocr
        """,
    )
    assert result.stdout.splitlines() == ["1", "1", "empty", "rocr"]
    assert "ROCR_VISIBLE_DEVICES renumbers devices" in result.stderr


@pytest.mark.parametrize(
    ("busy", "remove", "vram", "available"),
    [
        ("0", None, "0", True),
        ("7", None, "0", False),
        ("0", "gpu_busy_percent", "0", False),
        ("0", "mem_info_vram_total", "0", False),
        ("0", None, "garbage", False),
    ],
)  # type: ignore[misc]
def test_telemetry_must_be_complete_and_idle(
    sysfs: Path, busy: str, remove: str | None, vram: str, available: bool
) -> None:
    pci = sysfs / "pci" / "0000:05:00.0"
    (pci / "gpu_busy_percent").write_text(f"{busy}\n")
    if remove:
        (pci / remove).unlink()
    proc = sysfs / "proc" / "77"
    proc.mkdir()
    (proc / "vram_222").write_text(f"{vram}\n")
    result = run_rocm(sysfs, "is_available 1 18 && echo yes || echo no")
    assert result.stdout.strip() == ("yes" if available else "no")


def test_detection_defers_to_nvidia_tooling_and_explicit_state(sysfs: Path) -> None:
    script = f"""
        export VLLM_GR_KFD_TOPOLOGY={sysfs}/topology
        nvidia-smi() {{ echo 1; }}
        source tools/devices/resource.sh
        echo "$rocm_state"
        rocm_state=true
        source tools/devices/resource.sh
        echo "$rocm_state"
    """
    result = subprocess.run(
        ["bash", "-c", script], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10
    )
    assert result.stdout.splitlines() == ["false", "true"]


def test_shell_lock_passes_the_pci_address(sysfs: Path) -> None:
    result = run_rocm(
        sysfs,
        """
        python3() { echo "$@" > "$ARGS_FILE"; echo READY; exec sleep 3; }
        command() {
            if [[ "$*" = "-v python" ]]; then return 1; fi
            builtin command "$@"
        }
        pid=$(lock_gpu 1) || exit 1
        unlock_gpu "$pid"
        lock_gpu 9 || echo unknown
        cat "$ARGS_FILE"
        """.replace("$ARGS_FILE", str(sysfs / "args")),
    )
    lines = result.stdout.splitlines()
    assert "unknown" in lines
    assert lines[-1].endswith("1 --rocm-pci-address 0000:05:00.0")


def _hip(bus_ids: list[str]) -> Any:
    hip = Mock()
    hip.hipInit.return_value = hip.hipSetDevice.return_value = hip.hipMalloc.return_value = 0

    def device_count(pointer: Any) -> int:
        pointer._obj.value = len(bus_ids)
        return 0

    def bus_id(buffer: Any, _length: int, ordinal: int) -> int:
        buffer.value = bus_ids[ordinal].encode()
        return 0

    hip.hipGetDeviceCount.side_effect = device_count
    hip.hipDeviceGetPCIBusId.side_effect = bus_id
    return hip


def test_rocm_lock_matches_pci_address_and_exceeds_idle_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hip = _hip(["0000:75:00.0", "0000:05:00.0"])
    monkeypatch.setattr(lock_gpu, "_load_hip", lambda: hip)
    monkeypatch.setattr(lock_gpu.time, "sleep", lambda _: None)
    lock_gpu.lock_gpu_rocm("0000:05:00.0", 4)
    hip.hipSetDevice.assert_called_once_with(1)
    size = hip.hipMalloc.call_args.args[1]
    assert isinstance(size, ctypes.c_size_t)
    assert size.value == lock_gpu.ROCM_MIN_LOCK_MB << 20


def test_rocm_lock_refuses_a_device_outside_the_visible_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hip = _hip(["0000:75:00.0"])
    monkeypatch.setattr(lock_gpu, "_load_hip", lambda: hip)
    with pytest.raises(SystemExit, match="No HIP-visible device"):
        lock_gpu.lock_gpu_rocm("0000:05:00.0", 4)
    hip.hipMalloc.assert_not_called()
