# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import collect_env

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
SCRIPT = Path(collect_env.__file__).resolve()


@pytest.mark.parametrize("installed", [False, True], ids=["empty", "wheel-metadata"])
def test_standalone_report_without_site_packages(tmp_path, installed):
    script = tmp_path / "collect_env.py"
    script.write_bytes(SCRIPT.read_bytes())
    # Include installed wheel metadata without making any runtime importable.
    if installed:
        dist = tmp_path / "vllm_gr-1.2.3+npu.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm-gr\nVersion: 1.2.3+npu\n")
    env = {**os.environ, "PYTHONPATH": "", "PATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-E", "-S", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "System Info" in result.stdout
    assert "Python version" in result.stdout
    assert "vLLM Version                 : Not installed" in result.stdout
    expected = "1.2.3+npu" if installed else "Not installed"
    assert f"vLLM-GR Version              : {expected}\n" in result.stdout


def test_standalone_report_with_broken_runtime(tmp_path):
    script = tmp_path / "collect_env.py"
    script.write_bytes(SCRIPT.read_bytes())
    dist = tmp_path / "torch-2.10.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.10.0\n")
    (tmp_path / "torch.py").write_text("raise RuntimeError('driver initialization failed')")
    (tmp_path / "vllm.py").write_text("raise OSError('extension unavailable')")
    output = subprocess.check_output(
        [sys.executable, "-E", "-S", str(script)],
        env={**os.environ, "PATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": "0"},
        text=True,
        timeout=30,
    )
    assert "2.10.0 (import failed: RuntimeError)" in output
    assert "CUDA_VISIBLE_DEVICES=0" in output.splitlines()


def test_probe_timeout_does_not_prevent_the_next_probe(monkeypatch):
    monkeypatch.setattr(collect_env, "PROBE_TIMEOUT_SECONDS", 0.2)
    code, _, reason = collect_env.run([sys.executable, "-c", "import time; time.sleep(30)"])
    assert code != 0
    assert "timed out" in reason
    monkeypatch.setattr(collect_env, "PROBE_TIMEOUT_SECONDS", 10)
    assert collect_env.run([sys.executable, "-c", "print('next')"]) == (0, "next", "")


def test_empty_crash_directory_preserves_report(tmp_path, monkeypatch, capsys):
    crash = SimpleNamespace(DEFAULT_MINIDUMP_DIR=str(tmp_path))
    torch = SimpleNamespace(utils=SimpleNamespace(_crash_handler=crash))
    monkeypatch.setattr(collect_env, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(collect_env, "torch", torch, raising=False)
    monkeypatch.setattr(collect_env, "get_pretty_env_info", lambda: "report")
    collect_env.main()
    assert "report" in capsys.readouterr().out


def test_public_gr_environment_is_sorted_and_filtered(monkeypatch):
    # A plugin registration must not make a private GR variable reportable.
    registry = SimpleNamespace(environment_variables={"CUDA_VISIBLE_DEVICES", "VLLM_GR_PRIVATE"})
    monkeypatch.setitem(sys.modules, "vllm.envs", registry)
    monkeypatch.setattr(
        os,
        "environ",
        {
            "VLLM_GR_TARGET_DEVICE": "npu",
            "VLLM_GR_PRIVATE": "internal",
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_SECRET": "sensitive",
        },
    )
    assert collect_env.get_env_vars().splitlines() == [
        "CUDA_VISIBLE_DEVICES=0",
        "VLLM_GR_TARGET_DEVICE=npu",
    ]
