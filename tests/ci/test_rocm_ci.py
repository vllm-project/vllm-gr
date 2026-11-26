# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ROCm CI pipeline wiring and Docker invocation without an AMD agent."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.helpers.paths import REPO_ROOT
from tools.ci.check_onerec_accuracy import parse_beams

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

SCRIPTS = REPO_ROOT / ".buildkite" / "common" / "scripts"


def _expanded_ready_steps() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["python3", str(SCRIPTS / "upload_pipeline.py"), ".buildkite/test-ready.yml"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return list(yaml.safe_load(result.stdout)["steps"])


def test_rocm_step_is_gated_soft_and_runs_on_the_amd_queue() -> None:
    step = next(step for step in _expanded_ready_steps() if step.get("key") == "rocm-w7900")
    assert step["if"] == 'build.env("VLLM_GR_ROCM_CI") == "1"'
    assert step["soft_fail"] is True
    assert step["agents"] == {"queue": "amd-w7900"}
    assert {"exit_status": 75, "limit": 2} in step["retry"]["automatic"]
    assert step["command"] == "bash .buildkite/common/scripts/run-rocm-ci.sh"


@pytest.fixture  # type: ignore[misc]
def fake_docker(tmp_path: Path) -> Path:
    """A PATH directory whose docker and getent record their arguments."""
    if shutil.which("bash") is None:
        pytest.skip("Bash required")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'printf "%s\\0" "$@" > "$DOCKER_LOG.$1"\n'
    )
    (bin_dir / "getent").write_text(
        '#!/bin/bash\ncase "$2" in video) echo video:x:44:;; render) echo render:x:110:;; esac\n'
    )
    for tool in ("docker", "getent"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def _run(script: str, fake_docker: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    base = {
        # Never touch the checkout's real log directory from a test.
        "VLLM_GR_ROCM_CI_OUTPUT": f"ci-output/test-{fake_docker.parent.name}",
        "PATH": f"{fake_docker}:{os.environ['PATH']}",
        "DOCKER_LOG": str(fake_docker.parent / "docker.log"),
        "BUILDKITE_COMMIT": "abc123",
        "BUILDKITE_JOB_ID": "job1",
        "HOME": str(fake_docker.parent),
    }
    return subprocess.run(
        ["bash", str(SCRIPTS / script)],
        cwd=REPO_ROOT,
        env={**base, **env},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _docker_args(fake_docker: Path, command: str) -> list[str]:
    raw = (fake_docker.parent / f"docker.log.{command}").read_text()
    return raw.split("\0")[:-1]


def test_run_script_passes_only_the_agent_gpus_and_the_checkout(fake_docker: Path) -> None:
    render = fake_docker.parent / "renderD200"
    render.touch()
    sentinel = REPO_ROOT / "ci-output" / "rocm" / "sentinel"
    existed = sentinel.exists()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    result = _run(
        "run-rocm-ci.sh",
        fake_docker,
        {
            "VLLM_GR_ROCM_RENDER_DEVICES": str(render),
            "VLLM_GR_ROCM_DEVICE_ID": "0",
            "VLLM_GR_TEST_MODEL": "OpenOneRec/OneRec-1.7B",
            "HF_ENDPOINT": "https://mirror.example",
            # Removed options must not mount host paths.
            "VLLM_GR_HF_CACHE": str(fake_docker.parent / "hf"),
            "VLLM_GR_TEST_MODEL_DIR": "/data/OneRec-1.7B",
        },
    )
    assert result.returncode == 0, result.stderr
    args = _docker_args(fake_docker, "run")
    joined = " ".join(args)
    assert "--device /dev/kfd" in joined and f"--device {render}" in joined
    assert joined.count("--device ") == 2
    assert "--group-add 44" in joined and "--group-add 110" in joined
    assert f"--user {os.getuid()}:{os.getgid()}" in joined
    # Like the CUDA jobs, the model is downloaded inside the container: only the
    # checkout is mounted.
    mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "-v"]
    assert mounts == [f"{REPO_ROOT}:/workspace"]
    assert "/data/OneRec-1.7B" not in joined and "HF_HOME" not in joined
    assert "-e VLLM_GR_TEST_MODEL" in joined
    assert "-e HF_ENDPOINT" in joined and "HF_TOKEN" not in joined
    assert "vllm-gr-ci-rocm:gfx1100-abc123" in args
    assert args[-4:] == [
        "--output",
        f"/workspace/ci-output/test-{fake_docker.parent.name}",
        "--device-id",
        "0",
    ]
    assert sentinel.exists(), "the default log directory must survive a test run"
    if not existed:
        sentinel.unlink()
    log = (fake_docker.parent / "docker.log").read_text()
    assert "image rm vllm-gr-ci-rocm:gfx1100-abc123" in log


@pytest.mark.parametrize("output", ["/tmp/logs", "ci-output/../logs"])  # type: ignore[misc]
def test_run_script_requires_a_relative_log_directory(fake_docker: Path, output: str) -> None:
    result = _run("run-rocm-ci.sh", fake_docker, {"VLLM_GR_ROCM_CI_OUTPUT": output})
    assert result.returncode == 1
    assert "relative path" in result.stderr
    assert not (fake_docker.parent / "docker.log").exists()


def test_run_script_refuses_a_missing_render_node(fake_docker: Path) -> None:
    result = _run(
        "run-rocm-ci.sh",
        fake_docker,
        {"VLLM_GR_ROCM_RENDER_DEVICES": str(fake_docker.parent / "renderD999")},
    )
    assert result.returncode == 1
    assert "does not exist" in result.stderr
    assert not (fake_docker.parent / "docker.log.run").exists()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ["build"]),
        (
            {"VLLM_GR_ROCM_CI_CACHE_REF": "registry.example/cache", "VLLM_GR_ROCM_ARCH": "gfx1201"},
            ["buildx"],
        ),
        ({"VLLM_GR_ROCM_CI_PUSH": "1"}, ["build", "push"]),
    ],
)  # type: ignore[misc]
def test_build_script_uses_the_configured_cache(
    fake_docker: Path, env: dict[str, str], expected: list[str]
) -> None:
    result = _run("build-ci-image-rocm.sh", fake_docker, env)
    assert result.returncode == 0, result.stderr
    arch = env.get("VLLM_GR_ROCM_ARCH", "gfx1100")
    assert result.stdout.strip() == f"vllm-gr-ci-rocm:{arch}-abc123"
    commands = [
        line.split()[0] for line in (fake_docker.parent / "docker.log").read_text().splitlines()
    ]
    assert [c for c in commands if c in ("build", "buildx", "push")][-len(expected) :] == expected
    if "buildx" in expected:
        args = " ".join(_docker_args(fake_docker, "buildx"))
        assert "type=registry,ref=registry.example/cache-gfx1201" in args
        assert "PYTORCH_ROCM_ARCH=gfx1201" in args


def test_accuracy_check_reads_request_zero_in_rank_order(tmp_path: Path) -> None:
    def line(request: int, rank: int) -> str:
        return json.dumps({"request": request, "rank": rank, "beam_token_ids": [rank]})

    log = tmp_path / "quickstart.log"
    log.write_text("\n".join(["INFO start", line(0, 1), line(1, 0), line(0, 0), "done"]))
    assert [beam["rank"] for beam in parse_beams(log)] == [0, 1]


def test_build_script_passes_the_github_api_and_token_secret(fake_docker: Path) -> None:
    token_file = fake_docker.parent / "token"
    token_file.write_text("secret-value")
    result = _run(
        "build-ci-image-rocm.sh",
        fake_docker,
        {
            "VLLM_GR_GITHUB_API": "https://api-proxy.example",
            "VLLM_GR_GITHUB_TOKEN_FILE": str(token_file),
            "VLLM_GR_ROCM_BASE_IMAGE": "registry.example/rocm/pytorch:10.1",
            "VLLM_GR_ROCM_APT_MIRROR": "https://mirror.example/ubuntu",
            "VLLM_GR_ROCM_PIP_INDEX_URL": "https://mirror.example/pypi/simple",
        },
    )
    assert result.returncode == 0, result.stderr
    args = " ".join(_docker_args(fake_docker, "build"))
    assert "--build-arg GITHUB_API=https://api-proxy.example" in args
    assert f"--secret id=github_token,src={token_file}" in args
    assert "--build-arg BASE_IMAGE=registry.example/rocm/pytorch:10.1" in args
    assert "--build-arg APT_MIRROR=https://mirror.example/ubuntu" in args
    assert "--build-arg PIP_INDEX_URL=https://mirror.example/pypi/simple" in args
    assert "secret-value" not in args


FETCH = REPO_ROOT / "tools" / "ci" / "fetch_github_source.sh"


@pytest.fixture  # type: ignore[misc]
def fake_github(tmp_path: Path) -> Path:
    """A PATH directory whose curl serves tarballs and submodule metadata."""
    if shutil.which("bash") is None or shutil.which("git") is None:
        pytest.skip("Bash and Git required")
    for name, files in {"main": {"README.md": "main\n"}, "sub": {"kernel.py": "k\n"}}.items():
        root = tmp_path / "srv" / f"owner-{name}-0123abc"
        root.mkdir(parents=True)
        for file, text in files.items():
            (root / file).write_text(text)
        subprocess.run(
            [
                "tar",
                "-czf",
                str(tmp_path / "srv" / f"{name}.tar.gz"),
                "-C",
                str(root.parent),
                root.name,
            ],
            check=True,
        )
    (tmp_path / "srv" / "contents.json").write_text(
        json.dumps(
            {
                "type": "submodule",
                "sha": "feed",
                "submodule_git_url": "https://github.com/owner/sub.git",
            }
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" >> "$CURL_LOG"\nout=""\n'
        'for ((i=1; i<=$#; i++)); do case "${!i}" in\n'
        "  -o) j=$((i+1)); out=${!j};;\n"
        '  -H) j=$((i+1)); h=${!j}; cp "${h#@}" "$CURL_LOG.headers";;\n'
        "esac; done\nurl=${!#}\n"
        # Like the ROCm organization, owner/sub rejects the token.
        'if [[ $url == */owner/sub/* ]] && grep -q Authorization "$CURL_LOG.headers"; then exit 22; fi\n'
        'case "$url" in\n'
        '  */repos/owner/main/tarball/*) cp "$SRV/main.tar.gz" "$out";;\n'
        '  */repos/owner/sub/tarball/feed*) cp "$SRV/sub.tar.gz" "$out";;\n'
        '  */repos/owner/main/contents/third_party/sub*) cat "$SRV/contents.json";;\n'
        "  *) exit 22;;\nesac\n"
    )
    (bin_dir / "curl").chmod(0o755)
    return tmp_path


def _fetch(fake_github: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {
        "PATH": f"{fake_github / 'bin'}:{os.environ['PATH']}",
        "CURL_LOG": str(fake_github / "curl.log"),
        "SRV": str(fake_github / "srv"),
        "GITHUB_API": "https://api-proxy.example/",
        "HOME": str(fake_github),
    }
    return subprocess.run(
        ["bash", str(FETCH), *args],
        env={**base, **env},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_fetch_extracts_submodules_and_keeps_the_token_off_the_command_line(
    fake_github: Path,
) -> None:
    token_file = fake_github / "token"
    token_file.write_text("secret-value")
    dest = fake_github / "out"
    result = _fetch(
        fake_github,
        "owner/main",
        "abc",
        str(dest),
        "--submodule",
        "third_party/sub",
        "--git-init",
        GITHUB_TOKEN_FILE=str(token_file),
    )
    assert result.returncode == 0, result.stderr
    assert (dest / "README.md").read_text() == "main\n"
    assert (dest / "third_party" / "sub" / "kernel.py").read_text() == "k\n"
    files = subprocess.run(
        ["git", "-C", str(dest), "ls-files"], text=True, capture_output=True, check=True
    ).stdout.split()
    assert files == ["README.md", "third_party/sub/kernel.py"]
    argv = (fake_github / "curl.log").read_text()
    assert "secret-value" not in argv
    assert "https://api-proxy.example/repos/owner/main/tarball/abc" in argv
    assert "retrying without it" in result.stderr


def test_fetch_refuses_a_non_empty_destination(fake_github: Path) -> None:
    dest = fake_github / "out"
    dest.mkdir()
    (dest / "keep").touch()
    result = _fetch(fake_github, "owner/main", "abc", str(dest))
    assert result.returncode == 2
    assert "not empty" in result.stderr
    assert not (fake_github / "curl.log").exists()


@pytest.mark.parametrize(
    "repository", ["https://github.com/owner/main.git", "git@github.com:owner/main.git"]
)  # type: ignore[misc]
def test_checkout_hook_fetches_the_build_commit(tmp_path: Path, repository: str) -> None:
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    shutil.copy(REPO_ROOT / ".buildkite" / "common" / "agent-hooks" / "checkout", hooks)
    (hooks / "fetch_github_source.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS"\n')
    (hooks / "fetch_github_source.sh").chmod(0o755)
    checkout = tmp_path / "builds" / "job"
    checkout.mkdir(parents=True)
    (checkout / "stale").touch()
    result = subprocess.run(
        ["bash", str(hooks / "checkout")],
        env={
            **os.environ,
            "ARGS": str(tmp_path / "args"),
            "BUILDKITE_REPO": repository,
            "BUILDKITE_COMMIT": "0123abc",
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "args").read_text().split() == [
        "owner/main",
        "0123abc",
        str(checkout),
        "--git-init",
    ]
    assert not checkout.exists()
