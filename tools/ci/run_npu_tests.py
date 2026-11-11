#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run the full NPU suite, an incomplete candidate, or grouped P0 acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("npu_test_suites.json")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def device_environment(device: str, inherited: dict[str, str]) -> dict[str, str]:
    if not device.isascii() or not device.isdecimal() or str(int(device)) != device:
        raise ValueError("--device must be one canonical nonnegative device ordinal")
    visible = inherited.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is not None and visible != device:
        raise ValueError(
            "--device disagrees with ASCEND_RT_VISIBLE_DEVICES; refusing to reallocate"
        )
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        if inherited.get(name):
            raise ValueError(f"Unset {name}; inherited pytest controls are not accepted")
    env = dict(inherited)
    env.pop("PYTEST_CURRENT_TEST", None)
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=device,
        VLLM_GR_TARGET_DEVICE="npu",
        PYTORCH_NPU_ALLOW_FORK="true",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TORCH_MULTIPROCESSING_METHOD="spawn",
        PYTHONDONTWRITEBYTECODE="1",
    )
    # Keep vendor paths (including acl) after the tested source and observer.
    paths = [str(ROOT), str(Path(__file__).parent)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def candidate_nodes(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported NPU suite manifest schema")
    tier = manifest["tiers"]["p0-candidate"]
    nodes = []
    for group in tier["groups"].values():
        if len(group["nodes"]) != group["expected_count"]:
            raise ValueError("Candidate group count disagrees with its node list")
        nodes.extend(group["nodes"])
    if not nodes or len(nodes) != tier["expected_count"] or len(nodes) != len(set(nodes)):
        raise ValueError("Candidate node list is empty, duplicated, or has an incorrect count")
    if tier["allowed_skips"] != 0 or tier["required_passes"] != len(nodes):
        raise ValueError("Every candidate node must execute and pass")
    for node in nodes:
        path = Path(node.split("::", 1)[0])
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "tests":
            raise ValueError(f"Candidate node must be below tests/: {node}")
        if "::" not in node or not (ROOT / path).is_file():
            raise ValueError(f"Candidate test does not exist: {node}")
    return nodes


def p0_groups(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Require an explicitly frozen, exact selection before starting P0."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported NPU suite manifest schema")
    tier = manifest["tiers"]["p0"]
    if tier.get("frozen") is not True:
        raise ValueError("P0 selection is not frozen; collect and review its exact nodes first")
    groups = tier["groups"]
    nodes = []
    if not isinstance(groups, dict) or not groups:
        raise ValueError("P0 requires named test groups")
    for name, group in groups.items():
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in name):
            raise ValueError("P0 group names must be safe directory names")
        if not group["nodes"] or len(group["nodes"]) != group["expected_count"]:
            raise ValueError(f"P0 group count disagrees with its node list: {name}")
        for node in group["nodes"]:
            path = Path(node.split("::", 1)[0])
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "tests":
                raise ValueError(f"P0 node must be below tests/: {node}")
            if "::" not in node or not (ROOT / path).is_file():
                raise ValueError(f"P0 test does not exist: {node}")
        nodes.extend(group["nodes"])
    if len(nodes) != len(set(nodes)) or len(nodes) != tier["expected_count"]:
        raise ValueError("P0 node list is duplicated or has an incorrect count")
    if tier["allowed_skips"] != 0 or tier["required_passes"] != len(nodes):
        raise ValueError("Every P0 node must execute and pass")
    return groups


def source_identity() -> dict[str, Any]:
    files = {}
    for directory in ("tools", "tests", "vllm_gr", "requirements"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for name in ("pyproject.toml", "setup.py", ".pre-commit-config.yaml"):
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "git_head": commit,
        "content_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def owned_processes(token: str) -> list[int]:
    """Find only descendants carrying this command's unguessable environment tag."""
    if not Path("/proc").is_dir():
        return []
    owned = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if (b"VLLM_GR_NPU_TEST_TOKEN=" + token.encode()) not in (
                path / "environ"
            ).read_bytes().split(b"\0"):
                continue
            if (path / "stat").read_text().split(") ", 1)[1].startswith("Z "):
                continue
            owned.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return owned


def clean_owned_processes(token: str) -> list[int]:
    for sig, duration in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        for pid in owned_processes(token):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + duration
        while owned_processes(token) and time.monotonic() < deadline:
            time.sleep(0.1)
    return owned_processes(token)


def run_command(argv: list[str], env: dict[str, str], log: Path, timeout: int | None) -> int:
    token = uuid.uuid4().hex
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=dict(env, VLLM_GR_NPU_TEST_TOKEN=token),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name == "posix",
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        interrupted = False
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            interrupted = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            returncode = 124
        finally:
            # Model tests may create their own sessions. The environment tag
            # survives that boundary; never kill processes merely by name/card.
            leftovers = owned_processes(token)
            remaining = clean_owned_processes(token) if leftovers else []
            reader.join(timeout=10)
            write_json(
                log.with_suffix(".execution.json"),
                {
                    "argv": argv,
                    "returncode": returncode,
                    "interrupted_or_timed_out": interrupted,
                    "wall_seconds": time.time() - started,
                    "cleaned_owned_pids": leftovers,
                    "remaining_owned_pids": remaining,
                    "output_reader_finished": not reader.is_alive(),
                    "cleanup_scope": "exact command environment tag; caller still owns the device lease",
                },
            )
            if remaining or reader.is_alive() or (leftovers and not interrupted):
                raise RuntimeError(
                    "Test process cleanup was required or incomplete; inspect execution evidence"
                )
        return returncode


def audit_report(report: dict[str, Any], expected: list[str] | None, rc: int) -> dict[str, Any]:
    selected = [item["nodeid"] for item in report.get("selected", [])]
    problems = []
    if rc != 0 or report.get("exitstatus") != 0:
        problems.append("pytest did not complete successfully")
    if report.get("collect_only"):
        problems.append("collection is not execution")
    if not selected or len(set(selected)) != len(selected):
        problems.append("selection is empty or duplicated")
    if expected is not None and Counter(selected) != Counter(expected):
        problems.append("collected nodes differ from the mandatory manifest")
    if report.get("collection_errors"):
        problems.append("collection failed")
    events: dict[str, list[dict[str, Any]]] = {}
    for event in report.get("reports", []):
        events.setdefault(event["nodeid"], []).append(event)
    if set(events) - set(selected):
        problems.append("execution contains nodes outside the frozen selection")
    outcomes = {}
    for node in selected:
        phases = events.get(node, [])
        counts = Counter(event["phase"] for event in phases)
        valid = (
            set(counts) <= {"setup", "call", "teardown"}
            and counts["setup"] == counts["teardown"] == 1
            and counts["call"] in (0, 1)
            and all(event["outcome"] in ("passed", "failed", "skipped") for event in phases)
            and (
                counts["call"] == 1
                or any(
                    event["phase"] == "setup" and event["outcome"] != "passed" for event in phases
                )
            )
        )
        if not valid:
            outcome = "NOT_RUN_OR_INCOMPLETE"
        elif any(event["outcome"] == "failed" for event in phases):
            outcome = "FAILED"
        elif any(event.get("wasxfail") is not None for event in phases):
            outcome = "XFAIL_OR_XPASS"
        elif any(event["outcome"] == "skipped" for event in phases):
            outcome = "SKIPPED"
        elif counts == Counter({"setup": 1, "call": 1, "teardown": 1}) and all(
            event["outcome"] == "passed" for event in phases
        ):
            outcome = "PASSED"
        else:
            outcome = "NOT_RUN_OR_INCOMPLETE"
        outcomes[node] = outcome
    outcome_counts = dict(Counter(outcomes.values()))
    if expected is not None and any(value != "PASSED" for value in outcomes.values()):
        problems.append("a mandatory candidate did not pass all three phases")
    if not outcome_counts.get("PASSED", 0):
        problems.append("no test executed and passed")
    if outcome_counts.get("FAILED") or outcome_counts.get("NOT_RUN_OR_INCOMPLETE"):
        problems.append("failed or incomplete test execution")
    return {
        "status": "FAILED" if problems else "PASSED",
        "problems": problems,
        "counts": outcome_counts,
        "nodes": outcomes,
        "collection_skips": report.get("collection_skips", []),
    }


def run_p0_groups(
    groups: dict[str, dict[str, Any]],
    env: dict[str, str],
    output: Path,
    timeout: int,
) -> dict[str, Any]:
    """Isolate host fixtures from native/model tests without changing Python."""
    results: dict[str, Any] = {
        name: {
            "status": "NOT_RUN",
            "expected_count": group["expected_count"],
            "nodes": dict.fromkeys(group["nodes"], "NOT_RUN"),
        }
        for name, group in groups.items()
    }
    commands = []
    stopped = False
    for name, group in groups.items():
        if stopped:
            continue
        directory = output / "groups" / name
        directory.mkdir(parents=True, exist_ok=False)
        report_path = directory / "pytest-events.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "--run-slow",
            "--durations=10",
            "-s",
            "-p",
            "npu_pytest_report",
            "-p",
            "pytest_timeout",
            "-p",
            "pytest_asyncio.plugin",
            "--junitxml=" + str(directory / "junit.xml"),
            "--basetemp=" + str(directory / "basetemp"),
            *group["nodes"],
        ]
        commands.append(command)
        item = results[name]
        item.update(status="RUNNING", command=command, evidence_dir=str(directory))
        write_json(directory / "result.json", item)
        rc = 1
        try:
            rc = run_command(
                command,
                dict(env, VLLM_GR_NPU_TEST_REPORT=str(report_path)),
                directory / "pytest.log",
                timeout,
            )
            item["pytest_returncode"] = rc
        except (OSError, ValueError, RuntimeError) as error:
            item["command_error"] = str(error)
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            item.update(audit_report(report, group["nodes"], rc))
            item["phase_counts"] = dict(
                Counter(
                    f"{event['phase']}:{event['outcome']}" for event in report.get("reports", [])
                )
            )
            if report.get("collection_skips") or not (directory / "junit.xml").is_file():
                item["status"] = "FAILED"
                item["problems"].append("collection skipped a module or JUnit evidence is missing")
            if "command_error" in item:
                item["status"] = "FAILED"
                item["problems"].append("group command or process cleanup failed")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            item.update(status="FAILED", error=str(error))
        stopped = item["status"] != "PASSED"
        write_json(directory / "result.json", item)
    nodes = {
        node: item["nodes"].get(node, "MISSING")
        for name, group in groups.items()
        for node in group["nodes"]
        for item in (results[name],)
    }
    phase_counts: Counter[str] = Counter()
    for item in results.values():
        phase_counts.update(item.get("phase_counts", {}))
    problems = [
        f"{name}: {item['status']}" for name, item in results.items() if item["status"] != "PASSED"
    ]
    if any(outcome != "PASSED" for outcome in nodes.values()):
        problems.append("not every frozen P0 node passed")
    return {
        "status": "FAILED" if problems else "PASSED",
        "problems": problems,
        "counts": dict(Counter(nodes.values())),
        "nodes": nodes,
        "groups": results,
        "phase_counts": dict(phase_counts),
        "commands": commands,
        "pytest_returncode": 1 if problems else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tier", choices=("full", "p0-candidate", "p0"), default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeout",
        type=int,
        help="Pytest deadline in seconds (candidate/P0 group: 1800; full: job deadline)",
    )
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = args.output_dir.resolve()
    # mkdir is exclusive: stale results can never qualify a new run.
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "tier": args.tier,
        "started_unix": time.time(),
        "commands": [],
    }
    rc = 1
    try:
        env = device_environment(args.device, dict(os.environ))
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        expected = candidate_nodes(manifest) if args.tier == "p0-candidate" else None
        groups = p0_groups(manifest) if args.tier == "p0" else None
        if groups is not None:
            result["groups"] = {
                name: {
                    "status": "NOT_RUN",
                    "expected_count": group["expected_count"],
                    "nodes": dict.fromkeys(group["nodes"], "NOT_RUN"),
                }
                for name, group in groups.items()
            }
        write_json(output / "suite.json", manifest if args.tier != "full" else {"tier": "full"})
        before = source_identity()
        write_json(output / "source.json", before)
        result["device"] = args.device
        result["allocation_owner"] = (
            "caller; this runner does not acquire or release a device lease"
        )
        result["model"] = env.get("VLLM_GR_TEST_MODEL")
        if args.tier != "full":
            model = Path(env.get("VLLM_GR_TEST_MODEL", ""))
            if not env.get("VLLM_GR_TEST_MODEL") or not (model / "config.json").is_file():
                raise ValueError(
                    f"{args.tier} requires VLLM_GR_TEST_MODEL pointing to a local model"
                )
            env.update(
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
            )
        runtime = [
            sys.executable,
            str(ROOT / "tools/ci/resolve_runtime.py"),
            "resolve",
            "--device",
            "npu",
            "--output",
            str(output / "runtime-policy.json"),
        ]
        preflight = [
            sys.executable,
            str(ROOT / "tools/ci/npu_test_preflight.py"),
            str(output / "runtime.json"),
        ]
        for index, command in enumerate((runtime, preflight)):
            result["commands"].append(command)
            if run_command(
                command, env, output / f"preflight-{index}.log", min(args.timeout or 120, 120)
            ):
                raise RuntimeError(f"NPU preflight {index} failed; tests were not run")
        if groups is not None:
            grouped = run_p0_groups(groups, env, output, args.timeout or 1800)
            result["commands"].extend(grouped.pop("commands"))
            result.update(grouped)
            result["source_unchanged"] = (
                before["content_sha256"] == source_identity()["content_sha256"]
            )
            if not result["source_unchanged"]:
                result["status"] = "FAILED"
                result["problems"].append("source changed during P0 execution")
            rc = 0 if result["status"] == "PASSED" else 1
            return rc
        report_path = output / "pytest-events.json"
        env["VLLM_GR_NPU_TEST_REPORT"] = str(report_path)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "--run-slow",
            "--durations=10",
            "-s",
            "-p",
            "npu_pytest_report",
            "--junitxml=" + str(output / "junit.xml"),
        ]
        if expected is not None:
            command += ["-p", "pytest_timeout", "-p", "pytest_asyncio.plugin", *expected]
        else:
            command += ["tests"]
        result["commands"].append(command)
        write_json(output / "result.json", result)
        timeout = (
            args.timeout if args.timeout is not None else (1800 if expected is not None else None)
        )
        pytest_rc = run_command(command, env, output / "pytest.log", timeout)
        result["pytest_returncode"] = pytest_rc
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        result.update(audit_report(report, expected, pytest_rc))
        if expected is not None:
            result["groups"] = {
                name: {
                    "expected_count": group["expected_count"],
                    "counts": dict(
                        Counter(result["nodes"].get(node, "MISSING") for node in group["nodes"])
                    ),
                }
                for name, group in manifest["tiers"]["p0-candidate"]["groups"].items()
            }
        result["source_unchanged"] = before["content_sha256"] == source_identity()["content_sha256"]
        if not result["source_unchanged"] or not (output / "junit.xml").is_file():
            result["status"] = "FAILED"
            result["problems"].append("source changed or JUnit evidence is missing")
        rc = 0 if result["status"] == "PASSED" else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result["status"] = "FAILED" if "pytest_returncode" in result else "BLOCKED"
        result["error"] = str(error)
    finally:
        result["finished_unix"] = time.time()
        write_json(output / "result.json", result)
        print(f"NPU {args.tier}: {result['status']}; evidence: {output}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
