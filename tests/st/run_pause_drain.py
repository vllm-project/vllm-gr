# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Verify Sidecar pause/reload/resume while a real GR request drains.

The test launcher delays GR stages to expose an in-flight request and records
native drain/reload events. It does not change weights, outputs or pause logic.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import Future
from functools import wraps
from pathlib import Path
from typing import Any


def install_probes() -> None:
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.worker import model_runner_patch
    from vllm_gr.v1.worker.gpu.beam_stage_runner import GPUBeamStageRunner

    if getattr(GPUBeamStageRunner.execute, "_pause_drain_probe", False):
        return

    path = Path(os.environ["GR_PAUSE_DRAIN_EVENTS"])
    armed = path.with_suffix(".armed")

    def record(event: str, **fields: Any) -> None:
        with path.open("a") as stream:
            stream.write(
                json.dumps(dict(event=event, time_ns=time.monotonic_ns(), **fields)) + "\n"
            )

    execute = GPUBeamStageRunner.execute

    @wraps(execute)
    def delayed_stage(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = execute(self, *args, **kwargs)
        if armed.exists():
            record("stage_in_flight")
            time.sleep(1)
        return result

    GPUBeamStageRunner.execute = delayed_stage
    pause = EngineCoreProc.pause_scheduler

    @wraps(pause)
    def traced_pause(self: Any, *args: Any, **kwargs: Any) -> Any:
        had_work = self.has_work()
        result = pause(self, *args, **kwargs)
        record("pause", had_work=had_work, pending=isinstance(result, Future) and not result.done())
        if isinstance(result, Future):
            result.add_done_callback(
                lambda future: record("drain_complete", result=future.result())
            )
        return result

    EngineCoreProc.pause_scheduler = traced_pause
    resume = EngineCoreProc.resume_scheduler

    @wraps(resume)
    def traced_resume(self: Any, *args: Any, **kwargs: Any) -> Any:
        record("resume")
        return resume(self, *args, **kwargs)

    EngineCoreProc.resume_scheduler = traced_resume
    install = model_runner_patch.install_worker_utility_hooks

    @wraps(install)
    def install_worker(worker_cls: Any) -> None:
        install(worker_cls)
        native = worker_cls.reload_weights
        if getattr(native, "_pause_drain_probe", False):
            return

        @wraps(native)
        def reload_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            record("reload_begin")
            result = native(self, *args, **kwargs)
            record("reload_complete")
            return result

        setattr(reload_weights, "_pause_drain_probe", True)
        worker_cls.reload_weights = reload_weights

    model_runner_patch.install_worker_utility_hooks = install_worker
    setattr(delayed_stage, "_pause_drain_probe", True)


# Spawned Engine Core processes import this launcher as __mp_main__.
if os.environ.get("GR_PAUSE_DRAIN_EVENTS"):
    install_probes()


def main() -> None:
    from tests.st.run_checkpoint_reload import outputs, prepare, request, server
    from vllm_gr.sidecar import (
        ControlClient,
        Instance,
        PausedUpdateProcess,
        UpdateContext,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    first, second, config = prepare(root)
    events = root / "events.jsonl"
    events.write_text("")
    os.environ["GR_PAUSE_DRAIN_EVENTS"] = str(events)
    armed = events.with_suffix(".armed")
    armed.unlink(missing_ok=True)

    def records() -> list[dict[str, Any]]:
        return [json.loads(line) for line in events.read_text().splitlines()]

    with server(root, first, config, "pause-drain", Path(__file__).resolve()) as (
        stub,
        control,
    ):
        before = outputs(stub.GenerateBatch(request(), timeout=120))
        armed.touch()
        in_flight = stub.GenerateBatch.future(request(), timeout=120)
        deadline = time.monotonic() + 30
        while not any(row["event"] == "stage_in_flight" for row in records()):
            if in_flight.done():
                raise AssertionError("Request finished before the in-flight stage was observed")
            if time.monotonic() >= deadline:
                raise TimeoutError("No GR stage observed")
            time.sleep(0.01)

        instance = Instance("worker", str(control.base_url).rstrip("/"))
        context = UpdateContext("weights", "checkpoint", "b", second)
        process = PausedUpdateProcess(ControlClient(timeout_seconds=120))
        assert asyncio.run(process.run(instance, context))
        armed.unlink()
        assert outputs(in_flight.result(timeout=120)) == before
        status = control.get("/service/status").json()
        assert status["service_state"] == "running"
        assert status["checkpoint_valid"] and status["checkpoint_version"] == "b"
        assert outputs(stub.GenerateBatch(request(), timeout=120)) != before

    rows = records()
    pauses = [row for row in rows if row["event"] == "pause"]
    assert len(pauses) == 1 and pauses[0]["had_work"] and pauses[0]["pending"]
    drain = next(row for row in rows if row["event"] == "drain_complete")
    assert drain["result"] is None
    ordered = ["pause", "drain_complete", "reload_begin", "reload_complete", "resume"]
    assert [row["event"] for row in rows if row["event"] in ordered] == ordered
    timestamps = [row["time_ns"] for row in rows if row["event"] in ordered]
    assert timestamps == sorted(timestamps)
    result = dict(
        passed=True,
        batch=4,
        prompt_tokens=64,
        decode_steps=3,
        events=rows,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        sys.argv.pop(1)
        from vllm_gr.entrypoints.cli.main import dispatch_main

        dispatch_main()
    else:
        main()
