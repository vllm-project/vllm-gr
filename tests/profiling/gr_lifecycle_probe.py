# SPDX-License-Identifier: Apache-2.0
"""Full-model, in-process persistent-GR abort/reuse probe (not a benchmark).

Run on one GPU with explicit V1 and in-process environment settings. Injection
uses the real frontend abort API after an actual non-final engine step; no
fake final output, state deletion, or replacement Worker is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from tests.profiling import profile_onerec_b1_l1024_sid3 as baseline


def _source():
    paths = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "vllm_gr",
    ], text=True).splitlines()
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output([
            "git", "branch", "--show-current",
        ], text=True).strip(),
        "runtime_files_sha256": {
            name: baseline._sha256_file(Path(name))
            for name in sorted(set(paths)) if Path(name).is_file()
        },
        "probe_sha256": baseline._sha256_file(Path(__file__)),
    }


def _snapshot(engine, core, runner):
    import torch
    from vllm_gr.v1.worker.gr_model_runner_state import get_gr_model_runner_state_stats
    from vllm_gr.v1.worker.model_runner_common import get_beam_graph_runtime_stats

    context = runner._beam_context
    queued = []
    for future, scheduled, _ in core.batch_queue:
        queued.append({
            "future_done": future.done(),
            "request_ids": list(getattr(scheduled, "gr_outputs", {}) or {}),
        })
    return {
        "state": get_gr_model_runner_state_stats(runner),
        "control": runner._gr_cuda_output_transport.stats(),
        "beam_graph": get_beam_graph_runtime_stats(core.model_executor.driver_worker.worker),
        "scheduler_requests": len(core.scheduler.requests),
        "frontend_requests": len(engine.output_processor.request_states),
        "beam_sessions": len(runner._beam_sessions),
        "beam_slots_active": len(context.registry._slot_map),
        "beam_slots_free": len(context.registry._free_slots),
        "native_blocks_free": core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        "batch_queue": queued,
        "allocated_bytes": torch.cuda.memory_allocated(runner.device),
        "reserved_bytes": torch.cuda.memory_reserved(runner.device),
    }


def _assert_released(snapshot, reference, *, drained):
    assert snapshot["state"]["created"] == snapshot["state"]["freed"]
    for key in ("scheduler_requests", "frontend_requests", "beam_sessions", "beam_slots_active"):
        assert snapshot[key] == 0, (key, snapshot)
    assert snapshot["state"]["active"] == 0
    assert snapshot["native_blocks_free"] == reference["native_blocks_free"]
    assert snapshot["beam_slots_free"] == reference["beam_slots_free"]
    if drained:
        assert not snapshot["batch_queue"]
        assert snapshot["control"]["control_slots_occupied"] == 0


def _assert_abort_window(case, issued_target):
    injected = case["injected"]
    assert injected["issued"] == injected["device_step"] == issued_target
    assert injected["completed"] == issued_target - 1
    for label in ("before_abort", "after_abort"):
        assert any(
            injected["request_id"] in queued["request_ids"] and not queued["future_done"]
            for queued in case[label]["batch_queue"]
        ), (label, case)
    assert case["after_abort"]["control"]["control_slots_occupied"] == 1


def _assert_graph_delta(before, after, *, execution, width, decode_dispatches):
    enabled = execution == "graph"
    assert before["beam_graph_enabled"] == after["beam_graph_enabled"] == enabled
    if enabled:
        assert width in before["captured_widths"] and width in after["captured_widths"]
    expected = decode_dispatches if enabled else 0
    assert after["full_dispatches"] - before["full_dispatches"] == expected


def _run(args, report):
    baseline._require_v1_model_runner()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("probe requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    baseline._preflight_backend("gpu")

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.plugin import initialize_runtime
    from vllm_gr.sampling_params import BeamSearchParams
    from vllm_gr.v1.worker import gr_model_runner_state
    from vllm_gr.v1.worker.model_runner_common import get_beam_graph_runtime_stats

    initialize_runtime()
    kwargs = baseline._engine_kwargs(args)
    kwargs["additional_config"] = {
        "vllm_gr_persistent_request": True,
        "vllm_gr_persistent_runner_state": True,
        "vllm_gr_scheduling_mode": "async",
        "vllm_gr_beam_graph_enabled": args.execution == "graph",
    }
    report.update(
        source=_source(), config=baseline._jsonable(vars(args)),
        engine_kwargs=kwargs, dependencies=baseline._dependency_versions(),
        device=baseline._driver_accelerator_info("gpu"),
        catalog_sha256=baseline._sha256_file(args.catalog),
        prompt_sha256=baseline._sha256_file(args.prompt), cases=[], successful_graph_dispatches=[],
    )
    compact_digests = []
    original_unpack = gr_model_runner_state.unpack_gr_final_output

    def record_final(output):
        result = original_unpack(output)
        if result is not None:
            compact_digests.append(hashlib.sha256(output["buffer"]).hexdigest())
        return result

    gr_model_runner_state.unpack_gr_final_output = record_final
    try:
        with GRLLM(model=args.model, **kwargs) as llm:
            engine = llm.llm_engine
            core = engine.engine_core.engine_core
            runner = core.model_executor.driver_worker.worker.model_runner
            assert type(core.model_executor).__name__ == "UniProcExecutor"
            baseline._validate_v1_engine_config(engine.vllm_config)
            prompts, _ = baseline._prepare_prompt(llm, args.prompt, args.input_len)
            params = BeamSearchParams(
                beam_width=args.beam_width, max_tokens=args.generation_steps + 2,
                temperature=0.0, ignore_eos=False,
                begin_token="<|sid_begin|>", end_token="<|sid_end|>",
            )

            def drain():
                # FINAL_ONLY can return before a trailing zero-token queue
                # entry is reconciled. Drain that actual entry explicitly.
                for _ in range(8):
                    if not core.scheduler.has_requests() and not core.batch_queue:
                        return
                    assert not engine.step(), "drain emitted unexpected user output"
                raise AssertionError("BatchQueue failed to drain within eight steps")

            def success():
                before = len(compact_digests)
                graph_before = get_beam_graph_runtime_stats(core.model_executor.driver_worker.worker)
                outputs = llm.beam_search(prompts, params, use_tqdm=False)
                assert len(compact_digests) == before + 1
                drain()
                graph_after = get_beam_graph_runtime_stats(core.model_executor.driver_worker.worker)
                _assert_graph_delta(
                    graph_before, graph_after, execution=args.execution,
                    width=args.beam_width, decode_dispatches=args.generation_steps - 1,
                )
                report["successful_graph_dispatches"].append({
                    "before": graph_before, "after": graph_after,
                })
                return {
                    "public": baseline._output_fingerprints(outputs),
                    "compact_sha256": compact_digests[-1],
                }

            reference = success()
            clean = _snapshot(engine, core, runner)
            _assert_released(clean, clean, drained=True)
            report["reference"] = reference
            report["reference_snapshot"] = clean

            for stage, issued_target in (("prefill", 1), ("decode", 2)):
                for recovery in ("drain", "immediate_reuse"):
                    case = {"stage": stage, "recovery": recovery, "status": "running"}
                    report["cases"].append(case)
                    case["before"] = _snapshot(engine, core, runner)
                    original_step = engine.step
                    final_count = len(compact_digests)

                    def injecting_step():
                        outputs = original_step()
                        if "injected" in case:
                            return outputs
                        active = [request for request in core.scheduler.requests.values()
                                  if getattr(request, "gr_persistent_runner_state", False)]
                        if len(active) == 1 and active[0].gr_issued_steps == issued_target:
                            request = active[0]
                            assert not outputs, "abort injection must precede final output"
                            case["injected"] = {
                                "request_id": request.request_id,
                                "issued": request.gr_issued_steps,
                                "completed": request.gr_completed_steps,
                                "device_step": runner._gr_model_runner_states[request.request_id].device_step,
                            }
                            assert case["injected"]["device_step"] == issued_target
                            case["before_abort"] = _snapshot(engine, core, runner)
                            engine.abort_request([request.request_id], internal=True)
                            case["after_abort"] = _snapshot(engine, core, runner)
                        return outputs

                    engine.step = injecting_step
                    try:
                        llm.beam_search(prompts, params, use_tqdm=False)
                    except RuntimeError as error:
                        if "injected" not in case or "terminated before final output" not in str(error):
                            raise
                        case["frontend_error"] = str(error)
                    else:
                        raise AssertionError("aborted beam_search returned a synthetic result")
                    finally:
                        engine.step = original_step
                    assert len(compact_digests) == final_count
                    aborted = case["after_abort"]
                    _assert_abort_window(case, issued_target)
                    _assert_graph_delta(
                        case["before"]["beam_graph"], aborted["beam_graph"],
                        execution=args.execution, width=args.beam_width,
                        decode_dispatches=issued_target - 1,
                    )
                    _assert_released(aborted, clean, drained=False)
                    assert aborted["state"]["created"] - case["before"]["state"]["created"] == 1
                    assert aborted["state"]["freed"] - case["before"]["state"]["freed"] == 1

                    if recovery == "drain":
                        drain()
                        case["after_drain"] = _snapshot(engine, core, runner)
                        _assert_released(case["after_drain"], clean, drained=True)
                    # The immediate branch deliberately does not consume old
                    # controls before the next ordinary beam_search call.
                    assert success() == reference
                    case["after_success"] = _snapshot(engine, core, runner)
                    _assert_released(case["after_success"], clean, drained=True)
                    case["status"] = "PASS"
                    print(json.dumps({"case": stage, "recovery": recovery, "status": "PASS"}), flush=True)
            report["final_snapshot"] = _snapshot(engine, core, runner)
    finally:
        gr_model_runner_state.unpack_gr_final_output = original_unpack
    report["status"] = "PASS"


def main():
    if not __debug__:
        raise RuntimeError("lifecycle probe requires assertions; do not use python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.set_defaults(
        backend="gpu", prefix_cache="disabled", dtype="bfloat16",
        beam_width=128, generation_steps=3, input_len=1024,
        gpu_memory_utilization=0.7, async_scheduling=True, seed=0,
    )
    args = parser.parse_args()
    report = {"status": "FAIL"}
    try:
        _run(args, report)
    except BaseException as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
