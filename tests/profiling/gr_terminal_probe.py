# SPDX-License-Identifier: Apache-2.0
"""Retain real model/engine references to audit active-request shutdown.

The probe intentionally stops its own synchronous caller after a real stage,
without aborting the request. This tests shutdown, not cancellation delivery.
No request dictionary or device state is manually cleared.
Optional Worker faults execute real Decode leaf work before raising a unique
Python exception; they do not simulate a CUDA context failure.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from tests.profiling import profile_onerec_b1_l1024_sid3 as baseline
from tests.profiling.gr_lifecycle_probe import _assert_graph_delta, _snapshot, _source
from tests.profiling.gr_worker_fault import inject_first_decode_failure, injected_failure_delivery


class _CallerStopped(Exception):
    """Probe-owned suspension of the caller; never a product abort result."""


def _beam_slots_restored(registry, initial_free):
    """The live-slot dictionary is not the allocator's fixed capacity."""
    free = registry._free_slots
    return (
        not registry._slot_map and not registry._slots
        and len(free) == registry.max_batch
        and set(free) == set(range(registry.max_batch))
        and (initial_free is None or len(free) == initial_free)
    )


def _run(args, report):
    baseline._require_v1_model_runner()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("terminal probe requires the in-process EngineCore")
    baseline._preflight_backend("gpu")
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.plugin import initialize_runtime
    from vllm_gr.sampling_params import BeamSearchParams
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
        source=_source(), terminal_probe_sha256=baseline._sha256_file(Path(__file__)),
        worker_fault_probe_sha256=baseline._sha256_file(
            Path(__file__).with_name("gr_worker_fault.py")
        ),
        config=baseline._jsonable(vars(args)), engine_kwargs=kwargs,
        dependencies=baseline._dependency_versions(), device=baseline._driver_accelerator_info("gpu"),
        catalog_sha256=baseline._sha256_file(args.catalog),
        prompt_sha256=baseline._sha256_file(args.prompt),
    )
    llm = GRLLM(model=args.model, **kwargs)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    worker = core.model_executor.driver_worker.worker
    runner = worker.model_runner
    original_step = engine.step
    target = 1 if args.boundary == "prefill" else 2
    try:
        baseline._validate_v1_engine_config(engine.vllm_config)
        assert type(core.model_executor).__name__ == "UniProcExecutor"
        initial_free_blocks = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        # Eager initializes Beam context lazily on the first real Prefill.
        # Do not allocate a session or run an extra request to obtain baseline.
        initial_context = getattr(runner, "_beam_context", None)
        initial = {
            "beam_graph": get_beam_graph_runtime_stats(worker),
            "beam_context_initialized": initial_context is not None,
            "beam_slots_free": (
                len(initial_context.registry._free_slots) if initial_context is not None else None
            ),
            "native_blocks_free": initial_free_blocks,
        }
        report["initial"] = initial
        prompts, _ = baseline._prepare_prompt(llm, args.prompt, args.input_len)
        params = BeamSearchParams(
            beam_width=args.beam_width, max_tokens=args.generation_steps + 2,
            temperature=0.0, ignore_eos=False,
            begin_token="<|sid_begin|>", end_token="<|sid_end|>",
        )

        def stop_caller_after_stage():
            outputs = original_step()
            active = list(core.scheduler.requests.values())
            if len(active) == 1 and active[0].gr_issued_steps == target:
                request = active[0]
                assert not outputs
                assert request.gr_completed_steps == target - 1
                report["request_id"] = request.request_id
                report["native_status_before"] = request.status.name
                report["boundary"] = {
                    "issued": request.gr_issued_steps,
                    "completed": request.gr_completed_steps,
                    "device_step": runner._gr_model_runner_states[request.request_id].device_step,
                }
                assert report["boundary"]["device_step"] == target
                raise _CallerStopped("probe caller stops stepping, without abort")
            return outputs

        if args.fault == "none":
            engine.step = stop_caller_after_stage
            try:
                llm.beam_search(prompts, params, use_tqdm=False)
            except _CallerStopped:
                pass
            else:
                raise AssertionError("caller was not stopped at the requested boundary")
            finally:
                engine.step = original_step
        else:
            with inject_first_decode_failure(runner, report, args.fault) as expected:
                try:
                    llm.beam_search(prompts, params, use_tqdm=False)
                except RuntimeError as error:
                    report["worker_error_delivery"] = injected_failure_delivery(
                        error, expected, args.fault,
                    )
                    report["worker_error"] = repr(error)
                    if report["worker_error_delivery"] == "quarantined_previous_control":
                        failed_decode = [
                            future for future, scheduled, _ in core.batch_queue
                            if (sidecar := getattr(scheduled, "gr_outputs", {}).get(
                                report["worker_fault"]["request_id"],
                            )) is not None and sidecar.step == 1
                        ]
                        assert len(failed_decode) == 1 and failed_decode[0].done()
                        # Only inspect an already completed native Future;
                        # never drive delivery of a pending control here.
                        assert failed_decode[0].exception() is expected
                        report["failed_decode_future_preserves_exception"] = True
                else:
                    raise AssertionError("Worker fault did not reach the real frontend")
            assert report["worker_fault"]["fired"] == 1
            report["request_id"] = report["worker_fault"]["request_id"]
            # The failing engine must not be reused, even if the Worker has
            # already released its semantic state during exception unwinding.
            transport = runner._gr_cuda_output_transport
            assert transport._failure is expected
            try:
                transport.check_live()
            except RuntimeError as error:
                assert error.__cause__ is expected
            else:
                raise AssertionError("failed Worker transport remained reusable")
        before = _snapshot(engine, core, runner)
        report["before_shutdown"] = before
        assert before["scheduler_requests"] == before["frontend_requests"] == 1
        if args.fault != "none":
            from vllm_gr.v1.worker import cuda_async_output

            transport = runner._gr_cuda_output_transport
            entries = [
                {
                    "request_id": entry.lease.request_id,
                    "device_step": entry.lease.device_step,
                    "device_started": entry.device_started,
                    "producer_active": entry.producer_active,
                    "registered": entry.registered,
                }
                for _, pool in transport._pools.values()
                for entry in pool._entries if entry is not None
            ]
            report["failure_ownership"] = {
                "quarantined": transport in cuda_async_output._QUARANTINED,
                "retained_workspaces": len(transport._pools),
                "control_entries": entries,
            }
            assert report["failure_ownership"]["quarantined"]
            assert report["failure_ownership"]["retained_workspaces"] == 1
            if args.fault == "advance":
                assert any(
                    entry["request_id"] == report["request_id"]
                    and entry["device_step"] == 2 and entry["device_started"]
                    and not entry["producer_active"] and not entry["registered"]
                    for entry in entries
                ), entries
        if args.fault == "none":
            assert before["state"]["active"] == 1
            assert before["control"]["control_slots_occupied"] == 1
            assert any(
                report["request_id"] in entry["request_ids"] and not entry["future_done"]
                for entry in before["batch_queue"]
            )
        _assert_graph_delta(
            initial["beam_graph"], before["beam_graph"],
            execution=args.execution, width=args.beam_width,
            decode_dispatches=0 if args.fault == "prepare" else target - 1,
        )
        report["graph_dispatch_gate"] = "PASS"
        native_request = core.scheduler.requests[report["request_id"]]
        assert not native_request.is_finished()
        report["native_status_before"] = native_request.status.name
        llm.shutdown()
        # Deliberately keep all references. GC cannot hide missing logical
        # finish/free or frontend registry cleanup in this acceptance check.
        after = _snapshot(engine, core, runner)
        report["after_shutdown"] = after
        report["native_status_after"] = native_request.status.name
        report["external_ids_after"] = dict(engine.output_processor.external_req_ids)
        report["shutdown_complete"] = llm._shutdown_complete
        transport = runner._gr_cuda_output_transport
        report["native_completion_tickets_after"] = transport.api.lib.gr_ticket_live()
        report["gates"] = {
            "native_request_finished": native_request.is_finished(),
            "scheduler_requests_freed": after["scheduler_requests"] == 0,
            "native_blocks_restored": after["native_blocks_free"] == initial_free_blocks,
            "frontend_registry_empty": after["frontend_requests"] == 0,
            "frontend_external_ids_empty": not report["external_ids_after"],
            "state_balanced": after["state"]["created"] == after["state"]["freed"] == 1,
            "state_inactive": after["state"]["active"] == 0,
            "beam_sessions_released": after["beam_sessions"] == after["beam_slots_active"] == 0,
            "beam_slots_restored": _beam_slots_restored(
                runner._beam_context.registry, initial["beam_slots_free"],
            ),
            "control_storage_drained": after["control"]["control_slots_occupied"] == 0,
            "control_pool_pinned_ownership_released": after["control"]["control_pinned_bytes"] == 0,
            "transport_drained": transport._terminal_drained and not transport._pools,
            "native_completion_tickets_released": report["native_completion_tickets_after"] == 0,
            "shutdown_complete": llm._shutdown_complete and llm.llm_engine is None,
        }
        assert all(report["gates"].values()), report["gates"]
        llm.shutdown()  # A successful repeated shutdown must be a no-op.
        report["status"] = "PASS"
    finally:
        engine.step = original_step
        if not llm._shutdown_complete:
            llm.shutdown()


def main():
    if not __debug__:
        raise RuntimeError("terminal probe requires assertions; do not use python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--boundary", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--fault", choices=("none", "prepare", "advance"), default="none")
    parser.set_defaults(
        backend="gpu", prefix_cache="disabled", dtype="bfloat16",
        beam_width=128, generation_steps=3, input_len=1024,
        gpu_memory_utilization=0.7, async_scheduling=True, seed=0,
    )
    args = parser.parse_args()
    if args.fault != "none" and args.boundary != "decode":
        parser.error("Worker fault injection requires --boundary decode")
    report = {"status": "FAIL"}
    try:
        _run(args, report)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
