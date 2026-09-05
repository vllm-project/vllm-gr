# SPDX-License-Identifier: Apache-2.0
"""One-shot Python Worker faults after real first-Decode leaf operations.

These hooks do not replace execute_model/sample_tokens, synthesize outputs or
touch CUDA context health. Install only after real model initialization.
"""

from contextlib import contextmanager


class InjectedWorkerError(RuntimeError):
    """Unique expected fault; unrelated runtime errors must fail the probe."""


def injected_failure_delivery(error, expected, fault):
    """Accept only the injected object or its precise old-control guard.

    BatchQueue can consume Prefill output before the failed Decode sampling
    Future. Its pool must reject reading after the Decode poisoned transport.
    Exception context or a merely matching message is not sufficient evidence.
    """
    if error is expected:
        return "direct"
    if (
        fault == "advance" and type(error) is RuntimeError
        and error.args == ("control pool quarantined; device drain is required",)
        and error.__cause__ is expected
    ):
        return "quarantined_previous_control"
    raise AssertionError("frontend did not deliver the exact injected Worker fault") from error


def _target(runner, scheduled):
    data = getattr(scheduled, "beam_data", {}) or {}
    matches = [
        (request_id, metadata) for request_id, metadata in data.items()
        if metadata.get("persistent_request")
        and metadata.get("persistent_runner_state")
        and metadata.get("is_beam_decode")
        and metadata.get("generation_index") == 1
    ]
    if not matches:
        return None
    assert len(data) == len(matches) == 1
    request_id, metadata = matches[0]
    state = runner._gr_model_runner_states[request_id]
    assert metadata["decode_step"] == 0
    assert state.stage.name == "PREFILL_BOOTSTRAPPED"
    assert state.step == state.device_step == state._prepared_step == 1
    sidecar = scheduled.gr_outputs[request_id]
    assert sidecar.logical_token_count == 1
    assert sidecar.physical_row_count == state.beam_width
    return request_id, metadata, state


@contextmanager
def inject_first_decode_failure(runner, report, fault):
    if fault not in ("prepare", "advance"):
        raise ValueError(f"unknown Worker fault: {fault}")
    owner = runner if fault == "prepare" else runner._gr_device_backend
    name = "_prepare_inputs" if fault == "prepare" else "advance_beam"
    had_override = name in vars(owner)
    override = vars(owner).get(name)
    original = getattr(owner, name)
    record = report["worker_fault"] = {"kind": fault, "fired": 0}
    failure = InjectedWorkerError(f"injected persistent first Decode {fault} failure")

    def fire(target, result=None):
        request_id, metadata, state = target
        assert record["fired"] == 0
        record.update(
            fired=1, request_id=request_id, generation_index=metadata["generation_index"],
            decode_step=metadata["decode_step"], state_stage=state.stage.name,
            state_step=state.step, device_step=state.device_step,
            prepared_step=state._prepared_step,
            control=runner._gr_cuda_output_transport.stats(),
        )
        if result is not None:
            assert result.device_step == 2 and result.control_output is not None
            record["backend_result_device_step"] = result.device_step
        raise failure

    def prepare(scheduled, *args, **kwargs):
        result = original(scheduled, *args, **kwargs)
        target = _target(runner, scheduled)
        if target is not None:
            fire(target)
        return result

    def advance(current_runner, state, sampler_output, metadata, *, is_last_step):
        target = _target(runner, runner._current_scheduler_output)
        if target is not None:
            assert current_runner is runner and target[2] is state
            assert target[1] is metadata and not is_last_step
        result = original(
            current_runner, state, sampler_output, metadata, is_last_step=is_last_step,
        )
        if target is not None:
            fire(target, result)
        return result

    setattr(owner, name, prepare if fault == "prepare" else advance)
    try:
        yield failure
    finally:
        if had_override:
            setattr(owner, name, override)
        else:
            delattr(owner, name)
