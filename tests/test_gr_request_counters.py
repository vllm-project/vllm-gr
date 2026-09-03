# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from vllm_gr.v1.metrics.stats import GRRequestCounters


def test_gr_request_counters_default_to_zero() -> None:
    counters = GRRequestCounters()
    assert all(value == 0 for value in counters.to_dict().values())


def test_gr_request_counters_merge_process_values() -> None:
    engine = GRRequestCounters(
        native_request_created_per_session=1,
        scheduler_add_request_per_session=1,
        decode_scheduler_calls=3,
    )
    worker = GRRequestCounters(
        gr_state_created_per_session=1,
        gr_state_freed_per_session=1,
        decode_worker_dispatches=3,
        final_result_d2h_count=1,
        final_result_d2h_bytes=128,
    )
    engine.add(worker)
    assert engine.native_request_created_per_session == 1
    assert engine.gr_state_created_per_session == 1
    assert engine.decode_scheduler_calls == engine.decode_worker_dispatches == 3
    assert engine.final_result_d2h_count == 1
    assert engine.final_result_d2h_bytes == 128
