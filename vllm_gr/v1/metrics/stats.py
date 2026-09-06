# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, dataclass

from vllm.v1.metrics.stats import RequestStateStats as BaseRequestStateStats


@dataclass(slots=True)
class GRRequestCounters:
    """Direct counters for one stable GR request."""

    native_request_created_per_session: int = 0
    scheduler_add_request_per_session: int = 0
    gr_state_created_per_session: int = 0
    gr_state_freed_per_session: int = 0
    decode_request_objects_created: int = 0
    decode_prefix_cache_queries: int = 0
    intermediate_beam_state_d2h_bytes: int = 0
    decode_scheduler_calls: int = 0
    decode_worker_dispatches: int = 0
    final_result_d2h_count: int = 0
    final_result_d2h_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def add(self, other: "GRRequestCounters") -> None:
        for name, value in other.to_dict().items():
            setattr(self, name, getattr(self, name) + value)


@dataclass
class RequestStateStats(BaseRequestStateStats):
    """Stats that need to be tracked across delta updates."""

    # Beam search overhead
    beam_search_overhead: float = 0.0
    beam_search_prefill_time: float = 0.0
    beam_search_decode_time: float = 0.0
    beam_search_sort_time: float = 0.0
    num_generation_tokens: int = 0
    num_beams: int = 0
    gr_queue_time: float = 0.0
    gr_prepare_time: float = 0.0
    gr_transfer_time: float = 0.0
    gr_forward_time: float = 0.0
    gr_postprocess_time: float = 0.0
    gr_return_time: float = 0.0
