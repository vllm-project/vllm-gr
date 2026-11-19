# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared helpers for NPU ModelRunner patch tests."""

from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData

from vllm_gr.v1.worker import npu_model_runner_capture as capture_mod
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner


def _eager_stage_output(
    *,
    kind: str,
    batch_size: int = 2,
    beam_width: int = 3,
    decode_step: int = 0,
) -> SimpleNamespace:
    request_ids = tuple(f"request-{index}" for index in range(batch_size))
    prompt_lengths = tuple(range(2, batch_size + 2))
    stage_index = 0 if kind == "prefill" else decode_step + 1
    metadata = SimpleNamespace(
        session_id="batch-0",
        dispatch_id=stage_index,
        stage_index=stage_index,
        kind=kind,
        decode_step=None if kind == "prefill" else decode_step,
        owner_generation=1,
        request_ids=request_ids,
        num_prompt_tokens_by_request=prompt_lengths,
        num_prompt_tokens=sum(prompt_lengths),
        beam_params=SimpleNamespace(beam_width=beam_width),
        produces_output=True,
        execution_capacity=SimpleNamespace(beam_width=beam_width),
    )
    counts = prompt_lengths if kind == "prefill" else (1,) * batch_size
    return SimpleNamespace(
        gr_stage_metadata={metadata.session_id: metadata},
        num_scheduled_tokens=dict(zip(request_ids, counts)),
        total_num_scheduled_tokens=sum(counts),
        beam_data={},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )


def _cached_prefill_output(cached: tuple[int, ...]) -> SimpleNamespace:
    scheduled = _eager_stage_output(kind="prefill", batch_size=len(cached))
    metadata = next(iter(scheduled.gr_stage_metadata.values()))
    lengths = tuple(257 + 128 * index for index in range(len(cached)))
    metadata.num_prompt_tokens_by_request = lengths
    metadata.num_prompt_tokens = sum(lengths)
    scheduled.scheduled_new_reqs = [
        NewRequestData(
            req_id=request_id,
            prompt_token_ids=[1] * length,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([index],),
            num_computed_tokens=computed,
            lora_request=None,
        )
        for index, (request_id, length, computed) in enumerate(
            zip(metadata.request_ids, lengths, cached, strict=True)
        )
    ]
    scheduled.num_scheduled_tokens = {
        request.req_id: length - request.num_computed_tokens
        for request, length in zip(scheduled.scheduled_new_reqs, lengths, strict=True)
    }
    scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
    return scheduled


def _shortcut_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    rows: list[Any] | None,
) -> None:
    """Stub the short-circuit gates with controllable results.

    ``rows`` non-empty simulates a homogeneous constrained step; the built
    sampler output mirrors ``build_constrained_sampler_output``'s contract
    ([rows, W] tensors, top-1 candidate in column 0).
    """
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    from vllm_gr.v1.beam import beam_decision
    from vllm_gr.v1.worker import constraint_hook

    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda r: enabled)

    def fake_build(runner: Any, logits: Any, collected_rows: list[Any]) -> Any:
        width = 4
        top_tokens = torch.arange(2 * width, dtype=torch.int32).reshape(2, width)
        top_logprobs = -torch.rand(2, width, dtype=torch.float32)
        return SamplerOutput(
            sampled_token_ids=top_tokens[:, :1].clone(),
            logprobs_tensors=LogprobsTensors(
                top_tokens, top_logprobs, torch.ones(2, dtype=torch.int32)
            ),
        )

    if rows is None:
        monkeypatch.setattr(constraint_hook, "collect_constrained_override_rows", lambda r: [])
        monkeypatch.setattr(constraint_hook, "build_constrained_sampler_output", fake_build)
    else:
        monkeypatch.setattr(constraint_hook, "collect_constrained_override_rows", lambda r: rows)
        monkeypatch.setattr(constraint_hook, "build_constrained_sampler_output", fake_build)


def _decision_stub(
    monkeypatch: pytest.MonkeyPatch,
    decision: dict[str, Any] | None,
) -> None:
    """Stub the beam-decision gate + runner with a canned decision dict."""
    from vllm_gr.v1.beam import beam_decision

    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda r: decision is not None)
    monkeypatch.setattr(
        beam_decision,
        "run_beam_decision",
        lambda runner, session, sampler_output, is_last_step, result_width, v2_layout=False: (
            decision
        ),
    )


def _decision_runner(is_last_step: bool) -> tuple[Any, Any]:
    state = SimpleNamespace(is_last_step=is_last_step, kv_reorder=None)
    session = SimpleNamespace(_state=state)
    runner = SimpleNamespace(
        _current_beam_info={
            "beam_session": session,
            "beam_data": {"s": {"is_beam_decode": True, "result_width": None}},
        },
        _beam_decision_result="unset",
    )
    return runner, state


def _make_determine_wrapper(native_result: Any) -> Any:
    """Build a patched NPU determine with a stubbed native _orig_determine."""

    def _orig_determine(self: Any, *args: Any, **kwargs: Any) -> Any:
        return native_result

    return capture_mod.make_patched_determine_batch_execution_and_padding_npu(_orig_determine)


def _make_beam_info(session: Any, req_ids: list[str], beam_keys: list[str]) -> dict[str, Any]:
    width = int(getattr(session, "request_width", getattr(session, "beam_width", 4)))
    beam_data = {
        req_id: (
            {
                "is_beam_decode": True,
                "decode_step": 0,
                "beam_decode_steps": 1,
                "beam_width": width,
                "prefix_len": 128,
                "cache_len": 128,
            }
            if req_id in beam_keys
            else {}
        )
        for req_id in req_ids
    }
    return {
        "beam_data": beam_data,
        "req_ids": req_ids,
        "beam_session": session,
    }


def _identity_layout_runner(batch_size: int = 2, width: int = 4) -> SimpleNamespace:
    request_ids = [f"beam-{index}" for index in range(batch_size)]
    session = SimpleNamespace(
        is_active=True,
        slot=0,
        batch_size=batch_size,
        request_width=width,
        attention_width=batch_size * width,
    )
    return SimpleNamespace(
        _current_beam_info=_make_beam_info(session, request_ids, request_ids),
        _beam_graph_keys={batch_size * width},
        _gr_npu_beam_graph_shapes={(batch_size, width)},
        _current_scheduler_output=SimpleNamespace(
            num_scheduled_tokens=dict.fromkeys(request_ids, width)
        ),
        query_start_loc=SimpleNamespace(
            np=np.arange(0, (batch_size + 1) * width, width, dtype=np.int32)
        ),
    )


def _stub_gr_config(monkeypatch: pytest.MonkeyPatch, gr_cfg: Any) -> None:
    import vllm_gr.v1.worker.gr_context as gr_context

    monkeypatch.setattr(gr_context, "_get_gr_config", lambda r: gr_cfg)


def _stub_beam_graph_install(
    monkeypatch: pytest.MonkeyPatch, ready_widths: tuple[int, ...] = ()
) -> list[Any]:
    import sys
    from types import ModuleType

    calls: list[Any] = []
    module = ModuleType("vllm_gr.v1.worker.npu_beam_decode_graph")

    def install(runner: Any) -> Any:
        calls.append(runner)
        model = SimpleNamespace(graph_ready=lambda width: width in ready_widths)
        runner._gr_npu_graph_model = runner.model = model
        return model

    module.install_npu_beam_graph_model = install  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


class _NPUPositionTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        # Only the Python device guard is mocked; storage remains on CPU.
        return cast(torch.device, SimpleNamespace(type="npu"))


class _CUDAPositionTensor(torch.Tensor):  # type: ignore[misc]
    @property
    def device(self) -> torch.device:
        return cast(torch.device, SimpleNamespace(type="cuda"))


def _position_plan(
    prompt_lengths: tuple[int, ...],
    step: int,
    beam_width: int = 3,
    dynamic_beam_widths: tuple[int, ...] | None = None,
) -> NPUEagerStagePlan:
    output = _eager_stage_output(
        kind="decode", batch_size=len(prompt_lengths), beam_width=beam_width, decode_step=step
    )
    metadata = next(iter(output.gr_stage_metadata.values()))
    metadata.num_prompt_tokens_by_request = prompt_lengths
    metadata.num_prompt_tokens = sum(prompt_lengths)
    metadata.beam_params.dynamic_beam_widths = dynamic_beam_widths
    return NPUEagerStagePlan.from_scheduler_output(output)


class _CountingPositionTuple(tuple[int, ...]):
    reads: list[int]

    def __new__(cls, values: tuple[int, ...]) -> Self:
        instance = super().__new__(cls, values)
        instance.reads = []
        return instance

    def __getitem__(self, index: Any) -> Any:
        assert isinstance(index, int)
        self.reads.append(index)
        return super().__getitem__(index)


def _terminal_result_runner(
    batch: int,
    *,
    returned: int = 2,
    stop_ids: tuple[int, ...] = (),
    ignore_stops: bool = False,
    penalty: float = 1.0,
    include_stop: bool = True,
    include_score: bool = True,
    include_parents: bool = True,
) -> NPUEagerBeamStageRunner:
    request_ids = tuple(f"request-{row}" for row in range(batch))
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.pending_plan = SimpleNamespace(execution=SimpleNamespace(request_ids=request_ids))
    stage.session_request = SimpleNamespace(
        batch_id="terminal-batch",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=request_id, prompt_token_ids=[1] * (row + 1)),
            )
            for row, request_id in enumerate(request_ids)
        ),
        beam_params=SimpleNamespace(
            stop_token_ids=stop_ids, ignore_stop_tokens=ignore_stops, length_penalty=penalty
        ),
        output_options=SimpleNamespace(
            num_return_sequences=returned,
            include_stop_token=include_stop,
            include_cumulative_logprob=include_score,
            include_parent_beam_indices=include_parents,
        ),
    )
    return stage


def _parent_history_request(batch_size: int, include_parents: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch_id="batch-0",
        items=tuple(
            SimpleNamespace(
                item_index=row,
                request=SimpleNamespace(request_id=f"request-{row}", prompt_token_ids=[1, 2]),
            )
            for row in range(batch_size)
        ),
        beam_params=SimpleNamespace(
            beam_width=3,
            max_tokens=3,
            temperature=1.0,
            stop_token_ids=(),
            ignore_stop_tokens=False,
            length_penalty=0.0,
        ),
        output_options=SimpleNamespace(
            num_return_sequences=3,
            include_stop_token=False,
            include_cumulative_logprob=True,
            include_parent_beam_indices=include_parents,
        ),
    )
