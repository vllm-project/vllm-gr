# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import gc
import json
import math
import os
import sys
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    create_stat,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    mask_beam_star_candidates,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.utils.catalog import complete_catalog_beam, fill_catalog_beams
from vllm_gr.utils.env import is_truthy_env
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.metrics.beam_search_metrics import record_beam_search_metrics
from vllm_gr.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()
# asyncio only holds a weak reference to a task once nothing else references
# it; an unreferenced fire-and-forget task can be garbage-collected mid-run,
# silently dropping it (see the asyncio.create_task docs' "Important" note).
# _beam_request_cleanup is dispatched this way from several places below, so
# keep a strong reference here for the task's lifetime and let its own
# done-callback discard it.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background_task(coro: Any) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


_BEAMPROC_ENABLED_ENV = "VLLM_GR_BEAM_PROC_ENABLED"
_BEAMPROC_FINAL_TIMEOUT_S_ENV = "VLLM_GR_BEAM_PROC_FINAL_TIMEOUT_S"
_BEAMPROC_CONTEXT_TRACE_HEADER = "x-vllm-gr-beam-context"


# Evaluate once at import; no platform/clock checks on the request hot path.
_BEAMPROC_SHARED_CLOCK = (
    sys.platform == "linux"
    and time.get_clock_info("perf_counter").implementation == "clock_gettime(CLOCK_MONOTONIC)"
)


def _beamproc_timing_metrics(final_plan, start, end, beam_width, decode_steps):
    timing = final_plan.get("beam_timing")
    if not _BEAMPROC_SHARED_CLOCK or not isinstance(timing, dict):
        return None
    boundary = timing.get("prefill_end")
    prefill_only = timing.get("prefill_only")
    processed_tokens = final_plan.get("num_processed_tokens", 0)
    if type(processed_tokens) is not int or processed_tokens < 0:
        processed_tokens = 0
    if (
        timing.get("clock") != "CLOCK_MONOTONIC"
        or type(boundary) not in (int, float)
        or not math.isfinite(boundary)
        or not start <= boundary <= end
        or type(prefill_only) is not bool
    ):
        return None
    return RequestStateStats(
        beam_search_prefill_time=0.0 if prefill_only else boundary - start,
        beam_search_decode_time=end - start if prefill_only else end - boundary,
        num_generation_tokens=beam_width * max(decode_steps, 0),
        num_processed_tokens=processed_tokens,
        num_beams=beam_width,
    )


def _record_beam_metrics(serving, metrics):
    if metrics is None:
        return
    try:
        model_name = serving.models.model_name
        if callable(model_name):
            model_name = model_name()
        model_name = str(model_name)
    except Exception:
        model_name = "unknown"
    record_beam_search_metrics(metrics, engine="0", model_name=model_name)


def _beamproc_fast_path_supported(
    *,
    batch_submission_supported: bool,
    beamproc_transport_supported: bool,
    total_decode_steps: int,
) -> bool:
    return batch_submission_supported and beamproc_transport_supported and total_decode_steps > 0


def _beamproc_final_timeout_s() -> float:
    raw = os.getenv(_BEAMPROC_FINAL_TIMEOUT_S_ENV)
    if raw is None:
        return 600.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default 600s",
            _BEAMPROC_FINAL_TIMEOUT_S_ENV,
            raw,
        )
        return 600.0


def _new_beamproc_root_request_id(_request_id: str) -> str:
    return random_uuid()


def _materialize_beamproc_final_logprobs(
    final_plan: Mapping[str, Any],
    beam_index: int,
    *,
    sid_begin_token_id: int | None,
    sid_end_token_id: int | None,
) -> FlatLogprobs:
    token_ids_array = final_plan.get("final_logprob_token_ids")
    logprobs_array = final_plan.get("final_logprob_values")
    step_offsets = final_plan.get("final_logprob_step_offsets")
    beam_offsets = final_plan.get("final_logprob_beam_offsets")
    if all(
        value is not None for value in (token_ids_array, logprobs_array, step_offsets, beam_offsets)
    ):
        beam_step_start = int(beam_offsets[beam_index])
        beam_step_end = int(beam_offsets[beam_index + 1])
        value_start = int(step_offsets[beam_step_start])
        value_end = int(step_offsets[beam_step_end])
        token_ids = token_ids_array[value_start:value_end].tolist()
        logprobs = logprobs_array[value_start:value_end].tolist()
        start_indices = (step_offsets[beam_step_start:beam_step_end] - value_start).tolist()
        end_indices = (step_offsets[beam_step_start + 1 : beam_step_end + 1] - value_start).tolist()

        if sid_begin_token_id is not None:
            token_ids.insert(0, sid_begin_token_id)
            logprobs.insert(0, 0.0)
            start_indices = [0, *(index + 1 for index in start_indices)]
            end_indices = [1, *(index + 1 for index in end_indices)]
        if sid_end_token_id is not None:
            start_indices.append(len(logprobs))
            token_ids.append(sid_end_token_id)
            logprobs.append(0.0)
            end_indices.append(len(logprobs))

        return FlatLogprobs(
            start_indices=start_indices,
            end_indices=end_indices,
            token_ids=token_ids,
            logprobs=logprobs,
            ranks=[None] * len(token_ids),
            decoded_tokens=[None] * len(token_ids),
        )

    final_logprob_steps = final_plan.get("final_logprob_steps") or []
    beam_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        beam_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])
    if beam_index < len(final_logprob_steps):
        for step in final_logprob_steps[beam_index]:
            step_token_ids = [int(token_id) for token_id in step.get("token_ids", ())]
            step_logprobs = [float(logprob) for logprob in step.get("logprobs", ())]
            if len(step_token_ids) != len(step_logprobs):
                raise RuntimeError("BeamProc final logprob tokens and values are not aligned")
            beam_logprobs.append_fast(
                step_token_ids,
                step_logprobs,
                [None] * len(step_token_ids),
                [None] * len(step_token_ids),
            )
    if sid_end_token_id is not None:
        beam_logprobs.append_fast([sid_end_token_id], [0.0], [None], [None])
    return beam_logprobs


def _beamproc_plan_buffer(engine_client) -> dict[str, list[dict[str, Any]]]:
    buffer = getattr(engine_client, "beamproc_apiserver_plans_by_root", None)
    if buffer is None:
        buffer = {}
        engine_client.beamproc_apiserver_plans_by_root = buffer
    return buffer


def _beamproc_output_client(engine_client):
    return getattr(engine_client, "engine_core", engine_client)


def _ensure_beamproc_apiserver_output(engine_client, root_request_id: str) -> None:
    output_client = _beamproc_output_client(engine_client)
    _install_beamproc_apiserver_output_handler(output_client)
    try:
        from vllm_gr.v1.engine.core_client import ensure_beamproc_output_task

        ensure_beamproc_output_task(output_client)
    except Exception:
        logger.exception(
            "Failed to start BeamProc APIServer final receiver root=%s client=%s output_client=%s",
            root_request_id,
            type(engine_client).__name__,
            type(output_client).__name__,
        )
        raise


def _handle_beamproc_apiserver_output(output_client, plan: Any) -> None:
    if not isinstance(plan, dict):
        logger.warning("Ignoring non-dict BeamProc APIServer final payload: %r", plan)
        return
    root_request_id = plan.get("root_request_id") or plan.get("request_id")
    if not root_request_id:
        logger.warning("Ignoring BeamProc APIServer final without root/request id")
        return
    root_request_id = str(root_request_id)
    final_token_ids = plan.get("final_token_ids") or []
    logger.debug(
        "BEAMPROC_AS_FINAL_HANDLE root=%s request_id=%s beam_width=%s "
        "final_beams=%s first_final_len=%s",
        root_request_id,
        plan.get("request_id"),
        plan.get("beam_width"),
        len(final_token_ids),
        (
            len(final_token_ids[0])
            if final_token_ids and hasattr(final_token_ids[0], "__len__")
            else 0
        ),
    )
    # The normal path has one live waiter and no buffered results. Preserve
    # FIFO handling below for early replies or multiple registered waiters.
    waiters = _beamproc_waiters(output_client)
    entries = waiters.get(root_request_id)
    if (
        entries
        and len(entries) == 1
        and not entries[0][1].done()
        and int(plan.get("beam_width") or 0) == entries[0][0]
        and not _beamproc_plan_buffer(output_client).get(root_request_id)
    ):
        waiters.pop(root_request_id)
        entries[0][1].set_result(plan)
        return
    _cache_beamproc_apiserver_plan(output_client, plan)
    _wake_beamproc_apiserver_waiters(output_client, root_request_id)


def _install_beamproc_apiserver_output_handler(output_client) -> None:
    if getattr(output_client, "beamproc_apiserver_output_handler", None) is not None:
        return

    def handle(plan: Any) -> None:
        _handle_beamproc_apiserver_output(output_client, plan)

    output_client.beamproc_apiserver_output_handler = handle


def _trace_headers_with_beam_context(
    trace_headers: Mapping[str, str] | None,
    beam_context: dict[str, Any] | None,
) -> Mapping[str, str] | None:
    if not beam_context:
        return trace_headers
    headers = dict(trace_headers or {})
    headers[_BEAMPROC_CONTEXT_TRACE_HEADER] = json.dumps(
        beam_context,
        separators=(",", ":"),
        sort_keys=True,
    )
    return headers


def _cache_beamproc_apiserver_plan(engine_client, plan: Any) -> None:
    if not isinstance(plan, dict):
        return
    root_request_id = plan.get("root_request_id") or plan.get("request_id")
    if not root_request_id:
        return
    buffer = _beamproc_plan_buffer(engine_client)
    entries = buffer.setdefault(str(root_request_id), [])
    entries.append(plan)
    logger.debug(
        "BEAMPROC_AS_FINAL_CACHE root=%s entries=%s beam_width=%s final_beams=%s",
        root_request_id,
        len(entries),
        plan.get("beam_width"),
        len(plan.get("final_token_ids") or []),
    )
    if len(entries) > 8:
        del entries[:-8]
    if len(buffer) > 512:
        for stale_root in list(buffer)[:128]:
            buffer.pop(stale_root, None)


def _pop_beamproc_apiserver_plan(
    engine_client,
    root_request_id: str,
    beam_width: int,
) -> dict[str, Any] | None:
    buffer = _beamproc_plan_buffer(engine_client)
    entries = buffer.get(root_request_id)
    if not entries:
        return None
    for index, plan in enumerate(entries):
        if int(plan.get("beam_width") or 0) == beam_width:
            selected = entries.pop(index)
            if not entries:
                buffer.pop(root_request_id, None)
            return selected
    return None


def _beamproc_waiters(engine_client) -> dict[str, list[tuple[int, asyncio.Future]]]:
    waiters = getattr(engine_client, "beamproc_apiserver_waiters_by_root", None)
    if waiters is None:
        waiters = {}
        engine_client.beamproc_apiserver_waiters_by_root = waiters
    return waiters


def _wake_beamproc_apiserver_waiters(engine_client, root_request_id: str) -> None:
    waiters = _beamproc_waiters(engine_client)
    entries = waiters.get(root_request_id)
    if not entries:
        logger.warning("BEAMPROC_AS_FINAL_WAKE_NO_WAITER root=%s", root_request_id)
        return
    remaining: list[tuple[int, asyncio.Future]] = []
    for beam_width, future in entries:
        if future.done():
            continue
        plan = _pop_beamproc_apiserver_plan(engine_client, root_request_id, beam_width)
        if plan is None:
            remaining.append((beam_width, future))
        else:
            logger.debug(
                "BEAMPROC_AS_FINAL_WAKE_MATCH root=%s beam_width=%s final_beams=%s",
                root_request_id,
                beam_width,
                len(plan.get("final_token_ids") or []),
            )
            if not future.done():
                future.set_result(plan)
    if remaining:
        waiters[root_request_id] = remaining
    else:
        waiters.pop(root_request_id, None)


def _discard_beamproc_apiserver_state(engine_client, root_request_id: str) -> None:
    _beamproc_plan_buffer(engine_client).pop(root_request_id, None)
    waiters = getattr(engine_client, "beamproc_apiserver_waiters_by_root", None)
    if waiters is None:
        return
    for _beam_width, future in waiters.pop(root_request_id, []):
        if not future.done():
            future.cancel()


def _cleanup_beamproc_prefill_output_state(
    engine_client,
    internal_request_id: str | None,
) -> None:
    if not internal_request_id:
        return

    output_processor = getattr(engine_client, "output_processor", None)
    if output_processor is not None:
        req_state = getattr(output_processor, "request_states", {}).pop(
            internal_request_id,
            None,
        )
        if req_state is not None:
            external_req_ids = getattr(output_processor, "external_req_ids", None)
            external_req_id = getattr(req_state, "external_req_id", None)
            if external_req_ids is not None and external_req_id is not None:
                internal_ids = external_req_ids.get(external_req_id)
                if internal_ids is not None:
                    with contextlib.suppress(ValueError):
                        internal_ids.remove(internal_request_id)
                    if not internal_ids:
                        external_req_ids.pop(external_req_id, None)

            parent_req = getattr(req_state, "parent_req", None)
            parent_requests = getattr(output_processor, "parent_requests", None)
            if parent_req is not None and parent_requests is not None:
                parent_request_id = getattr(parent_req, "request_id", None)
                if parent_request_id is not None:
                    parent_requests.pop(parent_request_id, None)

            lora_states = getattr(output_processor, "lora_states", None)
            lora_name = getattr(req_state, "lora_name", None)
            if lora_states is not None and lora_name is not None:
                with contextlib.suppress(Exception):
                    lora_states.request_finished(internal_request_id, lora_name)

            request_states = getattr(output_processor, "request_states", None)
            requests_drained = getattr(output_processor, "_requests_drained", None)
            if request_states is not None and not request_states and requests_drained is not None:
                with contextlib.suppress(Exception):
                    requests_drained.set()

    output_client = _beamproc_output_client(engine_client)
    reqs_in_flight = getattr(output_client, "reqs_in_flight", None)
    if reqs_in_flight is not None:
        reqs_in_flight.pop(internal_request_id, None)


class _BeamProcRequestLifecycle:
    """Own EngineCore abort and local cleanup for one BeamProc request."""

    def __init__(self, engine_client: Any, root_request_id: str) -> None:
        self.engine_client = engine_client
        self.root_request_id = root_request_id
        self.prefill_internal_request_id: str | None = None

    def record_prefill(self, internal_request_id: str) -> None:
        self.prefill_internal_request_id = internal_request_id

    async def abort_prefill(self) -> None:
        if self.prefill_internal_request_id is None:
            return
        await asyncio.shield(
            self.engine_client.abort(self.prefill_internal_request_id),
        )

    def cleanup(self) -> None:
        _cleanup_beamproc_prefill_output_state(
            self.engine_client,
            self.prefill_internal_request_id,
        )
        _discard_beamproc_apiserver_state(
            _beamproc_output_client(self.engine_client),
            self.root_request_id,
        )


async def _await_beamproc_apiserver_plan(
    engine_client,
    root_request_id: str,
    beam_width: int,
    timeout_s: float,
) -> dict[str, Any] | None:
    # The request path starts the receiver before sending prefill.
    output_client = _beamproc_output_client(engine_client)

    plan = _pop_beamproc_apiserver_plan(output_client, root_request_id, beam_width)
    if plan is not None or timeout_s <= 0:
        return plan

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    waiters = _beamproc_waiters(output_client)
    waiters.setdefault(root_request_id, []).append((beam_width, future))
    from vllm_gr.entrypoints.openai.beam_proc_manager import beam_proc_exit_code

    deadline = loop.time() + timeout_s
    health_check = None

    def check_sidecar():
        nonlocal health_check
        if future.done():
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            future.set_result(None)
            return
        try:
            exit_code = beam_proc_exit_code()
        except Exception as exc:  # noqa: BLE001 - propagate callback failures to the request
            future.set_exception(exc)
            return
        if exit_code is not None:
            future.set_exception(
                RuntimeError(
                    "BeamProc sidecar exited while awaiting final output "
                    f"for root={root_request_id} with status {exit_code}"
                )
            )
            return
        health_check = loop.call_later(min(remaining, 0.1), check_sidecar)

    health_check = loop.call_later(min(timeout_s, 0.1), check_sidecar)
    try:
        return await future
    finally:
        health_check.cancel()
        entries = waiters.get(root_request_id)
        if entries is not None:
            remaining = [
                (entry_beam_width, entry_future)
                for entry_beam_width, entry_future in entries
                if entry_future is not future
            ]
            if remaining:
                waiters[root_request_id] = remaining
            else:
                waiters.pop(root_request_id, None)


def _beamproc_plan_flat_indices(
    plan: dict[str, Any],
    *,
    candidate_count: int,
    beam_width: int,
) -> list[int] | None:
    flat_indices = plan.get("flat_indices")
    if not isinstance(flat_indices, list) or len(flat_indices) < beam_width:
        return None
    selected: list[int] = []
    seen: set[int] = set()
    for value in flat_indices[:beam_width]:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= candidate_count or index in seen:
            return None
        selected.append(index)
        seen.add(index)
    return selected


# return rank for next beam search request in a round-robin manner across the data parallel ranks
async def _next_data_parallel_rank(data_parallel_size: int) -> int:
    global _dp_rank_counter

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank


async def _collect_beam_result(q):
    """Drain output queue until finished."""
    while True:
        out = await q.get()
        if out.finished:
            return out


async def _gather_beam_results(queues):
    """Gather results from multiple output queues."""
    tasks = [asyncio.create_task(_collect_beam_result(q)) for q in queues]
    return list(await asyncio.gather(*tasks))


async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    use_beam_req,
    trace_headers,
    beam_context=None,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0)."""
    if not (
        hasattr(engine_client, "prepare_request") and hasattr(engine_client, "_add_requests_batch")
    ):
        generators = [
            engine_client.generate(
                individual_prompt,
                beam_search_params,
                f"{request_id_batch}-beam-{i}",
                lora_request=lora_req,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch))
        ]

        async def collect_last(generator):
            final_output = None
            async for output in generator:
                final_output = output
            if final_output is None:
                raise RuntimeError("Beam search engine request produced no output")
            return final_output

        return list(await asyncio.gather(*(collect_last(gen) for gen in generators))), []

    queues = []
    engine_core_requests = []
    for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
        request_id_item = f"{request_id_batch}-beam-{i}"
        q, ec_req = await engine_client.prepare_request(
            request_id_item,
            individual_prompt,
            beam_search_params,
            lora_request=lora_req,
            trace_headers=_trace_headers_with_beam_context(trace_headers, beam_context),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        queues.append(q)
        engine_core_requests.append(ec_req)

    await engine_client._add_requests_batch(engine_core_requests, use_batch_message=use_beam_req)

    internal_ids = [r.request_id for r in engine_core_requests] if use_beam_req else []
    output = await _gather_beam_results(queues)
    return output, internal_ids


async def _send_beamproc_prefill(
    engine_client,
    prompt,
    request_id_batch,
    beam_search_params,
    lora_request,
    trace_headers,
    beam_context,
    priority=0,
    data_parallel_rank: int | None = None,
    lifecycle: _BeamProcRequestLifecycle | None = None,
):
    request_id_item = f"{request_id_batch}-beam-0"
    _, ec_req = await engine_client.prepare_request(
        request_id_item,
        prompt,
        beam_search_params,
        lora_request=lora_request,
        trace_headers=_trace_headers_with_beam_context(trace_headers, beam_context),
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )
    if lifecycle is not None:
        lifecycle.record_prefill(ec_req.request_id)
    await engine_client._add_requests_batch([ec_req], use_batch_message=True)
    return ec_req.request_id


async def _beam_request_step(
    engine_client,
    session_id: str,
    fork_info,
    prev_beam_internal_ids,
    all_beams,
    request_id_batch,
    beam_search_params,
    eos_token_id,
    lora_request,
    trace_headers,
    prefix_len: int,
    priority=0,
    data_parallel_rank: int | None = None,
    max_decode_steps: int = 0,
    parent_cum_logprobs: list[float] | None = None,
    target_beam_width: int = 0,
    is_last_step: bool = False,
):
    """BEAM_REQUEST path: single logical message update for all active branches."""
    parent_beam_ids = []
    child_beam_ids = []
    beam_tokens = []
    beam_width = len(fork_info)

    # Map the current generation step's routing array
    for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
        parent_id = prev_beam_internal_ids[parent_beam_idx]
        child_id = f"{request_id_batch}-beam-{new_idx}"
        parent_beam_ids.append(parent_id)
        child_beam_ids.append(child_id)

        full_tokens = all_beams[new_idx].tokens
        beam_tokens.append(full_tokens[prefix_len:])

    # SamplingParams.n retains its upstream meaning: this is one logical
    # request. The active beam count is carried explicitly by beam_width.
    decode_sampling_params = clone_beam_decode_sampling_params(beam_search_params)

    # Listen exclusively to the parent session_id stream queue
    q = engine_client.register_beam_output(
        session_id,
        all_beams[0].tokens,
        decode_sampling_params,
        eos_token_id=eos_token_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )

    # Detect branches that were pruned during top-K filtering
    used_parents = set(parent_beam_ids)
    pruned_ids = [pid for pid in prev_beam_internal_ids if pid not in used_parents]

    # Dispatch the consolidated persistent update to the client
    await engine_client.beam_request_step_update(
        BeamRequestStepUpdate(
            session_id=session_id,
            parent_beam_ids=parent_beam_ids,
            child_beam_ids=child_beam_ids,
            beam_tokens=beam_tokens,
            pruned_ids=pruned_ids,
            prefix_len=prefix_len,
            beam_width=beam_width,
            sampling_params=decode_sampling_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            parent_beam_indices=[parent_beam_idx for parent_beam_idx, _ in fork_info],
            max_decode_steps=max_decode_steps,
            parent_cum_logprobs=parent_cum_logprobs,
            target_beam_width=target_beam_width,
            is_last_step=is_last_step,
        )
    )

    # Await the singular unified engine execution response block
    single_output = await _collect_beam_result(q)

    # Return the singular raw output instance wrapped as a single-element list
    # to maintain backward-compatible pipeline execution with the loop below
    return [single_output], child_beam_ids


async def _beam_request_cleanup(engine_client, session_id, final_ids, rank):
    """Signals the core engine to fully purge all remaining session cache allocations."""
    if not final_ids:
        return
    try:
        await engine_client.beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=final_ids,
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
                data_parallel_rank=rank,
                is_last_step=True,
            )
        )
    except Exception as e:
        logger.error("Failed to clear beam_request cache for session %s: %s", session_id, e)


async def beam_search(
    self,
    prompt: EngineInput,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    logger.debug(
        "vllm_gr serving_engine.beam_search entered request_id=%s "
        "engine_client=%s beam_proc_enabled=%s",
        request_id,
        type(self.engine_client).__name__,
        os.environ.get(_BEAMPROC_ENABLED_ENV),
    )
    generation_time: float = 0.0
    num_generation_tokens: int = 0
    num_processed_tokens: int = 0
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    begin_token = params.begin_token
    end_token = params.end_token
    catalog_strategy = getattr(params, "catalog_strategy", "fill")
    beam_star = bool(getattr(params, "beam_star", False))
    beam_star_coef = float(getattr(params, "beam_star_coef", 2.0))
    return_logprobs = bool(getattr(params, "return_logprobs", True))

    # All beams from the same logical beam search will share the same priority value
    # (the prefill timestamp), so the scheduler will treat them as a group with equal priority

    MICROSECONDS = 1000000
    priority = int(time.perf_counter() * MICROSECONDS)  # us granularity
    if beam_star:
        stats = create_stat(max_tokens)

    rank = None
    if (
        hasattr(self.engine_client, "vllm_config")
        and self.engine_client.vllm_config.parallel_config is not None
    ):
        data_parallel_size = self.engine_client.vllm_config.parallel_config.data_parallel_size
        if data_parallel_size is not None and data_parallel_size > 1:
            # In DP mode, we assign ranks to beam search requests in a round-robin manner.
            rank = await _next_data_parallel_rank(data_parallel_size)
            logger.debug(
                f"rank for beam search: {rank} out of data_parallel_size: {data_parallel_size}"
            )

    include_stop_str_in_output = params.include_stop_str_in_output
    if beam_width == 0:
        raise VLLMValidationError(
            "Beam width must be greater than 0", parameter="beam_width", value=0
        )
    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError(
            "You cannot use beam search when `skip_tokenizer_init=True`",
            parameter="skip_tokenizer_init",
            value=True,
        )

    eos_token_id: int = tokenizer.eos_token_id
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        if sid_begin_token_id is None or sid_begin_token_id == -1:
            raise VLLMValidationError(
                "begin_token must be a valid token in the tokenizer vocabulary",
                parameter="begin_token",
                value=begin_token,
            )
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
        if sid_end_token_id is None or sid_end_token_id == -1:
            raise VLLMValidationError(
                "end_token must be a valid token in the tokenizer vocabulary",
                parameter="end_token",
                value=end_token,
            )

    if not isinstance(prompt, str) and prompt.get("type") == "enc_dec":
        raise NotImplementedError

    prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    prompt_token_ids = [] if isinstance(prompt, str) else prompt.get("prompt_token_ids", [])
    tokenized_length = len(prompt_token_ids)
    logprobs_num = beam_width

    beam_search_params = SamplingParams(
        logprobs=logprobs_num,
        max_tokens=1,
        temperature=temperature,
        detokenize=False,
        flat_logprobs=True,
    )
    initial_tokens = list(prompt_token_ids)
    # NOTE: FlatLogprobs intentionally passed where BeamSearchSequence
    # expects list[dict[int, Logprob]].  Intermediate logprobs are never
    # read as dicts; the final beam.logprobs is replaced during
    # reconstruction at the end.
    # This reference is shared read-only across all beams — do not mutate
    # after the initial append below.
    initial_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        initial_tokens.append(sid_begin_token_id)
        initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

    orig_prompt = prompt
    if isinstance(prompt, str) or "type" not in prompt:
        orig_prompt = tokens_input(initial_tokens, prompt=prompt_text)

    all_beams = [
        BeamSearchSequence(
            orig_prompt=orig_prompt,
            tokens=initial_tokens,
            cum_logprob=0,
            logprobs=initial_logprobs,
            lora_request=lora_request,
        )
    ]
    # Deferred logprobs: instead of copying 6 FlatLogprobs lists per beam
    # per step (O(beams * steps^2)), store parent pointers and reconstruct
    # only for the final best beams at the end.
    # These dynamic attrs live on BeamSearchSequence (a plain @dataclass
    # without __slots__).  If upstream adds __slots__, these will break —
    # upstream the fields if that happens.
    all_beams[0]._lp_parent = None
    all_beams[0]._lp_step_data = None
    completed = []

    beam_search_start: int = time.perf_counter()
    prefill_start = beam_search_start  # Track initial start for prefill measurement
    pre_calc = 0 if sid_begin_token_id is None else 1
    token_idx_offset = 2 if sid_begin_token_id is not None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        raise VLLMValidationError(
            f"max tokens should be not lower than {pre_calc} to accommodate reserved begin/end tokens.",
            parameter="max_tokens",
            value=max_tokens,
        )

    # Check once if the engine supports batch submission / beam fork.
    batch_submission_supported = hasattr(self.engine_client, "prepare_request") and hasattr(
        self.engine_client, "_add_requests_batch"
    )
    beamproc_transport_supported = bool(
        getattr(self.engine_client, "supports_beamproc_transport", False)
    )
    use_beam_request = batch_submission_supported and hasattr(
        self.engine_client, "beam_request_step_update"
    )

    cleaned_up = False
    prev_beam_internal_ids: list[str] = []
    beamproc_ec_root_request_id: str | None = None
    fork_info: list[tuple[int, int]] | None = None
    prefill_id: str | None = None

    total_decode_steps = max_tokens - pre_calc
    beamproc_prefill_context = {
        "global_max_tokens": max_tokens,
        "total_decode_steps": total_decode_steps,
        "step_index": 0,
        "is_final_step": total_decode_steps == 1,
        "ignore_eos": ignore_eos,
        "include_stop_token": params.include_stop_str_in_output,
        "beam_width": beam_width,
        "catalog_strategy": catalog_strategy,
        "return_logprobs": return_logprobs,
        "end_token_id": sid_end_token_id,
        "pre_calc_tokens": pre_calc,
        "prompt_token_count": tokenized_length,
    }
    ec_apply_plan_requested = is_truthy_env(_BEAMPROC_ENABLED_ENV)
    ec_apply_plan = ec_apply_plan_requested and _beamproc_fast_path_supported(
        batch_submission_supported=batch_submission_supported,
        beamproc_transport_supported=beamproc_transport_supported,
        total_decode_steps=total_decode_steps,
    )
    if ec_apply_plan_requested and total_decode_steps > 0 and not ec_apply_plan:
        raise RuntimeError(
            "BeamProc is enabled but the AS fast path cannot run: "
            f"engine_client={type(self.engine_client).__name__} "
            f"has_prepare_request={hasattr(self.engine_client, 'prepare_request')} "
            "has_add_requests_batch="
            f"{hasattr(self.engine_client, '_add_requests_batch')} "
            f"supports_beamproc_transport={beamproc_transport_supported}"
        )
    if ec_apply_plan:
        beamproc_ec_root_request_id = _new_beamproc_root_request_id(request_id)
        beamproc_prefill_context["root_request_id"] = beamproc_ec_root_request_id
        lifecycle = _BeamProcRequestLifecycle(
            self.engine_client,
            beamproc_ec_root_request_id,
        )
        logger.debug(
            "BeamProc EC-apply fast path enabled root=%s beam_width=%s",
            beamproc_ec_root_request_id,
            beam_width,
        )
        try:
            # Catalog, sidecar and final receiver are initialized in init_gr.
            prefill_internal_request_id = await _send_beamproc_prefill(
                self.engine_client,
                all_beams[0].get_prompt(),
                beamproc_ec_root_request_id,
                beam_search_params,
                lora_request,
                trace_headers,
                beamproc_prefill_context,
                priority=priority,
                data_parallel_rank=rank,
                lifecycle=lifecycle,
            )
            lifecycle.record_prefill(prefill_internal_request_id)
            final_plan = await _await_beamproc_apiserver_plan(
                self.engine_client,
                beamproc_ec_root_request_id,
                beam_width,
                _beamproc_final_timeout_s(),
            )
            if final_plan is None:
                raise RuntimeError(
                    "BeamProc EC-apply mode requires a final Rust reply, "
                    f"but none arrived for root={beamproc_ec_root_request_id}"
                )
            final_plan_token_ids = final_plan.get("final_token_ids") or []
            logger.debug(
                "BEAMPROC_AS_FINAL_AWAIT_DONE root=%s beam_width=%s "
                "final_beams=%s first_final_len=%s",
                beamproc_ec_root_request_id,
                final_plan.get("beam_width"),
                len(final_plan_token_ids),
                (
                    len(final_plan_token_ids[0])
                    if final_plan_token_ids and hasattr(final_plan_token_ids[0], "__len__")
                    else 0
                ),
            )
        except (asyncio.CancelledError, GeneratorExit):
            logger.info(
                "BeamProc EC-apply cancelled for root=%s; aborting prefill",
                beamproc_ec_root_request_id,
            )
            with contextlib.suppress(Exception):
                await lifecycle.abort_prefill()
            raise
        except Exception:
            logger.exception(
                "BeamProc EC-apply failed for root=%s while BeamProc is enabled; "
                "refusing Python fallback",
                beamproc_ec_root_request_id,
            )
            with contextlib.suppress(Exception):
                await lifecycle.abort_prefill()
            raise
        finally:
            lifecycle.cleanup()

        final_token_ids = final_plan.get("final_token_ids") or []
        final_scores = final_plan.get("final_scores") or final_plan.get("scores") or []
        final_finish_reasons = final_plan.get("final_finish_reasons") or []
        final_stop_token_ids = final_plan.get("final_stop_token_ids") or []
        final_token_ids_include_prompt = final_plan.get("final_token_ids_include_prompt", True)
        outputs = []
        output_decode_token_ids: list[list[int]] = []
        for index, tokens_value in enumerate(final_token_ids[:beam_width]):
            # Numeric wire versions have already materialized Python ints.
            # Copy for the end-token append without converting every token twice.
            tokens = (
                list(tokens_value)
                if final_plan.get("version") in (2, 3)
                else [int(token_id) for token_id in tokens_value]
            )
            finish_reason = (
                str(final_finish_reasons[index]) if index < len(final_finish_reasons) else "length"
            )
            if finish_reason not in ("stop", "length"):
                finish_reason = "length"
            if finish_reason != "stop" and sid_end_token_id is not None:
                tokens.append(sid_end_token_id)
            text_tokens = tokens[tokenized_length:] if final_token_ids_include_prompt else tokens
            decode_tokens = text_tokens
            if (
                finish_reason == "stop"
                and not ignore_eos
                and not include_stop_str_in_output
                and text_tokens
                and text_tokens[-1] == eos_token_id
            ):
                decode_tokens = text_tokens[:-1]
            score = float(final_scores[index]) if index < len(final_scores) else None
            stop_reason = (
                int(final_stop_token_ids[index])
                if finish_reason == "stop"
                and index < len(final_stop_token_ids)
                and final_stop_token_ids[index] is not None
                else None
            )
            beam_logprobs = None
            if return_logprobs:
                beam_logprobs = _materialize_beamproc_final_logprobs(
                    final_plan,
                    index,
                    sid_begin_token_id=sid_begin_token_id,
                    sid_end_token_id=(sid_end_token_id if finish_reason != "stop" else None),
                )
            outputs.append(
                CompletionOutput(
                    text="",
                    cumulative_logprob=score,
                    token_ids=text_tokens,
                    index=index,
                    logprobs=beam_logprobs,
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
            )
            output_decode_token_ids.append(decode_tokens)
        if output_decode_token_ids:
            decoded_outputs = tokenizer.batch_decode(output_decode_token_ids)
            for output, decoded_text in zip(outputs, decoded_outputs, strict=True):
                output.text = decoded_text
        metrics = _beamproc_timing_metrics(
            final_plan, prefill_start, time.perf_counter(), beam_width, total_decode_steps
        )
        _record_beam_metrics(self, metrics)
        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=outputs,
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            metrics=metrics,
        )
        return

    catalog_task: asyncio.Task[list[set[int]]] | None = None
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for token in range(total_decode_steps):
            if not all_beams:
                # Catalog filtering eliminated every active beam (no valid continuation).
                # This is not an engine error — fall through to the normal post-loop path
                # so any EOS beams already in `completed` are returned.
                if not cleaned_up and use_beam_request and prev_beam_internal_ids:
                    _spawn_background_task(
                        _beam_request_cleanup(
                            self.engine_client, prefill_id, prev_beam_internal_ids, rank
                        )
                    )
                    cleaned_up = True
                    prev_beam_internal_ids = []
                break
            if token == 1:
                beam_search_start = time.perf_counter()

            if not use_beam_request or fork_info is None:
                prompts_batch, lora_req_batch = zip(
                    *[
                        (
                            beam.get_prompt(),
                            beam.lora_request,
                        )
                        for beam in all_beams
                    ]
                )
            else:
                prompts_batch, lora_req_batch = (), ()
            num_processed_tokens += len(all_beams) * logprobs_num
            if token > 0:
                num_generation_tokens += len(all_beams)
            request_id_batch = f"{request_id}-{random_uuid()}"
            if ec_apply_plan and beamproc_ec_root_request_id is None:
                beamproc_ec_root_request_id = request_id_batch

            # Launch catalog filtering in parallel with the engine step.
            catalog_task = None
            if self.models.catalog is not None:

                def get_valid_tokens_set(beam) -> set[int]:
                    generated_tokens = beam.tokens[len(prompt_token_ids) :]
                    return self.models.catalog.valid(generated_tokens)

                async def _run_catalog(coros):
                    return list(await asyncio.gather(*coros))

                catalog_task = asyncio.create_task(
                    _run_catalog(
                        [asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams]
                    )
                )

            gen_start = time.perf_counter()
            valid_tokens_sets = None

            is_final_step = token == max_tokens - pre_calc - 1
            parent_cum_lps = None
            if use_beam_request and fork_info is not None:
                parent_cum_lps = (
                    [float(b.cum_logprob) for b in all_beams]
                    if (catalog_task is None and self.models.catalog is None)
                    else None
                )
                output, prev_beam_internal_ids = await _beam_request_step(
                    self.engine_client,
                    session_id=prefill_id,
                    fork_info=fork_info,
                    prev_beam_internal_ids=prev_beam_internal_ids,
                    all_beams=all_beams,
                    request_id_batch=request_id_batch,
                    beam_search_params=beam_search_params,
                    eos_token_id=eos_token_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prefix_len=len(initial_tokens),
                    priority=priority,
                    data_parallel_rank=rank,
                    max_decode_steps=max_tokens - pre_calc,
                    parent_cum_logprobs=parent_cum_lps,
                    target_beam_width=beam_width,
                    is_last_step=is_final_step,
                )
                if is_final_step:
                    cleaned_up = True
            else:
                output, new_ids = await _add_batch_step(
                    self.engine_client,
                    prompts_batch,
                    lora_req_batch,
                    request_id_batch,
                    beam_search_params,
                    use_beam_request,
                    trace_headers,
                    beam_context=beamproc_prefill_context if token == 0 else None,
                    priority=priority,
                    data_parallel_rank=rank,
                )
                if new_ids:
                    prev_beam_internal_ids = new_ids
                    if prefill_id is None:
                        prefill_id = new_ids[0]
            if catalog_task is not None and valid_tokens_sets is None:
                valid_tokens_sets = await catalog_task
            if token > 0:
                generation_time += time.perf_counter() - gen_start
            # Store all new tokens generated by beam
            all_beams_token_id = []
            # Store the cumulative probability of all tokens
            # generated by beam search
            all_beams_logprob = []
            # Explicit beam index for each entry in all_beams_token_id/logprob.
            # Replaces idx // logprobs_num so that null-strategy beams (empty lists)
            # don't shift the mapping for subsequent beams.
            all_beams_beam_idx: list[int] = []
            # Per-beam flat data cache: (token_ids, logprobs, ranks, decoded_tokens)
            # Used to build logprobs_entry dicts without calling FlatLogprobs.__getitem__
            beam_flat_cache: list[tuple | None] = []

            # -------------------------------------------------------------------------
            # STRIDE-AWARE METADATA EXTRACTION FOR GENERATED BEAMS
            # -------------------------------------------------------------------------
            if use_beam_request and fork_info is not None:
                # Under the optimized mode, 'output' contains exactly 1 combined RequestOutput instance
                beam_result = output[0]
                if beam_result.outputs[0].finish_reason == "error":
                    if not cleaned_up:
                        _spawn_background_task(
                            _beam_request_cleanup(
                                self.engine_client, prefill_id, prev_beam_internal_ids, rank
                            )
                        )
                        cleaned_up = True
                    prev_beam_internal_ids = []
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                        finished=True,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                    )
                    return

                flat = beam_result.outputs[0].logprobs
                if flat is not None:
                    # Extract tracking attributes cleanly
                    raw_token_ids = flat.token_ids
                    raw_logprobs = flat.logprobs
                    raw_ranks = getattr(flat, "ranks", None)
                    raw_decoded = getattr(flat, "decoded_tokens", None)

                    stride_k = logprobs_num + 1
                    num_beams = len(fork_info)
                    total_needed = num_beams * stride_k

                    is_pruned = (
                        valid_tokens_sets is None
                        and parent_cum_lps is not None
                        and raw_ranks is not None
                        and len(raw_token_ids) == len(raw_ranks)
                        and len(raw_token_ids) >= min(beam_width, len(all_beams))
                    )

                    if is_pruned:
                        assert raw_ranks is not None
                        k = min(beam_width, len(raw_token_ids))
                        raw_ranks_arr = np.asarray(raw_ranks[:k], dtype=np.intp)
                        # ``ranks`` doubles as the per-token logprob display rank on
                        # the non-pruned path (vllm_gr/logprobs.py) -- a value that
                        # is always >= 0 (vllm's batched_count_greater_than never
                        # returns negative). _prune_beam_candidates_gpu encodes its
                        # parent-beam indices as -(idx + 1), which is always < 0, so
                        # the sign is an unambiguous signal for which encoding this
                        # row actually carries -- unlike a length/bounds check, which
                        # a co-scheduled wider request in the same batch can satisfy
                        # by coincidence even when this request was NOT pruned
                        # (silently misreading display ranks as parent indices).
                        if raw_ranks_arr.size == 0 or not bool((raw_ranks_arr < 0).all()):
                            is_pruned = False
                        else:
                            cand_parents = -(raw_ranks_arr + 1)
                            # Defensive only: a correctly encoded parent index is
                            # always < len(all_beams) by construction. A value
                            # outside that range would mean a protocol/version
                            # mismatch between worker and API server -- fail safe
                            # (fall through to the stride-based path) rather than
                            # index all_beams out of bounds below.
                            if not bool((cand_parents < len(all_beams)).all()):
                                is_pruned = False

                    if is_pruned:
                        cand_tokens = np.asarray(raw_token_ids[:k], dtype=np.int64)
                        cand_lps = np.asarray(raw_logprobs[:k], dtype=np.float64)
                        if is_vocab_slice_enabled():
                            offset, slice_size, _ = get_vocab_slice_config()
                            if offset is not None and slice_size is not None and slice_size > 0:
                                slice_mask = (cand_tokens >= 0) & (cand_tokens < slice_size)
                                cand_tokens = cand_tokens.copy()
                                cand_tokens[slice_mask] += offset

                        cum_lps = np.fromiter(
                            (all_beams[p].cum_logprob for p in cand_parents),
                            dtype=np.float64,
                            count=len(cand_parents),
                        )
                        cand_scores = cum_lps + cand_lps

                        all_beams_token_id = cand_tokens
                        all_beams_logprob = cand_scores
                        all_beams_beam_idx_arr = cand_parents

                        beam_flat_cache = [None] * num_beams
                        for c in range(len(cand_parents)):
                            p = cand_parents[c]
                            if beam_flat_cache[p] is None:
                                beam_flat_cache[p] = (
                                    [int(cand_tokens[c])],
                                    [float(cand_lps[c])],
                                    # No real per-token rank survives worker-side
                                    # top-k pruning (raw_ranks was already spent
                                    # carrying the parent index -- see is_pruned
                                    # above); report that honestly as None rather
                                    # than fabricate a look-alike ordinal. A list
                                    # (not a bare None) so this stays correct
                                    # against the unpatched upstream append_fast
                                    # too, which zips ranks directly instead of
                                    # special-casing a None argument.
                                    [None],
                                    [None],
                                )
                            else:
                                prev = beam_flat_cache[p]
                                beam_flat_cache[p] = (
                                    prev[0] + [int(cand_tokens[c])],
                                    prev[1] + [float(cand_lps[c])],
                                    prev[2] + [None],
                                    prev[3] + [None],
                                )
                    elif valid_tokens_sets is None and len(raw_token_ids) >= total_needed:
                        raw_tokens_arr = np.asarray(
                            raw_token_ids[:total_needed], dtype=np.int64
                        ).reshape(num_beams, stride_k)
                        raw_logprobs_arr = np.asarray(
                            raw_logprobs[:total_needed], dtype=np.float64
                        ).reshape(num_beams, stride_k)

                        if is_vocab_slice_enabled():
                            offset, slice_size, _ = get_vocab_slice_config()
                            if offset is not None and slice_size is not None and slice_size > 0:
                                slice_mask = (raw_tokens_arr >= 0) & (raw_tokens_arr < slice_size)
                                raw_tokens_arr = raw_tokens_arr.copy()
                                raw_tokens_arr[slice_mask] += offset

                        sampled = raw_tokens_arr[:, 0:1]
                        topk_tokens = raw_tokens_arr[:, 1:stride_k]
                        topk_logprobs = raw_logprobs_arr[:, 1:stride_k]

                        matches = (topk_tokens == sampled).any(axis=1)
                        if matches.all():
                            dedup_tokens = topk_tokens
                            dedup_logprobs = topk_logprobs
                        else:
                            dedup_tokens = topk_tokens.copy()
                            dedup_logprobs = topk_logprobs.copy()
                            no_match_idx = np.where(~matches)[0]
                            dedup_tokens[no_match_idx] = raw_tokens_arr[no_match_idx, :logprobs_num]
                            dedup_logprobs[no_match_idx] = raw_logprobs_arr[
                                no_match_idx, :logprobs_num
                            ]

                        cum_lps = np.fromiter(
                            (b.cum_logprob for b in all_beams[:num_beams]),
                            dtype=np.float64,
                            count=num_beams,
                        )
                        scores_2d = cum_lps[:, None] + dedup_logprobs

                        all_beams_token_id = dedup_tokens.ravel()
                        all_beams_logprob = scores_2d.ravel()
                        all_beams_beam_idx_arr = np.repeat(
                            np.arange(num_beams, dtype=np.intp), logprobs_num
                        )

                        ranks_has_data = raw_ranks is not None and len(raw_ranks) >= total_needed
                        decoded_has_data = (
                            raw_decoded is not None and len(raw_decoded) >= total_needed
                        )

                        beam_flat_cache = []
                        for b in range(num_beams):
                            s = slice(1, stride_k) if matches[b] else slice(0, logprobs_num)
                            offset = b * stride_k
                            r = (
                                raw_ranks[offset + s.start : offset + s.stop]
                                if ranks_has_data
                                else None
                            )
                            d = (
                                raw_decoded[offset + s.start : offset + s.stop]
                                if decoded_has_data
                                else None
                            )
                            beam_flat_cache.append((dedup_tokens[b], dedup_logprobs[b], r, d))
                    else:
                        # Fallback for catalog filtering or non-standard payload shapes
                        for b_idx in range(len(fork_info)):
                            current_beam = all_beams[b_idx]

                            # Compute the true exact allocation footprint boundaries
                            # from the flat remuxed vector block for this specific beam
                            start_offset = b_idx * stride_k
                            end_offset = start_offset + stride_k

                            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                                extract_and_dedup_segment(
                                    raw_token_ids,
                                    raw_logprobs,
                                    raw_ranks,
                                    raw_decoded,
                                    start_offset,
                                    end_offset,
                                    logprobs_num,
                                )
                            )

                            is_terminal_beam = False
                            if valid_tokens_sets is not None:
                                valid_tokens_set = valid_tokens_sets[b_idx]
                                is_terminal_beam = (
                                    not valid_tokens_set
                                    and self.models.catalog.is_terminal(
                                        current_beam.tokens[len(prompt_token_ids) :]
                                    )
                                )
                                if not is_terminal_beam:
                                    logprobs_pos = self.models.catalog.apply_catalog_filter(
                                        token_ids_pos,
                                        logprobs_pos,
                                        valid_tokens_set,
                                    )
                            if is_terminal_beam:
                                completed.append(complete_catalog_beam(current_beam))
                                beam_flat_cache.append(None)
                                continue

                            # Save the clean segment safely into the historical timeline cache
                            beam_flat_cache.append(
                                (token_ids_pos, logprobs_pos, ranks_pos, decoded_pos)
                            )
                            all_beams_token_id.extend(token_ids_pos)
                            all_beams_logprob.extend(
                                current_beam.cum_logprob + lp for lp in logprobs_pos
                            )
                            all_beams_beam_idx.extend([b_idx] * len(token_ids_pos))
                else:
                    beam_flat_cache = [None] * len(fork_info)
            else:
                # Baseline execution path for prefill / batch loops
                for i, result in enumerate(output):
                    current_beam = all_beams[i]
                    if result.outputs[0].finish_reason == "error":
                        yield RequestOutput(
                            request_id=request_id,
                            prompt=prompt_text,
                            outputs=[
                                CompletionOutput(
                                    index=0,
                                    text="",
                                    token_ids=[],
                                    cumulative_logprob=None,
                                    logprobs=None,
                                    finish_reason="error",
                                )
                            ],
                            finished=True,
                            prompt_token_ids=prompt_token_ids,
                            prompt_logprobs=None,
                        )
                        return

                    if result.outputs[0].logprobs is not None:
                        flat = result.outputs[0].logprobs

                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_and_dedup_flat_logprobs(flat, logprobs_num)
                        )
                        is_terminal_beam = False
                        if valid_tokens_sets is not None:
                            valid_tokens_set = valid_tokens_sets[i]
                            is_terminal_beam = (
                                not valid_tokens_set
                                and self.models.catalog.is_terminal(
                                    current_beam.tokens[len(prompt_token_ids) :]
                                )
                            )
                            if not is_terminal_beam:
                                logprobs_pos = self.models.catalog.apply_catalog_filter(
                                    token_ids_pos,
                                    logprobs_pos,
                                    valid_tokens_set,
                                )
                        if is_terminal_beam:
                            completed.append(complete_catalog_beam(current_beam))
                            beam_flat_cache.append(None)
                            continue
                        beam_flat_cache.append(
                            (token_ids_pos, logprobs_pos, ranks_pos, decoded_pos)
                        )
                        all_beams_token_id.extend(token_ids_pos)
                        all_beams_logprob.extend(
                            current_beam.cum_logprob + lp for lp in logprobs_pos
                        )
                        all_beams_beam_idx.extend([i] * len(token_ids_pos))
                    else:
                        beam_flat_cache.append(None)

            # Handle the token for the end of sentence (EOS)
            if not isinstance(all_beams_token_id, np.ndarray):
                all_beams_token_id = np.array(all_beams_token_id)
                all_beams_logprob = np.array(all_beams_logprob)
                all_beams_beam_idx_arr = np.array(all_beams_beam_idx, dtype=np.intp)

            if not ignore_eos:
                stop_token_ids = {eos_token_id}
                if sid_end_token_id is not None:
                    stop_token_ids.add(sid_end_token_id)
                eos_beams, all_beams_logprob = complete_eos_candidates(
                    parents=all_beams,
                    token_ids=all_beams_token_id,
                    scores=all_beams_logprob,
                    beam_idx=all_beams_beam_idx_arr,
                    beam_flat_cache=beam_flat_cache,
                    stop_token_ids=stop_token_ids,
                    initial_logprobs=initial_logprobs,
                    include_eos_token=include_stop_str_in_output,
                    inherit_lora_request=False,
                )
                completed.extend(eos_beams)

            # Processing non-EOS tokens
            topn_idx = select_top_indices(
                all_beams_logprob,
                beam_width,
            )
            if beam_star:
                topn_idx = mask_beam_star_candidates(
                    all_beams_logprob,
                    topn_idx,
                    stats[token],
                    token,
                    token_idx_offset,
                    beam_star_coef,
                )

            new_beams, selected_fork_info = materialize_selected_beams(
                parents=all_beams,
                selected_indices=topn_idx,
                token_ids=all_beams_token_id,
                scores=all_beams_logprob,
                beam_idx=all_beams_beam_idx_arr,
                beam_flat_cache=beam_flat_cache,
                initial_logprobs=initial_logprobs,
            )

            # Build the routing metadata for the next grouped beam update.
            if use_beam_request:
                fork_info = selected_fork_info

            if not new_beams:
                all_beams = []
                if not cleaned_up and use_beam_request and prev_beam_internal_ids:
                    _spawn_background_task(
                        _beam_request_cleanup(
                            self.engine_client, prefill_id, prev_beam_internal_ids, rank
                        )
                    )
                    cleaned_up = True
                    prev_beam_internal_ids = []
                break
            all_beams = new_beams
        # Re-enable GC as soon as the decode loop itself is done (whether it
        # ran to completion or exited via `break`). The outer `finally`'s
        # gc.enable() below still covers the case where an exception
        # propagates out of the loop before reaching here; calling
        # gc.enable() again there when already enabled is a harmless no-op.
        # Everything past this point (sorting, catalog fill, logprob
        # reconstruction, batch decode) runs with normal GC -- it was never
        # part of the "decode loop" the disable was meant to bracket.
        if gc_was_enabled:
            gc.enable()
        if not cleaned_up and use_beam_request and prev_beam_internal_ids:
            _spawn_background_task(
                _beam_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)
            )
            cleaned_up = True
            prev_beam_internal_ids = []

        if sid_end_token_id is not None:
            for beam in all_beams:
                beam.tokens.append(sid_end_token_id)
        completed.extend(all_beams)

        # Measure beam sort time
        beam_search_sort_start = time.perf_counter()
        sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
        beam_search_sort_time = time.perf_counter() - beam_search_sort_start
        valid_completed = [beam for beam in sorted_completed if np.isfinite(beam.cum_logprob)]
        best_beams = valid_completed[:beam_width]
        if self.models.catalog is not None:
            best_beams = fill_catalog_beams(
                best_beams,
                beam_width,
                self.models.catalog,
                prompt_token_ids=initial_tokens,
                prompt_text=prompt_text,
                sid_begin_token_id=sid_begin_token_id,
                sid_end_token_id=sid_end_token_id,
                initial_logprobs=initial_logprobs,
                max_tokens=max_tokens,
                lora_request=lora_request,
            )

        # Reconstruct full logprobs only for the final best beams by walking
        # the parent-pointer tree.  This replaces the O(beams * steps^2)
        # per-step copying with a single O(beams * steps) pass at the end.
        for beam in best_beams:
            reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)

        tokens_list = [
            (
                beam.tokens[tokenized_length:-1]
                if beam.tokens[-1] == eos_token_id and not ignore_eos
                else beam.tokens[tokenized_length:]
            )
            for beam in best_beams
        ]
        if hasattr(tokenizer, "batch_decode"):
            texts = tokenizer.batch_decode(tokens_list)
            for beam, text in zip(best_beams, texts):
                beam.text = text
        else:
            for beam, tokens in zip(best_beams, tokens_list):
                beam.text = tokenizer.decode(tokens)

        beam_search_decode_time = time.perf_counter() - beam_search_start
        beam_search_prefill_time = beam_search_start - prefill_start
        beam_search_overhead = beam_search_decode_time - generation_time
        metrics = RequestStateStats(
            beam_search_overhead=beam_search_overhead,
            beam_search_prefill_time=beam_search_prefill_time,
            beam_search_decode_time=beam_search_decode_time,
            beam_search_sort_time=beam_search_sort_time,
            num_generation_tokens=num_generation_tokens,
            num_processed_tokens=num_processed_tokens,
            num_beams=beam_width,
        )

        out = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=(
                        beam.finish_reason if beam.finish_reason is not None else "length"
                    ),
                    stop_reason=beam.stop_reason,
                )
                for i, beam in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            metrics=metrics,
        )

        _record_beam_metrics(self, metrics)

        yield out
    finally:
        if catalog_task is not None and not catalog_task.done():
            catalog_task.cancel()
            try:
                await catalog_task
            except asyncio.CancelledError:
                pass
        if not cleaned_up and use_beam_request and prev_beam_internal_ids:
            _spawn_background_task(
                _beam_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)
            )
            cleaned_up = True
            prev_beam_internal_ids = []
        if gc_was_enabled:
            gc.enable()


beam_search.create_stat = create_stat
