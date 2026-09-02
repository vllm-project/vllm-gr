# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from functools import wraps
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.types import BeamRequestStepUpdate

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass

# Bounded buffering for beam-session outputs that arrive while the session is
# between step collectors (EngineCore engine-driven chaining runs the next
# step ahead of the frontend's register_beam_output).  Upstream drops outputs for
# request ids without state (output_processor "already-aborted" branch),
# which would hang the beam coroutine's un-timeouted q.get() forever.
# Bootstrap per-session cap; register_beam_output_fn raises it to
# beam_max_decode_steps + 2 once the config is reachable (a live session can
# legitimately buffer one output per engine-led decode step, i.e. up to
# beam_max_decode_steps; +2 covers the terminal output and a margin).
# The cap is an admission/backpressure signal, not a drop threshold: a
# tracked live session must never lose a finished output (that hangs q.get
# until the 120s collect timeout). Overflows are counted and logged.
_GR_BEAM_PENDING_PER_ID = 4
_GR_BEAM_PENDING_MAX_SESSIONS = 256
_PROCESS_OUTPUTS_PATCH_MARKER = "_vllm_gr_beam_pending_patch"


def _ensure_beam_session_ids(processor) -> set:
    session_ids = getattr(processor, "_gr_beam_session_ids", None)
    if session_ids is None:
        session_ids = set()
        processor._gr_beam_session_ids = session_ids
    return session_ids


def _stamp_pending_cap(processor, owner=None) -> int:
    """Raise the per-session buffer cap from config when reachable."""
    cap = getattr(processor, "_gr_beam_pending_cap", None)
    if cap is not None:
        return cap
    cap = _GR_BEAM_PENDING_PER_ID
    try:
        from vllm_gr.config import GRConfig

        vllm_config = getattr(processor, "vllm_config", None)
        if vllm_config is None and owner is not None:
            vllm_config = getattr(owner, "vllm_config", None)
        cap = GRConfig.from_vllm_config(vllm_config).beam_max_decode_steps + 2
    except Exception:  # pragma: no cover - config read is best-effort
        pass
    cap = max(cap, _GR_BEAM_PENDING_PER_ID)
    processor._gr_beam_pending_cap = cap
    return cap


def track_beam_session_fn(self, request_id: str) -> None:
    """Mark ``request_id`` as a live beam session before EngineCore dispatch.

    Must run before SUBMIT_SESSION / ADD_BATCH: engine-driven chaining can
    emit the first decode output while the frontend is still awaiting
    prefill, and process_outputs only buffers ids in this set.
    """
    processor = getattr(self, "output_processor", None)
    if processor is None or not request_id:
        return
    _ensure_beam_session_ids(processor).add(request_id)
    _stamp_pending_cap(processor, owner=self)


def make_patched_process_outputs(original_process_outputs):
    """Wrap OutputProcessor.process_outputs to buffer beam-session outputs.

    Outputs for a known beam session id with no request state are held in a
    per-session buffer instead of being dropped; the next
    register_beam_output for that session replays them.  Non-beam outputs
    keep upstream behavior exactly.  Tracked live sessions never drop:
    exceeding the cap logs and counts an overflow, then keeps the output.
    """
    if getattr(original_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False):
        return original_process_outputs

    @wraps(original_process_outputs)
    def patched_process_outputs(self, engine_core_outputs, *args, **kwargs):
        session_ids = getattr(self, "_gr_beam_session_ids", None)
        if not session_ids:
            return original_process_outputs(self, engine_core_outputs, *args, **kwargs)
        pending = getattr(self, "_gr_beam_pending", None)
        if pending is None:
            pending = {}
            self._gr_beam_pending = pending
        remaining = []
        per_id_cap = getattr(self, "_gr_beam_pending_cap", _GR_BEAM_PENDING_PER_ID)
        overflow = getattr(self, "_gr_beam_pending_overflows", 0)
        for ec_output in engine_core_outputs:
            req_id = getattr(ec_output, "request_id", None)
            if req_id in session_ids and req_id not in self.request_states:
                if req_id not in pending and len(pending) >= _GR_BEAM_PENDING_MAX_SESSIONS:
                    overflow += 1
                    logger.error(
                        "vllm-gr: beam pending session admission overflow "
                        "(sessions=%d, cap=%d); keeping output for live session %s",
                        len(pending),
                        _GR_BEAM_PENDING_MAX_SESSIONS,
                        req_id,
                    )
                buffer = pending.setdefault(req_id, [])
                if len(buffer) >= per_id_cap:
                    overflow += 1
                    logger.error(
                        "vllm-gr: beam pending buffer overflow for live session %s "
                        "(buffered=%d, per_id_cap=%d); keeping the output "
                        "(dropping it would hang the collector)",
                        req_id,
                        len(buffer),
                        per_id_cap,
                    )
                buffer.append(ec_output)
                continue
            remaining.append(ec_output)
        self._gr_beam_pending_overflows = overflow
        if remaining:
            return original_process_outputs(self, remaining, *args, **kwargs)
        return original_process_outputs(self, [], *args, **kwargs)

    setattr(patched_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, True)
    return patched_process_outputs


# ---------------------------------------------------------------------------
# AsyncLLM methods for ADD_BATCH / BEAM_REQUEST_STEP_UPDATE
# ---------------------------------------------------------------------------


async def prepare_request_fn(
    self,
    request_id,
    prompt,
    params,
    *,
    lora_request=None,
    tokenization_kwargs=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank=None,
    prompt_text=None,
    arrival_time=None,
):
    """Process input and register output queue, but DON'T send to
    EngineCore. Returns (output_queue, engine_core_request) for later
    batch send via _add_requests_batch().

    The 0.22 input processor validates requests against the asynchronously
    discovered supported-task set, so request preparation is asynchronous."""
    from vllm.pooling_params import PoolingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if self.errored:
        from vllm.v1.engine.exceptions import EngineDeadError

        raise EngineDeadError()

    is_pooling = isinstance(params, PoolingParams)

    if (
        self.vllm_config.cache_config.kv_sharing_fast_prefill
        and not is_pooling
        and params.prompt_logprobs
    ):
        raise ValueError(
            "--kv-sharing-fast-prefill produces incorrect logprobs for "
            "prompt tokens, please disable it when the requests need "
            "prompt logprobs"
        )

    # Convert Input --> Request.
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        if prompt_text is not None:
            raise ValueError("should only provide prompt_text with EngineCoreRequest")
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        from vllm.renderers.inputs.preprocess import extract_prompt_components

        prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

    self.input_processor.assign_request_id(request)

    self._run_output_handler()

    # Create a new output collector for the request.
    params = request.params
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # Register the request in OutputProcessor (this process) but
    # do NOT send to EngineCore yet.
    self.output_processor.add_request(request, prompt_text, None, 0, queue)

    if self.log_requests:
        logger.info("Prepared request %s.", request.request_id)

    return queue, request


async def _add_requests_batch_fn(
    self,
    requests,
    use_batch_message: bool = False,
) -> None:
    """Send multiple already-prepared requests to EngineCore.

    Args:
        use_batch_message: If True, always send as ADD_BATCH (needed for
            beam cache). If False, send individual ADD messages.
    """
    await self.engine_core.add_requests_async(requests, force_batch=use_batch_message)


def register_beam_output_fn(
    self,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params,
    eos_token_id=None,
    lora_request=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """Register output queue for a beam fork child (no ZMQ send).

    The actual request will be created by EngineCore's beam_request_step_update handler.
    We just need a queue in OutputProcessor to receive outputs."""
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if eos_token_id is not None:
        sampling_params.update_from_generation_config({}, eos_token_id)

    ec_request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=lora_request,
        cache_salt=None,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
        priority=priority,
    )
    # Set external_req_id directly (skip assign_request_id which
    # would mangle the ID that EngineCore already knows about).
    ec_request.external_req_id = request_id

    self._run_output_handler()
    queue = RequestOutputCollector(sampling_params.output_kind, request_id)
    self.output_processor.add_request(ec_request, None, None, 0, queue)

    # Engine-driven hardening: track the session and replay any outputs that
    # arrived while the session had no collector (the chained step can run
    # ahead of this registration; without the replay they would have been
    # dropped upstream and the beam coroutine would hang on q.get()).
    processor = self.output_processor
    _ensure_beam_session_ids(processor).add(request_id)
    _stamp_pending_cap(processor, owner=self)
    pending = getattr(processor, "_gr_beam_pending", None)
    if pending:
        buffered = pending.get(request_id)
        if buffered:
            # R1: release exactly the OLDEST buffered output. Every beam
            # step Request runs with max_tokens=1, so each EngineCoreOutput
            # arrives finished=True and _collect_beam_result consumes exactly
            # one output per beam_search loop iteration: register:consume is
            # 1:1, so a FIFO one-at-a-time release never starves or stalls.
            # Replaying the whole list in ONE process_outputs call would let
            # the first finished output pop the request state and every
            # remaining one would hit upstream's req_state-is-None drop -
            # the next loop iteration would then await a collector that
            # never fires (bounded only by the 120s collect timeout).
            oldest = buffered.pop(0)
            logger.debug(
                "vllm-gr: replaying oldest of %d buffered beam output(s) for session %s",
                len(buffered) + 1,
                request_id,
            )
            processor.process_outputs([oldest])
            if not buffered:
                pending.pop(request_id, None)
    return queue


async def beam_request_step_update_fn(self, update: BeamRequestStepUpdate) -> None:
    """Send BEAM_REQUEST_STEP_UPDATE to EngineCore to grouped processing."""
    if update.beam_width == 0:
        # Session teardown (the beam_search cleanup update): drop any outputs
        # still buffered for this session (early-EOS break / abort leftovers;
        # see the FIFO release in register_beam_output_fn) and stop tracking
        # the id so late outputs revert to upstream's normal dropping.
        processor = getattr(self, "output_processor", None)
        pending = getattr(processor, "_gr_beam_pending", None)
        if pending is not None:
            pending.pop(update.session_id, None)
        session_ids = getattr(processor, "_gr_beam_session_ids", None)
        if session_ids is not None:
            session_ids.discard(update.session_id)
    await self.engine_core.beam_request_step_update_async(update)


async def beam_session_template_fn(self, template: BeamRequestStepUpdate) -> None:
    """Send BEAM_SESSION_TEMPLATE to EngineCore (fire-and-forget).

    Must be sent after the session's ADD_BATCH and before awaiting the
    prefill output — the EngineCore consumes it when the prefill output
    iteration runs its engine-driven hook.
    """
    await self.engine_core.beam_session_template_async(template)


async def submit_session_fn(
    self,
    requests,
    template: BeamRequestStepUpdate,
) -> None:
    """Submit a beam session as one SUBMIT_SESSION message.

    Replaces the ADD_BATCH + BEAM_SESSION_TEMPLATE pair: the prefill batch
    and the session template reach the EngineCore atomically in a single
    wire message.  Must be called before awaiting the prefill outputs (the
    EngineCore consumes the template in the same busy-loop iteration that
    processes the prefill output).  InprocClient keeps its direct-call
    path; clients without the merged send fall back to the legacy
    two-message sequence.

    Tracks the session id before dispatch so a chained step-0 output that
    arrives during prefill collection is buffered instead of dropped.
    """
    session_id = getattr(template, "session_id", None) or (
        getattr(requests[0], "request_id", None) if requests else None
    )
    if session_id:
        track_beam_session_fn(self, session_id)
    submit = getattr(self.engine_core, "submit_session_async", None)
    if submit is not None:
        await submit(requests, template)
        return
    # InprocClient (no wire): direct engine-core handler call.
    inproc_submit = getattr(self.engine_core, "submit_session", None)
    if inproc_submit is not None:
        inproc_submit(requests, template)
        return
    # Legacy client path: two messages, same FIFO guarantees.
    await self._add_requests_batch(requests, use_batch_message=True)
    await self.engine_core.beam_session_template(template)
