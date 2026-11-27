# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from contextvars import ContextVar
from functools import wraps
from weakref import WeakValueDictionary

from vllm.logger import init_logger

logger = init_logger(__name__)


_PROCESS_OUTPUTS_PATCH_MARKER = "_vllm_gr_admission_rejection_outputs_patch"

# Terminal admission rejections, keyed by the request id the *engine* used,
# parked between the output that carries one and the serving layer that reports
# it. A request the engine refuses ends on a bare error output whose reason
# lives in ``kv_transfer_params``; the serving loop only ever sees the finish
# reason, so the message needs somewhere to wait in between.
#
# The engine's name for a request is not the name its caller is answered under.
# The frontend extends the id it reports before submitting it -- once per
# prompt and per parallel sample -- and the engine extends it again, so an
# ordinary request's refusal is filed under a name no serving loop ever sees.
# ``_ENGINE_REQUEST_IDS`` records the engine requests the request being handled
# became, and a claim is an exact lookup under those names -- never a prefix,
# since the caller supplies its own name and one may extend another's.
#
# In-flight OpenAI requests keep their refusal reason and non-refusal error
# flag on the ``_EngineRequestIds`` instance weakly referenced by
# ``_IN_FLIGHT_REQUESTS_ATTR`` until classification completes, so bounded pool
# eviction by siblings or other callers cannot hide a real engine fault. Outputs
# without an in-flight owner (such as refused V1 sessions or late siblings) fall
# back to a bounded single-id refusal pool; unowned queries never aggregate
# across multiple engine ids.
_ADMISSION_REJECTIONS_ATTR = "_gr_admission_rejections"
_IN_FLIGHT_REQUESTS_ATTR = "_gr_in_flight_requests"
_ADMISSION_REJECTION_MAX = 256


class _EngineRequestIds(list[str]):
    """Per-request engine id list holding in-flight error classification."""

    __slots__ = ("__weakref__", "has_other_error", "refusal_reason")

    def __init__(self) -> None:
        super().__init__()
        self.has_other_error = False
        self.refusal_reason: str | None = None


# The engine requests submitted for the OpenAI request handled in this task.
# ``patch_client_request_id`` starts the list where the frontend resolves the
# request's id and appends every engine request added under it; outside an
# OpenAI request it is unset, which keeps in-process callers and beam sessions
# on the engine's own names.
_ENGINE_REQUEST_IDS: ContextVar[_EngineRequestIds | None] = ContextVar(
    "vllm_gr_engine_request_ids", default=None
)


def start_client_request_ids() -> _EngineRequestIds:
    """Create a fresh per-request engine id tracker for the current context."""
    return _EngineRequestIds()


def track_engine_request_id(
    processor: object,
    request_id: str,
    engine_request_ids: _EngineRequestIds | None,
) -> None:
    """Append ``request_id`` and bind it to the caller's in-flight tracker."""
    if engine_request_ids is None:
        return
    engine_request_ids.append(request_id)
    in_flight = getattr(processor, _IN_FLIGHT_REQUESTS_ATTR, None)
    if in_flight is None:
        in_flight = WeakValueDictionary()
        setattr(processor, _IN_FLIGHT_REQUESTS_ATTR, in_flight)
    in_flight[request_id] = engine_request_ids
    forget_admission_rejection(processor, request_id)


def note_admission_rejections(processor, engine_core_outputs) -> None:
    """Record refused outputs and non-refusal engine faults for classification."""
    from vllm.v1.engine import FinishReason

    from vllm_gr.v1.beam.batch_contracts import admission_rejection

    in_flight = getattr(processor, _IN_FLIGHT_REQUESTS_ATTR, None)
    for ec_output in engine_core_outputs:
        request_id = getattr(ec_output, "request_id", "")
        message = admission_rejection(ec_output)
        finish_reason = getattr(ec_output, "finish_reason", None)
        owner = in_flight.get(request_id) if in_flight is not None else None
        if in_flight is not None and owner is not None:
            if message is not None:
                if owner.refusal_reason is None:
                    owner.refusal_reason = message
            elif finish_reason == FinishReason.ERROR:
                owner.has_other_error = True
            elif finish_reason is not None:
                in_flight.pop(request_id, None)
                forget_admission_rejection(processor, request_id)
            continue

        if message is None:
            # A finished output that is not itself a refusal says its id now
            # belongs to a request this pool was never told about. Whatever is
            # parked under that id described an earlier one and must not answer
            # for this one: the caller picks its own id and may reuse it.
            if finish_reason is not None:
                forget_admission_rejection(processor, request_id)
            continue
        rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
        if rejections is None:
            rejections = {}
            setattr(processor, _ADMISSION_REJECTIONS_ATTR, rejections)
        rejections.pop(request_id, None)
        while len(rejections) >= _ADMISSION_REJECTION_MAX:
            rejections.pop(next(iter(rejections)))
        rejections[request_id] = message


def forget_admission_rejection(processor, request_id: str) -> None:
    """Drop whatever is parked for ``request_id``, if anything is."""
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if rejections:
        rejections.pop(request_id, None)


def take_admission_rejection(
    engine_client, request_id: str, engine_request_ids: _EngineRequestIds | None = None
) -> str | None:
    """Consume the parked reason for the request this caller is asking about.

    A refused request may be parked under ``request_id`` (the name the serving
    layer reports) or under one of the engine request ids tracked in
    ``engine_request_ids`` (see ``patch_client_request_id``). Lookups match
    those exact ids and never a shared prefix: the caller supplies its own
    name -- an ``X-Request-Id`` header, or the body's ``request_id`` -- so a
    prefix match would let one caller's name claim another request's refusal and
    report a real engine fault as a client mistake.

    When ``engine_request_ids`` is provided, aggregate across its tracked ids
    and answer with a refusal only when none of them failed otherwise.
    """
    processor = getattr(engine_client, "output_processor", None)
    if processor is None:
        return None
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if engine_request_ids is not None:
        in_flight = getattr(processor, _IN_FLIGHT_REQUESTS_ATTR, None)
        if in_flight:
            for i in engine_request_ids:
                in_flight.pop(i, None)
        direct_message = rejections.pop(request_id, None) if rejections else None
        if rejections:
            for i in engine_request_ids:
                rejections.pop(i, None)
        owner_fault = engine_request_ids.has_other_error
        owner_message = engine_request_ids.refusal_reason
        engine_request_ids.refusal_reason = None
        if owner_fault:
            return None
        return direct_message if direct_message is not None else owner_message
    return rejections.pop(request_id, None) if rejections else None


def make_patched_process_outputs(original_process_outputs):
    """Preserve admission reasons while forwarding every output to vLLM."""
    if getattr(original_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False):
        return original_process_outputs

    @wraps(original_process_outputs)
    def patched_process_outputs(self, engine_core_outputs, *args, **kwargs):
        note_admission_rejections(self, engine_core_outputs)
        return original_process_outputs(self, engine_core_outputs, *args, **kwargs)

    setattr(patched_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, True)
    return patched_process_outputs
