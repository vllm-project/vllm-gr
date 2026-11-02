# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Driver-side injection of Constraint Tables resources into workers.

After the driver compiles a
:class:`~vllm_gr.constraints.compiler.ConstraintTableResource`, it broadcasts it
to every worker via ``EngineCore.collective_rpc`` (vLLM 0.30.0
``v1/engine/core.py``).  The worker method ``set_gr_constraint_resources`` is
mounted by :mod:`vllm_gr.constraints.worker_inject` during worker
``init_device``.

The payload is a base-type wire dict (see
:func:`~vllm_gr.constraints.compiler.to_wire`) — not the raw dataclass — because
vLLM v1 decodes utility payloads with an untyped msgpack decoder.  Two entry
points are provided:

- :func:`inject_constraint_resources` for the synchronous offline engine
  (``LLMEngine.engine_core.collective_rpc``).
- :func:`inject_constraint_resources_async` for the OpenAI async client
  (``AsyncEngineCore.collective_rpc_async``).

Both resolve the collective-RPC callable from the engine handle, so callers do
not need to know which concrete engine type they hold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource

logger = logging.getLogger(__name__)


def _resolve_engine_core(engine: Any) -> Any:
    """Return the EngineCore handle from an offline LLM or OpenAI client."""
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is None:
        engine_core = getattr(engine, "_engine_core", None)
    if engine_core is None:
        raise TypeError(
            f"cannot resolve EngineCore from {type(engine).__name__}; expected "
            "an offline LLM/LLMEngine or an OpenAI engine client"
        )
    return engine_core


def _resolve_collective(
    engine: Any,
    *,
    async_: bool,
) -> Callable[..., Any]:
    """Return ``collective_rpc`` (sync) or ``collective_rpc_async`` callable."""
    engine_core = _resolve_engine_core(engine)
    # AsyncLLM itself exposes collective_rpc/collective_rpc_async; prefer the
    # client-level method so both AsyncEngineCore and AsyncLLM work.
    method_name = "collective_rpc_async" if async_ else "collective_rpc"
    collective = getattr(engine_core, method_name, None)
    if not callable(collective):
        raise TypeError(
            f"engine core of type {type(engine_core).__name__} has no "
            f"{method_name}; vLLM 0.30.0 v1 EngineCore is required"
        )
    return collective


def inject_constraint_resources(
    engine: Any,
    resource: "ConstraintTableResource",
) -> list[dict]:
    """Synchronously broadcast constraint tables to every worker.

    Args:
        engine: offline LLM/LLMEngine (or anything exposing ``engine_core``
            with a synchronous ``collective_rpc``).
        resource: the compiled ConstraintTableResource.

    Returns:
        A list of per-worker status dicts from ``set_gr_constraint_resources``.
    """
    payload = _resource_to_wire(resource)
    collective = _resolve_collective(engine, async_=False)
    statuses = collective("set_gr_constraint_resources", args=(payload,))
    _verify_injected(resource, statuses)
    return statuses


async def inject_constraint_resources_async(
    engine: Any,
    resource: "ConstraintTableResource",
) -> list[dict]:
    """Asynchronously broadcast constraint tables to every worker.

    Uses ``collective_rpc_async`` on the OpenAI async engine client
    (``AsyncEngineCore`` / ``AsyncLLM``).

    Returns:
        A list of per-worker status dicts from ``set_gr_constraint_resources``.
    """
    payload = _resource_to_wire(resource)
    collective = _resolve_collective(engine, async_=True)
    awaited: Awaitable[list[dict]] = collective("set_gr_constraint_resources", args=(payload,))
    statuses = await awaited
    _verify_injected(resource, statuses)
    return statuses


def _resource_to_wire(resource: "ConstraintTableResource") -> dict:
    """Serialize a resource to a wire payload (base types only).

    The raw dataclass would be decoded into a plain dict / nested lists by the
    untyped msgpack decoder on EngineCore, so we send an explicit base-type
    payload instead.
    """
    from vllm_gr.constraints.compiler import to_wire

    return to_wire(resource)


def _verify_injected(
    resource: "ConstraintTableResource",
    statuses: list[dict],
) -> None:
    """Verify every worker ack'd the same digest, then log.

    TP/DP requires all workers to hold the same tables.  A missing status, a
    non-dict status, a mismatched digest, or an empty result is a hard failure.
    """
    if not statuses:
        raise RuntimeError(
            "constraint table injection returned no worker acknowledgements "
            "(expected at least one worker)"
        )
    for index, status in enumerate(statuses):
        if not isinstance(status, dict):
            raise RuntimeError(f"worker #{index} returned a non-dict ack: {type(status).__name__}")
        ack_digest = status.get("digest")
        if ack_digest != resource.digest:
            raise RuntimeError(
                f"worker #{index} ack'd digest {ack_digest!r}, expected "
                f"{resource.digest!r}; TP/DP workers must hold identical tables"
            )
    logger.info(
        "vllm-gr: injected constraint tables digest=%s into %d worker(s)",
        resource.digest,
        len(statuses),
    )
