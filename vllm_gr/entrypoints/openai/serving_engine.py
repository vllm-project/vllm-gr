# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.renderers.inputs.preprocess import prompt_to_seq
from vllm.sampling_params import BeamSearchParams

from vllm_gr.entrypoints.common.lifecycle import (
    frontend_request_scope,
    schedule_frontend_gc_freeze,
)

if TYPE_CHECKING:
    from vllm_gr.entrypoints.beam_search.protocol import AsyncBeamSearchFrontend

logger = init_logger(__name__)


def create_online_beam_search_client(self) -> "AsyncBeamSearchFrontend":
    """Create the online A2 request/session adapter."""
    from vllm_gr.entrypoints.beam_search.client import (
        AsyncBeamSearchClient,
    )

    return AsyncBeamSearchClient(self)


@frontend_request_scope
async def beam_search(
    self,
    prompt: EngineInput | Sequence[EngineInput],
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    session_id: str | None = None,
    *,
    trie_ids: list[int] | tuple[int, ...] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Submit one online V1 request and yield terminal outputs only.

    ``prompt`` is a single input or a sequence of them; either way there is one
    submit and one terminal output per input, in order.

    ``ignore_eos`` is not a client choice on this interface. V1 runs the whole
    beam group as one persistent session over a fixed budget, so a beam that
    stopped early would make the terminal step data-dependent and the engine
    rejects such a request outright -- there is exactly one value it can serve.
    Supplying it here is what makes an otherwise ordinary request work, and
    doing it at this boundary rather than in the shared parser is also what
    keeps it off an engine that is not running V1.
    """
    from vllm_gr.config.execution import require_beam_config

    require_beam_config(getattr(self.engine_client, "vllm_config", None))
    prompts = tuple(prompt_to_seq(prompt))
    if not prompts:
        raise VLLMValidationError("beam_search requires at least one prompt")
    for candidate in prompts:
        # A session runs the one token stream its decoder owns, and the engine
        # refuses the multimodal features an encoder-decoder request arrives
        # with. The speech-to-text endpoints reach beam search through here,
        # so name what is missing instead of failing on a schema the caller
        # never sees.
        if isinstance(candidate, Mapping) and candidate.get("type") == "enc_dec":
            raise NotImplementedError(
                "beam search is not implemented for encoder-decoder inputs such as "
                "speech-to-text requests"
            )
    from vllm_gr.entrypoints.beam_search import run_beam_search_async
    from vllm_gr.entrypoints.beam_search.protocol import OnlineBeamSearchRequest

    params = copy(params)
    params.ignore_eos = True

    priority = int(time.perf_counter() * 1_000_000)
    rank = await _next_data_parallel_rank(self.engine_client)

    call = OnlineBeamSearchRequest(
        prompts=prompts,
        request_id=request_id,
        params=params,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=rank,
        session_id=session_id,
        trie_ids=trie_ids,
    )
    outputs = await run_beam_search_async(
        create_online_beam_search_client(self),
        call,
    )
    # Multi-process deployments run this serving frontend in a process
    # separate from the EngineCore/worker, whose GC freeze cannot reach here.
    # Freeze the frontend heap once after the first beam request so the
    # driver's own gen-1 GC (heavy per-request beam bookkeeping) cannot spike
    # request latency on a fixed cadence.  Idempotent; spawns a daemon thread.
    schedule_frontend_gc_freeze()

    for output in outputs:
        yield output


_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


async def _next_data_parallel_rank(engine_client) -> int | None:
    """Round-robin over the ranks accepted by the native input processor."""
    global _dp_rank_counter

    vllm_config = getattr(engine_client, "vllm_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    data_parallel_size = getattr(parallel_config, "data_parallel_size", 1)
    if data_parallel_size is None or data_parallel_size <= 1:
        return None
    if getattr(parallel_config, "local_engines_only", False):
        data_parallel_size = parallel_config.data_parallel_size_local

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank
