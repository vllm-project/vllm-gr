# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A2 request construction and Server-to-EngineCore session adapters."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable
from typing import Any, Protocol

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import GRConfig
from vllm_gr.entrypoints.beam_search_v1 import (
    BeamSearchV1Session,
    OfflineBeamSearchV1Call,
    OnlineBeamSearchV1Call,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchFailure,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputOptions,
    validate_beam_batch_request,
)


class BeamSearchV1RequestError(ValueError):
    """A request that cannot be admitted as a B=1 GR session."""


class BeamSearchV1BackendError(RuntimeError):
    """A typed Server-side view of an EngineCore session failure."""

    def __init__(self, failure: BeamBatchFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class SyncGRSessionClient(Protocol):
    """A3 methods installed on the synchronous vLLM ``LLMEngine``."""

    def submit_gr_session(self, request: BeamBatchRequest) -> object: ...

    def wait_gr_session(self, session_id: str, handle: object) -> BeamBatchResult: ...

    def abort_gr_session(self, session_id: str) -> None: ...

    def release_gr_session(self, session_id: str) -> None: ...


class AsyncGRSessionClient(Protocol):
    """A3 methods installed on vLLM ``AsyncLLM`` for online serving."""

    def submit_gr_session(self, request: BeamBatchRequest) -> Awaitable[object]: ...

    def wait_gr_session(self, session_id: str, handle: object) -> Awaitable[BeamBatchResult]: ...

    def abort_gr_session(self, session_id: str) -> Awaitable[None]: ...

    def release_gr_session(self, session_id: str) -> Awaitable[None]: ...


def new_gr_session_id(external_request_id: str | None = None) -> str:
    """Create the only request/session ID used by all layers of one run."""
    prefix = external_request_id.strip() if external_request_id else "offline"
    if not prefix:
        raise BeamSearchV1RequestError("external request_id must not be empty")
    return f"{prefix}-gr-beam-v1-{random_uuid()}"


def _token_id(tokenizer: Any, token: str | None, name: str) -> int | None:
    if token is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == -1:
        raise BeamSearchV1RequestError(f"{name} must be a valid tokenizer token")
    return int(token_id)


def _sampling_and_batch_params(
    params: Any,
    tokenizer: Any,
) -> tuple[SamplingParams, BeamBatchParams, int | None]:
    if not params.ignore_eos or getattr(params, "early_stopping", False):
        raise BeamSearchV1RequestError("beam_search_v1 requires ignore_eos=True and fixed steps")
    begin_token_id = _token_id(tokenizer, getattr(params, "begin_token", None), "begin_token")
    end_token_id = _token_id(tokenizer, getattr(params, "end_token", None), "end_token")
    reserved_tokens = int(begin_token_id is not None) + int(end_token_id is not None)
    decode_steps = params.max_tokens - reserved_tokens
    if decode_steps <= 0:
        raise BeamSearchV1RequestError(
            "max_tokens must leave at least one Decode step after begin/end tokens"
        )

    stop_token_ids: tuple[int, ...] = ()
    sampling_params = SamplingParams(
        # Beam scores arrive only in A5's terminal result, never in control tokens.
        logprobs=None,
        max_tokens=decode_steps,
        temperature=params.temperature,
        ignore_eos=params.ignore_eos,
        stop_token_ids=list(stop_token_ids),
        detokenize=False,
        flat_logprobs=True,
        skip_clone=True,
    )
    batch_params = BeamBatchParams(
        beam_width=params.beam_width,
        max_tokens=decode_steps,
        temperature=params.temperature,
        stop_token_ids=stop_token_ids,
        ignore_stop_tokens=params.ignore_eos,
        constraint_resource_id=DEFAULT_CONSTRAINT_RESOURCE_ID,
        length_penalty=params.length_penalty,
        catalog_strategy=getattr(params, "catalog_strategy", None),
    )
    return sampling_params, batch_params, begin_token_id


def _append_begin_token(request: EngineCoreRequest, begin_token_id: int | None) -> None:
    if begin_token_id is None:
        return
    if request.prompt_token_ids is None:
        raise BeamSearchV1RequestError("embedding prompts are not supported by beam_search_v1")
    request.prompt_token_ids = [*request.prompt_token_ids, begin_token_id]


def _validate_runtime_request(request: BeamBatchRequest, vllm_config: Any) -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise BeamSearchV1RequestError("persistent beam_search_v1 currently requires CUDA")
    if vllm_config.compilation_config.mode != 0:
        raise BeamSearchV1RequestError(
            "beam_search_v1 precision baseline requires compilation_config.mode=0; "
            "torch.compile precision is not accepted yet"
        )
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        )
    ):
        raise BeamSearchV1RequestError("beam_search_v1 integration requires TP=PP=DP=1")
    if vllm_config.speculative_config is not None or vllm_config.model_config.is_hybrid:
        raise BeamSearchV1RequestError("beam_search_v1 does not support speculative/hybrid models")
    validate_beam_batch_request(request)
    if len(request.items) != 1:
        raise BeamSearchV1RequestError("beam_search_v1 currently requires exactly one prompt")

    gr_config = GRConfig.from_vllm_config(vllm_config)
    max_width = gr_config.beam_max_width
    if max_width is None:
        raise BeamSearchV1RequestError(
            "beam_search_v1 requires vllm-gr beam_max_width to be configured"
        )
    if request.beam_params.beam_width > max_width:
        raise BeamSearchV1RequestError(
            f"beam_width {request.beam_params.beam_width} exceeds runtime limit {max_width}"
        )
    if request.beam_params.beam_width != max_width:
        raise BeamSearchV1RequestError("beam_search_v1 requires the configured fixed Beam width")
    if request.beam_params.max_tokens > 3:
        raise BeamSearchV1RequestError(
            "canonical SID constraints support at most three generation stages"
        )
    if request.beam_params.max_tokens > gr_config.beam_max_decode_steps:
        raise BeamSearchV1RequestError(
            f"Decode steps {request.beam_params.max_tokens} exceed runtime limit "
            f"{gr_config.beam_max_decode_steps}"
        )

    engine_request = request.items[0].request
    if engine_request.lora_request is not None or engine_request.mm_features:
        raise BeamSearchV1RequestError(
            "beam_search_v1 currently requires text-only, non-LoRA input"
        )
    prompt_tokens = engine_request.prompt_token_ids
    if prompt_tokens is None:
        raise BeamSearchV1RequestError("embedding prompts are not supported by beam_search_v1")
    model_config = vllm_config.model_config
    total_length = len(prompt_tokens) + request.beam_params.max_tokens
    if total_length > model_config.max_model_len:
        raise BeamSearchV1RequestError(
            f"prompt plus Decode length {total_length} exceeds max_model_len "
            f"{model_config.max_model_len}"
        )
    scheduler_config = vllm_config.scheduler_config
    if scheduler_config.max_num_seqs != 1:
        raise BeamSearchV1RequestError("beam_search_v1 integration requires max_num_seqs=1")
    if max_width > scheduler_config.max_num_batched_tokens:
        raise BeamSearchV1RequestError("Beam width exceeds the Worker token buffer budget")
    if len(prompt_tokens) > scheduler_config.max_num_batched_tokens:
        raise BeamSearchV1RequestError(
            f"Prefill length {len(prompt_tokens)} exceeds max_num_batched_tokens "
            f"{scheduler_config.max_num_batched_tokens}"
        )


def _make_batch_request(
    *,
    session_id: str,
    engine_request: EngineCoreRequest,
    batch_params: BeamBatchParams,
    params: Any,
    vllm_config: Any,
) -> BeamBatchRequest:
    if engine_request.request_id != session_id:
        raise BeamSearchV1RequestError("EngineCoreRequest.request_id must equal the GR session ID")
    output_options = BeamOutputOptions(
        num_return_sequences=batch_params.beam_width,
        include_stop_token=bool(getattr(params, "include_stop_str_in_output", False)),
        include_cumulative_logprob=True,
        include_parent_beam_indices=True,
    )
    request = BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=session_id,
        items=(BeamBatchItem(item_index=0, request=engine_request),),
        beam_params=batch_params,
        output_options=output_options,
    )
    _validate_runtime_request(request, vllm_config)
    return request


def _required_method(owner: Any, name: str) -> Any:
    method = getattr(owner, name, None)
    if method is None:
        raise NotImplementedError(f"A3 must install {type(owner).__name__}.{name}")
    return method


def _failure_result(batch_id: str, code: str, exc: Exception) -> BeamBatchResult:
    return BeamBatchResult(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=batch_id,
        output=None,
        failure=BeamBatchFailure(code=code, message=str(exc), item_index=None),
    )


def _raise_failure(result: BeamBatchResult) -> None:
    if result.failure is not None:
        raise BeamSearchV1BackendError(result.failure)


def _decode_beam_texts(
    tokenizer: Any, prompt_token_ids: list[int], suffix_token_ids: list[list[int]]
) -> list[str]:
    """Decode the shared OneRec prompt once, without changing text boundaries."""
    from tokenizers.decoders import ByteLevel
    from transformers import Qwen2TokenizerFast

    if not suffix_token_ids:
        return []
    # ByteLevel concatenates bytes before UTF-8 decoding. An ASCII-ending
    # prompt token (including OneRec's SID begin marker) is a safe split:
    # no partial UTF-8 character can join across it. Cleanup must be disabled
    # because it can rewrite whitespace across the prompt/suffix boundary.
    # Other tokenizers retain full-sequence decoding, not assumed additivity.
    shared_prompt = False
    if (
        isinstance(tokenizer, Qwen2TokenizerFast)
        and isinstance(tokenizer.backend_tokenizer.decoder, ByteLevel)
        and not tokenizer.clean_up_tokenization_spaces
        and prompt_token_ids
    ):
        last_token = tokenizer.convert_ids_to_tokens(prompt_token_ids[-1])
        shared_prompt = bool(last_token) and last_token.isascii()
    if shared_prompt:
        prompt_text = tokenizer.decode(prompt_token_ids)
        return [prompt_text + text for text in tokenizer.batch_decode(suffix_token_ids)]
    return [tokenizer.decode(prompt_token_ids + suffix) for suffix in suffix_token_ids]


class OfflineBeamSearchV1ServerImpl:
    """Offline A2 adapter. A3 supplies the EngineCore transport methods."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.engine = llm.llm_engine

    def prepare_request(self, call: OfflineBeamSearchV1Call) -> BeamBatchRequest:
        if len(call.prompts) != 1:
            raise BeamSearchV1RequestError("beam_search_v1 currently requires one prompt")
        if call.concurrency_limit is not None and call.concurrency_limit <= 0:
            raise BeamSearchV1RequestError("concurrency_limit must be positive or None")
        tokenizer = self.llm.get_tokenizer()
        sampling_params, batch_params, begin_token_id = _sampling_and_batch_params(
            call.params, tokenizer
        )
        engine_inputs = self.llm._preprocess_cmpl(call.prompts)
        lora_requests = self.llm._lora_request_to_seq(call.lora_request, 1)
        external_session_id = new_gr_session_id()
        engine_request = self.engine.input_processor.process_inputs(
            external_session_id,
            engine_inputs[0],
            sampling_params,
            supported_tasks=self.engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=lora_requests[0],
        )
        _append_begin_token(engine_request, begin_token_id)
        self._prompt_tokens = list(engine_request.prompt_token_ids)
        self._original_prompt = engine_inputs[0]
        self.engine.input_processor.assign_request_id(engine_request)
        return _make_batch_request(
            session_id=engine_request.request_id,
            engine_request=engine_request,
            batch_params=batch_params,
            params=call.params,
            vllm_config=self.engine.vllm_config,
        )

    def submit_once(self, request: BeamBatchRequest) -> BeamSearchV1Session:
        try:
            handle = _required_method(self.engine, "submit_gr_session")(request)
        except Exception as exc:
            abort = getattr(self.engine, "abort_gr_session", None)
            if abort is not None:
                try:
                    abort(request.batch_id)
                except Exception as abort_exc:  # noqa: BLE001 - cleanup boundary
                    exc.add_note(f"GR session cleanup also failed: {abort_exc}")
            failure = BeamBatchFailure("submit_failed", str(exc), None)
            raise BeamSearchV1BackendError(failure) from exc
        return BeamSearchV1Session(
            batch_id=request.batch_id,
            session_id=request.batch_id,
            handle=handle,
            data_parallel_rank=request.items[0].request.data_parallel_rank,
        )

    def wait_final(self, session: BeamSearchV1Session) -> BeamBatchResult:
        try:
            result = _required_method(self.engine, "wait_gr_session")(
                session.session_id, session.handle
            )
            if not isinstance(result, BeamBatchResult):
                raise TypeError("EngineCore returned a non-BeamBatchResult terminal value")
            session.terminal = True
            session.failed = result.failure is not None
            return result
        except Exception as exc:  # noqa: BLE001 - EngineCore failure boundary
            session.failed = True
            return _failure_result(session.batch_id, "execution_failed", exc)

    def build_output(self, call: OfflineBeamSearchV1Call, result: BeamBatchResult) -> list[Any]:
        _raise_failure(result)
        from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput, BeamSearchSequence

        tokenizer = self.llm.get_tokenizer()
        end = _token_id(tokenizer, getattr(call.params, "end_token", None), "end_token")
        sequences = []
        for item in result.output.items:
            for sequence in item.sequences:
                tokens = (
                    self._prompt_tokens + list(sequence.token_ids) + ([] if end is None else [end])
                )
                sequences.append(
                    BeamSearchSequence(
                        orig_prompt=self._original_prompt,
                        tokens=tokens,
                        logprobs=[],
                        cum_logprob=sequence.cumulative_logprob or 0.0,
                        text="",
                        finish_reason=sequence.finish_reason,
                        stop_reason=sequence.stop_token_id,
                    )
                )
        texts = _decode_beam_texts(
            tokenizer,
            self._prompt_tokens,
            [sequence.tokens[len(self._prompt_tokens) :] for sequence in sequences],
        )
        for sequence, text in zip(sequences, texts, strict=True):
            sequence.text = text
        return [BeamSearchOutput(sequences=sequences)]

    def abort(self, session: BeamSearchV1Session) -> None:
        if session.terminal or session.closed:
            return
        _required_method(self.engine, "abort_gr_session")(session.session_id)
        session.failed = True

    def close(self, session: BeamSearchV1Session) -> None:
        if session.closed:
            return
        try:
            if session.terminal:
                release = getattr(self.engine, "release_gr_session", None)
                if release is not None:
                    release(session.session_id)
            else:
                self.abort(session)
        finally:
            session.closed = True


class AsyncBeamSearchV1ServerImpl:
    """Online A2 adapter using the same request and EngineCore session ID."""

    def __init__(self, serving: Any) -> None:
        self.serving = serving
        self.engine = serving.engine_client

    async def prepare_request(self, call: OnlineBeamSearchV1Call) -> BeamBatchRequest:
        tokenizer = self.serving.input_processor.tokenizer
        if tokenizer is None:
            raise BeamSearchV1RequestError("beam_search_v1 requires tokenizer initialization")
        sampling_params, batch_params, begin_token_id = _sampling_and_batch_params(
            call.params, tokenizer
        )
        external_session_id = new_gr_session_id(call.request_id)
        engine_request = self.engine.input_processor.process_inputs(
            external_session_id,
            call.prompt,
            sampling_params,
            supported_tasks=await self.engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=call.lora_request,
            trace_headers=call.trace_headers,
            priority=call.priority,
            data_parallel_rank=call.data_parallel_rank,
        )
        _append_begin_token(engine_request, begin_token_id)
        self._prompt_tokens = list(engine_request.prompt_token_ids)
        self._begin_token_id = begin_token_id
        self.engine.input_processor.assign_request_id(engine_request)
        return _make_batch_request(
            session_id=engine_request.request_id,
            engine_request=engine_request,
            batch_params=batch_params,
            params=call.params,
            vllm_config=self.engine.vllm_config,
        )

    async def submit_once(self, request: BeamBatchRequest) -> BeamSearchV1Session:
        try:
            handle = _required_method(self.engine, "submit_gr_session")(request)
            if inspect.isawaitable(handle):
                handle = await handle
        except asyncio.CancelledError:
            await self._abort_id(request.batch_id)
            raise
        except Exception as exc:
            await self._abort_id(request.batch_id)
            failure = BeamBatchFailure("submit_failed", str(exc), None)
            raise BeamSearchV1BackendError(failure) from exc
        return BeamSearchV1Session(
            batch_id=request.batch_id,
            session_id=request.batch_id,
            handle=handle,
            data_parallel_rank=request.items[0].request.data_parallel_rank,
        )

    async def wait_final(self, session: BeamSearchV1Session) -> BeamBatchResult:
        try:
            result = _required_method(self.engine, "wait_gr_session")(
                session.session_id, session.handle
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, BeamBatchResult):
                raise TypeError("EngineCore returned a non-BeamBatchResult terminal value")
            session.terminal = True
            session.failed = result.failure is not None
            return result
        except Exception as exc:  # noqa: BLE001 - EngineCore failure boundary
            session.failed = True
            return _failure_result(session.batch_id, "execution_failed", exc)

    def build_output(self, call: OnlineBeamSearchV1Call, result: BeamBatchResult) -> list[Any]:
        _raise_failure(result)
        from vllm.outputs import CompletionOutput, RequestOutput

        tokenizer = self.serving.input_processor.tokenizer
        end = _token_id(tokenizer, getattr(call.params, "end_token", None), "end_token")
        begin = [] if self._begin_token_id is None else [self._begin_token_id]
        prompt = self._prompt_tokens if not begin else self._prompt_tokens[:-1]
        completions = []
        for item in result.output.items:
            for sequence in item.sequences:
                tokens = begin + list(sequence.token_ids) + ([] if end is None else [end])
                completions.append(
                    CompletionOutput(
                        index=len(completions),
                        text=tokenizer.decode(tokens),
                        token_ids=tokens,
                        cumulative_logprob=sequence.cumulative_logprob,
                        logprobs=None,
                        finish_reason=sequence.finish_reason,
                        stop_reason=sequence.stop_token_id,
                    )
                )
        return [
            RequestOutput(
                request_id=call.request_id,
                prompt=tokenizer.decode(prompt),
                prompt_token_ids=prompt,
                prompt_logprobs=None,
                outputs=completions,
                finished=True,
            )
        ]

    async def abort(self, session: BeamSearchV1Session) -> None:
        if session.terminal or session.closed:
            return
        await self._abort_id(session.session_id, required=True)
        session.failed = True

    async def _abort_id(self, session_id: str, *, required: bool = False) -> None:
        method = getattr(self.engine, "abort_gr_session", None)
        if method is None:
            if required:
                _required_method(self.engine, "abort_gr_session")
            return
        try:
            result = method(session_id)
            if inspect.isawaitable(result):
                await result
        except Exception:
            if required:
                raise

    async def close(self, session: BeamSearchV1Session) -> None:
        if session.closed:
            return
        try:
            if session.terminal:
                method = getattr(self.engine, "release_gr_session", None)
                if method is not None:
                    result = method(session.session_id)
                    if inspect.isawaitable(result):
                        await result
            else:
                await self.abort(session)
        finally:
            session.closed = True


__all__ = [
    "AsyncBeamSearchV1ServerImpl",
    "AsyncGRSessionClient",
    "BeamSearchV1BackendError",
    "BeamSearchV1RequestError",
    "OfflineBeamSearchV1ServerImpl",
    "SyncGRSessionClient",
    "new_gr_session_id",
]
