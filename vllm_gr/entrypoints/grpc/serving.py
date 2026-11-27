# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR-only protobuf conversion and beam-session dispatch."""

import asyncio
import itertools
import time
import uuid

import grpc
import numpy as np
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind

from vllm_gr.config import GRConfig
from vllm_gr.entrypoints.beam_search.client import AsyncBeamSearchClient
from vllm_gr.entrypoints.beam_search.request_params import apply_beam_request_defaults
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam.batch_contracts import (
    ADMISSION_REJECTED_CODE,
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamEngineCoreRequest,
    BeamOutputOptions,
    BeamSequenceColumns,
    sequence_columns,
)
from vllm_gr.v1.engine.beam_admission import (
    validate_gr_generation_budget,
    validate_gr_request,
)

from .generated import vllm_engine_pb2 as pb


class BeamVllmEngineServicer(VllmEngineServicer):
    """GR generation with upstream health and model/server metadata RPCs."""

    # Accept only explicitly mapped or neutral fields; reject everything else.
    MAPPED = frozenset(
        {"temperature", "max_tokens", "n", "ignore_eos", "seed", "truncate_prompt_tokens"}
    )
    TOKEN_OUTPUT_FLAGS = frozenset({"skip_special_tokens", "spaces_between_special_tokens"})
    NEUTRAL = {"top_p": 1.0, "repetition_penalty": 1.0}

    def __init__(self, engine, start_time):
        super().__init__(engine, start_time)
        self.client = AsyncBeamSearchClient(engine_client=engine)
        self._beam_tasks = {}

    def decode_tokens(self, tokenized):
        if tokenized.HasField("input_ids_i64_le"):
            if tokenized.input_ids:
                raise ValueError("input_ids and input_ids_i64_le are mutually exclusive")
            raw = tokenized.input_ids_i64_le
            if not raw or len(raw) % 8:
                raise ValueError(
                    "input_ids_i64_le must be nonempty with byte length divisible by 8"
                )
            tokens = np.frombuffer(raw, dtype="<i8")
        else:
            # upb's native __array__ conversion avoids an intermediate Python list.
            # This still copies into NumPy-owned storage, unlike the binary view.
            tokens = np.asarray(tokenized.input_ids, dtype=np.int64)
        return tokens

    def build_batch(self, messages):
        if not messages:
            raise ValueError("requests must not be empty")
        config = getattr(self.engine, "vllm_config", None)
        gr_config = GRConfig.from_vllm_config(config) if config is not None else None
        params_key = None
        members, identities = [], []
        batch_id = uuid.uuid4().hex
        for index, message in enumerate(messages):
            if not message.HasField("beam_search"):
                raise ValueError("GenerateBatch requires explicit beam_search on every member")
            if message.WhichOneof("input") != "tokenized":
                raise ValueError("GR beam requires tokenized input")
            if message.stream:
                raise ValueError("incremental stream=true is not supported by terminal GR sessions")
            if message.HasField("mm_inputs") or message.HasField("kv_transfer_params"):
                raise ValueError("GR beam does not support mm_inputs or kv_transfer_params")
            source = message.sampling_params
            for field, value in source.ListFields():
                if field.name in self.MAPPED or field.name in self.TOKEN_OUTPUT_FLAGS:
                    continue
                if field.name in self.NEUTRAL and value == self.NEUTRAL[field.name]:
                    continue
                raise ValueError(f"sampling_params.{field.name} is not supported by GR beam")
            if source.temperature != 0 or not source.ignore_eos:
                raise ValueError("GR beam requires temperature=0 and ignore_eos=true")
            if not source.HasField("max_tokens") or source.max_tokens < 1:
                raise ValueError("sampling_params.max_tokens must be explicit and positive")
            if gr_config is not None:
                validate_gr_generation_budget(gr_config, source.max_tokens)
            options = message.beam_search
            width = options.beam_width or source.n or 1
            if source.n and source.n != width:
                raise ValueError("sampling_params.n must match beam_search.beam_width")
            params = apply_beam_request_defaults(
                config,
                BeamSearchParams(
                    beam_width=width,
                    max_tokens=source.max_tokens,
                    dynamic_beam_widths=list(options.dynamic_beam_widths) or None,
                    result_width=options.result_width if options.HasField("result_width") else None,
                ),
            )
            final_width = params.dynamic_beam_widths[-1] if params.dynamic_beam_widths else width
            beam = BeamBatchParams(
                beam_width=width,
                max_tokens=source.max_tokens,
                temperature=0.0,
                stop_token_ids=(),
                ignore_stop_tokens=True,
                constraint_resource_id=options.constraint_resource_id or "default",
                length_penalty=options.length_penalty
                if options.HasField("length_penalty")
                else 1.0,
                dynamic_beam_widths=tuple(params.dynamic_beam_widths)
                if params.dynamic_beam_widths is not None
                else None,
                per_parent_beam_widths=tuple(params.per_parent_beam_widths)
                if params.per_parent_beam_widths is not None
                else None,
                candidate_width=gr_config.beam_candidate_width if gr_config is not None else None,
            )
            output = BeamOutputOptions(
                num_return_sequences=final_width
                if params.result_width is None
                else params.result_width,
                include_stop_token=False,
                include_cumulative_logprob=True,
                include_parent_beam_indices=options.include_parent_beam_indices,
            )
            key = (beam, output)
            if params_key is not None and key != params_key:
                raise ValueError("members of one GR batch must have identical beam/output options")
            params_key = key
            tokens = self.decode_tokens(message.tokenized)
            tokenizer = self.engine.input_processor.tokenizer
            if source.HasField("truncate_prompt_tokens"):
                length = source.truncate_prompt_tokens
                if length == -1:
                    length = self.engine.model_config.max_model_len - source.max_tokens
                    if length <= 0:
                        raise ValueError("prompt plus generation exceeds model length")
                elif length <= 0:
                    raise ValueError("truncate_prompt_tokens must be positive or -1")
                if tokenizer is not None and tokenizer.truncation_side == "left":
                    tokens = tokens[-length:]
                else:
                    tokens = tokens[:length]
            if not tokens.size or tokens.min() < 0:
                raise ValueError("token IDs must be nonempty and nonnegative")
            # Match native InputProcessor: tokenizer-only IDs may exceed the
            # model vocabulary; skip this upper-bound check without a tokenizer.
            if tokenizer is not None:
                max_token_id = max(
                    tokenizer.max_token_id, self.engine.model_config.get_vocab_size() - 1
                )
                if tokens.max() > max_token_id:
                    raise ValueError("token IDs are out of vocabulary")
            if len(tokens) + source.max_tokens > self.engine.model_config.max_model_len:
                raise ValueError("prompt plus generation exceeds model length")
            # Beam fanout belongs to BeamBatchParams, not native sampling n.
            native_params = SamplingParams(
                temperature=0.0,
                max_tokens=source.max_tokens,
                ignore_eos=True,
                detokenize=False,
                seed=source.seed if source.HasField("seed") else None,
                skip_special_tokens=source.skip_special_tokens,
                spaces_between_special_tokens=source.spaces_between_special_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            identity = message.request_id or uuid.uuid4().hex
            identities.append(identity)
            members.append(
                BeamBatchItem(
                    item_index=index,
                    trie_id=message.trie_id,
                    request=BeamEngineCoreRequest(
                        request_id=batch_id if index == 0 else f"{batch_id}-{index}",
                        external_req_id=identity,
                        prompt_token_ids=tokens,
                        mm_features=None,
                        sampling_params=native_params,
                        pooling_params=None,
                        arrival_time=time.time(),
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                    ),
                )
            )
        if len(set(identities)) != len(identities):
            raise ValueError("request_id must be unique within a batch")
        assert params_key is not None
        request = BeamBatchRequest(
            protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
            batch_id=batch_id,
            items=tuple(members),
            beam_params=params_key[0],
            output_options=params_key[1],
        )
        validate_gr_request(request, config=config)
        return request, identities

    def build_response(
        self, request, identities, result, response_batch_id="", *, compact=True, binary=False
    ):
        response = pb.BatchGenerateResponse(batch_id=response_batch_id or request.batch_id)
        for identity, source, item in zip(
            identities, request.items, result.output.items, strict=True
        ):
            target = response.items.add(request_id=identity)
            sequences = item.sequences
            if isinstance(sequences, BeamSequenceColumns):
                if compact and binary:
                    beams = target.binary_beams
                    beams.count = sequences.count
                    beams.steps = sequences.steps
                    beams.prompt_tokens = len(source.request.prompt_token_ids)
                    beams.output_ids_i32_le = sequences.token_ids
                    if sequences.cumulative_logprobs is not None:
                        beams.cumulative_logprobs_f32_le = sequences.cumulative_logprobs
                    if sequences.parent_beam_indices is not None:
                        beams.parent_beam_indices_i32_le = sequences.parent_beam_indices
                    continue
                if not compact:
                    raise TypeError("non-compact replies require materialized Beam sequences")
                tokens, scores, parents = sequence_columns(sequences)
                count, steps = sequences.count, sequences.steps
                beams = target.beams
                beams.prompt_tokens = len(source.request.prompt_token_ids)
                beams.output_ids.extend(tokens.reshape(-1).tolist())
                beams.output_offsets.extend(range(0, count * steps + 1, steps))
                beams.cumulative_logprobs.extend(
                    itertools.repeat(0.0, count) if scores is None else scores.tolist()
                )
                beams.has_cumulative_logprob.extend(itertools.repeat(scores is not None, count))
                beams.finish_reasons.extend(itertools.repeat("length", count))
                if parents is not None:
                    beams.parent_beam_indices.extend(parents.reshape(-1).tolist())
                beams.parent_offsets.extend(
                    itertools.repeat(0, count + 1)
                    if parents is None
                    else range(0, count * steps + 1, steps)
                )
                beams.stop_token_ids.extend(itertools.repeat(-1, count))
                continue
            if compact:
                beams = target.beams
                beams.prompt_tokens = len(source.request.prompt_token_ids)
                beams.output_ids.extend(
                    itertools.chain.from_iterable(s.token_ids for s in sequences)
                )
                beams.output_offsets.extend(
                    itertools.accumulate((len(s.token_ids) for s in sequences), initial=0)
                )
                beams.cumulative_logprobs.extend(
                    s.cumulative_logprob if s.cumulative_logprob is not None else 0.0
                    for s in sequences
                )
                beams.has_cumulative_logprob.extend(
                    s.cumulative_logprob is not None for s in sequences
                )
                beams.finish_reasons.extend(s.finish_reason for s in sequences)
                beams.parent_beam_indices.extend(
                    itertools.chain.from_iterable(s.parent_beam_indices for s in sequences)
                )
                beams.parent_offsets.extend(
                    itertools.accumulate((len(s.parent_beam_indices) for s in sequences), initial=0)
                )
                beams.stop_token_ids.extend(
                    s.stop_token_id if s.stop_token_id is not None else -1 for s in sequences
                )
            else:
                for index, sequence in enumerate(sequences):
                    complete = target.responses.add().complete
                    complete.output_ids.extend(sequence.token_ids)
                    complete.finish_reason = sequence.finish_reason
                    complete.prompt_tokens = len(source.request.prompt_token_ids)
                    complete.completion_tokens = len(sequence.token_ids)
                    complete.index = index
                    complete.parent_beam_indices.extend(sequence.parent_beam_indices)
                    if sequence.cumulative_logprob is not None:
                        complete.cumulative_logprob = sequence.cumulative_logprob
                    if sequence.stop_token_id is not None:
                        complete.matched_token_id = sequence.stop_token_id
        return response

    async def _execute(
        self, messages, context, response_batch_id="", *, compact=True, binary=False
    ):
        try:
            request, identities = self.build_batch(messages)
            if any(identity in self._beam_tasks for identity in identities):
                raise ValueError("request_id already has an active beam session")
        except (VLLMValidationError, ValueError, TypeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        task = asyncio.current_task()
        for identity in identities:
            self._beam_tasks[identity] = task
        session = None
        try:
            try:
                session = await self.client.submit_once(request)
                result = await self.client.wait_final(session, materialize=not compact)
            except VLLMValidationError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                raise
            except Exception as exc:
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
                raise
            if result.failure is not None:
                status = (
                    grpc.StatusCode.INVALID_ARGUMENT
                    if result.failure.code == ADMISSION_REJECTED_CODE
                    else grpc.StatusCode.INTERNAL
                )
                await context.abort(status, result.failure.message)
            return self.build_response(
                request, identities, result, response_batch_id, compact=compact, binary=binary
            )
        except asyncio.CancelledError:
            await context.abort(grpc.StatusCode.CANCELLED, "GR beam session cancelled")
            raise
        finally:
            try:
                if session is not None:
                    await self.client.close(session)
            finally:
                for identity in identities:
                    self._beam_tasks.pop(identity, None)

    async def Generate(self, request, context):
        if not request.HasField("beam_search"):
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "Only GR beam generation is supported"
            )
            return
        result = await self._execute([request], context, compact=False)
        for response in result.items[0].responses:
            yield response

    async def GenerateBatch(self, request, context):
        return await self._execute(
            request.requests, context, request.batch_id, binary=request.binary_beam_output
        )

    async def Embed(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Embedding is not supported by GR")

    async def Abort(self, request, context):
        # A member belongs to an atomic session: aborting it cancels the batch.
        tasks = {self._beam_tasks[rid] for rid in request.request_ids if rid in self._beam_tasks}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return pb.AbortResponse()
