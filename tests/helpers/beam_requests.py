# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam batch requests shared by tests across components.

Scheduler, entrypoint, constraint, and kernel tests all need a valid
``BeamBatchRequest``. Keeping the builders here lets each suite import them
without depending on the scheduler test module that first needed them.
"""

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
)


def make_batch_request(max_tokens: int = 3) -> BeamBatchRequest:
    request = EngineCoreRequest(
        request_id="session-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="session-0",
        items=(BeamBatchItem(item_index=0, request=request),),
        beam_params=BeamBatchParams(
            beam_width=4,
            max_tokens=max_tokens,
            temperature=0.0,
            stop_token_ids=(2,),
            ignore_stop_tokens=True,
        ),
        output_options=BeamOutputOptions(
            num_return_sequences=4,
            include_stop_token=True,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )


def make_multi_batch_request(max_tokens: int = 3) -> BeamBatchRequest:
    first = make_batch_request(max_tokens)
    second = EngineCoreRequest(
        request_id="session-0:1",
        prompt_token_ids=[4, 5],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=first.protocol_version,
        batch_id=first.batch_id,
        items=(first.items[0], BeamBatchItem(item_index=1, request=second)),
        beam_params=first.beam_params,
        output_options=first.output_options,
    )
