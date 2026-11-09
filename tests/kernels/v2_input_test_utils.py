# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native worker and Prefill metadata builders for V2 state and input tests."""

# mypy: disable-error-code="no-untyped-def,no-untyped-call"

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch

from vllm_gr.v1.beam.batch_contracts import BeamBatchParams, BeamOutputOptions
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRStageKind,
    GRStageMetadata,
)


def make_prefill_metadata() -> GRStageMetadata:
    return GRStageMetadata(
        session_id="session-0",
        dispatch_id=0,
        stage_index=0,
        kind=GRStageKind.PREFILL,
        decode_step=None,
        is_last_stage=False,
        input_device_state=None,
        output_device_state=GRDeviceStateRef("session-0", 0),
        beam_params=BeamBatchParams(
            beam_width=4,
            max_tokens=3,
            temperature=0.0,
            stop_token_ids=(),
            ignore_stop_tokens=True,
        ),
        output_options=BeamOutputOptions(
            num_return_sequences=4,
            include_stop_token=True,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
        num_prompt_tokens=7,
        produces_output=True,
        request_ids=("session-0",),
        num_prompt_tokens_by_request=(7,),
    )


def make_native_v2_worker(prompt: list[int], computed: int = 0, capacity: int = 2) -> Any:
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.states import RequestState

    worker: Any = GPUModelRunner.__new__(GPUModelRunner)
    worker.device = torch.device("cuda", torch.accelerator.current_device_index())
    worker.max_num_reqs = capacity
    worker.use_dcp = False
    worker.input_buffers = InputBuffers(capacity, 64, worker.device)
    worker.req_states = RequestState(capacity, 16, 64, 0, 512, worker.device)
    worker.req_states.add_request("session-0", len(prompt), prompt, computed, max_tokens=3)
    worker.req_states.apply_staged_writes()
    worker.block_tables = SimpleNamespace(apply_staged_writes=Mock())
    worker.encoder_cache = None
    worker.prompt_logprobs_worker = None
    worker.lora_state = SimpleNamespace(remove_request=Mock())
    worker.sampler = None
    worker.pooling_runner = None
    worker.pp_handler = None
    worker.adaptive_verification = None
    worker.fast_prefill = None
    worker.pcp_manager = None
    worker.decode_query_len = 1
    worker.model_config = SimpleNamespace(rswa_window=None)
    worker.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1, remove_request=Mock())
    worker.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    return worker
