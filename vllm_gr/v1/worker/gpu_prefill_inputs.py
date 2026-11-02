# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded single-request GR Prefill inputs for the native V1/V2 contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import BatchReqState


def try_prepare_gr_prefill_inputs(
    runner: Any, scheduler_output: Any, num_scheduled_tokens: NDArray[np.int32]
) -> torch.Tensor | None:
    """Return logits indices, or decline without changing any runner state.

    The native execute_model input-prep fence still encloses this function.
    Both CPU mirrors and fixed-address GPU buffers retain their native ABI.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    if not stages or len(stages) != 1 or getattr(scheduler_output, "beam_data", None):
        return None
    batch = runner.input_batch
    if batch.num_reqs != 1 or len(num_scheduled_tokens) != 1:
        return None
    req_id = batch.req_ids[0]
    stage = stages.get(req_id)
    if stage is None or stage.kind is not GRStageKind.PREFILL:
        return None
    if (
        runner.device.type != "cuda"
        # vLLM 0.30 represents all multi-dimensional RoPE through uses_mrope.
        or runner.uses_mrope
        or runner.enable_prompt_embeds
        or batch.req_prompt_embeds
        or runner.lora_config
        or runner.use_async_spec_decode
        or scheduler_output.scheduled_spec_decode_tokens
        or runner.num_accepted_tokens_event is not None
        or runner.mamba_prev_last_scheduled_idx is not None
        or batch.prev_sampled_token_ids is not None
        # Native input preparation waits for an outstanding PP token broadcast.
        or runner._pp_recv_work is not None
    ):
        return None
    scheduled = int(num_scheduled_tokens[0])
    computed = int(batch.num_computed_tokens_cpu[0])
    end = computed + scheduled
    if (
        scheduled <= 0
        or scheduled != scheduler_output.total_num_scheduled_tokens
        or scheduled > runner.max_num_tokens
        or computed < 0
        or end > batch.num_prompt_tokens[0]
        or stage.num_prompt_tokens != batch.num_prompt_tokens[0]
    ):
        return None

    from vllm_gr.ops.cuda.gr_prefill_inputs import prepare_gr_prefill_metadata

    batch.block_table.commit_block_table(1)
    # A single Prefill row is contiguous, including prefix-cache hits and
    # partial Prefill chunks. Avoid the general repeat/cumsum/index_select.
    runner.input_ids.np[:scheduled] = batch.token_ids_cpu[0, computed:end]
    runner.query_pos.np[:scheduled] = runner.arange_np[:scheduled]
    runner.req_indices.np[:scheduled] = 0
    runner.query_start_loc.np.fill(scheduled)
    runner.query_start_loc.np[0] = 0
    seq_lens_cpu = runner.optimistic_seq_lens_cpu.numpy()
    seq_lens_cpu.fill(0)
    seq_lens_cpu[0] = end
    runner._compute_prev_positions(1)
    num_tokens = runner.requests[req_id].num_tokens
    runner.discard_request_mask.np[0] = end < num_tokens
    runner.num_accepted_tokens.np.fill(1)
    runner.num_scheduled_tokens.np[0] = scheduled

    # Do not reuse logits_indices: native execute state can retain the result.
    logits_indices = torch.empty(1, dtype=torch.int32, device=runner.device)
    prepare_gr_prefill_metadata[((max(scheduled, runner.max_num_reqs + 1) + 255) // 256,)](
        runner.positions,
        runner.query_start_loc.gpu,
        runner.seq_lens,
        runner.num_computed_tokens,
        runner.req_indices.gpu,
        runner.query_pos.gpu,
        runner.num_scheduled_tokens.gpu,
        runner.discard_request_mask.gpu,
        runner.num_accepted_tokens.gpu,
        logits_indices,
        computed,
        scheduled,
        num_tokens,
        MAX_REQS=runner.max_num_reqs,
        BLOCK=256,
    )
    # Keep native block-table handling, including current block IDs, group
    # layout, slot padding and any supported context-parallel mapping.
    batch.block_table.compute_slot_mapping(
        1, runner.query_start_loc.gpu[:2], runner.positions[:scheduled]
    )
    runner.input_ids.copy_to_gpu(scheduled)
    return logits_indices


def try_prepare_gr_prefill_inputs_v2(
    runner: Any,
    scheduler_output: Any,
    batch_req_state: BatchReqState,
    batch_desc: BatchExecutionDescriptor,
) -> InputBatch | None:
    """Return the native V2 InputBatch, or decline before changing any state.

    Only input preparation is specialized. Native request registration, KV
    updates, attention preparation and native computed-token updates remain in charge
    of their state. Graph input addresses are the native persistent buffers.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    scheduled_tokens = scheduler_output.num_scheduled_tokens
    if (
        not stages
        or len(stages) != 1
        or len(scheduled_tokens) != 1
        or getattr(scheduler_output, "beam_data", None)
        or batch_desc.num_reqs not in (None, 1)
    ):
        return None
    req_id, scheduled = next(iter(scheduled_tokens.items()))
    stage = stages.get(req_id)
    if stage is None or stage.kind is not GRStageKind.PREFILL:
        return None
    if (
        runner.device.type != "cuda"
        or runner.use_pp
        or runner.use_dcp
        or runner.dp_size != 1
        or runner.supports_mm_inputs
        or runner.is_encoder_decoder
        or runner.is_pooling_model
        or runner.lora_config
        or runner.speculative_config
        or runner.fast_prefill is not None
        or runner.pcp_manager is not None
        or runner.model_config.rswa_window is not None
        or scheduler_output.scheduled_spec_decode_tokens
        or scheduler_output.scheduled_encoder_inputs
        or scheduler_output.has_structured_output_requests
    ):
        return None

    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    if (
        type(runner.model_state) is not DefaultModelState
        or runner.model_state.rope_state is not None
    ):
        return None
    states = runner.req_states
    req_index = states.req_id_to_index.get(req_id)
    if req_index is None:
        return None
    computed = int(states.num_computed_tokens_np[req_index])
    prompt = int(states.prompt_len.np[req_index])
    end = computed + scheduled
    padded = batch_desc.num_tokens
    if (
        scheduled <= 0
        or batch_req_state.req_ids != [req_id]
        or batch_req_state.num_tokens != scheduled
        or scheduled != scheduler_output.total_num_scheduled_tokens
        or not scheduled <= padded <= runner.max_num_tokens
        or computed < 0
        or end > prompt
        or stage.num_prompt_tokens != prompt
    ):
        return None

    from vllm_gr.ops.cuda.gr_prefill_inputs import prepare_gr_prefill_inputs_v2

    buffers = runner.input_buffers
    from vllm import envs

    if envs.VLLM_MOE_SKIP_PADDING:
        buffers.is_padding[:scheduled].fill_(False)
        buffers.is_padding[scheduled:padded].fill_(True)
    # One fused launch also writes the small index tensors, avoiding separate
    # arange/zeros kernels and index/query H2D copies. Keep native dtypes and
    # per-dispatch ownership instead of reusing mutable sampling indices.
    batch_indices = torch.empty(4, dtype=torch.int32, device=runner.device)
    logits_indices = torch.empty(1, dtype=torch.int64, device=runner.device)
    prepare_gr_prefill_inputs_v2[((max(scheduled, runner.max_num_reqs + 1) + 255) // 256,)](
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        states.all_token_ids.gpu,
        states.num_computed_tokens.gpu,
        states.prefill_len.gpu,
        states.next_prefill_tokens,
        batch_indices,
        logits_indices,
        req_index,
        scheduled,
        TOKEN_STRIDE=states.all_token_ids.gpu.stride(0),
        MAX_REQS=runner.max_num_reqs,
        BLOCK=256,
    )
    idx_mapping = batch_indices[:1]
    return InputBatch(
        req_ids=[req_id],
        num_reqs=1,
        num_reqs_after_padding=1,
        idx_mapping=idx_mapping,
        idx_mapping_np=batch_req_state.idx_mapping_np,
        expanded_idx_mapping=idx_mapping,
        expanded_local_pos=batch_indices[3:4],
        num_scheduled_tokens=batch_req_state.num_scheduled_tokens,
        num_tokens=scheduled,
        num_tokens_after_padding=padded,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=buffers.query_start_loc[:2],
        query_start_loc_np=np.array([0, scheduled], dtype=np.int32),
        seq_lens=buffers.seq_lens[:1],
        seq_lens_cpu_upper_bound=torch.from_numpy(np.array([end], dtype=np.int32)),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=states.num_computed_tokens_np[batch_req_state.idx_mapping_np],
        prefill_len_np=batch_req_state.prefill_len_np,
        num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
        is_prefilling_np=batch_req_state.is_prefilling_np,
        has_prefill=batch_req_state.has_prefill,
        input_ids=buffers.input_ids[:padded],
        positions=buffers.positions[:padded],
        is_padding=buffers.is_padding[:padded],
        logits_indices=logits_indices,
        cu_num_logits=batch_indices[1:3],
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
        max_query_len=batch_desc.max_query_len,
    )
