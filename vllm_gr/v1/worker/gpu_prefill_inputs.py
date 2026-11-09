# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded GR Prefill inputs for native V1 single-request and V2 batch contracts."""

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

    This hook specializes InputBatch construction. Request state, KV updates,
    attention preparation and computed-token updates retain native ownership.
    Graph input addresses are the native persistent buffers.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    scheduled_tokens = scheduler_output.num_scheduled_tokens
    if (
        not stages
        or len(stages) != 1
        or not scheduled_tokens
        or getattr(scheduler_output, "beam_data", None)
    ):
        return None
    stage = next(iter(stages.values()))
    if stage.kind is not GRStageKind.PREFILL or set(stage.request_ids) != scheduled_tokens.keys():
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
    if len(scheduled_tokens) != 1 or batch_desc.num_reqs not in (None, 1):
        return _try_prepare_gr_prefill_batch_inputs_v2(
            runner, scheduler_output, batch_req_state, batch_desc, stage
        )
    req_id, scheduled = next(iter(scheduled_tokens.items()))
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


def _try_prepare_gr_prefill_batch_inputs_v2(
    runner: Any,
    scheduler_output: Any,
    batch_req_state: BatchReqState,
    batch_desc: BatchExecutionDescriptor,
    stage: Any,
) -> InputBatch | None:
    """Fuse pure Prefill rows in native order, including reused request slots."""
    req_ids = batch_req_state.req_ids
    num_reqs = len(req_ids)
    padded_reqs = batch_desc.num_reqs or num_reqs
    scheduled = batch_req_state.num_scheduled_tokens
    num_tokens = batch_req_state.num_tokens
    padded_tokens = batch_desc.num_tokens
    states = runner.req_states
    idx = batch_req_state.idx_mapping_np
    prompt_counts = dict(zip(stage.request_ids, stage.num_prompt_tokens_by_request))
    if (
        not 0 < num_reqs <= padded_reqs <= runner.max_num_reqs
        or len(set(req_ids)) != num_reqs
        or set(req_ids) != scheduler_output.num_scheduled_tokens.keys()
        or len(idx) != num_reqs
        or len(scheduled) != num_reqs
        or any(
            states.req_id_to_index.get(req_id) != idx[i]
            or scheduler_output.num_scheduled_tokens[req_id] != scheduled[i]
            or prompt_counts.get(req_id) != states.prompt_len.np[idx[i]]
            for i, req_id in enumerate(req_ids)
        )
        or np.any(scheduled <= 0)
        or num_tokens != scheduler_output.total_num_scheduled_tokens
        or num_tokens != int(scheduled.sum())
        or not num_tokens <= padded_tokens <= runner.max_num_tokens
    ):
        return None
    computed = states.num_computed_tokens_np[idx]
    ends = computed + scheduled
    if np.any(computed < 0) or np.any(ends > states.prompt_len.np[idx]):
        return None

    from vllm import envs
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_gr.ops.cuda.gr_prefill_inputs import prepare_gr_prefill_batch_inputs_v2
    from vllm_gr.v1.worker.numpy_staging import array_h2d

    # Pack only the small row descriptors; tokens stay in native GPU/UVA state.
    # Native gather_batch_req_state owns ordering, independently of slot IDs or
    # logical Beam member order. Keep host mirrors for native attention code.
    payload = np.empty(2 * num_reqs + 1, dtype=np.int32)
    payload[:num_reqs] = idx
    payload[num_reqs] = 0
    np.cumsum(scheduled, out=payload[num_reqs + 1 :])
    descriptors = array_h2d(payload, torch.int32, runner.device)
    query_start_loc_np = np.full(padded_reqs + 1, num_tokens, dtype=np.int32)
    query_start_loc_np[: num_reqs + 1] = payload[num_reqs:]
    seq_lens_cpu = np.zeros(padded_reqs, dtype=np.int32)
    seq_lens_cpu[:num_reqs] = ends
    # Fresh sampling indices survive asynchronous consumption of this batch.
    idx_mapping = torch.empty(num_reqs, dtype=torch.int64, device=runner.device)
    batch_indices = torch.empty(2 * num_reqs + 1, dtype=torch.int32, device=runner.device)
    logits_indices = torch.empty(num_reqs, dtype=torch.int64, device=runner.device)
    buffers = runner.input_buffers
    grid_tokens = max(int(scheduled.max()), runner.max_num_reqs + 1, padded_tokens - num_tokens)
    prepare_gr_prefill_batch_inputs_v2[(num_reqs, (grid_tokens + 255) // 256)](
        buffers.input_ids,
        buffers.positions,
        buffers.query_start_loc,
        buffers.seq_lens,
        buffers.is_padding,
        states.all_token_ids.gpu,
        states.num_computed_tokens.gpu,
        states.prefill_len.gpu,
        states.next_prefill_tokens,
        descriptors,
        idx_mapping,
        batch_indices,
        logits_indices,
        num_tokens,
        padded_tokens,
        NUM_REQS=num_reqs,
        TOKEN_STRIDE=states.all_token_ids.gpu.stride(0),
        MAX_REQS=runner.max_num_reqs,
        WRITE_PADDING=envs.VLLM_MOE_SKIP_PADDING,
        BLOCK=256,
    )
    return InputBatch(
        req_ids=req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=padded_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx,
        expanded_idx_mapping=idx_mapping,
        expanded_local_pos=batch_indices[num_reqs + 1 :],
        num_scheduled_tokens=scheduled,
        num_tokens=num_tokens,
        num_tokens_after_padding=padded_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=buffers.query_start_loc[: padded_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:padded_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=computed,
        prefill_len_np=batch_req_state.prefill_len_np,
        num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
        is_prefilling_np=batch_req_state.is_prefilling_np,
        has_prefill=batch_req_state.has_prefill,
        input_ids=buffers.input_ids[:padded_tokens],
        positions=buffers.positions[:padded_tokens],
        is_padding=buffers.is_padding[:padded_tokens],
        logits_indices=logits_indices,
        cu_num_logits=batch_indices[: num_reqs + 1],
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
        max_query_len=batch_desc.max_query_len,
    )
