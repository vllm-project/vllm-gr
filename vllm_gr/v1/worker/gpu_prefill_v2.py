# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native V2 GR Prefill registration, fused state writes and input preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import BatchReqState


def _can_fuse_prefill_state_writes(
    worker: Any, scheduler_output: "SchedulerOutput", requests: list["NewRequestData"]
) -> bool:
    """Check the complete fresh-batch write contract before registration."""
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.states import RequestState

    states = worker.req_states
    blocks = worker.block_tables
    if (
        worker.device.type != "cuda"
        or worker.use_pp
        or worker.use_dcp
        or worker.dp_size != 1
        or worker.parallel_config.tensor_parallel_size != 1
        or worker.parallel_config.prefill_context_parallel_size != 1
        or worker.supports_mm_inputs
        or worker.is_encoder_decoder
        or type(worker.model_state) is not DefaultModelState
        or worker.model_state.rope_state is not None
        or type(states) is not RequestState
        or type(blocks) is not BlockTables
        or states.num_reqs != 0
        or scheduler_output.scheduled_cached_reqs.req_ids
        or scheduler_output.scheduled_spec_decode_tokens
        or scheduler_output.scheduled_encoder_inputs
        or scheduler_output.has_structured_output_requests
        or blocks.num_kv_cache_groups != 1
        or blocks.blocks_per_kv_block != [1]
    ):
        return False
    for request in requests:
        if request.num_computed_tokens != 0 or len(request.block_ids) != 1:
            return False
        num_tokens = len(request.prefill_token_ids)
        num_blocks = len(request.block_ids[0])
        if (
            num_tokens != len(request.prompt_token_ids)
            or num_tokens != scheduler_output.num_scheduled_tokens[request.req_id]
            or not 0 < num_tokens <= states.all_token_ids.gpu.shape[1]
            or not 0 < num_blocks <= blocks.block_tables[0].gpu.shape[1]
        ):
            return False
    # The fused flush must own every queued write. A pending update from any
    # other dispatch keeps the existing per-tensor flush path.
    return not any(
        tensor._staged_write_indices
        for tensor in (
            states.total_len,
            states.all_token_ids,
            states.num_computed_tokens,
            blocks.block_tables[0],
        )
    )


def _apply_fused_prefill_state_writes(worker: Any) -> None:
    """Flush native registration writes with one H2D and one ordered kernel."""
    from vllm_gr.ops.cuda.gr_prefill_inputs import (
        apply_gr_prefill_batch_state_writes_v2,
        apply_gr_prefill_state_writes_v2,
    )

    states = worker.req_states
    block_table = worker.block_tables.block_tables[0]
    tokens = states.all_token_ids._staged_write_contents
    block_ids = block_table._staged_write_contents
    from vllm_gr.v1.worker.numpy_staging import NumpyStagedContents, array_h2d

    token_array = tokens.array() if isinstance(tokens, NumpyStagedContents) else np.asarray(tokens)
    count = len(states.total_len._staged_write_indices)
    targets = (
        states.total_len.gpu,
        states.all_token_ids.gpu,
        states.num_computed_tokens.gpu,
        block_table.gpu,
    )
    strides = dict(
        TOKEN_STRIDE=states.all_token_ids.gpu.stride(0),
        BLOCK_TABLE_STRIDE=block_table.gpu.stride(0),
        BLOCK=256,
    )
    if count == 1:
        req_index = states.total_len._staged_write_indices[0]
        payload: npt.NDArray[np.int32] = np.empty(2 + len(tokens) + len(block_ids), dtype=np.int32)
        payload[0] = states.total_len._staged_write_contents[0]
        payload[1] = states.num_computed_tokens._staged_write_contents[0]
        payload[2 : 2 + len(tokens)] = token_array
        payload[2 + len(tokens) :] = block_ids
        packed = array_h2d(payload, torch.int32, worker.device)
        apply_gr_prefill_state_writes_v2[((max(len(tokens), len(block_ids)) + 255) // 256,)](
            packed, *targets, req_index, len(tokens), len(block_ids), **strides
        )
    else:
        # Registration appends one start-zero write per tensor and native slot.
        # Pack offsets with the contents so the GPU needs no per-tensor UVA
        # descriptors. Each row owns a distinct slot, including after reuse.
        # Header columns: slot, total length, computed tokens, token offset,
        # token count, block offset and block count. The kernel uses this ABI.
        header_size = 7 * count
        payload = np.empty(header_size + len(tokens) + len(block_ids), dtype=np.int32)
        header = payload[:header_size].reshape(count, 7)
        header[:, 0] = states.total_len._staged_write_indices
        header[:, 1] = states.total_len._staged_write_contents
        header[:, 2] = states.num_computed_tokens._staged_write_contents
        token_ends = np.asarray(states.all_token_ids._staged_write_cu_lens, dtype=np.int32)
        block_ends = np.asarray(block_table._staged_write_cu_lens, dtype=np.int32)
        header[0, 3] = header_size
        header[1:, 3] = header_size + token_ends[:-1]
        header[:, 4] = header_size + token_ends - header[:, 3]
        header[0, 5] = header_size + len(tokens)
        header[1:, 5] = header_size + len(tokens) + block_ends[:-1]
        header[:, 6] = header_size + len(tokens) + block_ends - header[:, 5]
        payload[header_size : header_size + len(tokens)] = token_array
        payload[header_size + len(tokens) :] = block_ids
        extent = max(int(header[:, 4].max()), int(header[:, 6].max()))
        packed = array_h2d(payload, torch.int32, worker.device)
        apply_gr_prefill_batch_state_writes_v2[(count, (extent + 255) // 256)](
            packed, *targets, **strides
        )
    # Clear only after successful submission. The native block-table flush
    # still follows update_requests and refreshes num_blocks, but must not
    # submit the same block IDs twice. CUDA errors propagate normally.
    for tensor in (
        states.total_len,
        states.all_token_ids,
        states.num_computed_tokens,
        block_table,
    ):
        tensor.clear_staged_writes()


def try_add_gr_prefill_requests(worker: Any, scheduler_output: "SchedulerOutput") -> bool:
    """Register an admitted GR Prefill without unused native sampler state.

    Keep the native request/model/KV lifecycle, including slot reuse. GR owns
    sampling after this forward, so native sampling parameters are neither
    initialized nor flushed. Decline before mutation for ordinary or mixed
    batches and for features requiring the native sampling path.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    current_output = getattr(worker, "_current_scheduler_output", None)
    if not stages or current_output is None or current_output is not scheduler_output:
        return False
    new_reqs = scheduler_output.scheduled_new_reqs
    if len(stages) != 1 or not new_reqs:
        return False
    stage = next(iter(stages.values()))
    request_ids = {request.req_id for request in new_reqs}
    if (
        stage.kind is not GRStageKind.PREFILL
        or len(request_ids) != len(new_reqs)
        or request_ids != set(stage.request_ids)
        or request_ids != scheduler_output.num_scheduled_tokens.keys()
        or scheduler_output.scheduled_cached_reqs.req_ids
        or scheduler_output.scheduled_spec_decode_tokens
        or scheduler_output.scheduled_encoder_inputs
        or scheduler_output.has_structured_output_requests
        or worker.encoder_cache is not None
        or worker.lora_config is not None
        or worker.speculative_config is not None
        or worker.is_pooling_model
        or not worker.is_first_pp_rank
        or not worker.is_last_pp_rank
    ):
        return False
    # Validate every member before removing or registering any native slot.
    # Partial/new-plus-cached batches retain native registration.
    for request in new_reqs:
        params = request.sampling_params
        if (
            scheduler_output.num_scheduled_tokens[request.req_id] <= 0
            or request.prompt_token_ids is None
            or request.prefill_token_ids is None
            or request.mm_features
            or request.lora_request is not None
            or params is None
            or params.prompt_logprobs is not None
        ):
            return False

    fuse_writes = _can_fuse_prefill_state_writes(worker, scheduler_output, new_reqs)
    # Keep native CPU registration and slot lifecycle. The optional fused
    # flush writes the same native state; other GR Prefills retain individual
    # staged writes. Native sampler initialization remains unnecessary here.
    for request in new_reqs:
        worker._remove_request(request.req_id)
        worker.req_states.add_request(
            req_id=request.req_id,
            prompt_len=len(request.prompt_token_ids),
            all_token_ids=request.prefill_token_ids,
            num_computed_tokens=request.num_computed_tokens,
            max_tokens=request.sampling_params.max_tokens,
        )
        req_index = worker.req_states.req_id_to_index[request.req_id]
        worker.model_state.add_request(req_index, request)
        worker.block_tables.append_block_ids(req_index, request.block_ids, overwrite=True)
        worker.lora_state.add_request(request.req_id, req_index, request.lora_request)
        # Reset per-slot flags even when the reused slot previously requested
        # prompt logprobs. This worker is not used by GR sampling itself.
        assert worker.prompt_logprobs_worker is not None
        worker.prompt_logprobs_worker.add_request(
            request.req_id, req_index, request.sampling_params
        )
    if fuse_writes:
        worker.req_states.prompt_len.copy_to_uva()
        worker.req_states.prefill_len.copy_to_uva()
        _apply_fused_prefill_state_writes(worker)
    else:
        worker.req_states.apply_staged_writes()
    worker.model_state.apply_staged_writes()
    return True


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
