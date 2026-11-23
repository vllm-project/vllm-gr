# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Guarded single-request GR Prefill inputs for native ModelRunner V1."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR, GRStageKind


def try_prepare_gr_prefill_inputs(
    runner: Any, scheduler_output: Any, num_scheduled_tokens: NDArray[np.int32]
) -> torch.Tensor | None:
    """Return logits indices, or decline without changing any runner state.

    The native execute_model input-prep fence still encloses this function.
    Both CPU mirrors and fixed-address GPU buffers retain their native ABI.
    """
    stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
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
