# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent, fixed-width CUDA Beam postprocessing for one active session."""

from __future__ import annotations

import torch
import triton

from vllm_gr.ops.cuda import beam_state as ops
from vllm_gr.ops.cuda.beam_kv_reorder import (
    _select_kv_gather_kernel,
    _select_kv_scatter_kernel,
)
from vllm_gr.ops.cuda.beam_selection import (
    _group_beam_outputs_kernel,
    _hierarchical_topk_out,
)
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState


class BeamSearchRunner:
    """Enqueue Beam decisions and same-step KV reorder on the current stream.

    The caller has already written this round's Decode KV at the device
    ``state.num_decode_steps`` position. This runner does not run a forward,
    read scheduling metadata, encode results, or allocate a KV pool. Only one
    runner may execute against the shared workspace/context at a time.

    Capture binds this session's KV slot, state row, workspace and fixed
    configuration. Candidate addresses must remain fixed across replay;
    eager callers may supply other tensors with the same validated layout.
    Consumers on another stream must wait on an event recorded after advance.
    """

    def __init__(
        self,
        session: BeamSearchSession,
        state: BeamSearchState,
        workspace: BeamPostprocessWorkspace,
        *,
        state_index: int = 0,
    ) -> None:
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            raise TypeError("state_index must be an integer")
        if not 0 <= state_index < state.max_batch_size:
            raise ValueError("state_index exceeds state capacity")
        context = session.context
        width = state.beam_width
        if state.device.type != "cuda":
            raise ValueError("BeamSearchRunner requires CUDA state")
        if state.decision is not context.decision:
            raise ValueError("state must alias this session's decision buffers")
        if not session.is_active or session.beam_width != width:
            raise ValueError("session must be active with the configured beam width")
        if (
            workspace.key.beam_width != width
            or workspace.key.top_k != width
            or workspace.max_batch < 1
            or workspace.topk_token_ids.device != state.device
            or workspace.max_decode_steps < state.sequence.shape[2]
        ):
            raise ValueError("workspace geometry/device does not match Beam state")
        if not context.pools:
            raise ValueError("session must provide at least one KV layer")
        first = context.pools[0].unshared_key
        if (
            first.ndim != 5
            or first.dtype not in (torch.float16, torch.bfloat16)
            or first.device != state.device
            or first.shape[1] < width
            or first.shape[3] < state.max_new_tokens - 1
            or not first.is_contiguous()
            or not 0 <= session.slot < first.shape[0]
        ):
            raise ValueError("KV pool capacity, dtype, device, or layout is incompatible")
        for pool in context.pools:
            for tensor in (pool.unshared_key, pool.unshared_value):
                if (
                    tensor.shape != first.shape
                    or tensor.stride() != first.stride()
                    or tensor.dtype != first.dtype
                    or tensor.device != first.device
                ):
                    raise ValueError("all KV layers must have identical layouts")
        layers = len(context.pools)
        aux = context.aux_key_buf
        expected_aux = (layers, first.shape[1], *first.shape[2:])
        for tensor in (aux, context.aux_value_buf):
            if (
                tuple(tensor.shape) != expected_aux
                or not tensor.is_contiguous()
                or tensor.dtype != first.dtype
                or tensor.device != first.device
            ):
                raise ValueError("KV reorder requires compatible per-layer scratch")
        for ptrs in (context.pool_key_ptrs, context.pool_value_ptrs):
            if (
                ptrs is None
                or ptrs.shape != (layers,)
                or ptrs.dtype != torch.int64
                or ptrs.device != first.device
                or not ptrs.is_contiguous()
            ):
                raise ValueError("KV reorder requires persistent layer pointer tables")

        self.session = session
        self.state = state
        self.workspace = workspace
        self.state_index = state_index
        self._closed = False
        # Construct device views only once. State and workspace rows need not
        # equal the KV slot; the workspace's first row serves this B=1 runner.
        self._state = {
            name: value[state_index : state_index + 1].view(*value.shape[1:])
            if value.ndim > 1
            else value[state_index : state_index + 1]
            for name, value in vars(state).items()
            if isinstance(value, torch.Tensor)
        }
        if workspace.state_scratch is None:
            workspace.state_scratch = ops.BeamStateScratch.allocate(
                workspace.max_batch, width, state.device
            )
        scratch = workspace.state_scratch
        self._scratch = {
            name: getattr(scratch, name)[0]
            if getattr(scratch, name).ndim > 1
            else getattr(scratch, name)[:1]
            for name in scratch.__dataclass_fields__
        }
        self._selected_tokens = workspace.selected_token_ids[0]
        self._selected_parents = workspace.selected_parent_ids[0]
        self._selected_scores = workspace.selected_scores[0]
        self._selected_flat = workspace.selected_flat_indices[0]
        self._next_sequence = workspace.next_sequence[0]
        self._kv_capacity = first.shape[3]
        self._kv_grid = (layers * width * first.shape[2],)
        self._kv_args = (
            context.pool_key_ptrs,
            context.pool_value_ptrs,
            aux.data_ptr(),
            context.aux_value_buf.data_ptr(),
            session.slot * first.stride(0) * first.element_size(),
            width,
            self._state["num_decode_steps"],
        )
        self._kv_kwargs = dict(
            H=first.shape[2],
            BLOCK_D=triton.next_power_of_2(first.shape[4]),
            D=first.shape[4],
            stride_pw=first.stride(1),
            stride_ph=first.stride(2),
            stride_ps=first.stride(3),
            stride_al=aux.stride(0),
            stride_aw=aux.stride(1),
            stride_ah=aux.stride(2),
            stride_as=aux.stride(3),
            active_mask=self._state["active_mask"],
            execute=self._scratch["reorder"],
            DEVICE_STEPS=True,
        )

    def initialize_from_prefill(
        self, token_ids: torch.Tensor, logprobs: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        """Initialize once from one GPU parent row; generated values stay on GPU."""
        self._advance(token_ids, logprobs, valid_mask, prefill=True)

    def advance(
        self, token_ids: torch.Tensor, logprobs: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        """Advance one Decode round; all runtime decisions are device-side."""
        self._advance(token_ids, logprobs, valid_mask, prefill=False)

    def reset(self) -> None:
        """Reuse this row after the caller has ordered all prior consumers.

        Same-stream consumers need no wait. For another stream, enqueue its
        completion event with current_stream().wait_event(...) before reset.
        Configuration and resource addresses remain unchanged.
        """
        if self._closed:
            raise RuntimeError("Beam runner's session has been released")
        self.state.reset(self.state_index)

    def end_session(self, consumer_done: torch.Event | None = None) -> None:
        """Release the existing slot only after device work and final consumers.

        This lifecycle boundary is outside graph capture and the steady path.
        Pass the final consumer's event when it ran on another stream.
        """
        if self._closed:
            return
        stream = torch.cuda.current_stream(self.state.device)
        if consumer_done is not None:
            stream.wait_event(consumer_done)
        done = torch.Event()
        done.record(stream)
        done.synchronize()
        self.session.end_session()
        self._closed = True

    def _advance(
        self,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        prefill: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError("Beam runner's session has been released")
        width = self.state.beam_width
        rows = 1 if prefill else width
        for name, tensor, dtype in (
            ("token_ids", token_ids, torch.int32),
            ("logprobs", logprobs, torch.float32),
            ("valid_mask", valid_mask, torch.bool),
        ):
            if (
                tensor.shape != (rows, width)
                or tensor.dtype != dtype
                or tensor.device != self.state.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CUDA {dtype} [{rows}, {width}]")
        s, w = self._state, self._scratch
        block = triton.next_power_of_2(width)
        ops._prepare_candidates_kernel[(rows,)](
            token_ids,
            logprobs,
            valid_mask,
            s["beam_scores"],
            s["active_mask"],
            s["num_generated_tokens"],
            s["max_new_tokens_tensor"],
            s["stop_token_ids"],
            s["num_stop_tokens"],
            s["ignore_stop_tokens"],
            w["parent_scores"],
            w["selection_mask"],
            w["terminal_ranks"],
            w["finish_reasons"],
            w["cumulative_scores"],
            w["row_counts"],
            w["row_errors"],
            w["row_valid_counts"],
            WIDTH=width,
            BLOCK=block,
            PREFILL=prefill,
        )
        ops._validate_step_kernel[(1,)](
            s["is_initialized"],
            s["is_finished"],
            s["error_code"],
            s["num_generated_tokens"],
            s["num_decode_steps"],
            s["max_new_tokens_tensor"],
            s["num_completed"],
            w["row_counts"],
            w["row_errors"],
            w["row_valid_counts"],
            w["row_offsets"],
            w["execute"],
            w["reorder"],
            w["next_completed"],
            ROWS=rows,
            BLOCK=block,
            CAPACITY=self.state.completed_capacity,
            KV_CAPACITY=self._kv_capacity,
            PREFILL=prefill,
        )
        selected_scores, selected_flat, stride = _hierarchical_topk_out(
            w["parent_scores"],
            logprobs,
            w["selection_mask"],
            1,
            rows,
            width,
            width,
            self.workspace.selection_scratch,
            mask_invalid_indices=True,
        )
        _group_beam_outputs_kernel[(1,)](
            selected_scores,
            selected_flat,
            token_ids,
            self._selected_tokens,
            self._selected_parents,
            self._selected_scores,
            self._selected_flat,
            width,
            stride,
            WRITE_FLAT_INDICES=True,
            BLOCK=block,
        )
        ops._append_completed_kernel[(rows,)](
            token_ids,
            w["terminal_ranks"],
            w["finish_reasons"],
            w["cumulative_scores"],
            w["row_offsets"],
            s["num_generated_tokens"],
            w["execute"],
            s["completed_token_ids"],
            s["completed_scores"],
            s["completed_parent_ids"],
            s["completed_lengths"],
            s["completed_finish_reasons"],
            s["completed_tie_break_indices"],
            WIDTH=width,
            BLOCK=block,
            PREFILL=prefill,
        )
        extent = self.state.sequence.shape[2]
        ops._stage_sequence_kernel[(width,)](
            s["sequence"],
            self._selected_tokens,
            self._selected_parents,
            self._selected_flat,
            self._next_sequence,
            s["num_generated_tokens"],
            w["execute"],
            WIDTH=width,
            SEQUENCE_STRIDE=extent,
            OUT_STRIDE=self.workspace.max_decode_steps,
            BLOCK=triton.next_power_of_2(extent),
            PREFILL=prefill,
        )
        ops._commit_beams_kernel[(width,)](
            self._selected_tokens,
            self._selected_parents,
            self._selected_scores,
            self._selected_flat,
            self._next_sequence,
            s["sequence"],
            s["beam_scores"],
            s["input_ids"],
            s["parent_ids"],
            s["active_mask"],
            s["sequence_lengths"],
            s["history_token_ids"],
            s["history_parent_ids"],
            s["num_generated_tokens"],
            w["execute"],
            WIDTH=width,
            SEQUENCE_STRIDE=extent,
            NEXT_STRIDE=self.workspace.max_decode_steps,
            BLOCK=triton.next_power_of_2(extent),
            PREFILL=prefill,
        )
        if not prefill:
            _select_kv_gather_kernel[self._kv_grid](
                *self._kv_args, s["parent_ids"], **self._kv_kwargs
            )
            _select_kv_scatter_kernel[self._kv_grid](*self._kv_args, **self._kv_kwargs)
        ops._commit_step_kernel[(1,)](
            s["active_mask"],
            s["num_generated_tokens"],
            s["num_decode_steps"],
            s["num_active_beams"],
            s["num_completed"],
            s["is_initialized"],
            s["is_finished"],
            w["next_completed"],
            w["execute"],
            WIDTH=width,
            BLOCK=block,
            PREFILL=prefill,
        )
