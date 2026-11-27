# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Stream a CUDA LM head into canonical candidate logits and full-window LSE."""

from typing import Any

import torch
import triton

from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
from vllm_gr.ops.cuda.rec_constrained_topk import (
    _launch_rec_constrained_topk_cuda_out,
    _stream_constraint_logits_kernel,
)
from vllm_gr.ops.sliced_logits import is_supported_unquantized_lm_head


class StreamingConstrainedHead:
    """One reusable logits tile and legal-candidate storage per fixed geometry."""

    def __init__(self, weight: torch.Tensor, backend: Any, rows: int, top_k: int) -> None:
        self.weight, self.backend = weight, backend
        self.rows, self.top_k = rows, top_k
        self.vocab_offset: int
        self.vocab_size: int
        self.vocab_offset, self.vocab_size = backend.vocab_slice or (
            0,
            backend.info.table_metadata.vocab_size,
        )
        info = backend.info.table_metadata
        first_degree = (
            backend._max_first_degree if info.trie_count > 1 else backend._tables()[0].numel()
        )
        self.candidate_columns = triton.next_power_of_2(
            max(top_k, first_degree, info.max_prefix1_degree, info.max_prefix2_degree, 1)
        )
        self.cols = min(32768 if self.vocab_size >= 65536 else 8192, self.vocab_size)
        self.tile = torch.empty((rows, self.cols), dtype=weight.dtype, device=weight.device)
        self.candidates = torch.empty(
            (rows, self.candidate_columns), dtype=weight.dtype, device=weight.device
        )
        self.stats = torch.empty((rows, 2), dtype=torch.float32, device=weight.device)

    @classmethod
    def for_worker(
        cls, worker: Any, backend: Any, rows: int, top_k: int
    ) -> "StreamingConstrainedHead | None":
        if not isinstance(backend, CUDAConstraintBackend):
            return None
        model = worker.model
        head = getattr(model, "lm_head", None)
        processor = getattr(model, "logits_processor", None)
        if (
            head is None
            or not is_supported_unquantized_lm_head(head)
            or getattr(head, "bias", None) is not None
            or processor is None
            or getattr(processor, "logits_as_input", False)
            or getattr(processor, "scale", 1.0) != 1.0
            or getattr(processor, "soft_cap", None) is not None
        ):
            return None
        offset, size = backend.vocab_slice or (0, backend.info.table_metadata.vocab_size)
        if size <= 8192:
            return None
        cached = getattr(head, "_sliced_weight", None)
        key = getattr(head, "_sliced_cache_key", None)
        if cached is not None and key is not None and key[:2] == (offset, size):
            start = head._sliced_rel_start
            weight = cached[start : start + size]
        else:
            weight = head.weight[offset : offset + size]
        if (
            weight.ndim != 2
            or weight.shape[0] != size
            or weight.device.type != "cuda"
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            return None
        info = backend.info.table_metadata
        first_degree = (
            backend._max_first_degree if info.trie_count > 1 else backend._tables()[0].numel()
        )
        columns = triton.next_power_of_2(
            max(top_k, first_degree, info.max_prefix1_degree, info.max_prefix2_degree, 1)
        )
        tile_columns = min(32768 if size >= 65536 else 8192, size)
        if (columns + tile_columns) * weight.element_size() + 8 >= size * weight.element_size():
            return None
        return cls(weight, backend, rows, top_k)

    @property
    def workspace_bytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in (self.tile, self.candidates, self.stats)
        )

    def supports(self, hidden: torch.Tensor) -> bool:
        return (
            hidden.ndim == 2
            and 0 < hidden.shape[0] <= self.rows
            and hidden.shape[1] == self.weight.shape[1]
            and hidden.dtype == self.weight.dtype
            and hidden.device == self.weight.device
            and hidden.is_contiguous()
        )

    def select(
        self,
        hidden: torch.Tensor,
        prefixes: torch.Tensor,
        temperatures: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        counts: torch.Tensor,
        device_step: torch.Tensor | None,
        current_step: int = 0,
        trie_ids: torch.Tensor | None = None,
    ) -> None:
        rows = hidden.shape[0]
        if not self.supports(hidden):
            raise ValueError("streaming head requires matching contiguous hidden states")
        if any(tuple(output.shape) != (rows, self.top_k) for output in outputs):
            raise ValueError("streaming head candidate output geometry changed")
        backend = self.backend
        first_ids, first_offsets, first_values, pair_keys, second_offsets, second_values = (
            backend._tables()
        )
        root_offsets = backend._first_token_offsets
        for start in range(0, self.vocab_size, self.cols):
            cols = min(self.cols, self.vocab_size - start)
            tile = self.tile.view(-1)[: rows * cols].view(rows, cols)
            torch.mm(hidden, self.weight[start : start + cols].T, out=tile)
            _stream_constraint_logits_kernel[(rows,)](
                tile,
                self.candidates,
                self.stats,
                prefixes,
                trie_ids if trie_ids is not None else prefixes,
                root_offsets if root_offsets is not None else first_offsets,
                first_ids,
                first_offsets,
                first_values,
                pair_keys,
                second_offsets,
                second_values,
                temperatures,
                device_step if device_step is not None else prefixes,
                start,
                cols,
                self.vocab_size,
                self.vocab_offset,
                prefix_stride=prefixes.stride(0),
                candidate_stride=self.candidate_columns,
                table_vocab_size=backend.info.table_metadata.vocab_size,
                pair_count=pair_keys.numel(),
                first_count=first_ids.numel(),
                temperature_count=temperatures.numel(),
                TRIE_COUNT=backend.info.table_metadata.trie_count,
                step=current_step,
                DEVICE_STEP=device_step is not None,
                BLOCK=self.candidate_columns,
                WINDOW_BLOCK=2048,
                num_warps=8 if self.candidate_columns >= 2048 else 4,
            )
        _launch_rec_constrained_topk_cuda_out(
            tile,
            first_ids,
            first_offsets,
            first_values,
            pair_keys,
            second_offsets,
            second_values,
            temperatures,
            current_step,
            self.top_k,
            *outputs[:2],
            outputs[2],
            counts,
            self.vocab_offset,
            None,
            device_step=device_step,
            rows=rows,
            block=self.candidate_columns,
            flat_prefixes=prefixes,
            table_vocab_size=backend.info.table_metadata.vocab_size,
            trie_ids=trie_ids,
            first_token_offsets=root_offsets,
            candidate_logits=self.candidates,
            denominator_stats=self.stats,
            logits_vocab_size_override=self.vocab_size,
        )


@torch.inference_mode()
def prewarm_streaming_constrained_head(worker: Any, backend: Any, top_k: int) -> None:
    """Prepare the Decode Triton variants before publishing a compatible Trie."""
    head = StreamingConstrainedHead.for_worker(worker, backend, 1, top_k)
    if head is None:
        return

    # Grid size and counts do not specialize either kernel. One row with the
    # production prefix stride and a device step warms every Decode batch size.
    device = head.weight.device
    hidden = torch.zeros((1, head.weight.shape[1]), dtype=head.weight.dtype, device=device)
    prefix = torch.full((1, 3), -1, dtype=torch.int32, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    trie_ids = torch.zeros_like(step) if backend.info.table_metadata.trie_count > 1 else None
    outputs = (
        torch.empty((1, top_k), dtype=torch.int32, device=device),
        torch.empty((1, top_k), dtype=torch.float32, device=device),
        torch.empty((1, top_k), dtype=torch.bool, device=device),
    )
    counts = torch.empty(1, dtype=torch.int32, device=device)
    head.select(hidden, prefix, temperature, outputs, counts, step, trie_ids=trie_ids)
    torch.accelerator.synchronize(device)
