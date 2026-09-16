# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BeamAttentionContext — worker-level shared resource container for incremental decode.

Holds the KV pools, session registry, and suffix temporary buffers needed
by the attention layer during beam decode.  All per-request state (decode_step,
beam_width, prefix_len, is_last_step) is managed by BeamSessionRegistry,
NOT by this context.

This context is a worker-level singleton.  BeamSearchSession instances
reference it but do not own it.
"""

from __future__ import annotations

import torch

from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool, _validate_pool_dtype
from vllm_gr.v1.beam.beam_session_registry import BeamSessionRegistry


class BeamDecisionBuffers:
    """Device-resident beam-decision state.

    Holds ``sequence`` / ``beam_scores`` / ``group_token_num`` so that the
    beam decision (``beam_search_group`` / ``beam_search_rec_final_select`` +
    ``select_unshared_kv``) can run on the NPU worker with stable addresses,
    matching the device Beam update order. The
    buffers are sized to (max_batch, max_beams, max_decode_steps) and are
    never reallocated after construction — this keeps every address stable
    for ACL graph capture/replay and for the fused xllm decision operators.
    """

    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        max_decode_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_beams = max_beams
        self.device = device
        # Sequence history: [max_batch, max_beams, max_decode_steps]
        self.sequence = torch.zeros(
            (max_batch, max_beams, max_decode_steps),
            dtype=torch.int32,
            device=device,
        )
        # Cumulative log-probs per beam: [max_batch, max_beams]
        self.beam_scores = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)
        # Per-target-beam source-beam mapping (prefix sums consumed by the
        # fused select_unshared_kv kernel): [max_batch, max_beams]
        self.group_token_num = torch.zeros((max_batch, max_beams), dtype=torch.int32, device=device)
        # Host-side token count for the worker decision (next sequence slot to
        # write).  Mirrors the session decode_step but lives with the buffers
        # so the decision module owns it entirely and reset() clears it.
        self.worker_decode_steps = 0

    def reset(self, batch_index: int | None = None) -> None:
        """Clear all legacy state, or only one physical device row.

        A row reset leaves the legacy shared host counter untouched. The
        persistent CUDA runner owns separate per-row device counters.
        """
        if batch_index is None:
            self.sequence.zero_()
            self.beam_scores.zero_()
            self.group_token_num.zero_()
            self.worker_decode_steps = 0
        else:
            if isinstance(batch_index, bool) or not isinstance(batch_index, int):
                raise TypeError("batch_index must be an integer")
            if not 0 <= batch_index < self.sequence.shape[0]:
                raise ValueError("batch_index exceeds decision capacity")
            self.sequence[batch_index : batch_index + 1].zero_()
            self.beam_scores[batch_index : batch_index + 1].zero_()
            self.group_token_num[batch_index : batch_index + 1].zero_()


class BeamAttentionContext:
    """Worker-level shared resource container for incremental decode.

    Responsibilities:
    - Own the per-layer KV pools (shared across all sessions).
    - Own the BeamSessionRegistry (session_id → slot mapping + state).
    - Own the suffix KV temporary buffers (shared, partitioned by slot).
    - Provide get_suffix_buf(slot, beam_width) for per-session buffer slices.

    Exposed to:
    - context.pools[i]            → forward / cascade_attn
    - context.registry            → session management
    - context.get_suffix_buf(...) → cascade_attn
    """

    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        _validate_pool_dtype(dtype)

        self.max_batch = max_batch
        self.max_beams = max_beams
        self.max_decode_steps = max_decode_steps
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # Per-request session registry
        self.registry = BeamSessionRegistry(max_batch)

        # Per-layer KV pools (shared across all sessions)
        self.pools: list[BeamAttentionPool] = [
            BeamAttentionPool(
                max_batch,
                max_beams,
                num_kv_heads,
                max_decode_steps,
                head_dim,
                device,
                dtype,
            )
            for _ in range(num_layers)
        ]

        # Stable CUDA pointer tables consumed by the fused all-layer KV
        # reorder.  Building these once removes two per-step device tensors and
        # the matching in-place pointer-offset kernels from postprocessing.
        if device.type == "cuda":
            self.pool_key_ptrs = torch.tensor(
                [pool.unshared_key.data_ptr() for pool in self.pools],
                dtype=torch.int64,
                device=device,
            )
            self.pool_value_ptrs = torch.tensor(
                [pool.unshared_value.data_ptr() for pool in self.pools],
                dtype=torch.int64,
                device=device,
            )
        else:
            self.pool_key_ptrs = None
            self.pool_value_ptrs = None

        # Suffix KV temporary buffers (for cascade attention)
        # Partitioned by slot: each slot gets max_beams * max_decode_steps rows
        self.suffix_k_buf = torch.zeros(
            (max_batch * max_beams * max_decode_steps, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.suffix_v_buf = torch.zeros(
            (max_batch * max_beams * max_decode_steps, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # --- CUDA/ACL Graph fixed-address device buffers ---
        # decode_step_buf: device-side scalar, read by Triton cache_unshared_kv kernel
        # and filled each step before model forward / graph replay.
        self.decode_step_buf = torch.zeros(1, dtype=torch.int32, device=device)

        # decode_step_1based_buf: device-side 1-based step count consumed by the
        # xllm cache_unshared_kv / x_attention kernels (their ABI takes an int32
        # [1] device tensor).  Refreshed each step outside the graph so a
        # captured ACL graph reads the fresh value at replay.
        self.decode_step_1based_buf = torch.zeros(1, dtype=torch.int32, device=device)

        # xllm_block_table_buf: pool-slot block table for the xllm
        # cache_unshared_kv kernel (replaces the per-call torch.tensor([slot])
        # allocation; request i maps to pool slot xllm_block_table_buf[i]).
        self.xllm_block_table_buf = torch.zeros(max_batch, dtype=torch.int32, device=device)

        # Whether the attached worker consumes the NPU xllm device buffers
        # (decode_step_1based_buf / xllm_block_table_buf).  Only NPU kernels
        # read them; on GPU the Triton copy_suffix/cache_unshared kernels read
        # ``decode_step_buf`` alone, so the per-step fill_ of the xllm buffers
        # (two redundant launches on the same stream) is skipped there.
        self._gr_uses_xllm_buffers = getattr(device, "type", None) == "npu"

        # aux_key_buf / aux_value_buf: pre-allocated PER-LAYER scratch buffers
        # for select_unshared_kv. The leading dim is num_layers: the fused
        # Triton path reorders all layers in one launch, so each layer must
        # stage through its own slice (a single shared buffer would make
        # layers clobber each other's staging data). The Python fallback
        # and the cascade reorder path index the same buffer with [layer].
        self.aux_key_buf = torch.zeros(
            (num_layers, max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )
        self.aux_value_buf = torch.zeros(
            (num_layers, max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )

        # Device-resident beam-decision state .
        self.decision = BeamDecisionBuffers(max_batch, max_beams, max_decode_steps, device, dtype)

    def get_suffix_buf(self, slot: int, beam_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get suffix KV buffer slice for a specific slot.

        Returns (k_buf, v_buf) each of shape [beam_width * max_decode_steps, H, D].
        """
        offset = slot * self.max_beams * self.max_decode_steps
        size = beam_width * self.max_decode_steps
        return self.suffix_k_buf[offset : offset + size], self.suffix_v_buf[offset : offset + size]
