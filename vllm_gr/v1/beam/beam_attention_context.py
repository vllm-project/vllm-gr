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

        # --- CUDA Graph fixed-address device buffers ---
        # decode_step_buf: device-side scalar, read by Triton cache_unshared_kv kernel
        # and filled each step before model forward / graph replay.
        self.decode_step_buf = torch.zeros(1, dtype=torch.int32, device=device)

        # aux_key_buf / aux_value_buf: pre-allocated buffers for select_unshared_kv
        # (replace per-step .clone() allocation). Shape matches one pool slot.
        # WARNING: these are SINGLE shared buffers aliased across all layers in
        # BeamSearchSession.step_complete (via [same_buf]*num_layers). This is
        # safe ONLY because select_unshared_kv processes layers sequentially.
        # Do NOT vectorize or parallelize the per-layer loop without giving each
        # layer its own aux_buf slot.
        self.aux_key_buf = torch.zeros(
            (max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )
        self.aux_value_buf = torch.zeros(
            (max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )

    def get_suffix_buf(self, slot: int, beam_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get suffix KV buffer slice for a specific slot.

        Returns (k_buf, v_buf) each of shape [beam_width * max_decode_steps, H, D].
        """
        offset = slot * self.max_beams * self.max_decode_steps
        size = beam_width * self.max_decode_steps
        return self.suffix_k_buf[offset : offset + size], self.suffix_v_buf[offset : offset + size]
