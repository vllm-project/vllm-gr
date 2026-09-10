# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm_gr.v1.attention.backends.beam_attn import (
    BeamAttentionMetadataBuilder,
)


def test_resumed_suffix_slots_distinguish_cached_and_chunk_tokens() -> None:
    builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
    builder.block_size = 4

    mappings = builder._build_beam_mappings(
        beam_req_idxs=[0, 1],
        beam_prefix_lens=[4, 4],
        beam_widths=[2, 2],
        beam_decode_steps=[2, 2],
        beam_cache_lens=[4, 5],
        beam_q_start_offsets=[0, 0],
        q_starts=[0, 4, 7],
        common_attn_metadata=SimpleNamespace(),
    )
    suffix_locations = mappings[6]

    assert suffix_locations == [
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (-1, 3),
        (1, 4),
        (-1, 4),
        (-1, 5),
        (-1, 6),
    ]
