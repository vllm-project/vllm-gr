# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager grouped-query attention over a shared prompt and private Beam suffixes."""

import torch


def eager_beam_decode(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    prefix_length: int,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend one token per beam to the paged prompt and its own suffix.

    Query is [beam, query_heads, dim], pages are [page, token, kv_heads, dim],
    and suffixes are [beam, kv_heads, token, dim]. Suffixes include the current
    token and contain only live positions.
    """
    beam_width, query_heads, head_dim = query.shape
    _, page_size, kv_heads, _ = key_pages.shape
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    if not 0 <= prefix_length <= block_table.numel() * page_size:
        raise ValueError("Beam prefix length is outside the block table")
    if suffix_key.shape[2] == 0:
        raise ValueError("Beam suffix must include the current token")

    num_pages = (prefix_length + page_size - 1) // page_size
    pages = block_table[:num_pages].long()
    prefix_key = key_pages.index_select(0, pages).flatten(0, 1)[:prefix_length].float()
    prefix_value = value_pages.index_select(0, pages).flatten(0, 1)[:prefix_length].float()
    grouped_query = query.float().reshape(beam_width, kv_heads, -1, head_dim)
    prefix_scores = torch.einsum("bhgd,thd->bhgt", grouped_query, prefix_key)
    suffix_scores = torch.einsum("bhgd,bhtd->bhgt", grouped_query, suffix_key.float())
    weights = torch.softmax(torch.cat((prefix_scores, suffix_scores), dim=-1) * scale, dim=-1)
    prefix_weight, suffix_weight = weights.split((prefix_length, suffix_key.shape[2]), dim=-1)
    prefix_output = torch.einsum("bhgt,thd->bhgd", prefix_weight, prefix_value)
    suffix_output = torch.einsum("bhgt,bhtd->bhgd", suffix_weight, suffix_value.float())
    return (prefix_output + suffix_output).reshape_as(query).to(query.dtype)
