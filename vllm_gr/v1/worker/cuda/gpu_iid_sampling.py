# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Persistent per-prefix sampling state for one catalog-constrained CUDA request."""

from dataclasses import replace
from typing import Any

import torch
from msgspec.structs import replace as replace_struct

from vllm_gr.v1.beam.dynamic_width import prune_stage_pools, reorder_stage_kv
from vllm_gr.v1.worker.cuda.gpu_dynamic_beam import GPUDynamicBeam


def draw_children(
    ids: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    total: int,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample each parent count independently and coalesce equal (parent, child) pairs."""
    scores = scores / temperature
    scores, order = scores.sort(dim=1, stable=True)
    ids = ids.gather(1, order)
    if 0 < top_k < scores.shape[1]:
        scores.masked_fill_(scores < scores[:, -top_k, None], -torch.inf)
    if top_p < 1:
        discard = scores.softmax(1).cumsum(1) <= 1 - top_p
        discard[:, -1] = False
        scores.masked_fill_(discard, -torch.inf)
    probs = scores.softmax(1)
    cdf = probs.cumsum(1)
    cdf[:, -1] = 1
    parents = torch.repeat_interleave(
        torch.arange(len(counts), device=ids.device), counts, output_size=total
    )
    uniforms = torch.rand(total, generator=generator, device=ids.device)
    # Each draw searches its parent's CDF. Construct and normalize each distribution once.
    if cdf.is_cuda:
        from vllm_gr.ops.cuda.iid_sampling import search_parent_cdfs

        columns = search_parent_cdfs(cdf, parents, uniforms)
    else:
        columns = torch.searchsorted(
            cdf.index_select(0, parents), uniforms[:, None], right=True
        ).flatten()
    parent, column, child_counts = _group_draws(parents, columns, ids.shape[1])
    return parent, ids[parent, column], child_counts, probs[parent, column].log()


def _group_draws(
    parents: torch.Tensor, columns: torch.Tensor, degree: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Include the parent in the key so equal tokens from different paths stay distinct."""
    packed, counts = (parents * degree + columns).unique(return_counts=True)
    return packed // degree, packed % degree, counts


class GPUIIDSampling(GPUDynamicBeam):
    """Use the existing variable-width attention and terminal payload in a single GR session."""

    def trim_idle_buffers(self) -> None:
        # Branch counts depend on draws. Keep the previous two decode geometries
        # for reuse without caching a KV bank for every width ever sampled.
        keep = set(self.widths[:2])
        prune_stage_pools(self.state.context, keep)
        self.buffers = {width: value for width, value in self.buffers.items() if width in keep}

    def start(self, options: tuple[int, int, float, int], trie_ids: tuple[int, ...]) -> None:
        self.total, self.top_k, self.top_p, seed = options
        device = self.owner.beam.state.device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.batch_size = len(trie_ids)
        self.trie_ids = torch.tensor(trie_ids, device=device, dtype=torch.long)
        self.members = torch.arange(self.batch_size, device=device)
        self.columns = torch.zeros_like(self.members)
        self.counts = torch.full((self.batch_size,), self.total, device=device)
        self.sequence = torch.zeros((self.batch_size, 3), device=device, dtype=torch.int32)
        self.lineage = torch.zeros_like(self.sequence)
        self.scores = torch.zeros(self.batch_size, device=device)
        self.widths: list[int] = []

    def local_metadata(self, metadata: Any) -> Any:
        """The scheduler reserves capacity. The worker binds only the sampled prefix rows."""
        widths = (*self.widths, *([metadata.beam_params.beam_width] * (3 - len(self.widths))))
        return replace(
            metadata,
            beam_params=replace_struct(
                metadata.beam_params, dynamic_beam_widths=widths, per_parent_beam_widths=None
            ),
        )

    def sample(self, logits: torch.Tensor, step: int, temperature: float) -> None:
        target = self.owner.beam.state
        batch = self.batch_size
        previous_width = self.widths[-1] if self.widths else 1
        # Attention is rectangular. Selection reads only actual prefixes, never padding.
        live_rows = self.members * previous_width + self.columns
        ids, logits = self.owner.backend.get().iid_candidates(
            logits.index_select(0, live_rows), self.sequence, step, self.trie_ids[self.members]
        )
        bad = ~logits.isfinite().any(1) | logits.isnan().any(1) | logits.isposinf().any(1)
        errors = target.error_code[:batch]
        errors.scatter_reduce_(0, self.members, bad.to(errors.dtype) * 3, reduce="amax")
        # A bad row still follows a legal edge until the normal error receipt.
        logits = logits.masked_fill(bad[:, None], -torch.inf)
        logits[:, 0] = torch.where(bad, 0.0, logits[:, 0])
        parents, tokens, self.counts, local_scores = draw_children(
            ids,
            logits,
            self.counts,
            batch * self.total,
            temperature,
            self.top_k,
            self.top_p,
            self.generator,
        )
        # Unique parent/child pairs are ordered by parent. All children of a
        # member therefore remain contiguous in the compact sampling state.
        parent_columns = self.columns[parents]
        self.members = self.members[parents]
        self.sequence = self.sequence[parents]
        self.sequence[:, step] = tokens
        self.lineage = self.lineage[parents]
        self.lineage[:, step] = parent_columns
        self.scores = self.scores[parents] + local_scores

        sizes = torch.zeros(batch, device=parents.device, dtype=torch.long)
        sizes.scatter_add_(0, self.members, torch.ones_like(self.members))
        if batch == 1:
            width = parents.numel()
        elif parents.numel() == batch * self.total:
            width = self.total
        else:
            width = int(sizes.max().item())
        self.widths.append(width)
        offsets = sizes.cumsum(0) - sizes
        self.columns = torch.arange(parents.numel(), device=parents.device) - offsets[self.members]
        if step == 1:
            # The next attention stage needs the suffix KV already written for
            # each child's parent. Indices stay relative to that member's KV slot.
            parent_map = torch.zeros((batch, width), dtype=torch.long, device=parents.device)
            parent_map[self.members, self.columns] = parent_columns
            reorder_stage_kv(self.state.context, previous_width, width, parent_map, step)
        last = step == 2
        if last:
            leaves = torch.repeat_interleave(
                torch.arange(parents.numel(), device=parents.device),
                self.counts,
                output_size=batch * self.total,
            )
            # Expand only the terminal payload to use the existing result transport.
            self.state.scores = self.scores[leaves].view(batch, self.total)
            self.state.sequence = self.sequence[leaves].view(batch, self.total, 3)
            self.state.parent_history = self.lineage[leaves].view(batch, self.total, 3)
        target.active_mask[:batch].zero_()
        target.active_mask[self.members, self.columns] = True
        target.input_ids[:batch, :width].zero_()
        target.input_ids[self.members, self.columns] = tokens.to(target.input_ids.dtype)
        target.sequence[:batch, :width, :3].zero_()
        target.sequence[self.members, self.columns, :3] = self.sequence
        target.num_active_beams[:batch].copy_(sizes)
        target.is_initialized[:batch].fill_(True)
        target.num_generated_tokens[:batch].fill_(step + 1)
        target.num_decode_steps[:batch].fill_(step)
        target.is_finished[:batch].fill_(last)
