# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam sampler bookkeeping for the NPU V1 runner."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _remux_logprobs_tensors(
    request_row_configs: list[tuple[Any, int, int]],
    max_chained_dim: int,
    target_K: int,
    orig_lps: torch.Tensor,
    orig_lp_ids: torch.Tensor | None,
    orig_ranks: torch.Tensor | None,
    st_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Merges and aligns sampler tensor slices sequentially across runtime waves."""
    if max_chained_dim == 0 or orig_lps.numel() == 0:
        return st_ids, orig_lps, orig_lp_ids, orig_ranks

    new_st_ids, new_lps, new_lp_ids, new_ranks = [], [], [], []
    sampler_row_cursor = 0
    logical_row_cursor = 0
    logical_st_ids = st_ids.shape[0] == sum(
        req_id is not None for req_id, *_ in request_row_configs
    )

    for req_id, _, W_logits, *_ in request_row_configs:
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        if W_logits == 0:
            dummy_st = torch.tensor([[-1]], dtype=st_ids.dtype, device=st_ids.device)
            new_st_ids.append(dummy_st)

            new_lps.append(
                torch.full(
                    (1, max_chained_dim),
                    float("-inf"),
                    dtype=orig_lps.dtype,
                    device=orig_lps.device,
                )
            )
            if orig_lp_ids is not None:
                new_lp_ids.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_lp_ids.dtype, device=orig_lp_ids.device
                    )
                )
            if orig_ranks is not None:
                new_ranks.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_ranks.dtype, device=orig_ranks.device
                    )
                )
            continue

        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        sampled_row = logical_row_cursor if logical_st_ids else sampler_row_cursor
        new_st_ids.append(st_ids[sampled_row : sampled_row + 1, :])
        logical_row_cursor += 1

        req_lps = orig_lps[sampler_row_cursor : sampler_row_cursor + W_logits, :target_K].reshape(
            1, W_logits * target_K
        )
        if W_logits * target_K < max_chained_dim:
            req_lps = torch.nn.functional.pad(
                req_lps, (0, max_chained_dim - (W_logits * target_K)), value=float("-inf")
            )
        new_lps.append(req_lps)

        if orig_lp_ids is not None:
            req_lp_ids = orig_lp_ids[
                sampler_row_cursor : sampler_row_cursor + W_logits, :target_K
            ].reshape(1, W_logits * target_K)
            if W_logits * target_K < max_chained_dim:
                req_lp_ids = torch.nn.functional.pad(
                    req_lp_ids, (0, max_chained_dim - (W_logits * target_K)), value=0
                )
            new_lp_ids.append(req_lp_ids)

        if orig_ranks is not None:
            req_ranks = orig_ranks[sampler_row_cursor : sampler_row_cursor + W_logits].reshape(
                1, W_logits
            )
            if W_logits < max_chained_dim:
                req_ranks = torch.nn.functional.pad(
                    req_ranks, (0, max_chained_dim - W_logits), value=0
                )
            new_ranks.append(req_ranks)

        sampler_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None,
    )


def _beam_collapse_bookkeeping(
    runner: Any,
    request_row_configs: Any,
    beam_data: Any,
    sampler_output: Any,
) -> tuple[list[Any], dict[Any, int], int]:
    """Collapse per-beam sampler output back onto logical (session) requests.

    Remuxes the sampler logprobs tensors in place so that each logical request
    owns one flattened ``W * K`` row, and resizes the sampler output tensors
    accordingly.

    Returns:
        ``(collapsed_req_ids, collapsed_req_id_to_index, num_logical_requests)``.
    """
    collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
    collapsed_req_id_to_index = {req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)}
    num_logical_requests = len(collapsed_req_ids_copy)

    if sampler_output is not None:
        st_ids = sampler_output.sampled_token_ids
        lp_tensors = sampler_output.logprobs_tensors
        if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
            orig_lps = lp_tensors.logprobs
            orig_lp_ids = lp_tensors.logprob_token_ids
            orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)
            actual_K = orig_lps.shape[1]

            max_chained_dim = 0
            for req_id, _, W, *_ in request_row_configs:
                if req_id is not None and req_id in beam_data:
                    m_width = beam_data[req_id].get("beam_width", 1)
                    needed_dim = m_width * actual_K
                else:
                    needed_dim = W * actual_K
                max_chained_dim = max(max_chained_dim, needed_dim)

            final_st_ids, final_lps, final_lp_ids, final_ranks = _remux_logprobs_tensors(
                request_row_configs,
                max_chained_dim,
                actual_K,
                orig_lps,
                orig_lp_ids,
                orig_ranks,
                st_ids,
            )
            if not hasattr(runner, "_beam_logprobs_cache"):
                runner._beam_logprobs_cache = {}

            req_id_to_config = {r[0]: (r[2], r[3]) for r in request_row_configs if r[0] is not None}

            for idx, req_id in enumerate(collapsed_req_ids_copy):
                mdata = beam_data.get(req_id)
                if mdata and mdata.get("is_beam_decode", False):
                    logits_num, active_beams = req_id_to_config.get(req_id, (0, []))
                    total_beam_width = mdata.get("beam_width", 1)

                    if logits_num > 0:
                        # FAST-PATH: This chunk contains ALL beams directly.
                        if logits_num == total_beam_width:
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]
                            continue

                        # SLOW-PATH: Handle split chunked prefills across steps.
                        actual_lps = final_lps[idx, : logits_num * actual_K].view(
                            logits_num, actual_K
                        )
                        actual_lp_ids = (
                            final_lp_ids[idx, : logits_num * actual_K].view(logits_num, actual_K)
                            if final_lp_ids is not None
                            else None
                        )
                        actual_ranks = (
                            final_ranks[idx, :logits_num] if final_ranks is not None else None
                        )

                        if req_id not in runner._beam_logprobs_cache:
                            runner._beam_logprobs_cache[req_id] = {
                                "lps": torch.full(
                                    (total_beam_width, actual_K),
                                    float("-inf"),
                                    device=orig_lps.device,
                                    dtype=orig_lps.dtype,
                                ),
                                "lp_ids": (
                                    torch.zeros(
                                        (total_beam_width, actual_K),
                                        dtype=(
                                            orig_lp_ids.tensor.dtype
                                            if hasattr(orig_lp_ids, "tensor")
                                            else torch.long
                                        ),
                                        device=orig_lps.device,
                                    )
                                    if orig_lp_ids is not None
                                    else None
                                ),
                                "ranks": (
                                    torch.zeros(
                                        total_beam_width, dtype=torch.long, device=orig_lps.device
                                    )
                                    if orig_ranks is not None
                                    else None
                                ),
                                "completed_beams": set(),
                            }

                        cache_entry = runner._beam_logprobs_cache[req_id]

                        for local_row_idx, physical_beam_id in enumerate(active_beams):
                            if physical_beam_id not in cache_entry["completed_beams"]:
                                cache_entry["lps"][physical_beam_id] = actual_lps[local_row_idx]
                                if actual_lp_ids is not None:
                                    cache_entry["lp_ids"][physical_beam_id] = actual_lp_ids[
                                        local_row_idx
                                    ]
                                if actual_ranks is not None:
                                    cache_entry["ranks"][physical_beam_id] = actual_ranks[
                                        local_row_idx
                                    ]
                                cache_entry["completed_beams"].add(physical_beam_id)

                        if len(cache_entry["completed_beams"]) >= total_beam_width:
                            final_lps[idx].copy_(cache_entry["lps"].view(-1))
                            if final_lp_ids is not None:
                                final_lp_ids[idx].copy_(cache_entry["lp_ids"].view(-1))
                            if final_ranks is not None:
                                final_ranks[idx, :total_beam_width].copy_(cache_entry["ranks"])

                            del runner._beam_logprobs_cache[req_id]

            try:
                st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
                lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

                if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                    lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
                if orig_ranks is not None and final_ranks is not None:
                    clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                    lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(
                        clean_ranks_slice
                    )
            except Exception as e:
                logger.critical("Inplace copy failed during beam collapse: %s", e)
                raise RuntimeError("Inplace copy failed during beam collapse") from e

    return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests


def _attach_beam_search_result(runner: Any, model_output: Any) -> None:
    """Attach the worker beam decision onto a ModelRunnerOutput, if present.

    ``ModelRunnerOutput`` is a mutable dataclass (no ``__slots__``), so the
    decision dict is attached as a dynamic ``beam_search_result`` attribute.
    Under async scheduling the runner returns an ``AsyncModelRunnerOutput``
    wrapper; the scheduler eventually sees ``get_output()`` which returns the
    inner ``ModelRunnerOutput``, so the attribute must land on that inner
    object.  This matches the asynchronous scheduling ``ModelRunnerOutput.beam_search_result``
    save-flow used by the scheduler ``kv_transfer_params`` attachment.

    Async ``execute_model`` returns ``None`` / an ``ExecuteModelState`` (the
    sampling + decision happen later in ``sample_tokens``), which are skipped
    here; the ``sample_tokens`` hook attaches the decision to the final output.
    """
    decision = getattr(runner, "_beam_decision_result", None)
    if decision is None:
        return
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    if not isinstance(model_output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
        # Async execute_model: sampling (and the decision) has not run yet; the
        # sample_tokens hook attaches the decision to the real output.
        return

    target = model_output
    if isinstance(target, AsyncModelRunnerOutput):
        target = getattr(target, "model_runner_output", None) or getattr(
            target, "_model_runner_output", None
        )
        if target is None:
            logger.warning(
                "vllm-gr: cannot attach beam_search_result to AsyncModelRunnerOutput "
                "(no inner ModelRunnerOutput); decision dropped"
            )
            return
    target.beam_search_result = decision
