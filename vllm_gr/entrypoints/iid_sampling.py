# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Catalog-constrained IID sampling in one persistent GR session."""

from __future__ import annotations

import math
import secrets
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_gr.config import GRConfig
from vllm_gr.entrypoints.beam_search import run_beam_search
from vllm_gr.entrypoints.beam_search.protocol import OfflineBeamSearchRequest
from vllm_gr.sampling_params import BeamSearchParams

if TYPE_CHECKING:
    from vllm_gr.entrypoints.llm import GRLLM
    from vllm_gr.outputs import CompactBeamSearchOutput


@dataclass
class IIDSampleResult:
    """Unordered SID triples with multiplicities and distinct input rows per step."""

    counts: dict[tuple[int, ...], int]
    rows_per_stage: tuple[int, int, int]


def run_iid_sampling(
    llm: GRLLM,
    prompts: Sequence[Sequence[int]],
    num_samples: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
    trie_ids: Sequence[int] | None,
) -> list[IIDSampleResult]:
    """Draw three-token SIDs with replacement. The prompt excludes the new begin marker."""
    from vllm_gr.config.models import (
        SUPPORTED_BEAM_BATCH_SIZES,
        ConstraintMode,
        configured_constraint_mode,
    )

    config = llm.llm_engine.vllm_config
    gr = GRConfig.from_vllm_config(config)
    if len(prompts) not in SUPPORTED_BEAM_BATCH_SIZES:
        raise ValueError(f"unsupported IID prompt count: {len(prompts)}")
    capacity = gr.beam_max_width
    if type(num_samples) is not int or capacity is None or not 1 <= num_samples <= capacity:
        raise ValueError(f"num_samples must be between 1 and the configured width ({capacity})")
    if (
        config.device_config.device_type != "cuda"
        or configured_constraint_mode(gr) is not ConstraintMode.TABLE
    ):
        raise ValueError("IID sampling requires CUDA and a GR constraint table")
    if config.use_v2_model_runner or not config.model_config.enforce_eager:
        raise ValueError("IID sampling requires ModelRunner V1 with enforce_eager=True")
    if not math.isfinite(temperature) or temperature <= 0 or temperature > 2:
        raise ValueError("temperature must be finite and in (0, 2]")
    if type(top_k) is not int or top_k < -1 or not 0 < top_p <= 1:
        raise ValueError("expected top_k >= -1 and top_p in (0, 1]")
    if seed is None:
        seed = secrets.randbits(63)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    params = BeamSearchParams(
        beam_width=capacity,
        dynamic_beam_widths=[capacity] * 3,
        result_width=num_samples,
        temperature=temperature,
        max_tokens=4,
        begin_token="<|sid_begin|>",
        end_token=None,
        ignore_eos=True,
        catalog_strategy=None,
    )
    call = OfflineBeamSearchRequest(
        prompts=[tokens_input(list(tokens)) for tokens in prompts],
        params=params,
        trie_ids=list(trie_ids) if trie_ids is not None else [0] * len(prompts),
        output_mode="compact",
        iid_sampling=(num_samples, top_k, top_p, seed),
    )
    return [
        _collect_samples(output, num_samples)
        for output in run_beam_search(llm._create_beam_search_client(), call)
    ]


def _collect_samples(output: CompactBeamSearchOutput, num_samples: int) -> IIDSampleResult:
    counts = Counter(seq.token_ids[1:] for seq in output.sequences)
    if sum(counts.values()) != num_samples or any(len(sid) != 3 for sid in counts):
        raise RuntimeError("incomplete IID sampling result")
    return IIDSampleResult(
        dict(counts), (1, len({sid[0] for sid in counts}), len({sid[:2] for sid in counts}))
    )
