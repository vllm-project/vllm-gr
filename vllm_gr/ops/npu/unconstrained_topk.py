# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Native full-vocabulary candidate baseline for Ascend Beam V1."""

from vllm_gr.ops.unconstrained_topk_torch import select_unconstrained_candidates

__all__ = ["select_unconstrained_candidates"]
