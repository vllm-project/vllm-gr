# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Resolve the class name stored in pickles of pre-migration Beam results."""

from vllm_gr.entrypoints.beam_search.lazy_output import _LazyMaterializedBeamSequence

__all__ = ["_LazyMaterializedBeamSequence"]
