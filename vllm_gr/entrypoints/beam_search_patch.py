# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gr.entrypoints.llm import beam_search
from vllm.entrypoints.llm import LLM


def patch_offline_beam_search():
    LLM.beam_search = beam_search
