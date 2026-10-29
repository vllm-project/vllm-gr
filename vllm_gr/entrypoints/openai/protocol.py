# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py

from vllm_gr.sampling_params import BeamSearchParams


def to_beam_search_params(self, max_tokens: int, default_sampling_params: dict) -> BeamSearchParams:
    extra_args = self.vllm_xargs or {}
    n = self.n if self.n is not None else 1
    if (temperature := self.temperature) is None:
        temperature = default_sampling_params.get(
            "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
        )
    return BeamSearchParams(
        beam_width=n,
        max_tokens=max_tokens,
        ignore_eos=self.ignore_eos,
        temperature=temperature,
        length_penalty=self.length_penalty,
        include_stop_str_in_output=self.include_stop_str_in_output,
        begin_token=extra_args.get("begin_token"),
        end_token=extra_args.get("end_token"),
        # Absent → None: the coordinator resolves the backend-appropriate
        # default (Trie: "fill"; Constraint Tables: None).
        catalog_strategy=extra_args.get("catalog_strategy"),
        dynamic_beam_widths=extra_args.get("dynamic_beam_widths"),
        result_width=extra_args.get("result_width"),
        per_parent_beam_widths=extra_args.get("per_parent_beam_widths"),
    )
