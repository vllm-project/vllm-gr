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
    requested_logprobs = bool(getattr(self, "logprobs", False))
    requested_top_logprobs = getattr(self, "top_logprobs", None) or 0
    explicit_mode = extra_args.get("logprobs_mode")
    if explicit_mode is None:
        if not requested_logprobs:
            logprobs_mode = "none"
        elif requested_top_logprobs > 0:
            logprobs_mode = "top_k"
        else:
            logprobs_mode = "selected"
    else:
        logprobs_mode = explicit_mode
    top_logprobs = (
        int(extra_args.get("top_logprobs", requested_top_logprobs))
        if logprobs_mode == "top_k"
        else 0
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
        catalog_strategy=extra_args.get("catalog_strategy", "fill"),
        result_width=extra_args.get("result_width", n),
        logprobs_mode=logprobs_mode,
        top_logprobs=top_logprobs,
    )
