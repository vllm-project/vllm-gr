# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from recif_utils import build_prefix, generated_tokens_to_sid

from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams


def main():
    parser = argparse.ArgumentParser(description="RECIF offline beam search")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument(
        "--attention-backend",
        choices=("AUTO", "CUSTOM"),
        default="AUTO",
        help="Use CUSTOM to enable the grouped-beam attention backend",
    )
    args = parser.parse_args()

    history_sids = [int(value) for value in args.history.split(",")]
    attention_kwargs = (
        {"attention_config": {"backend": "CUSTOM"}} if args.attention_backend == "CUSTOM" else {}
    )
    llm = GRLLM(
        model=args.model_path,
        max_logprobs=args.beam,
        beam_max_width=args.beam,
        skip_tokenizer_init=True,
        max_num_seqs=args.beam,
        logprobs_mode="processed_logprobs",
        **attention_kwargs,
    )
    params = BeamSearchParams(
        beam_width=args.beam,
        max_tokens=3,
        temperature=0.0,
        ignore_eos=True,
    )
    prompt_token_ids = build_prefix(history_sids)
    outputs = llm.beam_search(
        [{"prompt_token_ids": prompt_token_ids}],
        params,
    )

    for rank, sequence in enumerate(outputs[0].sequences):
        generated = sequence.tokens[len(prompt_token_ids) :]
        print(rank, generated_tokens_to_sid(generated), float(sequence.cum_logprob))


if __name__ == "__main__":
    main()
