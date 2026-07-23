# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm_gr.entrypoints.beam_search_config import BeamSearchModel
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams


def main():
    parser = argparse.ArgumentParser(description="RECIF offline beam search")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--bf", default="8,8,8")
    parser.add_argument("--beam", type=int, default=64)
    args = parser.parse_args()

    history_sids = [int(value) for value in args.history.split(",")]
    bf = tuple(int(value) for value in args.bf.split(","))

    llm = GRLLM(
        model=args.model_path,
        beam_search_model=BeamSearchModel.RECIF,
        max_logprobs=max(bf),
        skip_tokenizer_init=True,
        max_num_seqs=max(args.beam, bf[0]),
        logprobs_mode="processed_logprobs",
    )
    params = BeamSearchParams(
        beam_width=args.beam,
        max_tokens=3,
        temperature=0.0,
        ignore_eos=True,
        branch_factors=bf,
    )
    outputs = llm.beam_search(
        [{"history_sids": history_sids}],
        params,
    )

    for rank, sequence in enumerate(outputs[0].sequences):
        print(rank, getattr(sequence, "sid"), float(sequence.cum_logprob))


if __name__ == "__main__":
    main()
