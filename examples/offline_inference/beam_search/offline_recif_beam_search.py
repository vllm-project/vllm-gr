import argparse

from vllm import LLM

from vllm_gr.models.recif import beam_search, bytes_to_sid


def main():
    parser = argparse.ArgumentParser(description="RECIF offline beam search")
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--bf", default="8,8,8")
    parser.add_argument("--beam", type=int, default=64)
    args = parser.parse_args()

    history_sids = [int(value) for value in args.history.split(",")]
    bf = tuple(int(value) for value in args.bf.split(","))

    llm = LLM(
        model=args.model,
        max_logprobs=max(bf),
        skip_tokenizer_init=True,
        max_num_seqs=args.beam,
    )
    triples, scores = beam_search(
        llm,
        history_sids,
        bf=bf,
        beam=args.beam,
    )

    for rank, ((sa, sb, sc), score) in enumerate(zip(triples, scores)):
        print(
            rank,
            (int(sa), int(sb), int(sc)),
            bytes_to_sid(sa, sb, sc),
            float(score),
        )


if __name__ == "__main__":
    main()
