# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Send contiguous int64 token buffers through the GR GenerateBatch RPC."""

import argparse
import json
import uuid

import grpc
import numpy as np

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc


def generate(stub, rows, *, width=4, max_new_tokens=3):
    batch_id = uuid.uuid4().hex
    requests = []
    for index, row in enumerate(rows):
        tokens = np.asarray(row, dtype="<i8")
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError("each prompt must be a nonempty one-dimensional token array")
        requests.append(
            pb.GenerateRequest(
                request_id=f"{batch_id}-{index}",
                tokenized=pb.TokenizedInput(input_ids_i64_le=tokens.tobytes()),
                sampling_params=pb.SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    ignore_eos=True,
                    n=width,
                ),
                beam_search=pb.BeamSearchOptions(beam_width=width),
            )
        )
    response = stub.GenerateBatch(
        pb.BatchGenerateRequest(batch_id=batch_id, requests=requests),
        timeout=120,
    )
    if response.batch_id != batch_id or len(response.items) != len(requests):
        raise ValueError("response batch identity or cardinality mismatch")
    items = []
    for index, item in enumerate(response.items):
        if item.request_id != requests[index].request_id:
            raise ValueError("response item order changed")
        beams = item.beams
        flat, offsets = list(beams.output_ids), list(beams.output_offsets)
        if len(offsets) != width + 1 or not all(beams.has_cumulative_logprob):
            raise ValueError("incomplete beam output")
        items.append(
            dict(
                token_ids=[flat[a:b] for a, b in zip(offsets, offsets[1:])],
                scores=list(beams.cumulative_logprobs),
            )
        )
    return dict(batch_id=batch_id, items=items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()
    rows = np.random.default_rng(7).integers(
        3,
        args.vocab_size,
        size=(args.batch_size, args.tokens),
        dtype=np.int64,
    )
    with grpc.insecure_channel(args.address) as channel:
        result = generate(
            rpc.VllmEngineStub(channel), rows, width=args.width, max_new_tokens=args.max_new_tokens
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
