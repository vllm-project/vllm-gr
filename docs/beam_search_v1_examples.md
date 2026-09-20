# Beam Search V1 calls

Use a canonical Constraint Table configuration, for example `beam_v1.json`:

```json
{
  "schema_version": 1,
  "attention_backend": "CUSTOM",
  "beam": {
    "execution_mode": "v1",
    "worker_decision": true,
    "graph_enabled": true,
    "max_width": 128,
    "max_decode_steps": 3
  },
  "constraint_table": {
    "enabled": true,
    "backend": "cuda",
    "max_top_k": 128,
    "path": "/absolute/path/table.safetensors",
    "format": "constraint_table_v1",
    "artifact_digest": "sha256:REPLACE_WITH_ARTIFACT_DIGEST",
    "tokenizer_digest": "sha256:REPLACE_WITH_TOKENIZER_DIGEST"
  }
}
```

Build the artifact with `python tools/build_constraint_table.py --help` and use
its actual digests. The model and table must share a tokenizer and vocabulary.
The system test builds a small legal OneRec table without an external catalog.

## Offline

```python
import json

from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams

with open("beam_v1.json") as file:
    config = json.load(file)
params = BeamSearchParams(
    beam_width=128, max_tokens=5, ignore_eos=True, temperature=0.0,
    begin_token="<|sid_begin|>", end_token="<|sid_end|>",
)
with GRLLM(
    model="/path/to/OneRec-1.7B", trust_remote_code=True,
    dtype="bfloat16", vllm_gr_config=config, async_scheduling=True,
    enable_chunked_prefill=False, enable_prefix_caching=True,
    max_num_seqs=1, max_model_len=8192, max_num_batched_tokens=8192,
    max_logprobs=128,
) as llm:
    output = llm.beam_search_v1(["Your prompt"], params, use_tqdm=False)[0]
    for sequence in output.sequences:
        print(sequence.tokens, sequence.cum_logprob, sequence.finish_reason)
```

The budget above is effective N=3 plus the two explicit boundary tokens. Offline
sequences contain prompt plus generated tokens and boundary tokens. Token/text
fields materialize on first access. As in upstream #396, offline `logprobs` is
empty and online `logprobs` is `None`; cumulative scores remain unnormalized; equal-length survivors retain
the Beam ranking for any supported length penalty.

## Online engine

```python
import asyncio
import json

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.inputs import tokens_input
from vllm_gr.plugin import initialize_runtime
from vllm_gr.sampling_params import BeamSearchParams

async def main():
    initialize_runtime()
    model = "/path/to/OneRec-1.7B"
    with open("beam_v1.json") as file:
        config = json.load(file)
    args = AsyncEngineArgs(
        model=model, trust_remote_code=True, dtype="bfloat16",
        additional_config={"vllm_gr": config}, async_scheduling=True,
        enable_chunked_prefill=False, enable_prefix_caching=True,
        max_num_seqs=1, max_model_len=8192, max_num_batched_tokens=8192,
        max_logprobs=128,
    )
    async with build_async_engine_client_from_engine_args(args) as engine:
        serving = OpenAIServing(
            engine_client=engine,
            models=OpenAIServingModels(
                engine_client=engine,
                base_model_paths=[BaseModelPath(name=model, model_path=model)],
                lora_modules=None,
            ),
            request_logger=None,
        )
        ids = serving.input_processor.tokenizer.encode("Your prompt")
        params = BeamSearchParams(
            beam_width=128, max_tokens=3, ignore_eos=True, temperature=0.0,
        )
        async for output in serving.beam_search_v1(tokens_input(ids), "client-id", params):
            assert output.request_id == "client-id" and output.finished
            for completion in output.outputs:
                print(completion.token_ids, completion.cumulative_logprob)

asyncio.run(main())
```

Online completion tokens exclude the prompt. Cancel the task consuming the
iterator when a connection closes, and close the iterator in its owning task.
Cancellation cleanup is retained until retirement is acknowledged. A returned
terminal result already has retirement proof; subsequent calls can reuse the
Worker workspace. This entrypoint yields one final RequestOutput per B=1 call.

## Result validation

Both offline and online calls validate result types at the transport boundary,
including in-process results. Normal output conversion checks protocol and request
identity, success/failure status, candidate counts, generation lengths, non-negative
token IDs, and present, finite cumulative scores.

For debugging or regression validation, set
`VLLM_GR_BEAM_STRICT_RESULT_VALIDATION=1` in the frontend process before making
calls. This additionally runs the full result contract validator and checks parent
lineage lengths/ranges and descending score order. These internal diagnostics are
disabled by default; the public result retains the Worker-provided candidate order
and scores in either mode. The option does not change retirement guarantees.
