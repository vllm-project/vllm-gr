# Completions API

Send one prompt to an OpenAI-compatible endpoint and read the ranked Beam
candidates from the response. The example below uses OneRec with a constraint
table. [Prepare a persistent service](../../examples/online_serving/beam_search/README.md#prepare-a-persistent-service)
to generate the catalog configuration, then
[start the HTTP server](../../examples/online_serving/beam_search/README.md#start-the-server)
with the generated configuration and served model name `onerec`. The
[API server overview](README.md#start-and-verify-the-server) shows how to check
readiness.

## Request

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "onerec",
    "prompt": "Recommend the next video.",
    "use_beam_search": true,
    "n": 128,
    "temperature": 0.0,
    "max_tokens": 5,
    "ignore_eos": true,
    "vllm_xargs": {
      "begin_token": "<|sid_begin|>",
      "end_token": "<|sid_end|>"
    }
  }'
```

### Beam parameters

| Field | Meaning |
| --- | --- |
| `model` | Model name advertised by `GET /v1/models` |
| `prompt` | Text to complete |
| `use_beam_search` | Selects Beam generation when set to `true`; requires `stream: false` (the default) |
| `n` | Request width. Set it to the maximum value in the server's `gr.beam_width` list. |
| `max_tokens` | Token budget. For this catalog, five covers the two boundary markers and three generated SID components. |
| `vllm_xargs.begin_token`, `vllm_xargs.end_token` | SID boundary markers used for this request |
| `vllm_xargs.result_width` | Optional number of candidates to return, up to the final retained width |

See [Beam settings](../configuration/README.md#beam-settings) for deployment
capacity, dynamic width schedules, and the relationship between request
width and returned candidates.

## Response

Read each candidate from `choices[].text`. With the catalog loaded, a
returned SID has this form:

```text
<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>
```

The catalog and prompt determine the actual results. To check membership,
use the [response verifier](../../examples/online_serving/catalog/README.md#verify-a-saved-response).
In full-vocabulary mode, candidates can contain tokens outside the catalog.

## Batching several prompts in one HTTP request

For a compatible deployment, send a non-streaming `POST /v1/completions`
request with a list of prompts and shared Beam settings. This example uses
two prompts and a server configured with width 128, requesting up to two
results per prompt.

```json
{
  "model": "onerec",
  "prompt": [
    "Recommend the next video.",
    "Recommend a video about cooking."
  ],
  "stream": false,
  "use_beam_search": true,
  "n": 128,
  "temperature": 0.0,
  "ignore_eos": true,
  "max_tokens": 5,
  "vllm_xargs": {
    "begin_token": "<|sid_begin|>",
    "end_token": "<|sid_end|>",
    "result_width": 2
  }
}
```

The prompt count must be in the configured `gr.beam_batch_sizes`.
The list must fit `max_num_seqs` and the engine token budget, including
the begin token appended to each prompt. Every prompt must have the same
effective generation budget. The engine still validates KV capacity when it
admits the session.

Prompt lists that fail the batching conditions use independent Beam sessions.
Look for `Beam batch declined` in the server log when diagnosing this path.
See [Beam settings](../configuration/README.md#beam-settings) for capacity
and graph configuration.

### Match results to inputs

The response flattens candidates in input-prompt order. `choices[].index`
is a zero-based position in that flattened list, not the input-prompt index.
For two prompts that each produce two candidates, the response has this
shape (SIDs and token counts are illustrative; optional fields are omitted):

```json
{
  "id": "cmpl-example",
  "object": "text_completion",
  "created": 1700000000,
  "model": "onerec",
  "choices": [
    {"index": 0, "text": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>", "logprobs": null, "finish_reason": "length"},
    {"index": 1, "text": "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>", "logprobs": null, "finish_reason": "length"},
    {"index": 2, "text": "<|sid_begin|><s_a_7><s_b_8><s_c_9><|sid_end|>", "logprobs": null, "finish_reason": "length"},
    {"index": 3, "text": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>", "logprobs": null, "finish_reason": "length"}
  ],
  "usage": {"prompt_tokens": 16, "completion_tokens": 20, "total_tokens": 36}
}
```

Here choices 0 and 1 belong to the first prompt, and 2 and 3 to the second.
`usage.prompt_tokens` sums the input tokens across prompts;
`usage.completion_tokens` sums generated tokens across all candidates.
The default JSON does not expose a per-candidate cumulative Beam score.

The returned count can be smaller than the requested result width, for
example with a small catalog. The response provides neither a `prompt_index`
field nor per-prompt candidate counts. Do not split choices by HTTP `n`:
it is the Beam search width, and even `result_width` is only an upper bound.
When counts can vary and input attribution is required, send one prompt per
HTTP request and retain its application ID, or use the
[gRPC request identities and results](grpc.md).

## Model prompt templates

Send the final model input as `prompt` to `POST /v1/completions`. If your
workload starts from conversation messages, render them locally with the
serving model's tokenizer and chat template before sending the request.
Set `add_special_tokens: false` when the rendered prompt already contains
the model's special tokens. Read generated text from `choices[].text` for
both ordinary and streaming responses.

The OneRec benchmark clients prepare their prompts with the bundled template;
see [benchmark input preparation](../benchmarking/README.md#online-server-mode).

## Authentication

For a server launched with `--api-key` or `VLLM_API_KEY`, add
`-H "Authorization: Bearer $VLLM_API_KEY"` to the HTTP request. See the
[security policy](../../SECURITY.md) for deployment guidance.
