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
| `use_beam_search` | Selects Beam generation when set to `true` |
| `n` | Request width. Set it to the maximum value in the server's `gr.beam_width` list. |
| `max_tokens` | Token budget. For this catalog, five covers the two boundary markers and three generated SID components. |
| `vllm_xargs.begin_token`, `vllm_xargs.end_token` | SID boundary markers used for this request |
| `vllm_xargs.result_width` | Optional number of candidates to return, up to the final retained width |

!!! tip "Request width and result count"

    The example uses a fixed width of 128. With `gr.beam_width: [8, 4, 2]`,
    send `n: 8`. A three-step request returns two candidates by default or one
    with `result_width: 1`. See [Beam settings](../configuration/README.md#beam-settings)
    for deployment defaults and candidate widths.

## Response

Read each candidate from `choices[].text`. With the catalog loaded, a
returned SID has this form:

```text
<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>
```

The catalog and prompt determine the actual results. To check membership,
use the [response verifier](../../examples/online_serving/catalog/README.md#verify-a-saved-response).
In full-vocabulary mode, candidates can contain tokens outside the catalog.

## Chat completions

Use `POST /v1/chat/completions` with the same Beam settings and replace
`prompt` with `messages`, for example:

```json
{"messages": [{"role": "user", "content": "Recommend the next video."}]}
```

The server renders the conversation with its chat template. For the OneRec
accuracy workload, add `--chat-template benchmarks/open_one_rec/qwen3_soft_switch.jinja2`
to the same server command. Read each candidate from
`choices[].message.content`. See the [accuracy benchmark](../benchmarking/accuracy.md)
for the dataset-backed client.

## Authentication

For a server launched with `--api-key` or `VLLM_API_KEY`, add
`-H "Authorization: Bearer $VLLM_API_KEY"` to the HTTP request. See the
[security policy](../../SECURITY.md) for deployment guidance.
