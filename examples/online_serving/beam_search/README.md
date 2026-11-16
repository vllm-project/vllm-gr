# Online Beam Search V1

Install the [CUDA runtime](../../../docs/quickstart.md) or
[Ascend runtime](../../../docs/quickstart-ascend.md) first. Run commands from
the repository root with a local model snapshot and its matching tokenizer.

## Prepare the canonical configuration

Follow the [artifact build instructions](../../offline_inference/beam_search/README.md#constraint-table-artifact)
to compile the real catalog into a `.safetensors` Constraint Table. Copy the
[engine configuration](../../offline_inference/beam_search/README.md#engine-configuration)
into `vllm_gr_config.yaml`, replacing the artifact path and both digests.
Use `constraint_table.backend: cuda` on GPU or `npu` on Ascend. Keep
`beam.max_width: 128`, `beam.max_decode_steps: 3`, and `execution_mode: v1`
for this example. NPU execution choices are documented in the NPU quickstart.

## Start the server

The HTTP server uses vLLM's optional `--api-key` or `VLLM_API_KEY` and accepts
`Authorization: Bearer <key>` on protected routes. Without a key, that check
is disabled. In the supported vLLM runtime, the check covers paths beginning
with `/v1`, `/v2`, `/inference`, and `/cohere`; other routes, such as `/health`
and `/invocations`, are not covered, and `OPTIONS` requests are skipped. Verify
the installed vLLM version and enabled routes. Restrict access to the backend
and use the deployment's
gateway for TLS, authentication, authorization, and route restrictions.
See the [security policy](../../../SECURITY.md).

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
vllm serve "$ONEREC_MODEL" --gr \
  --served-model-name onerec \
  --vllm-gr-config ./vllm_gr_config.yaml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-logprobs 128 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --async-scheduling \
  --no-enable-chunked-prefill
```

GR patches load through the vLLM plugin. Do not combine the canonical config
with legacy GR CLI options such as `--catalog-path` or `--beam-max-width`.
Native prefix caching may be enabled without changing the artifact contract.

## Send a request

From another terminal:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{
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

The public budget includes the SID begin/end markers, leaving three generated
SID components. The server returns complete candidates after matching resource
release; candidate scores and latency depend on model, prompt, and hardware.
For supported prompt-list batching, see the
[V1 batching guide](../../../docs/onerec_v1_async_quickstart.md#batching-several-prompts-in-one-http-request).
