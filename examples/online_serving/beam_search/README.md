# Online Beam Search V1

Serve OneRec over HTTP with catalog-constrained Beam Search. Check the
returned SIDs against the source catalog.

## Prepare the environment and inputs

Install the [CUDA runtime](../../../docs/quickstart.md) or
[Ascend runtime](../../../docs/quickstart-ascend.md). Run the commands below
from the repository root in Bash with Python and curl 7.76 or later.
Apply the [vLLM CLI dispatch patch](../../../docs/cli.md#command-form-and-installation).
Check the installed runtime with `vllm check-env --gr` on CUDA or
`vllm check-env --gr --npu` on Ascend.

Follow the [shared Constraint Table setup](../catalog/README.md#set-the-model-and-dataset-paths)
to set the model and data paths, compile the table, and write the server YAML.

### Beam configuration parameters

The values come from the example YAML passed to `--config`. The HTTP
client flags configure the request. Library defaults can differ.

| Parameter | Example value | Server YAML and client CLI |
| --- | --- | --- |
| `gr.beam_graph_enabled` | `true` | Set in YAML to enable Beam graphs. |
| `gr.worker_decision` | `true` | Set in YAML to enable Beam selection on the Worker. |
| `gr.beam_width` | `[128, 128, 128]` | Set the client flag `-b` to the largest configured width. |
| `gr.max_decode_steps` | `3` | Set the effective token capacity in YAML. The client sends `max_tokens: 5`, which includes two SID markers and three generated tokens. |

## Start the server

CUDA defaults to ModelRunner V2. See
[ModelRunner selection](../../README.md#choose-a-modelrunner) to override it.
The command forces ModelRunner V1 on Ascend.

```bash
export ONEREC_SERVED_MODEL_NAME=onerec
onerec_runner="${VLLM_USE_V2_MODEL_RUNNER:-1}"
if [ "$ONEREC_BACKEND" = npu ]; then
  onerec_runner=0
fi
VLLM_USE_V2_MODEL_RUNNER="$onerec_runner" vllm serve "$ONEREC_MODEL" --gr \
  --host 127.0.0.1 --port 8000 \
  --served-model-name "$ONEREC_SERVED_MODEL_NAME" \
  --config "$ONEREC_WORKDIR/vllm_gr_config.yaml" \
  --trust-remote-code \
  --max-logprobs 128 \
  --max-num-seqs 128 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --async-scheduling \
  --no-enable-chunked-prefill
```

The context limit is 2,048 tokens. With chunked Prefill disabled, keep
`--max-num-batched-tokens` at least as large as `--max-model-len`.

Leave the server running. Model loading, kernel compilation, and graph capture
can take several minutes. In a second terminal at the repository root, set
`ONEREC_WORKDIR` first if you prepared the catalog in a different directory.
The readiness check returns HTTP 200 with an empty body.

```bash
export ONEREC_WORKDIR="${ONEREC_WORKDIR:-$PWD/results/onerec-serving}"
curl --fail --silent --show-error --output /dev/null http://127.0.0.1:8000/health
```

GR patches load through the vLLM plugin. Use `--gr` to enable GR serving
and `--config` to load the YAML. For authentication and deployment guidance, see
the [security policy](../../../SECURITY.md). If the server uses `--api-key`
or `VLLM_API_KEY`, set `VLLM_API_KEY` in the client terminal too.

## Send a request and verify the result

```bash
bash examples/online_serving/beam_search/client.sh \
  > "$ONEREC_WORKDIR/result.json" \
&& python examples/online_serving/catalog/verify_catalog.py \
  "$ONEREC_WORKDIR/result.json" "$ONEREC_WORKDIR/catalog.json"
```

The client sends a plain prompt to `/v1/completions`. Use `-p` to pass a
UTF-8 prompt file. The request uses `ignore_eos: true` and `max_tokens: 5`
for two boundary markers and three SID components.

- To change the served name from `onerec`, set `ONEREC_SERVED_MODEL_NAME` in
  both terminals. Start the server with that value in `--served-model-name`.
  If you change the server address or port, set `ONEREC_BASE_URL` on the client.
- To change the width, [update the server YAML and capacity flags](../catalog/README.md#change-the-beam-width),
  then pass the matching request width with `-b`.

Each `choices[].text` contains a complete SID, for example
`<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>`.
The model, catalog, and prompt determine the SIDs and their order. The verifier
reports the actual number of SIDs when all returned choices pass. See
[saved response verification](../catalog/README.md#verify-a-saved-response)
to also assert an expected count.

Press Ctrl+C in the server terminal to stop the service.

## Troubleshooting

- If `/health` cannot connect, check the server terminal for the startup error
  or wait for model loading and graph capture to finish.
- If the table fails to load, check the absolute artifact path and rebuild
  with the same tokenizer snapshot used by the server.
- If startup reports an error binding an `ipc://` socket on a shared
  filesystem, set `VLLM_RPC_BASE_PATH` to a short, writable directory on the
  serving node that supports Unix sockets.
- If the request returns a model or width error, check the served model name
  and match the request width to `max(gr.beam_width)`.
- If startup runs out of device memory, use a device with enough free memory
  for the model and Beam workspace.
