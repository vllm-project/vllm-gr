# OneRec Beam Search over HTTP

Run one verified request against a temporary local server, or
[keep a server running](#prepare-a-persistent-service) for your own clients.

## Prepare the environment and inputs

Install the [CUDA runtime](../../../docs/getting_started/installation/gpu.md) or
[Ascend runtime](../../../docs/getting_started/installation/npu.md). Set `ONEREC_MODEL` to a
[local OneRec snapshot with its tokenizer](../../README.md#run-onerec-offline).
Run the commands below from the repository root on Linux with Python, Bash
and curl 7.76 or later.
Apply the [vLLM CLI dispatch patch](../../../docs/cli/README.md#enable-gr-dispatch).
Check the installed runtime with `vllm check-env --gr` on CUDA or
`vllm check-env --gr --npu` on Ascend.

## Run one request

```bash
python examples/online_serving/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

On Ascend, append `--constraint-backend npu`. The quickstart builds a
Constraint Table from the [two-SID sample](../catalog/sample_catalog.json),
starts `vllm serve` on a loopback port, and sends `Recommend the next video.`
at its natural length. It requests width 128 with three SID components.

### Expected output

The quickstart prints candidates and verifies every complete SID against the
raw catalog. When both sample SIDs are returned, the output includes:

```text
<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>
<|sid_begin|><s_a_1><s_b_2><s_c_4><|sid_end|>
SUCCESS: All 2 generated SIDs exist in the catalog.
```

The count reflects the returned choices. The model and prompt determine their
order. The quickstart stops its server and workers after the request finishes
or when interrupted by Ctrl+C or `SIGTERM`. It prints the run directory with
`server.log` and `result.json` (once a response is received). Pass `--workdir DIR`
to keep each run in a separate subdirectory of your chosen location. The
default parent is the system temporary directory.

### Use your own inputs

Pass `--prompt FILE` to read a UTF-8 prompt. Both quickstarts use
`--catalog FILE` for **constraint triples**. For example, reuse the
[offline two-SID example](../../offline_inference/beam_search/README.md#run-with-a-constraint-table)
from the repository root:

```bash
export ONEREC_WORKDIR="$PWD/results/onerec-example"
python examples/online_serving/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --catalog "$ONEREC_WORKDIR/constraint_triples.json"
```

With custom triples, the quickstart looks for the original five-token SID
sequences in `catalog.json` beside the triples. If the raw catalog is
elsewhere, pass `--verify-catalog /path/to/raw-catalog.json` too. To compare
offline and HTTP results, give both commands the same model, prompt and
triples. The [offline quickstart](../../offline_inference/beam_search/README.md)
searches the full vocabulary by default.

## Prepare a persistent service

Follow the [shared preparation command](../catalog/README.md#prepare-a-serving-configuration)
to set `ONEREC_MODEL`, `ONEREC_WORKDIR` and `ONEREC_BACKEND`, then build
`$ONEREC_WORKDIR/vllm_gr_config.yaml`. The same YAML serves HTTP and gRPC.

### Beam configuration parameters

The values come from the example YAML passed to `--config`. The HTTP
client flags configure the request. Library defaults can differ.

| Parameter | Example value | Server YAML and client CLI |
| --- | --- | --- |
| `gr.beam_graph_enabled` | `true` | Set in YAML to enable Beam graphs. |
| `gr.worker_decision` | `true` | Set in YAML to enable Beam selection on the Worker. |
| `gr.beam_width` | `[128, 128, 128]` | Pass the maximum width with the client flag `-b`. |
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
can take several minutes. In a second terminal at the repository root, use the
same `ONEREC_WORKDIR` and `ONEREC_VERIFY_CATALOG` as in the
[preparation step](../catalog/README.md#prepare-a-serving-configuration).
For the default sample, use the values below.
The readiness check returns HTTP 200 with an empty body.

```bash
export ONEREC_WORKDIR="${ONEREC_WORKDIR:-$PWD/results/onerec-serving}"
export ONEREC_VERIFY_CATALOG="${ONEREC_VERIFY_CATALOG:-$PWD/examples/online_serving/catalog/sample_catalog.json}"
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
  "$ONEREC_WORKDIR/result.json" "$ONEREC_VERIFY_CATALOG"
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
`<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>`.
The model, catalog, and prompt determine the SIDs and their order. The verifier
reports the actual number of SIDs when all returned choices pass. See
[saved response verification](../catalog/README.md#verify-a-saved-response)
to also assert an expected count.

Press Ctrl+C in the server terminal to stop the service.

## Troubleshooting

- If the one-request quickstart fails, inspect `server.log` in the printed run
  directory. The quickstart waits up to 600 seconds for its own server.
- If `/health` cannot connect, check the server terminal for the startup error
  or wait for model loading and graph capture to finish.
- If the table fails to load, check the absolute artifact path and rebuild
  with the same tokenizer snapshot used by the server.
- If startup reports an error binding an `ipc://` socket on a shared
  filesystem, set `VLLM_RPC_BASE_PATH` to a short, writable directory on the
  serving node that supports Unix sockets.
- If the request returns a model or width error, check the served model name
  and match the request width to the maximum value in `gr.beam_width`.
- If startup runs out of device memory, use a device with enough free memory
  for the model and Beam workspace.
