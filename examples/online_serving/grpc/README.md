# OneRec Beam Search over gRPC

Install the [CUDA runtime](../../../docs/getting_started/installation/gpu.md) and the optional gRPC
dependencies from the repository root.

```bash
VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e ".[grpc]"
```

## Check the gRPC installation

Run the smoke test to exercise the local gRPC server and client on CUDA. It
creates a tiny random Llama locally and uses ModelRunner V1. You can run it
before downloading OneRec or preparing a catalog.

```bash
python examples/online_serving/grpc/run_smoke.py --output results/grpc-smoke
```

On success it prints and saves `results/grpc-smoke/result.json` with
`"calls": 2` and `"batch_size": 4`. The two replies have identical token
IDs and scores for the same input. The test then stops its server.

## Serve OneRec over gRPC

Apply the [vLLM CLI dispatch patch](../../../docs/cli/README.md#enable-gr-dispatch).
Set `ONEREC_MODEL` to your
[local OneRec snapshot with its tokenizer](../../README.md#run-onerec-offline).
Run the [shared preparation command](../catalog/README.md#prepare-a-serving-configuration)
with `ONEREC_BACKEND=cuda` to compile the sample or your own catalog. A
configuration already prepared for HTTP on the same model and backend can
be used directly.

### Beam configuration parameters

The server loads the shared YAML through `--config`.
The values below apply to that configuration. Library defaults can differ.

| Parameter | Example value | Server YAML and client CLI |
| --- | --- | --- |
| `gr.beam_graph_enabled` | `true` | Set in YAML to enable Beam graphs. |
| `gr.worker_decision` | `true` | Set in YAML to enable Beam selection on the Worker. |
| `gr.beam_width` | `[128, 128, 128]` | Pass the maximum width with the client flag `--width 128`. |
| `gr.max_decode_steps` | `3` | Set the effective token capacity in YAML. Use `--max-new-tokens 3` for the three-layer Constraint Table. The client handles the boundary markers separately. |

### Start the server

CUDA defaults to ModelRunner V2. See
[ModelRunner selection](../../README.md#choose-a-modelrunner) to select V1.

```bash
export ONEREC_SERVED_MODEL_NAME=onerec
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" vllm serve "$ONEREC_MODEL" --gr \
  --grpc --host 127.0.0.1 --port 50051 \
  --served-model-name "$ONEREC_SERVED_MODEL_NAME" \
  --config "$ONEREC_WORKDIR/vllm_gr_config.yaml" \
  --trust-remote-code \
  --max-logprobs 128 --max-num-seqs 128 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --async-scheduling --no-enable-chunked-prefill
```

Leave the server running. In a second terminal at the repository root, set
`ONEREC_MODEL`, `ONEREC_WORKDIR` and `ONEREC_VERIFY_CATALOG` to the paths
used during [preparation](../catalog/README.md#prepare-a-serving-configuration).
For the default sample, the raw catalog is
`$PWD/examples/online_serving/catalog/sample_catalog.json`.
The readiness check waits up to 300 seconds.

```bash
python -c 'import grpc; grpc.channel_ready_future(grpc.insecure_channel("127.0.0.1:50051")).result(timeout=300)'
```

If the check times out, inspect the server terminal for startup errors.

### Send one request and verify the result

```bash
printf '%s' 'Recommend the next video.' > "$ONEREC_WORKDIR/prompt.txt"
python examples/online_serving/grpc/client.py \
  --model "$ONEREC_MODEL" --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --batch-size 1 --width 128 --max-new-tokens 3 \
  > "$ONEREC_WORKDIR/grpc-result.json" \
&& python examples/online_serving/catalog/verify_catalog.py \
  "$ONEREC_WORKDIR/grpc-result.json" "$ONEREC_VERIFY_CATALOG"
```

Before calling `GenerateBatch`, the client appends the SID begin token to the
prompt. The server returns three generated token IDs per beam. The client wraps
them in boundary tokens for verification. When both sample SIDs are returned,
the verifier prints `SUCCESS: All 2 generated SIDs exist in the catalog.`
Pass a different UTF-8 file to `--prompt` to try your own input. See
[saved response verification](../catalog/README.md#verify-a-saved-response)
to also assert an expected count.
To change the width, [update the server YAML and capacity flags](../catalog/README.md#change-the-beam-width)
and pass the matching value with `--width`.

Press Ctrl+C in the server terminal to stop the service.
