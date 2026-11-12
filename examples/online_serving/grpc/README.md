# OneRec Beam Search over gRPC

Install the [CUDA runtime](../../../docs/getting_started/installation/cuda.md) and the optional gRPC
dependencies from the repository root.

```bash
VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e ".[grpc]"
```

Apply the [vLLM CLI dispatch patch](../../../docs/cli/README.md#enable-gr-dispatch).

Follow the [shared Constraint Table setup](../catalog/README.md#set-the-model-and-dataset-paths)
to set the model and data paths, compile a CUDA table, and write the server YAML.

## Beam configuration parameters

The server loads the shared YAML through `--config`.
The values below apply to that configuration. Library defaults can differ.

| Parameter | Example value | Server YAML and client CLI |
| --- | --- | --- |
| `gr.beam_graph_enabled` | `true` | Set in YAML to enable Beam graphs. |
| `gr.worker_decision` | `true` | Set in YAML to enable Beam selection on the Worker. |
| `gr.beam_width` | `[128, 128, 128]` | Match the first width with the client flag `--width 128`. |
| `gr.max_decode_steps` | `3` | Set the effective token capacity in YAML. Use `--max-new-tokens 3` for the three-layer Constraint Table. The client handles the boundary markers separately. |

## Start the server

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
`ONEREC_MODEL` and `ONEREC_WORKDIR` to the paths used during setup. The
readiness check waits up to 300 seconds.

```bash
python -c 'import grpc; grpc.channel_ready_future(grpc.insecure_channel("127.0.0.1:50051")).result(timeout=300)'
```

If the check times out, inspect the server terminal for startup errors.

## Send one request and verify the result

```bash
printf '%s' 'Recommend the next video.' > "$ONEREC_WORKDIR/prompt.txt"
python examples/online_serving/grpc/client.py \
  --model "$ONEREC_MODEL" --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --batch-size 1 --width 128 --max-new-tokens 3 \
  > "$ONEREC_WORKDIR/grpc-result.json" \
&& python examples/online_serving/catalog/verify_catalog.py \
  "$ONEREC_WORKDIR/grpc-result.json" "$ONEREC_WORKDIR/catalog.json"
```

Before calling `GenerateBatch`, the client appends the SID begin token to the
prompt. The server returns three generated token IDs per beam. The client wraps
them in boundary tokens for verification. The verifier reports the actual
number of SIDs when all returned choices pass. See
[saved response verification](../catalog/README.md#verify-a-saved-response)
to also assert an expected count.
To change the width, [update the server YAML and capacity flags](../catalog/README.md#change-the-beam-width)
and pass the matching value with `--width`.

Press Ctrl+C in the server terminal to stop the service.

For a separate CUDA smoke test that builds a tiny random model, run
`python examples/online_serving/grpc/run_smoke.py --output results/grpc-smoke`.
