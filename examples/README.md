# Examples

## Run OneRec offline

Install the [CUDA runtime](../docs/getting_started/installation/gpu.md) or
[Ascend runtime](../docs/getting_started/installation/npu.md). Run these commands from the
repository root in Bash. The first run needs only a local OneRec-1.7B model
snapshot with its tokenizer.

If you already have the snapshot, set `ONEREC_MODEL` to its directory.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
```

To download the snapshot with the Hugging Face CLI, use a directory outside
the checkout.

```bash
export ONEREC_MODEL="$HOME/.cache/vllm-gr/models/OneRec-1.7B"
hf download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
```

### CUDA

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

### Ascend

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --constraint-backend npu --execution eager
```

### Expected output

With the defaults, `request=0 returned_beams=128` means the first request
returned 128 candidates. The script shows the five highest-ranked candidates
with their `beam_tokens` and `cumulative_logprob`. It uses
`Recommend the next video.` by default. Pass `--prompt /path/to/prompt.txt`
to use a UTF-8 file at its natural token length.

Full-vocabulary search can return candidates outside your catalog.
[Try a small catalog](offline_inference/beam_search/README.md#run-with-a-constraint-table)
to restrict the output to known items.

## Run OneRec over HTTP

Use the same local model and apply the
[CLI dispatch patch](../docs/cli/README.md#enable-gr-dispatch). On Linux
with Bash and curl 7.76 or later, run:

```bash
python examples/online_serving/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

On Ascend, append `--constraint-backend npu`. The command starts a local
server, sends one prompt, prints the returned SIDs, verifies them against the
two-entry sample catalog, and stops the server. If both SIDs are returned, the
last verification line reads `SUCCESS: All 2 generated SIDs exist in the catalog.`

Both quickstarts accept `--prompt FILE` for a natural-length UTF-8 prompt
and `--catalog FILE` for constraint triples. The offline default searches the
full vocabulary. The HTTP default searches the sample catalog. To compare
the two, supply the same model, prompt and triples to both. See the
[HTTP guide](online_serving/beam_search/README.md) for custom inputs and
instructions for keeping a server running.

## Run gRPC

- [Try gRPC with a tiny local model](online_serving/grpc/README.md#try-grpc-with-a-tiny-local-model)
  by sending two RPCs to a randomly initialized Llama.
- [Serve a real OneRec model over gRPC](online_serving/grpc/README.md#serve-onerec-over-grpc)
  with the same catalog and server configuration as HTTP.

## Run Sidecar hot updates

Follow the [Sidecar example](sidecar/README.md) to configure checkpoint and
Trie updates for running gRPC instances and check the version on each instance.

## Choose a ModelRunner

| Entry point | Default with `VLLM_USE_V2_MODEL_RUNNER` unset |
| --- | --- |
| Library and general CLI | ModelRunner V1 |
| CUDA direct offline OneRec, HTTP serving and OneRec gRPC examples | ModelRunner V2 |
| CUDA offline thinking and tiny-model gRPC smoke test | ModelRunner V1 |
| Ascend Beam examples | ModelRunner V1 |

Set `VLLM_USE_V2_MODEL_RUNNER=0` before startup to run a CUDA example with V1.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

The CUDA examples listed under V2 also select V2 for an empty value. Restart
the process after changing the setting. The Ascend Beam examples and tiny-model
gRPC smoke test explicitly select V1. See [ModelRunner configuration](../docs/configuration/README.md#modelrunner-selection)
for library and CLI use.

Both runners use the same Beam Search API and `gr` configuration.
CUDA Prefill and Decode capture policies differ by runner. See the
[V2 integration guide](../docs/design/module/model_runner_v2.md) and
[Prefill capture guide](../docs/design/feature/cuda_prefill_graph.md#configuration).

## Measure performance

- [Measure offline Beam Search latency](offline_inference/beam_search/README.md#measure-latency-without-profiling)
- [Measure offline or online throughput](../docs/benchmarking/throughput.md)
