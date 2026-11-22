# vllm serve

`vllm serve ... --gr` starts a GR server. The same command serves the
OpenAI-compatible HTTP API; add `--grpc` for the gRPC transport.
For OneRec Beam serving, use the
[prepared serving configuration](../../examples/online_serving/catalog/README.md#prepare-a-serving-configuration)
in place of `engine.yaml`. On Ascend, select
[ModelRunner V1](../configuration/README.md#modelrunner-selection) by setting
`VLLM_USE_V2_MODEL_RUNNER=0` before starting the server or running its help.

## Usage

=== "HTTP"

    ```bash
    vllm serve /models/OneRec --gr \
      --config engine.yaml \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --max-model-len 8192 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 128 --max-logprobs 128 \
      --gpu-memory-utilization 0.9 \
      --async-scheduling --no-enable-chunked-prefill
    ```

=== "gRPC"

    ```bash
    vllm serve /models/OneRec --gr --config engine.yaml --grpc --port 8000
    ```

## Arguments

The model may be positional or supplied with `--model`. Native server options
retain the installed vLLM argument contract. Use `--help=all` for the complete
version-specific inventory, including authentication, TLS, parallelism, tokenizer,
cache, scheduling, and transport options. This CLI does not maintain a second copy
of those definitions.

| Option | Form and purpose |
| --- | --- |
| `--gr` | Boolean switch selecting the GR command |
| `--grpc` | Select the GR gRPC server; requires the gRPC extra |
| `--host`, `--port` | Listening address and integer port |
| `--served-model-name` | Model name exposed to clients |
| `--revision` | Model revision or immutable snapshot |
| `--trust-remote-code` | Allow the model's custom Python implementation |
| `--dtype` | Native model dtype, such as `auto` or `bfloat16` |
| `--max-model-len` | Maximum model sequence length |
| `--max-num-seqs` | Scheduler sequence capacity |
| `--max-num-batched-tokens` | Scheduler token budget |
| `--max-logprobs` | Candidate/logprob capacity |
| `--gpu-memory-utilization` | Device memory utilization fraction |
| `--async-scheduling` | Enable asynchronous scheduling |
| `--enable-prefix-caching`, `--no-enable-prefix-caching` | Explicit prefix-cache setting |
| `--enable-chunked-prefill`, `--no-enable-chunked-prefill` | Explicit chunked-Prefill setting |
| `--enforce-eager` | Disable graph execution |
| `--attention-config` | Native JSON attention configuration |

!!! note "Beam serving"

    Start the Beam server with `--async-scheduling --no-enable-chunked-prefill`.

## Configuration

Pass a startup YAML file with `--config`. Native vLLM settings are top-level
keys and GR settings go under `gr`. An explicit CLI option overrides the value
from the file. For example, given an `engine.yaml` that specifies the model:

```bash
vllm serve --gr --config engine.yaml --port 9000
```

Override a single GR field with `--gr.FIELD`, or pass a JSON object with
`--gr-config`. These two commands set the same Beam batch sizes:

```bash
vllm serve --gr --config engine.yaml --gr.beam_batch_sizes '[1,2,4]'
vllm serve --gr --config engine.yaml --gr-config '{"beam_batch_sizes":[1,2,4]}'
```

See [Configuration](../configuration/README.md) for YAML examples, defaults
and validation rules.

For the HTTP request and response shape, see
[the completions API](../serving/completions.md). For the gRPC protocol,
see [gRPC token inputs](../serving/grpc.md).
