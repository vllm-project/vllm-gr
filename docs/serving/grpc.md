# gRPC API

Use the experimental GR gRPC service when the client already has token IDs. A
`GenerateBatch` request carries one or more tokenized prompts and returns
candidates grouped by input. Clients that need server-side text or chat
rendering can use the [HTTP completions API](completions.md).

<a id="usage-and-tests"></a>

## Start the server

Follow the [CUDA OneRec gRPC example](../../examples/online_serving/grpc/README.md)
to install the `grpc` extra, prepare a constraint table and start the server on
`127.0.0.1:50051`.

!!! warning "gRPC transport security"

    The gRPC listener uses an unencrypted channel and does not enforce the
    HTTP API key. Keep it on a trusted network and use an authenticated gateway
    with TLS for remote access. See the [security policy](../../SECURITY.md).

## Send a request

From a second terminal, set `ONEREC_MODEL` and `ONEREC_WORKDIR` to the paths
used during setup. This example matches a server with widths `[128, 128, 128]`.

```bash
printf '%s' 'Recommend the next video.' > "$ONEREC_WORKDIR/prompt.txt"
python examples/online_serving/grpc/client.py \
  --address 127.0.0.1:50051 --model "$ONEREC_MODEL" \
  --prompt "$ONEREC_WORKDIR/prompt.txt" --batch-size 1 \
  --width 128 --max-new-tokens 3
```

The client tokenizes the prompt locally and appends the SID begin token.
The three generated tokens are the SID components. The client adds both
boundary markers when decoding the result for catalog verification.

## RPC methods

| RPC | Use |
| --- | --- |
| `Generate` | Submit one tokenized request. |
| `GenerateBatch` | Submit multiple inputs with shared Beam and output options. Each input can select its own `trie_id`. |
| `Abort` | Cancel an in-flight request. |
| `HealthCheck`, `GetModelInfo`, `GetServerInfo` | Inspect the running service. |

<a id="protocol-and-input-handling"></a>

## Token inputs

Supply either repeated unsigned 32-bit token IDs or `input_ids_i64_le`, a
contiguous buffer of little-endian signed 64-bit token IDs. The buffer holds
one prompt, with no shape or length prefix. It must be nonempty, have a byte
length divisible by eight, and contain nonnegative token IDs. Sending both
encodings is an error.

Omitting `trie_id` selects Trie 0. With a multi-Trie table, different inputs
can select different Tries. A single-Trie or unconstrained deployment accepts
only ID 0. See [constraint tables](../configuration/constraint_tables.md).
The [service schema](../../vllm_gr/entrypoints/grpc/proto/vllm_engine.proto)
defines field types and optional metadata.

## Binary Beam results

By default, each batch item carries flat token IDs plus offsets that delimit
its candidates. Set `binary_beam_output` to request little-endian int32
columns for eligible complete, equal-length sequences. Small, stopped or
variable-length results may use the ordinary fields in the same batch.
Decode each item according to the representation it contains.

```python
import numpy as np

request.binary_beam_output = True
response = stub.GenerateBatch(request)
for item in response.items:
    if item.HasField("binary_beams"):
        columns = item.binary_beams
        ids = np.frombuffer(columns.output_ids_i32_le, dtype="<i4")
        candidates = ids.reshape(columns.count, columns.steps).tolist()
    else:
        beams = item.beams
        offsets = beams.output_offsets
        candidates = [
            list(beams.output_ids[start:end])
            for start, end in zip(offsets, offsets[1:])
        ]
```

Binary scores, when present, are little-endian FP32 with one value per
candidate. Parent indices, when requested and present, are little-endian int32
with the same shape as the tokens.

<a id="historical-http-and-grpc-comparison"></a>

See the [HTTP and gRPC benchmark](../benchmarking/throughput.md#historical-http-and-grpc-comparison)
for measurements from a specific workload.
