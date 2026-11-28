# gRPC API

The experimental GR gRPC service accepts token IDs and returns ranked Beam
candidates grouped by input. To send text, use
[`POST /v1/completions`](completions.md).

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

Run the [OneRec gRPC client example](../../examples/online_serving/grpc/README.md)
against the configured server. It tokenizes locally, appends the SID begin
token, and decodes the three generated SID components with boundary markers
for catalog verification. The wire formats are defined below.

## RPC methods

| RPC | Use |
| --- | --- |
| `Generate` | Submit one tokenized request. |
| `GenerateBatch` | Submit multiple inputs with shared Beam and output options. Each input can select its own `trie_id`. |
| `Abort` | Cancel an in-flight request. Aborting a member cancels its entire Beam batch. |
| `HealthCheck`, `GetModelInfo`, `GetServerInfo` | Inspect the running service. |

<a id="protocol-and-input-handling"></a>

## Token inputs

Choose one input encoding per prompt.

| Encoding | Representation |
| --- | --- |
| Repeated token IDs | Unsigned 32-bit integers |
| `input_ids_i64_le` | Nonempty, contiguous little-endian signed 64-bit integers with nonnegative values. Encode one prompt with no shape or length prefix. The byte length is divisible by eight. |

Omitting `trie_id` selects Trie 0. With a multi-Trie table, different inputs
can select different Tries. A single-Trie or unconstrained deployment accepts
only ID 0. See [constraint tables](../configuration/constraint_tables.md).
The [service schema](../../vllm_gr/entrypoints/grpc/proto/vllm_engine.proto)
defines field types and optional metadata.

## Binary Beam results

Decode each batch item using the fields the server returned.

| Fields | Representation |
| --- | --- |
| `beams` | Flat token IDs with offsets that delimit candidates. |
| `binary_beams` | Little-endian int32 columns for eligible complete, equal-length sequences. Request this format with `binary_beam_output`. |

A batch can contain both formats because small, stopped, or variable-length
results may use `beams` when binary output was requested.

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
