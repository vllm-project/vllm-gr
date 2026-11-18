# GR gRPC and NumPy token inputs

## Protocol and input handling

GR gRPC accepts token prompts and keeps them as NumPy arrays through EngineCore
and Worker staging.

- The schema extends SMG protobuf 0.4.6 with beam options, `GenerateBatch` and optional
  `input_ids_i64_le`, preserving original field numbers and method names.
  Schema headers identify the source and additions.
- The service accepts either repeated uint32 tokens via `np.asarray(..., dtype=np.int64)`
  or raw little-endian int64 bytes via `np.frombuffer`. Each binary buffer holds
  one prompt without a header or shape prefix; token count is byte length / 8.
  Conflicting fields, empty/unaligned data and negative tokens fail. After
  truncation, vocabulary validation matches native InputProcessor: use the
  larger tokenizer/model bound when a tokenizer exists, otherwise skip the
  vocabulary upper-bound check. Output vocabulary slicing does not restrict
  prompt IDs. Truncation follows the tokenizer's configured side.
- `BeamVllmEngineServicer` constructs the existing `BeamBatchRequest`, reuses
  submit/wait/close sessions, and fills flat per-sample protobuf beam outputs.
  GR Generate, GenerateBatch and Abort are supported. The servicer inherits
  upstream HealthCheck, GetModelInfo and GetServerInfo; Embed remains disabled.
  GetTokenizer remains UNIMPLEMENTED, as in the pinned upstream version.
- `GenerateRequest.trie_id` selects a Trie for each input; omitted selects Trie 0.
  `GenerateBatch` members can select different Tries while sharing beam/output
  options. CUDA and NPU support Multi-Trie execution with a multi-Trie constraint
  table. Engine Core validates IDs against the loaded table before admission.
  A single-Trie or unconstrained deployment accepts only ID 0, matching
  Python/HTTP. See [Multi-Trie constraint tables](multi_root_constraint_table.md)
  for backend scoring semantics and execution limits.
- GR installs the bindings and servicer during upstream gRPC startup. Upstream owns
  engine/listener initialization, standard health, reflection, keepalive,
  statistics and shutdown. GR supplies extended-schema reflection and drains
  session cleanup before engine shutdown.
- `BeamEngineCoreRequest` carries NumPy prompts. Core token storage and
  scheduler adapters live in `v1/beam/token_buffer.py`; V2 GPU staging lives in
  `v1/worker/numpy_staging.py`. Prefix hashes preserve baseline cache identity.

```text
protobuf repeated / bytes -> NumPy -> BeamBatchRequest
 -> msgpack/ndarray IPC -> Core token buffer -> scheduler -> Worker staging/H2D
 -> model + beam -> protobuf response
```

Frontend arrays pass through without a defensive snapshot or forced read-only
flag. Legacy list inputs convert at the batch boundary. Core-owned storage,
CPU staging and H2D still copy; this is not end-to-end zero copy. HTTP/offline
output boundaries retain their list/tokenizer interfaces. Internal batch wire
version is 2; frontend, Core and Workers must restart with the same revision.
The GR batch contract requires nonempty contiguous nonnegative int64 arrays;
`None` and embedding-only prompts are rejected.

In vLLM 0.30, V2 can flush staged token contents through a native UVA buffer
instead of `async_tensor_h2d`. `NumpyStagedContents` implements `__array__`
so the native NumPy assignment materializes the chunks once. Falling back to
per-token `__getitem__` would concatenate every chunk for every token in a
multi-request Prefill. Both transfer paths retain native write ordering and
buffer ownership; read-only token inputs remain supported. Regression tests
cover the UVA assignment, buffer reuse, and array dtype/copy semantics.

## Usage and tests

**Security:** In the supported vLLM runtime, the upstream gRPC listener is
unencrypted and has no built-in authentication or authorization. The HTTP
API key does not protect gRPC, including health and reflection. Do not expose
the port to untrusted networks. Restrict backend access with network controls;
use a trusted gateway when remote access requires TLS and access control. See
the [security policy](../SECURITY.md) and the
[vLLM security guide](https://docs.vllm.ai/en/latest/usage/security.html).
Behavior depends on the installed vLLM version. In the reference implementation,
omitting `--host` binds to `0.0.0.0`. The example below binds to loopback only;
this limits reachability but does not add authentication.

Install `pip install -e '.[grpc]'`, then use existing GR settings:

```bash
vllm serve /path/to/model --gr --vllm-gr-config /path/to/gr-config.json --grpc --host 127.0.0.1 --port 50051
python examples/online_serving/grpc/client.py --address 127.0.0.1:50051
python examples/online_serving/grpc/run_smoke.py --output /tmp/gr-grpc-smoke
```

The CI pre-commit job installs `.[dev,grpc]` and checks runtime imports before
running tests. Local gRPC tests skip when optional dependencies are missing or
grpcio/protobuf are below the generated bindings' required versions.

The GPU example creates random tiny Llama weights and checks real production
CLI requests. Generate bindings with `python tools/generate_grpc_proto.py`
after installing `grpcio-tools==1.84.0`.

Focused tests cover pinned upstream wire compatibility, both token encodings,
input/beam validation, batch/single output fields, cancellation/error cleanup,
standard health/reflection/stats, array passthrough, codec transport, prefix
hash identity, token growth and Worker staging.

## Historical HTTP and gRPC comparison

> Measured on NVIDIA H20 / date not recorded / ModelRunner V1 eager, FP16
> (runtime versions not recorded). Not a performance guarantee.

The [original gRPC validation](https://github.com/JiusiServe/vllm-gr/pull/542)
used a [decode_graph snapshot](https://github.com/JiusiServe/vllm-gr/commit/2b84265)
for HTTP and the gRPC implementation under review for both gRPC inputs. The paths shared random Qwen3 MoE weights (4.149B total / 0.311B
active), batch size 4, 4096 input tokens per prompt and Beam width 256. Each
request ran one Prefill and three Decode stages. Prefix Cache, Chunked Prefill
and constraints were disabled. Each path used 10 warmups and 200 measured
requests, alternating GPU probes. All 600 outputs matched.

| Per batch (ms) | Baseline HTTP/JSON/list | gRPC binary | gRPC repeated |
| --- | ---: | ---: | ---: |
| Python frontend, input + output | 16.755 | 1.897 | 2.117 |
| Model forward, excluding logits | 186.342 | 187.474 | 187.372 |
| Beam search, including logits | 11.475 | 11.491 | 11.508 |
| Client RTT minus Python server time | 1.339 | 0.719 | 0.646 |
| Other server/framework time | 5.465 | 8.345 | 4.836 |
| Client RTT | 221.375 | 209.925 | 206.479 |
| RTT median, GPU probes off | 220.944 | 206.324 | 206.476 |
| RTT P95, GPU probes off | 223.107 | 206.931 | 207.105 |

The first six rows are means from the 100 GPU-probed requests per path; the
first five sum to RTT. The last two rows use 100 GPU-probe-off controls;
Python server boundary hooks remain enabled. Client request construction,
encoding and reply parsing are outside RTT. Model and beam times sum four
batch forwards/stages, without multiplying again by batch size.

Other framework time is server time minus frontend, model and beam. The
client/server difference includes transport/runtime scheduling, not just wire
protocol work. HTTP returns text while gRPC returns token arrays, so frontend
savings include decoding and response formatting. All samples are retained;
isolated gRPC wall-time spikes affect means, so median/P95 are also reported.

## Binary Beam results

The default `GenerateBatch` response retains the existing flat Beam fields.
For large complete results, the internal result can travel as owned columns
without constructing one Python object per candidate. This representation
does not depend on input token encoding or a particular model architecture.

Clients can additionally set `BatchGenerateRequest.binary_beam_output = true`.
Eligible items then return `binary_beams` with `count`, `steps`, `prompt_tokens`,
and little-endian int32 `output_ids_i32_le` in `[count, steps]` row order.
Requested scores use optional FP32 `cumulative_logprobs_f32_le`; requested parent
indices use optional int32 `parent_beam_indices_i32_le` with the same row shape.

Clients opting in must handle both `binary_beams` and ordinary `beams`: small,
stopped, or variable-length items retain the existing response. The binary
option changes the output representation only; default gRPC, HTTP and offline
outputs keep their existing behavior.

For an existing `BatchGenerateRequest` and generated gRPC stub:

```python
request.binary_beam_output = True
response = stub.GenerateBatch(request)
for item in response.items:
    if item.HasField("binary_beams"):
        columns = item.binary_beams
        tokens = np.frombuffer(columns.output_ids_i32_le, dtype="<i4").reshape(
            columns.count, columns.steps
        )
    else:
        # Existing flat Beam fields, including the small/STOP fallback.
        tokens = item.beams.output_ids
```
