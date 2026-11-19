# GPU Prefill capture configuration

ModelRunner V1 and V2 have separate capture policies. Selecting V2 must not
change V1's configuration or execution behavior. Both can capture B=1/2/4
where admitted by `max_num_seqs`.

Both runners default to the request-count policy `beam.online_batch_sizes`,
whose default is `[1, 2, 4]`. An explicit policy such as `[1, 3, 7]` selects
those capture counts without changing sequence or token capacity. A non-default
policy must fit `max_num_seqs`. The Prefill-only environment override
`VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=3,7` takes precedence over this policy;
it accepts increasing counts from 1 through 8. It does not change HTTP batching
or Decode capture. Capture counts are filtered by `max_num_seqs` in both runners.

## ModelRunner V1: preserve the GR grid

The default V1 GR grid runs from 128 to 8192
in steps of 128. `VLLM_GR_PREFILL_GRAPH_BUCKET_STEP` and
`VLLM_GR_PREFILL_GRAPH_MAX_BUCKET` configure that grid. Native
`cudagraph_capture_sizes` does not replace it. Explicit internal constructor
`buckets` retain their existing precedence.

V1 clamps buckets to the smaller of `max_num_batched_tokens` and the
per-request `max_model_len`; clamping can add that limit as an endpoint.
For `max_model_len=2048`, the GR Prefill runner therefore does not capture
T=4096 even if several requests make that total valid. Such requests retain
the original fallback path.

`VLLM_GR_PREFILL_GRAPH_KV_BOUND` sets the captured Attention bound to
`max(2 * token_bucket, override)` and retains the existing small-bucket
opt-in behavior. V1 async GR Prefill also retains its original requirement
that native input tensors have the actual scheduled token length. GR can
pad its own static buffers after admission.

## Dense CUDA Prefill (opt-in)

Captured CUDA Prefill attention reads K/V from the paged cache, which selects
FA2's split-KV kernel. `VLLM_GR_DENSE_PREFILL=fa2` instead attends over the
layer's freshly projected contiguous K/V with the dense FA2 kernel;
`VLLM_GR_DENSE_PREFILL=cudnn` uses cuDNN SDPA for single-request Prefill of at
least 64 tokens and dense FA2 otherwise. The KV cache is still written. The
default `off` keeps the paged path.

The mode applies only when prefix caching, chunked Prefill and KV connectors
are all disabled, because only then does every captured Prefill hold a fresh
prompt whose K/V rows align with its query rows. Each call also checks the CPU
lengths: the pure-Prefill entry also serves eager batches and non-Beam decode
rows, and any request whose context is longer than its scheduled tokens keeps
the whole batch on the paged path. `VLLM_BATCH_INVARIANT`, layers with sliding
windows, ALiBi, softcap or sinks, quantized KV caches, KV sharing and NPU also
keep the paged path. If cuDNN cannot serve the shape, dense FA2 is used.

A CUDA graph freezes the choice made at capture, so the CPU check alone is not
enough: native decode capture uses `seq_len == query_len == 1` dummies and then
replays over cached context. While a stream is capturing, dense attention is
selected only inside the fresh-Prefill scope that the GR Prefill runners (V1 and
V2) open; all other captures keep the paged path. With dense Prefill requested,
those runners replay a FULL Prefill graph only when every scheduled request has
no computed tokens; otherwise V2 uses its PIECEWISE graph and V1 the native path,
where attention runs eagerly with the CPU check.

## ModelRunner V2: use the native capture plan

V2 consumes the resolved `compilation_config.cudagraph_capture_sizes` list.
It filters impossible sizes without adding capacity endpoints or extra small
buckets. An empty native list stays empty. Native defaults may cover only
small batches, so configure larger sizes explicitly when needed.

`max_model_len` bounds each member; `max_num_batched_tokens` bounds the entire
iteration. For each B, capture candidates must fit both the iteration budget
and `B * max_model_len`. FULL candidates must also fit the native dummy split:
`T // B + T % B <= max_model_len`, because native capture places the entire
remainder in the last request. At B=4 and a model limit of 2048, T=8190/8191
are excluded from FULL capture, while T=8189/8192 remain valid. Filtered FULL
shapes retain their token buckets for same-bucket PIECEWISE fallback, or eager
execution when PIECEWISE is unavailable; dispatch does not try a larger FULL
bucket.

For `max_num_seqs=4`, `max_model_len=2048` and
`max_num_batched_tokens=4096`, this configuration permits `(B=2,T=4096)` and
`(B=4,T=4096)` while excluding `(B=1,T=4096)`:

```python
compilation_config={
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, *range(128, 4097, 128)],
}
```

Setting either GR bucket step/max variable explicitly makes V2 use the GR
grid instead. V2 still filters the resulting sizes against multi-request
capacity. Unset those two variables to use native capture sizes. The KV-bound
variable alone does not extend V2's native list. V2's Attention capture
builder establishes per-request context bounds.

## Compare actual execution

The same native capture configuration no longer implies equal V1/V2 Prefill
coverage. Record actual `(B,T)` shapes, fallback reasons and replay counts.
V1 also retains runtime Decode capture after batch changes, while V2 uses its
startup captures. Separate this capture work from steady-state timing when
that distinction matters to the experiment.

Use `tests.st.model_runner_policy_isolation` for V1 baseline/current behavior
checks and V2 batch-switch checks. Historical experiments that aligned the
runners' capture plans are not proof of V1 baseline compatibility.
