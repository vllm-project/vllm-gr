# GPU Prefill capture configuration

ModelRunner V1 and V2 have separate capture policies. Selecting V2 must not
change V1's configuration or execution behavior. Both can capture B=1/2/4
where admitted by `max_num_seqs`.

Both CUDA ModelRunners, for Prefill and Decode, capture request counts from `gr.beam_batch_sizes`
(default `[1, 2, 4]`), filtered by `max_num_seqs`. The Beam completion route uses the same list to admit a prompt list as one session. Sizes must be strictly increasing integers from 1 to 8.

## ModelRunner V1

The packaged `gr.prefill_token_buckets` list contains multiples of 128 up to
8192. Replace the list to choose different token lengths. With a null list,
V1 uses its built-in 128 to 8192 range. Explicit internal constructor
`buckets` take precedence.

For GPU V1 Beam execution, every non-null configured list, including the
packaged default, represents total tokens across the batch. Before allocating
static buffers, V1 builds one capture plan shared with graph capture. Each
`(B,T)` must satisfy `B <= max_num_seqs`, `B <= T`,
`T <= max_num_batched_tokens`, and `T <= B * max_model_len`.
Buckets with no feasible request count receive no buffers. For model length
512, token budget 8192, and capture batches `[1,2,4]`, the default grid allocates
16 buckets through 2048 instead of 64 through 8192.

With `gr.prefill_token_buckets: null`, V1 retains its per-context limit:
clamp to the smaller of the token budget and model length, potentially adding
that limit as an endpoint. V2's null policy below remains separate.

`gr.prefill_kv_bound` sets the captured Attention bound to
`max(2 * token_bucket, override)` and retains the existing small-bucket
opt-in behavior. V1 async GR Prefill also retains its original requirement
that native input tensors have the actual scheduled token length. GR can
pad its own static buffers after admission.

## Dense CUDA Prefill (opt-in)

Captured CUDA Prefill attention reads K/V from the paged cache, which selects
FA2's split-KV kernel. Setting `gr.dense_prefill` to `fa2` attends over the
layer's freshly projected contiguous K/V with the dense FA2 kernel;
`gr.dense_prefill: cudnn` uses cuDNN SDPA for single-request Prefill of at
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

V2 uses the same packaged token list as V1. Set `gr.prefill_token_buckets: null`
to use the resolved `compilation_config.cudagraph_capture_sizes` list.
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

With `gr.prefill_token_buckets: null`, for `max_num_seqs=4`, `max_model_len=2048` and
`max_num_batched_tokens=4096`, this configuration permits `(B=2,T=4096)` and
`(B=4,T=4096)` while excluding `(B=1,T=4096)`:

```python
compilation_config={
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, *range(128, 4097, 128)],
}
```

V2 filters an explicit GR list against multi-request capacity as well.
The KV-bound setting alone does not extend the native list.

## Compare actual execution

The same native capture configuration no longer implies equal V1/V2 Prefill
coverage. Record actual `(B,T)` shapes, fallback reasons and replay counts.
Both runners pre-capture configured Decode batches. V1 can recapture a
configured shape when the underlying buffers change. Separate this capture work from steady-state timing when
that distinction matters to the experiment.

Use `tests.st.model_runner_policy_isolation` for V1 baseline/current behavior
checks and V2 batch-switch checks. Historical experiments that aligned the
runners' capture plans are not proof of V1 baseline compatibility.
