# GPU Prefill capture configuration

ModelRunner V1 and V2 have separate capture policies. Selecting V2 must not
change V1's configuration or execution behavior. Both can capture B=1/2/4
where admitted by `max_num_seqs`; V1 legacy Beam execution remains B=1.

Both runners default to the request-count policy `beam.online_batch_sizes`,
whose default is `[1, 2, 4]`. An explicit policy such as `[1, 3, 7]` selects
those capture counts without changing sequence or token capacity. A non-default
policy must fit `max_num_seqs`. The Prefill-only environment override
`VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=3,7` takes precedence over this policy;
it accepts increasing counts from 1 through 8. It does not change HTTP batching
or Decode capture. Capture counts are filtered by `max_num_seqs` in both runners.

## ModelRunner V1: preserve the GR grid

V1 retains the policy of `7ee1d95`. Its default GR grid runs from 128 to 8192
in steps of 128. `VLLM_GR_PREFILL_GRAPH_BUCKET_STEP` and
`VLLM_GR_PREFILL_GRAPH_MAX_BUCKET` configure that grid. Native
`cudagraph_capture_sizes` does not replace it. Explicit internal constructor
`buckets` retain their existing precedence.

`VLLM_GR_PREFILL_GRAPH_BUCKETS=128,256,512` selects an explicit, strictly
increasing list of positive token buckets instead of the step/max grid.
When the list is set, the step/max variables are ignored. An invalid or empty
list is rejected rather than falling back to another capture policy.

By default, V1 clamps buckets to the smaller of `max_num_batched_tokens` and the
per-request `max_model_len`; clamping can add that limit as an endpoint.
For `max_model_len=2048`, the GR Prefill runner therefore does not capture
T=4096 even if several requests make that total valid. Such requests retain
the original fallback path.

For an explicit bucket list with Beam V1 execution, buckets describe the total
tokens across requests and are bounded by `max_num_batched_tokens` instead.
Each request still has to fit `max_model_len`.

`VLLM_GR_PREFILL_GRAPH_KV_BOUND` sets the captured Attention bound to
`max(2 * token_bucket, override)` and retains the existing small-bucket
opt-in behavior. V1 async GR Prefill also retains its original requirement
that native input tensors have the actual scheduled token length. GR can
pad its own static buffers after admission.

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

Setting `VLLM_GR_PREFILL_GRAPH_BUCKETS` or either GR bucket step/max variable
explicitly makes V2 use the GR grid instead, with the explicit list taking
precedence over step/max. V2 still filters the resulting sizes against
multi-request capacity. Unset all three bucket variables to use native capture
sizes. The KV-bound variable alone does not extend V2's native list. V2's
Attention capture builder establishes per-request context bounds.

## Compare actual execution

The same native capture configuration no longer implies equal V1/V2 Prefill
coverage. Record actual `(B,T)` shapes, fallback reasons and replay counts.
V1 also retains runtime Decode capture after batch changes, while V2 uses its
startup captures. Separate this capture work from steady-state timing when
that distinction matters to the experiment.

Use `tests.st.model_runner_policy_isolation` for V1 baseline/current behavior
checks and V2 batch-switch checks. Historical experiments that aligned the
runners' capture plans are not proof of V1 baseline compatibility.
