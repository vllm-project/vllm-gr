# Short suffix attention on CUDA

Enabled by default on eligible CUDA CUSTOM Beam engines, incremental beam
decode with one query per beam uses a fused Triton suffix/merge kernel for any
configured positive KV pool capacity. It reads K/V from the existing pool, accumulates suffix scores in
FP32, and merges them
with the prefix attention output and natural-log LSE. It writes the final
attention output directly.

Prefix attention exposes canonical out + FP32 natural-log LSE states; persistent KV writes are unchanged.
With `beam.enable_kv_pool_ping_pong` enabled (the CUDA fixed-width default),
KV writes and both suffix implementations select one of two fixed-address
banks from the device Decode step. Selection gathers history into the other
bank for the next step, omitting the write-back scatter. The step counter
already advances after selection, so bank selection adds no kernel or graph
recapture. Set the field to `False` to restore the original single-bank
gather/scatter path. Dynamic-width and NPU execution keep their existing KV
reorder paths. The Qwen benchmark documents the controlled A/B switch.
The fast path removes packed suffix copies, suffix FlashAttention,
intermediate suffix outputs, FA2 GQA LSE conversion, and a separate merge.
It supports CUDA graph replay: the step and active mask are device inputs,
and inactive beams retain the old zero-K/V suffix semantics.
The pool slot is a runtime scalar with Triton value specialization disabled,
so changing slots at a fixed tensor geometry reuses the compiled kernel.
A CUDA graph captures its scalar slot argument; step and mask remain mutable
device inputs during replay.

The dispatch is restricted to pure incremental decode and unwindowed
attention without ALiBi, softcap, or sinks. Head dimensions of 64/128/256 use
the FA2-aligned MMA fast path; other head sizes that FlashAttention 2 accepts
(a multiple of 8, at most 256) use the generic Triton merge kernel.
With the optimization disabled, configurations use the original implementation.
The kernel itself supports FP32 for independent correctness validation, in addition
to the FP16/BF16 activation types used by the model runner.

## Configuration

The fused implementation is enabled by default for CUDA CUSTOM attention with
an active Beam pool. To opt out, set the field explicitly in the engine's
existing `vllm_gr_config` mapping before startup:

```python
vllm_gr_config["beam"]["enable_short_suffix_attention"] = False
# Pass the mapping to GRLLM(..., vllm_gr_config=vllm_gr_config).
```

Setting the field to `False` restores packed suffix KV and separate merging.
Omitting it or setting it to `True` enables fusion. The typed `GRConfig`
exposes the same boolean field; the legacy additional-config key is
`vllm_gr_enable_short_suffix_attention`. Do not mix canonical and legacy keys.

With fusion enabled, `attention_config.flash_attn_version` retains vLLM's
hardware-dependent selection; supported explicit FA2/FA3/FA4 overrides remain
valid. Suffix fusion does not force FA2. See [prefix optimization](beam_prefix_optimization.md)
for optional measured per-shape plans. NPU engines and engines without an active
Beam pool do not activate this optimization.

Enabling requires a positive pool capacity and an explicit maximum beam width.
Model/context initialization validates CUDA, head dimensions, GQA grouping,
dtype, canonical prefix states and absence of window/softcap/ALiBi/sinks. Unsupported
static configurations raise an error before execution; explicitly disable
fusion for those configurations.

At runtime, the GPU dispatcher uses the fused kernel only for pure beam decode
batches with one query per beam. Mixed standard/beam batches, prefix
recomputation and other batch shapes use the existing packed-suffix and
merge path without raising a metadata error. Fixed-membership multi-request
decode batches remain eligible: the check uses the session's flattened
`B * beam_width` rows. Prefill selection is independent of decode prefix plans.

Restart the engine when changing the configuration so CUDA graphs capture
the selected implementation. CUDA graph enablement is independent.
There is no fixed suffix-length upper limit imposed by this optimization.
Performance gains depend on the actual suffix length and should be measured
for the workload. Tiny
FP16/BF16 pools preserve FA2's rounding stages; larger pools loop over the
actual device-side suffix extent, avoiding computation for unused capacity.
Compare both paths on the same checkout and workload, starting a fresh engine
for each value of `beam.enable_short_suffix_attention`. Keep the model,
input/output token interval, context length, beam width, generation length,
warmup count, and CUDA graph settings identical.

## Numerical validation

### FA2 arithmetic compatibility

For FP16/BF16 inputs with configured pool capacity 1/2/3, head dimensions
64/128/256 and at most 16 query heads per KV head, the fused path preserves
the disabled FA2 suffix attention and CUDA merge arithmetic:

- Use FA2's `kWidth=2` MMA operand layout and softmax/LSE arithmetic.
- Round probabilities to the activation dtype before PV, and round the
  normalized suffix output before prefix/suffix merging.
- Preserve the CUDA merge's exponential, round-to-nearest division and FMA
  ordering.

Both rounding stages stay in registers. The kernel still reads the KV pool
directly and produces the merged output in one launch, without packed suffix
KV or intermediate suffix output/LSE tensors. FP32 inputs, larger capacities
and groups above 16 retain the existing online FP32 implementation.

`tests/kernels/attention/test_beam_short_suffix.py` compares this tiny-pool path
with the actual packed FA2 suffix, GQA LSE conversion and CUDA merge using
`atol=0, rtol=0`. It covers FP16/BF16, strided tensors, partial tiles, GQA
layouts, runtime slots and graph replay with changing step, mask and KV data.
Matching FA2's arithmetic is distinct from minimizing error against a dense
reference; the independent reference checks keep their existing tolerances.

### Independent dense references

`tests/kernels/attention/test_beam_short_suffix.py` compares with an
independent dense attention reference over the concatenated prefix and
suffix. The reference uses FP32 with TF32 disabled; it does not use
FlashAttention as its oracle. FP32 inputs and outputs are tested directly,
not just FP16/BF16 agreement between two approximate implementations.

Coverage includes:

- Pool capacities 1/2/3 and each valid step; capacities 4/8/16/32/128
  with early, intermediate, full and masked bootstrap extents.
- Widths 1/3/7/128/256, MHA/GQA/MQA and partial program tiles.
- Nonzero pool slots and excess pool capacity.
- Strided query, output and LSE tensors; inactive and unmasked beams.
- Empty prefixes including the actual FA2 `+inf` LSE sentinel, large logits,
  and negative masked bootstrap steps.
- Large-value cancellation and preservation of small FP32 residuals, using
  an independent CPU FP64 average and an operation-count roundoff bound.
- Graph replay with changing step, mask and reordered KV at fixed addresses.
- Actual FA2 prefix integration and initialization rejection for unsupported settings.

Run the kernel tests and relevant regression tests:

```bash
.venv/bin/pytest -q -s \
  tests/kernels/attention/test_beam_short_suffix.py \
  tests/test_short_suffix_config.py \
  tests/test_gr_config.py \
  tests/test_gr_config_v1.py \
  tests/test_beam_attention_platform_execution.py \
  tests/test_beam_decode_graph.py \
  tests/test_beam_attention_pool.py \
  tests/test_beam_attention_metadata.py \
  tests/test_incremental_beam_attention.py
```

On the validation machine, the regular FP32 matrix had maximum absolute
error `1.3113e-6`, with `atol=2e-6, rtol=2e-5`. The large-logit cases use
`atol=2e-5, rtol=2e-4`. FP16/BF16 are also compared with the FP32 oracle,
allowing intermediate prefix-output and final-output rounding.

## Historical performance validation (2026-09-28)

Measured on NVIDIA L20 with Qwen3-0.6B BF16, PyTorch 2.11.0+cu130 and
vLLM 0.22.1. Both runs used this change on top of `decode_graph` commit
`74c4a9e`, switching only `beam.enable_short_suffix_attention`.
The workload used context length 128, beam width 128, batch size 1,
and three generated stages (one prefill and two decode stages).

Input IDs were sampled with seed 0 from `[1000, 2000)`. The synthetic
three-stage constraint table used the same interval and one continuation
per beam after the first stage. CUDA graphs were enabled and prefix caching
was disabled. Each fresh engine ran 10 warmups, 100 unprofiled measurements,
and five separate complete timeline runs. The Qwen benchmark runner from
commit `435eb2a` was used with the configuration field inserted before engine
startup; that benchmark is maintained separately from this kernel change.

| Metric | Disabled | Enabled |
| --- | ---: | ---: |
| Median request latency | 13.663 ms | 10.859 ms |
| P95 request latency, nearest rank | 14.205 ms | 11.113 ms |
| Median sequential requests/s | 73.19 | 92.09 |
| GPU kernel duration/request, profiled | 10.764 ms | 8.453 ms |
| GPU kernel count/request | 2057 | 1609 |

Median latency decreased by 20.52% in this workload. The fused kernel
accounted for 0.126 ms across 56 calls/request. Suffix FlashAttention
(1.542 ms, 56 calls), KV packing (0.315 ms, 112 calls), and separate merging
(0.096 ms, 56 calls) disappeared. Layout and output copies were also removed;
persistent KV writes remained.

Profiler wall time is excluded from latency samples. Summed GPU kernel
execution time and unprofiled request latency are different measurements.
Results depend on the model, token interval, workload and hardware. All
measured requests returned 128 three-token beams; this checks the execution
contract, not bitwise BF16 equivalence. Numerical correctness is validated
against the independent FP32 reference described above.

## Integration validation on updated decode_graph

The change was rebased onto `decode_graph` commit `9c71c9d` on 2026-10-08.
The expanded attention, configuration, model-runner, startup and graph
regression suite passed 467 tests, with three platform skips. This includes
the independent FP32 kernel comparisons and the new ModelRunner V2 tests.

Qwen3-0.6B BF16 smoke runs exercised both ModelRunner V1 and V2 with the
optimization disabled and enabled: context 128, beam width 128, batch 1,
and three generated stages. Each arm ran three warmups, three unprofiled
requests and one separate timeline request. All requests returned 128
three-token beams. Both runners' enabled traces contained 56 fused-kernel
calls and no suffix FA2, packed suffix copy or independent merge calls;
disabled traces contained no fused-kernel calls. These smoke runs establish
integration and dispatch behavior, not a new latency distribution.

The historical BF16 OneRec cache-miss A/B validation had reproducible
Top-1 changes on 2 of 100 requests before upstream `9fc66d3` changed selected
candidate scoring to normalization over the shared logits window. Updated
same-revision comparisons and independent references are described below.

## Real-model numerical validation (2026-10-08)

The final regression matrix passed 485 tests, with three platform skips.
The attention/configuration subset passed 163 tests. An additional empty
prefix regression first reproduced NaNs with FP32, FP16 and BF16 inputs;
canonicalizing FA2's `+inf` LSE sentinel to `-inf` fixes the absent-prefix state.
Nine large-value cancellation cases protect FP32 intermediate precision.

Eager diagnostics replay both implementations on identical real query,
paged-prefix cache, suffix pool, active mask and actual FA2 prefix state.
Independent dense FP32 references disable TF32. Four trajectories cover
22,400 attention calls on 100 OneRec-1.7B inputs (1024 tokens including SID
begin) and 224 calls on the fixed Qwen3-0.6B 128-token input. Beam width is
128, batch size is one and three stages are generated.

The maximum full-reference FP32 absolute error was `1.2970e-4` for OneRec
and `3.8147e-5` for Qwen; the maximum per-call relative RMSE was below
`1.0e-6` and `3.7e-7`, respectively. The original pointwise tolerances were
retained: some real-value/cancellation elements exceed the regular threshold,
and a few OneRec elements exceed the large-logit threshold. These comparisons
are not reported as universal pointwise passes.

Independent CPU FP64 replays cover all 56 saved Qwen calls and the 13 OneRec
calls selected by those large-logit failures. The fused FP32 outputs have no
large-logit tolerance failures against FP64 in this audit; the GPU dense FP32
references themselves retain several. The audit separately holds the actual
FA2 prefix state fixed to isolate suffix/merge error. Its engineering
roundoff budget uses the weighted absolute values being combined and keeps
all original tolerance counts; this budget is not a formal error guarantee
or a replacement for the pointwise checks.

Actual FA2 replay matches the saved generic output exactly. Rounding the
unnormalized exp weights to BF16 before the value product, then rounding the
suffix output, reproduces 99.9942% of its suffix elements on the saved Qwen
inputs. Rounding only the suffix output reproduces 93.7622%. The existing GQA
LSE conversion is correct. The fused path retains those intermediate
computations in FP32 and rounds the final merged output once.

A separate production-CUDA-Graph matrix runs disabled, enabled, FP32 suffix
reference and FP64 suffix reference, with ten warmups and 100 cache-miss
requests per arm. Both models execute 200 measured decode graph replays per
arm. The oracle is installed before startup graph capture; step and active
mask remain dynamic GPU inputs. Weights, prefill and the actual FA2 prefix
state retain the production BF16 behavior, so this is a suffix-reference
model replay rather than an entirely FP32/FP64 model.

OneRec's disabled/enabled controls exactly reproduce the earlier 100-request
benchmark, including all 128 candidate IDs, ordering and scores. They differ
in Top-1 on eight inputs. Against the graph FP64 suffix-reference trajectory,
Top-1 differs on 7/100 inputs for the generic path, 5/100 for the fused path,
and 4/100 for the FP32 reference. Of the eight cross-toggle changes, five
agree with the fused winner and three with the generic winner; one reference
case has tied best candidates. These are reference-agreement observations,
not business-quality measurements or proof of unchanged beam ordering.

Qwen retains the same Top-1 and candidate set across all four graph arms;
ranks and scores differ. Its control scores also differ from the earlier
benchmark despite matching recorded configuration, so only comparisons
within this new matrix are used for its reference agreement. Changing
intermediate rounding can alter BF16 model trajectories. Enabling this
optimization does not promise bitwise output or beam-order equivalence.

## Suffix-length validation (2026-10-08)

The explicit toggle accepts any positive configured pool capacity; no
three-token upper limit is imposed by the fused implementation. Pools of
up to three tokens retain static unrolling. Larger pools use a device-side
loop bounded by the actual suffix extent, with the same stable addresses
and active-mask semantics during CUDA graph replay.

The attention/configuration/beam regression suite passed 488 tests with
three platform skips. Long-suffix comparisons cover capacities 4/8/16/32/128,
early/intermediate/full extents, negative bootstrap steps, empty/nonempty
prefixes, MHA/GQA/MQA and FP32/FP16/BF16 inputs. The FP32 cases retain
`atol=2e-6, rtol=2e-5`; the maximum long-case absolute error was `3.1591e-6`.
Graph replay covers capacities 3/4/32/128 with changing step, mask and KV
histories. Real FA2 prefix integration covers capacities 3/8/32. Production
pool initialization is tested with FP16 and BF16.

A separate CUDA-graph kernel microbenchmark on NVIDIA L20 used BF16,
beam width 128, 16 query heads, eight KV heads and head dimension 128.
With capacity 3 and actual suffix length 3, median kernel execution time
was 2.377 microseconds before and 2.378 microseconds after this extension.
With capacity 128 and actual length 1, the extended kernel took 1.917
microseconds; with actual length 128, it took 36.884 microseconds. These
are operator measurements, not request latency or comparisons against
generic attention for long suffixes. Users should measure the toggle on
their workload to select the implementation.
