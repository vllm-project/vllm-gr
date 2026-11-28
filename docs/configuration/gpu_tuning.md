# GPU performance tuning

Use this guide to tune vLLM-GR Beam inference on NVIDIA GPUs. Choose settings
from the actual model geometry, request distribution, device resources, and
measured execution path. The same configuration principles apply across
supported models, while the optimal values and eligibility for individual
kernels depend on model, hardware, and runtime.

The [configuration reference](README.md) describes the startup YAML and
precedence. The [packaged defaults](../../vllm_gr/config/default_config.yaml)
list the public settings. This guide explains their performance effects and
how to measure and select values. See the
[CUDA installation guide](../getting_started/installation/gpu.md) and
[feature compatibility matrix](../user_guide/feature_compatibility.md) for
runtime requirements.

## Describe the workload

Record the parameters below for each workload and each generation stage.

| Symbol | Meaning |
| --- | --- |
| `B` | Number of requests in one atomic Beam batch. |
| `L` | Input length per request, including cached tokens when applicable. |
| `W` | Retained Beam width per request at a generation step. |
| `K` | Candidates selected per parent Beam at that step. |
| `R` | Actual number of rows projected by the LM head or processed by Top-K. |
| `V` | Effective vocabulary width of that projection or selection. |
| `H` | Hidden dimension of the LM head. |
| `s` | Bytes per logits element. |

In a fixed-width Beam workload, Prefill typically produces one logits row per
request (`R = B`). Decode has one row per surviving parent Beam
(`R = B * W`). With dynamic widths, use the preceding step's retained width
for Decode. Record the actual worker row count and any padding. The Prefill
graph token bucket uses the sum of scheduled input tokens across the batch,
which may be much larger than the number of logits rows. Prefix cache hits
can reduce the number of scheduled input tokens without reducing the full
attention context length.

Even when two workloads have the same number of Decode rows, they can have
different K, score distributions, attention work and output costs. Benchmark
the production distribution of batch sizes, lengths, widths, generation steps,
and output sizes. Hold search semantics and the timing boundary constant when
comparing implementations.

## Establish a reference configuration

Use a small set of fixed workloads before enabling multiple graph shapes or
changing several tuning parameters together. This example uses B4, W256,
K256, three generation steps and 1,024 input tokens per request. Adjust the
numbers to your workload. Save it as `gpu-tuning.yaml`.

```yaml
dtype: bfloat16
tensor-parallel-size: 1
max-model-len: 2048
max-num-batched-tokens: 8192
max-num-seqs: 4
max-logprobs: 256
async-scheduling: true
enable-chunked-prefill: false
enable-prefix-caching: false
enforce-eager: false
attention-config:
  backend: CUSTOM

gr:
  constraint_mode: none
  beam_width: [256, 256, 256]
  candidate_width: [256, 256, 256]
  max_decode_steps: 3
  result_width: null
  worker_decision: true
  beam_batch_sizes: [4]
  beam_graph_enabled: true
  prefill_graph_enabled: true
  prefill_token_buckets: [1024, 2048, 4096, 8192]
  dense_prefill: "off"
  prefill_flash_attn_version: auto
  enable_short_suffix_attention: true
  prefix_policy: "off"
  unconstrained_head_layout: transposed
  head_slice: auto
  head_slice_min_rows: 1025
  head_slice_max_mb: 128
  topk_keep: 8
  topk_select: i64
```

The example uses a full-projection reference for up to 1,024 Decode rows.
The threshold is one greater than the maximum projected row count. It also
disables cross-request prefix caching for cache-miss timing. Use a separate
cache-hit scenario if cache reuse is relevant to deployment.

Use the default CUDA runner selection for the installed revision. After enabling the
[CLI dispatch integration](../cli/README.md#command-form-and-installation),
start the server with a supported model.

```bash
unset VLLM_USE_V2_MODEL_RUNNER
vllm serve /path/to/model --gr --config gpu-tuning.yaml
```

Use the same startup YAML with the
[offline GRLLM API](README.md#python) for synchronous batch measurements.
See the [offline Beam example](../../examples/offline_inference/beam_search/README.md)
and [HTTP completion contract](../serving/completions.md) for requests and
output. See [ModelRunner selection](README.md#modelrunner-selection) for the
defaults of the current checkout. Leaving the variable unset preserves native
fallback on releases with CUDA automatic selection. For a controlled runner
comparison, set `VLLM_USE_V2_MODEL_RUNNER=0` or `=1` before process startup:
explicit V2 selection can fail for unsupported configurations rather than
falling back to V1. Record the effective runner for each experiment and see
the [ModelRunner V2 design](../design/module/model_runner_v2.md) for its
supported execution path.

## Search and capacity settings

These parameters define the search, engine capacity, and result contract.
Reducing a width or returned result count can reduce latency and memory, and
can also change the recommendation or generation quality. Evaluate those
changes against the application's quality requirement.

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.beam_width` | Retained width per generation step. The packaged default is `[128, 128, 128]`; each width is in 1..1024. Higher W increases Decode rows, KV storage, selection work and potential result count. Choose by quality and then by measured latency/capacity. |
| `gr.candidate_width` | Per-parent candidate count, in 1..1024; packaged default `[128, 128, 128]`. The first entry equals the first retained width. A later W must fit the preceding retained rows times the current K. Increasing K increases candidate selection and Beam merging. Compare K independently from W only when the quality contract allows it. |
| `gr.max_decode_steps` | Generation capacity excluding boundary markers. Default 3, at least the schedule length. It sizes persistent history and KV resources. Match actual generation stages. Dynamic schedules support up to three effective steps; fixed-width unconstrained CUDA supports longer generation. Table mode requires three effective steps. |
| `gr.result_width` | Sequences returned, up to the final W. Default `null` uses the final W. A smaller value reduces terminal payload, transfer and text processing. Hold it constant when comparing end-to-end latency. |
| `gr.worker_decision` | Default `null` selects the platform default (enabled for CUDA Beam). The worker performs Beam selection and KV reordering. Keep worker decision enabled for supported CUDA Beam execution; an explicit `false` does not select an alternative legacy Beam engine. |
| `gr.beam_max_width`, `gr.beam_candidate_width` | Optional scalar capacities in the typed configuration for deployments without request width schedules. They size persistent execution and candidate buffers. Schedule lists derive the capacities when present, and conflicts are rejected. Size for the maximum admitted request geometry. |

See the [Beam settings contract](README.md#beam-settings) and
[batch execution design](../design/feature/beam_batch_execution.md) for
request width validation. An oversized capacity can increase allocations and
graph capture work even if many requests use narrower widths.

### Constraint settings

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.constraint_mode` | Explicit `none` (the packaged startup default) selects over the full vocabulary. `table` selects legal candidates from a catalog constraint table. Choose the mode for application semantics. A comparison between modes also changes the search and scoring domain. |
| `gr.constraint_table.path` | Model-bound table artifact. Its catalog size, token coverage, and legal fanout affect resident table memory, projected vocabulary, and selection cost. Use the actual deployment catalog. |
| `gr.constraint_table.backend` | `auto` resolves the production backend for the target device. CUDA uses the CUDA backend. CPU is a reference backend. Measure the actual accelerator implementation. |
| `gr.constraint_table.format` | `constraint_table_v1` stores one Trie; `constraint_table_v2` supports multiple Tries. Select the format and number of Tries needed by request routing. Measure table size and the shared token window. |
| `gr.constraint_table.max_top_k` | Per-parent candidate capacity; must cover all admitted K. Current CUDA table operators cap it at 256. Size from the required workload, since table selection and scratch depend on this capacity. |
| `gr.constraint_table.digests.artifact`, `gr.constraint_table.digests.tokenizer` | Bind the table to its contents and tokenizer. Generate them with the builder so the serving workload and its candidate identity are reproducible. |

CUDA table mode derives a contiguous LM head vocabulary window from the loaded
tables. The projected window and legal fanout affect the selection workload.
Changing the catalog or projection window can change the candidate results and
normalization domain. The unconstrained LM head slicing and MCAP Top-K settings
in the following sections tune the full-vocabulary path. See
[constraint table configuration](constraint_tables.md) and
[constraint table design](../design/module/constraint_tables.md).

## Native vLLM settings

| Setting | Principle and tuning guidance |
| --- | --- |
| `max-model-len` | Maximum per-sequence token length including generation. Set it to cover the workload and output budget. Actual context length, KV dtype and model layout drive KV memory and attention cost. |
| `max-num-batched-tokens` | Scheduler token budget. Cover the largest scheduled Prefill token total and at least the model length for unchunked Prefill. A larger budget can also admit more graph capture shapes. |
| `max-num-seqs` | Logical request capacity. Cover the largest batch; each Beam request counts once for admission even though Decode expands rows. |
| `max-logprobs` | Native capacity for returned candidate log probabilities. Set it at least to the maximum per-parent K, including every configured schedule. |
| `gpu-memory-utilization` | Native engine memory budget used in cache sizing. Set from available device memory and peak memory for model, KV, graphs and postprocess resources. Measure remaining headroom after warmup and at production geometry. |
| `dtype` | Current persistent Beam execution requires FP16 or BF16. The dtype changes memory traffic, GEMM and attention kernels, and numerical behavior. Validate quality when changing precision. |
| Parallel sizes, quantization, LoRA and speculative decoding | Current asynchronous Beam execution requires TP/PP/DP/DCP/PCP=1 and a uniform text decoder. Quantization, LoRA, speculation, transfer and microbatching are unsupported in this path. These are compatibility boundaries rather than optimization candidates. |
| `async-scheduling`, `enable-chunked-prefill` | Beam requires asynchronous scheduling and complete Prefill. Keep `true` and `false`, respectively. Partial Prefill scheduling is unsupported. |
| `enable-prefix-caching` | Cross-request reuse of already computed prefix KV. It can reduce Prefill model work for repeated prefixes. Measure cache-miss and cache-hit workloads separately and record cache state. |
| `enforce-eager` | `true` disables graph execution for a diagnostic reference. For replay measurements use `false` and check actual graph dispatch. |
| `attention-config.backend` | Select the supported attention backend. The CUDA Beam optimizations use `CUSTOM`. Use the native attention configuration rather than placing the backend under `gr`. |

The supported execution constraints are enforced in the
[runtime validation](../../vllm_gr/config/execution.py). See
[asynchronous scheduling and budgets](README.md#asynchronous-scheduling)
for the native engine and GR capacity relationship.

## LM head projection and slicing

The LM head maps each hidden state to vocabulary logits. Full projection
produces all R by V values. Vocabulary slicing computes C columns at a
time, consumes them in Top-K, and reuses the temporary buffer. The entire
vocabulary participates in projection and normalization. Slicing can reduce
peak storage and intermediate logits traffic at the expense of extra GEMMs,
kernel launches, merging and repair.

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.unconstrained_head_layout` | `transposed` (default) caches a contiguous transposed LM head weight for full and sliced Decode projections. The current path requires compatible unquantized Qwen3 with contiguous FP16/BF16 weights, TP=PP=1 and no LoRA, among other eligibility checks. `null` uses the model's original layout. On models without this optimization, verify the native projection. |
| `gr.head_slice` | `auto` (default) selects eligible Decode slicing by row count. A positive integer requests the slice width and bypasses the automatic row threshold; the implementation aligns and clamps it. `0` uses native `compute_logits`. CUDA V2 Prefill uses native full logits. To isolate slicing while retaining the transposed weight path, use `auto` and raise the threshold for the full-projection control. |
| `gr.head_slice_min_rows` | Default 1024; a nonnegative minimum actual R for automatic slicing. Determine the point at which slicing is beneficial from measured workload shapes. For a full-projection reference, set it above the largest admitted R. |
| `gr.head_slice_max_mb` | Default 128 MiB. Target budget for a single logits slice; 0 disables budget-based width reduction. This controls temporary buffer size, not the full engine memory limit. |

See the [CUDA stage runner](../../vllm_gr/v1/worker/cuda/gpu_beam_stage_runner.py)
and [V2 Prefill projection](../../vllm_gr/v1/worker/cuda/v2/gpu_beam_stage_runner_v2.py)
for the projection eligibility and planning logic.

### Calculate the buffer cost

```text
Full logits bytes  = R * V * s
Slice logits bytes = R * C * s
Number of slices   = ceil(V / C)
LM head FLOPs     ~ 2 * R * H * V
Transposed weight bytes = H * V * bytes_per_weight
```

The current automatic planner starts by rounding `floor(V / 6)` up to a
1,024-column boundary. It constrains widths to 16,384 through 32,768 columns,
subject to actual vocabulary size. An explicit positive width is also aligned
and clamped. If the candidate covers the vocabulary, full projection is used.
When a positive budget is smaller than the slice buffer, the planner halves
the width down to its minimum and aligns it. The buffer can exceed a requested
budget below that minimum.

For a target temporary budget Q in bytes and a 1,024-column alignment A,
calculate an approximate aligned upper bound:

```text
C_budget = floor(Q / (R * s * A)) * A
```

Check the implementation limits and actual slice plan. If even the minimum
slice width exceeds the available budget, reduce the admitted workload or
adjust the other memory allocations. Account for the LM head weight copy,
model weights, KV, CUDA Graph private pools, Top-K scratch, and allocator
reservations. Record allocated and reserved memory without adding them as
independent costs, and leave headroom for workload variation.

### Choose full projection or slicing

1. Establish the same-layout full-projection reference where feasible.
2. Compare feasible slice settings for the same R, V, K, dtype and selector.
   Record the actual slice width, slice count, and tail width.
3. Measure GEMM time, DRAM traffic, cache behavior, merge and repair time,
   and peak memory. A bandwidth-bound workload can benefit from reduced
   intermediate traffic. A workload dominated by GEMM and additional launches
   may favor full projection. The GPU's advertised bandwidth and compute
   throughput alone cannot select the faster path.
4. Measure several R values for the same other geometry, and choose a row
   threshold near an observed latency crossover. Choose a memory-feasible
   configuration when the full-projection control cannot fit. Use different
   deployment profiles if one threshold does not serve the workload mix.

## Top-K selection and merging

For unconstrained CUDA selection, the MCAP implementation divides each logits
row into 1,024-column vocabulary tiles. It selects local candidates, merges
them, and repairs the result by scanning tiles when omitted candidates may
enter the final top K. A concentrated score distribution can require repairs
even when the initial candidate pool contains many more than K elements.
The final score normalization covers the entire vocabulary.

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.topk_keep` | Default 8, range 1..256. Local retention per row and vocabulary tile, adjusted for K and coverage. More retained entries can reduce repair scans while increasing persistent scratch, sorting work, registers and shared memory. This setting applies to Prefill and Decode. |
| `gr.topk_select: i64` | Default full 64-bit score and token-ID ordering keys. Use as the first reference for candidate quality and performance. |
| `gr.topk_select: i32` | Short keys for local selection. Keys can collide near the cutoff and require extra boundary evidence and repair. Test independently with actual scores and K. Check the same logits against the reference before using this path. |
| `gr.topk_select: flash` | Opt-in FlashInfer Top-K for eligible unsliced CUDA logits. Reserves a worker-shared logits buffer and computes normalization over the full vocabulary. Compare it on eligible small row counts and validate output. |

Flash selection requires a compatible FilteredTopK implementation, at most 256
rows within the reserved geometry, and temperatures of 0 or 1. Single-row
calls with K below 1024 retain the regular selector. The hardware probe checks
at least 128 KiB of shared memory per SM. Ineligible calls fall back to `i64`; for
example, L20 lacks the required capacity. Forced positive head-slice widths
disable the Flash buffer reservation. Record actual dispatch and include the
reserved buffer in memory measurements.

See the [selection implementation](../../vllm_gr/ops/cuda/unconstrained_topk_mcap.py)
and [postprocess workspace](../../vllm_gr/v1/worker/cuda/gpu_postprocess_workspace.py).

### Calculate merge and scratch dimensions

For the full-projection MCAP path, use these estimates for the tile and merge
geometry. Here T is the vocabulary tile count, e is the effective retention
per tile, and G is the number of merge chunks.

```text
T = ceil(V / 1024)
e = min(K, max(topk_keep, ceil(K / T)))
G = ceil(T * e / 1024)
Persistent key bytes = 8 * R * (T * e + G * min(K, 1024))
Final merge work width = next_pow2(G * min(K, 1024))
```

The key estimate covers two persistent tensors; the implementation also
allocates other scratch. The sliced path uses a different streaming merge
geometry. One array's merge work width times eight bytes provides a
working-set estimate, while the compiler determines actual register and shared
memory allocation. Check the compiled resources and device's opt-in
shared-memory-per-block limit for every specialization, including the Prefill
rows and both regular and tail vocabulary tiles.

### Select a keep value

1. Start with `i64` and `keep=8`. Compare 16 and 32, removing candidates
   that produce the same effective e for the stage.
2. For K at or below the requested retention, `i64` can avoid local-truncation
   repair. For larger K, evaluate the number of repaired tiles per row and
   rows requiring repair.
3. Use the merge-width formula to find the next discrete power-of-two work
   geometry. Compare values near the top of the current geometry and at the
   next geometry if the kernels compile and memory permits.
4. Measure total selection, merge and repair time along with peak scratch.
   Larger keep is useful only when the saved work offsets its resource cost.
   Confirm token IDs, ordering, count, valid mask and normalized scores against
   a reference built using identical logits.

## CUDA Graphs and execution reuse

CUDA Graph capture records work with fixed resource addresses and shape
constraints. Replay reduces CPU launch overhead. Capturing too many shapes
increases startup time and private memory. Capture the common admitted shapes
and inspect actual replay rather than inferring it from enabled settings.

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.beam_batch_sizes` | Default `[1, 2, 4]`; increasing unique integers in 1..8. Defines CUDA graph batch sizes and eligible atomic multi-prompt HTTP batches. Separate HTTP requests are not combined automatically. Include production batch shapes and measure startup and graph memory. |
| `gr.beam_graph_enabled` | Default true, permits Beam Decode graph capture and replay. Captured batch and Beam geometry must match. Compare against eager execution for diagnosis, and record actual Decode replay. |
| `gr.prefill_graph_enabled` | Default true, permits Prefill graph capture and replay. Check FULL, PIECEWISE and eager dispatch on the intended runner. |
| `gr.prefill_token_buckets` | Packaged default grid covers 128..8192 in steps of 128 scheduled batch tokens, subject to admission capacity. A finer grid can reduce padding and improve hits, while more captures consume memory and startup time. `null` uses the built-in V1 grid or native V2 capture sizes. A native explicitly empty list remains empty on V2. |
| `gr.prefill_kv_bound` | Default `null`. CUDA V1 normally captures a KV length bound of twice the token bucket. A positive value widens this bound to at least that value to admit cache-hit tails with a longer total context. It also adds short buckets of 1 and 16 on the shared grid where applicable. V2 does not widen the attention extent through this setting. Measure hot-prefix replay and capture memory by runner. |
| `gr.adapter_cache_enabled` | Default false. Permits retaining stage adapters by B to preserve graph-referenced input and attention buffer addresses between batches. Some multi-batch graph paths already retain their adapters. Compare mixed-batch workloads for reuse, peak memory, and rebuilds. |
| Native `compilation-config.cudagraph_capture_sizes` | V2 uses native capture sizes when GR Prefill buckets are null. Choose native capture policy together with the runner and GR settings. Verify the effective capture and dispatch plan. |

See [CUDA Prefill graph configuration](../design/feature/cuda_prefill_graph.md#configuration)
and [ModelRunner V2](../design/module/model_runner_v2.md) for shape eligibility.
Change one of capture geometry, caching or runner at a time.

## Attention settings

### Prefill and short suffix

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.dense_prefill` | Default `off`. On eligible fresh prompts, `fa2` or `cudnn` can read contiguous K/V during Prefill. cuDNN additionally requires an eligible single-request shape and minimum token count. Compatible dense batches can use FA2; other shapes use paged attention. Compare actual kernels and graph eligibility on fresh inputs. |
| `gr.prefill_flash_attn_version` | Default `auto` can select FA3 for eligible FA2 Prefill shapes. Explicit 2 or 3 selects the version subject to support; `null` follows Decode. Compare supported versions with equal input shape and dtype and check actual backend dispatch. |
| Native `attention-config.flash_attn_version` | Controls the Decode FlashAttention version. Eligible FA2 shapes allow the GR prefix segmentation strategy, while FA3 and FA4 use their native splitting. Test version changes together with their actual prefix execution mode. |
| `gr.enable_short_suffix_attention` | Default true. Permits the fused short Decode suffix attention and prefix-state merge for eligible pure single-query Beam shapes. Its benefit depends on suffix length, KV/head geometry, backend version and kernel resources. Compare against the packed path when investigating Decode attention. |

### Shared prefix segmentation

Beam Decode can share prefix KV pages among beams within the same request.
The GR prefix planner may split the context into parallel attention segments
and merge partial outputs using log-sum-exp. This can improve parallelism for
long prefixes when a native attention chain is slow. Segmentation also adds
merge and scheduling work. Cross-request prefix caching is a separate
native setting.

| Setting | Principle and tuning guidance |
| --- | --- |
| `gr.prefix_policy` | Default `off` keeps native attention. `auto` selects calibrated segmentation for eligible paged FA2 shapes. Compare long-context workloads with `auto` and `off` at fixed cache state and Beam geometry. |
| `gr.prefix_min_chain_us` | Default 150 microseconds. Requires the predicted unsplit prefix chain to reach this threshold before considering segmentation. Sweep around the predicted chain costs of production shapes and measure actual segments, attention latency and merge overhead. |
| `gr.prefix_calibration` | Default true. Enables use or creation of compatible device/geometry calibration data. Setting false disables segmentation. Keep enabled for an `auto` experiment. |
| `gr.prefix_calibration_path` | Default null, using the standard user cache location. Selects the calibration YAML. Use compatible hardware, software and model geometry; do not reuse incompatible coefficients across devices. Check resolved calibration and actual dispatch. |

The auto policy requires compatible calibration, sufficient predicted gain and
parallel headroom, compatible page and metadata geometry, and at least two
segments. FA3/FA4 and batch-invariant mode use the native path. See the
[CUDA Decode attention design](../design/feature/cuda_decode_attention.md#configuration)
for the eligibility, calibration and fallback details.

## Online catalog update memory

`gr.trie_retention_seconds` (default 60 seconds) retains retired constraint
tables after hot replacement. Allow memory for overlapping tables and measure
peak device memory if the catalog is refreshed online. See
[weight and catalog updates](../user_guide/weight_update.md).

For settings outside CUDA tuning, refer to the [configuration guide](README.md)
and [packaged defaults](../../vllm_gr/config/default_config.yaml). The
[Ascend quickstart](../getting_started/quickstart.md#ascend) links its runtime
configuration and usage guidance.

## Host and output overhead

When GPU work is short, use a timeline to measure request admission,
scheduler-to-worker time, metadata preparation, host-to-device transfers,
first graph launch, inter-stage gaps and terminal output. CPU single-thread
performance, NUMA placement, affinity, concurrent work and process mode
affect these intervals. Hold them constant during device comparisons. The
intervals can overlap with GPU execution, so use measured end-to-end latency.

The output contract also matters. Full offline Beam outputs can materialize
text lazily; compact output contains token IDs and scores. Online output has
a batched tokenizer path. Include text materialization and the same number
of returned candidates in every timed call if text is required. Compare
token-only measurements separately. See the
[output processor](../../vllm_gr/entrypoints/beam_search/output_processor.py).

## Cross-device tuning procedure

1. **Record the effective execution.** Pin the model, tokenizer, input samples,
   runtime revision, device and software environment. Confirm native kernel
   compilation and execution, actual attention and selection paths, and graph
   replay. See the [benchmark guide](../benchmarking/README.md) for setup.
2. **Establish a reference.** Use i64 selection and a feasible full-projection
   control where supported. Measure startup, first request and steady-state
   memory and latency for representative workload shapes. Capture a separate
   profile of CPU gaps, GEMM, Top-K, KV and attention.
3. **Isolate variables.** Hold projection constant when sweeping keep and
   selector. Hold the selected keep constant when comparing slice widths and
   budgets. Test graph shape grids, attention versions and prefix settings
   separately. Expand to production workload distributions after selecting
   feasible candidates.
4. **Verify resource feasibility.** Compare actual compiled shared memory,
   registers, spills and occupancy against the device resources. Check
   deployment memory headroom including KV, graph pools and concurrent
   allocations. Validate mixed-shape transitions.
5. **Verify quality and performance.** Compare identical logits against a
   stable reference, then compare per-request candidate IDs, ranks, scores
   and output text. Run the application's accepted quality metric. For
   finalists repeat matching requests in independent processes in A/B/B/A
   order. Use equal synchronization boundaries and output scope. Record batch
   mean, P50, P95, maximum, paired differences, and serving throughput where
   relevant. Collect profiling in separate runs and expand the sample count
   for tail-latency estimates.

For GPU bottleneck attribution, use
[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) for
kernel launch resources, memory traffic, and achieved throughput and
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
for CPU, transfer, and graph execution timelines. The repository's
[benchmark reference](../cli/bench.md) describes available workloads.
Check workload conditions and use identical search and output contracts before
comparing results.

## Worked example using historical OneRec measurements

OneRec-1.7B on L20 provides a concrete example of the procedure. The
[archived experiment] used GR commit `539a30f2ca98fedc5b2323419c282a0142ea9a56`
with runtime patches, CUDA ModelRunner V2, BF16, 1,024 input tokens, three
generated tokens, one process, and CUDA Graphs. Cross-request prefix caching
was disabled and all candidate text was included in timing. Each comparison
used the same 24 input batches.

| Workload | Configuration | Batch mean (ms) |
| --- | --- | ---: |
| B1/W32 | Optimized GR | 49.52 |
| B1/W32 | SID-GR, same experiment | 49.13 |
| B4/W1024 | GR i32 baseline | 704.60 |
| B4/W1024 | GR i64 alone | 580.00 |
| B4/W1024 | GR i64, repair optimization, batched text decoding | 558.59 |

The small-width profile measured 3.63 ms before the first Prefill graph
launch for GR and 0.91 ms for SID-GR. The three GR GPU Top-K stages together
took approximately 0.25 ms. The large-width experiment instead benefited most
from changing the selector. The earlier Prefill initialization patch was
common to all three B4 GR runs. The combined repair and text changes saved
another 21.41 ms after switching to i64. The i32 and i64 paths replaced 36 to
96 candidates in each of four audited requests, so quality verification
remains important. Independent process repeats are needed to resolve the
sub-millisecond B1 difference.

With this model's H=2048, V=176384 and BF16 logits, B4/W1024 has 4,096 Decode
rows. The projection formulas give the following candidate settings.

| `head_slice` | `head_slice_min_rows` | `head_slice_max_mb` | Calculated Decode buffer |
| --- | ---: | ---: | --- |
| `auto` | 4097 | 128 | Full projection, 1,378 MiB |
| `auto` | 1024 | 128 | 16,384 columns, 11 slices, 128 MiB |
| `auto` | 1024 | 256 | 29,696 columns, 6 slices, 232 MiB |
| `32768` | 1024 | 256 | 32,768 columns, 6 slices, 256 MiB |

The historical measurements used full projection with i64 and keep32 in the
optimized configuration. The archived estimator rounded the 256 MiB
automatic width down to 28,672 columns (seven slices, 224 MiB). The pinned GR
runtime rounds the starting width up, giving the values in the table. These
are derived buffer sizes, so record actual shapes and performance on the
target device. At this V and K, keep47 remains in the same final merge work
width as keep32 while keep48 increases it from 8,192 to 16,384. Apply the
formula for different vocabulary and candidate sizes.

The [reproducibility package] contains the historical planning tools, benchmark
adapter, source patch and result summaries. The archived source patch and
batched text option belong to that experiment. The measurements retain their
recorded revision and configuration, and the recommended starting points for
other workloads and GPU models require new measurements.

## Recommended starting points by GPU

Start with i64, a feasible full-projection reference, and the packaged
`topk_keep: 8`. Compare 16 and 32 at fixed workload geometry. For a small K,
local retention up to K can remove i64 truncation repairs. For larger K, use
the merge geometry and compiled resource checks to select additional values.
Set the automatic slicing threshold above the largest reference R. Choose
slice budgets from actual available memory and compare full and sliced
projection when both fit.

| GPU | Nominal memory | Opt-in shared memory per thread block | Initial tuning priority |
| --- | ---: | ---: | --- |
| L4 / A10 | 24 GB | 99 KiB | Measure global memory headroom first; compare a feasible slice and smaller scratch if full projection is too costly. |
| L20 / L40 / L40S | 48 GB | 99 KiB | Establish full projection when feasible, compare slicing and keep at the current merge geometry, and check CTA resources. |
| A100 | 40 or 80 GB | 163 KiB | Compare full and sliced projection; screen the next merge geometry after resource checks. |
| H20 | Typically 96 GB | 227 KiB | Measure local GEMM and repair performance; compare feasible larger keep values and slices. |
| H100 | 80 or 94 GB | 227 KiB | Compare full and sliced projection and eligible larger keep values, measuring the actual device and software stack. |
| H200 | 141 GB | 227 KiB | Compare intermediate DRAM traffic, GEMM, and selection time; screen larger keep values if resources allow. |
| B200 | 180 GB | 227 KiB | Confirm native kernel and attention support first, then compare projection and selection geometry. |

These are starting experiments based on device resources. The GPU's actual
SM count, memory bandwidth, precision throughput and CPU environment affect
performance. Query the actual instance's shared-memory-per-block opt-in
attribute and available global memory; variants can differ. Per-SM capacity
for the optional Flash path is a separate limit. For example, Blackwell
compute capability 12.x devices have different per-block resources from B200
compute capability 10.0. See [CUDA compute capabilities] and
[CUDA architectural limits]. Device memory references are
[A10 and L-series specifications] and [A100, Hopper, and B200 specifications].

[archived experiment]: https://github.com/vllm-project/vllm-gr/blob/aaed8326a835df62a302215608f9286ff7237536/docs/benchmarks/onerec-gpu-tuning/OneRec_GPU_Tuning.md
[reproducibility package]: https://github.com/vllm-project/vllm-gr/blob/aaed8326a835df62a302215608f9286ff7237536/docs/benchmarks/onerec-gpu-tuning/onerec-tuning-20261120.zip
[CUDA compute capabilities]: https://developer.nvidia.com/cuda/gpus
[CUDA architectural limits]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications
[A10 and L-series specifications]: https://docs.nvidia.com/vgpu/packaging-pricing-licensing-guide/latest/
[A100, Hopper, and B200 specifications]: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html
