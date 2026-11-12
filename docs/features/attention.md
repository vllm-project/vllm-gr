# Attention optimizations

The following optimizations have different device and runtime requirements.
Choose the relevant section before enabling one; their presence in this guide
does not imply that every backend supports every optimization.

- [CUDA prefix attention](#cuda-prefix-attention)
- [CUDA short-suffix attention](#cuda-short-suffix-attention)
- [NPU BSND Prefill](#npu-bsnd-prefill)
- [NPU XAttention output binding](#npu-xattention-output-binding)

## CUDA prefix attention

Prefix attention returns activation-dtype `out[tokens, query_heads, head_dim]`
and FP32 natural-log `LSE[query_heads, tokens]`. The existing suffix kernel
consumes this interface without knowing which FlashAttention backend produced it.
Empty paged prefixes are normalized before merging, including FA2 builds that
leave empty-request LSE uninitialized. Prefill and persistent KV writes are independent.

### Selection and compatibility

Without an explicit choice, vLLM selects a supported FA version as usual. The
suffix fusion setting no longer forces FA2. The prefix policy compares:

| Plan | Execution | Capability constraint |
| --- | --- | --- |
| Native | Backend default splitting | Any available FA2/FA3/FA4 adapter |
| Native split | Explicit `num_splits`, including `1` to avoid automatic splitting | FA3/FA4 adapter; FA2 varlen rejects values above 1 |
| Framework segmented | One virtual-batch FA call over request × KV segment, followed by stable LSE merge | Plain paged FP16/BF16 prefix; no window, ALiBi, softcap or sinks |

Native CUDA attention is the default. Set `gr.prefix_policy: auto` to enable
CUDA/FA2 segmentation with the device-local cost model described in the next
section. The cost model measures execution time, so validate fixed-sample model
Recall before enabling a plan that changes the attention reduction order.
FA3/FA4 are never overridden, because
their own split-K heuristic already pays a single fused combine. There are no
L20, SM89 or SM90 dispatch branches.

Framework segmentation reuses KV page IDs and copies only queries and compact
metadata. All segments enter a **single** FA call, exposing additional parallel
work; sequential Python calls would not achieve this. The merge uses FP32 online
log-sum-exp accumulation and masks empty segments before loading their outputs
or LSE. It handles uneven final segments and arbitrary request query lengths.
Each merge program owns one (request, head, 256-element tile); a single program
per (request, head) leaves too few CTAs and costs as much as the FA call at W=64.
Workspace and merge cost can outweigh the benefit; segmentation is not enabled
just because it is supported.

Batch-invariant mode retains the original version with splitting disabled, and an
explicit `flash_attn_version` captured at start-up always wins. Length buckets are
1024-aligned and clamped to the model bound, so every supported segment count
divides the bucket evenly.

### Device calibration

The four cost-model coefficients (serial chain, attention FLOPs, KV bytes and
per-segment overhead) are measured once per `(device, geometry)` at engine
start-up, before any Beam graph is captured, and cached in a commented YAML file
(`$XDG_CACHE_HOME/vllm_gr/prefix_calibration.yaml`, overridable with
`gr.prefix_calibration_path`). Every worker measures its own device and
writes the same entry, so no lock is needed; a torn file simply fails to parse
and is re-measured. The chain and split probes use the largest length up to
`max_model_len` (capped at 16384) whose page count divides by 8, so the 8-way
split probe fits the block table for any `max_model_len`.

A plan is installed only when the model predicts at least a 5% win and the
unsplit chain is at least 1.5x above the reachable floor, so compute-bound
wide-beam shapes stay native. The unsplit chain must also reach 150 us, a bound
measured on L20 for 1024/B1/W128. `gr.prefix_min_chain_us` overrides it
with a finite value >= 0; other values log a warning and keep 150. On RTX PRO
6000 a value of 8 lets W=64 prefixes of 512-4096 tokens split, where native FA2
launches only `heads` CTAs per request.

The file records a schema version that changes whenever a measured kernel
changes cost; entries from another schema are ignored and re-measured. Schema 2
follows the per-tile merge, whose per-segment overhead is far lower.

`gr.prefix_calibration: false` skips measurement and `gr.prefix_policy: "off"`
keeps the default native path. Set `gr.prefix_policy: auto` to opt in to
measured segmentation; delete the file to re-measure. Every
failure path logs a warning and continues on the native path.

### CUDA graphs and fallback

The asynchronous V1/V2 Beam stage path obtains lengths from existing CPU
metadata, never from device `.item()` calls. Only a shape present in the matching
profile activates a power-of-two, page-aligned length bucket, capped at model
capacity. Graph signatures include `(B, W, prefix bucket)`; the profile is immutable
so a captured bucket also fixes its backend/split plan. Replay changes lengths,
page IDs, queries and active rows at stable addresses, without selecting or tuning
kernels on the device.

New buckets capture lazily against real request buffers. This introduces a
one-time capture cost and additional graph memory for each measured shape.
Startup retains the original model-capacity fallback graph. Replacing V1 input
storage discards every length bucket for the affected geometry. Dynamic-beam
buffer invalidation continues to clear its graphs; V2 recoverable capture failures
are isolated by geometry and bucket. Legacy paths without length-bucket metadata
retain their original graph bounds and only match profiles for those bounds.

### Validation status

See the PR description and its dated validation note for checks actually run.
Backend capability filtering and CPU graph-cache tests do not establish GPU
performance on untested architectures. FA4 requires its optional CuTe dependencies.
Numerical comparisons use tolerances; changing backend or reduction order does
not promise bitwise equality or identical final beam ranking.

## CUDA short-suffix attention

Short suffix attention fuses suffix attention and prefix/suffix merging into
one Triton kernel for eligible CUDA CUSTOM Beam decode. It reads the persistent
suffix KV pool directly, avoiding packed suffix copies and separate merge
work. Prefix attention and persistent KV writes retain their existing roles.

### Configuration

The optimization is enabled by default for eligible engines. To disable it,
set this field in the configuration passed to `GRLLM` before engine startup:

```python
gr["enable_short_suffix_attention"] = False
# Pass this mapping to GRLLM(..., gr=gr).
```

Disabling it uses packed suffix KV and separate attention/merge operations.
Restart the engine after changing the setting so CUDA graphs capture the
selected implementation. CUDA graph enablement is a separate setting.

### Scope and limitations

- The fused path requires CUDA CUSTOM attention, an active Beam session and
  KV pool, and incremental decode with one query per beam. NPU, ordinary
  requests and Prefill do not use this path.
- Initialization checks head dimensions, dtype, GQA layout and prefix states.
  Windowed attention, ALiBi, softcap and sinks are not supported by the fused
  path; disable it for unsupported configurations.
- Fixed-membership multi-request Beam decode is supported. Valid mixed
  standard/Beam batches use the generic path. Missing sessions or multi-query
  Beam suffixes are rejected.
- Any positive configured pool capacity is supported. Graph replay reads the
  changing step, mask and KV contents from device buffers.

### Validation and performance

[Kernel coverage](https://github.com/JiusiServe/vllm-gr/blob/main/tests/kernels/attention/test_beam_short_suffix.py) compares
with independent dense references and the applicable packed attention path,
including CUDA graph replay. [Configuration coverage](https://github.com/JiusiServe/vllm-gr/blob/main/tests/v1/attention/test_short_suffix_config.py)
checks activation and rejection conditions.

Arithmetic agreement for covered kernel cases does not guarantee identical
model outputs or Beam ordering. Intermediate rounding can change scores and
selected tokens. Measure enabled and disabled paths on the same code, model,
inputs and configuration, with a fresh engine for each setting; performance
depends on the workload and suffix length.

See [prefix optimization](#cuda-prefix-attention) for the related prefix
attention path.

## NPU BSND Prefill

BSND Prefill graphs avoid per-layer FIA task parameter updates for complete
prompts without cached history. The feature is disabled by default and uses
the normal vllm-gr modules. No import overlay or `sitecustomize` is required.

### Configuration

Enable the existing Prefill graph facility and select explicit total-token /
request-count pairs before starting the engine:

```yaml
gr:
  prefill_graph_enabled: true
  npu_bsnd_capture_specs: "512,4;1024,4"
```

Each entry is `T,B`, where `P=T/B` is the capacity of each request. The example
supports four requests with individual lengths up to 128 or 256 tokens.
The smallest captured graph that fits every request is selected. A total
length that fits T is insufficient if any individual length exceeds P.

An empty or null `gr.npu_bsnd_capture_specs` disables BSND. B must be 1, 2 or 4; T must
be positive, divisible by B, and present in the runner's configured Prefill
buckets. Invalid configuration raises an error during runner initialization.
Restart the engine after changing either setting.

These graphs require the Beam/CUSTOM attention metadata builder for every
attention group. Native attention and mixed-backend models skip the Beam
Prefill captures and retain their native execution path, even when BSND specs
are configured. Capture failures on eligible Beam backends still propagate.

Deployments previously using `tools/bsnd_v41` must remove that entry from
PYTHONPATH and stop using the historical import overlay. Normal package
installation/imports now select the implementation.

### Execution and fallback

Every request must have zero computed tokens and its complete prompt scheduled
in the current batch. Lengths include tokens added by the input processor.
Requests are mapped in actual InputBatch row order, not scheduler dictionary
insertion order.

The runner pads each request on the right into a fixed P-token region. Input
IDs, positions and slot mappings keep stable addresses. Padding slots are -1.
The model still writes real K/V to its paged cache, but BSND FIA reads the
current layer's contiguous Q/K/V rather than paged history. The existing causal
mask isolates padding keys from real queries in this full-prompt setting.

After replay, basic-slice device copies compact the real hidden states into an
independent persistent output buffer. Existing logits indices, stage scheduling,
constraint selection and Decode execution remain unchanged. Workspace and LSE
buffers are retained by captured metadata rather than process-global caches.
Buffer reuse assumes the existing serial compute-stream execution.

Prefix-cache hits, historical KV, mismatched batch sizes and lengths outside
the configured shapes use the existing TND graph/native fallback. Unsupported
BSND attention configurations, including KV sharing, sliding windows, softcap,
sinks, ALiBi and FP8, are rejected by the BSND guard. Device capture/replay
errors are propagated rather than treated as ordinary graph misses. The TND
update-before-replay stream ordering is retained.

Padding and input/output copies add device work and memory use. Reduced FIA
submission overhead does not guarantee lower latency for every length distribution.

### Validation

Dependency-free host tests, run from the repository root:

```bash
python -m tests.v1.worker.test_npu_prefill_bsnd
```

In the supported vLLM/Ascend environment:

```bash
python -m pytest -q tests/v1/attention/test_beam_attention_platform_execution.py \
  tests/v1/worker/test_prefill_graph_runner.py tests/v1/worker/test_npu_model_runner_patch.py \
  tests/v1/worker/test_npu_prefill_bsnd.py
python -m ruff check vllm_gr/v1/worker/npu/npu_prefill_graph_runner.py \
  vllm_gr/v1/worker/npu/npu_model_runner_patch.py \
  vllm_gr/v1/attention/backends/beam_attn_metadata.py \
  vllm_gr/v1/attention/backends/beam_attn.py \
  vllm_gr/v1/attention/backends/beam_attn_npu.py \
  tests/v1/worker/test_npu_prefill_bsnd.py tests/v1/worker/test_prefill_graph_runner.py
```

Run `ruff format --check` on the same files. The generic Prefill unit tests
clear deployment BSND specs in their fixture; dedicated configuration tests
explicitly set their own specs. Production bucket validation remains strict.

Run the repository strict type-check wrapper on the host regressions:

```bash
python tools/pre_commit/mypy.py 1 3.12 tests/v1/worker/test_npu_prefill_bsnd.py
```

When collecting replay evidence, ensure the EngineCore logging configuration
attaches a handler to `vllm_gr` as well as `vllm`. If an existing custom logging
configuration is used, preserve its handlers and register `vllm_gr` with an
INFO-level handler. Pass the absolute JSON path through
`VLLM_LOGGING_CONFIG_PATH` before starting each engine. Raising the logging level
alone does not attach a missing handler. Missing replay logs are inconclusive;
capture counts and Decode dispatches do not establish Prefill replay.

Host tests do not establish NPU numerical correctness. Device validation should
cover B=1/2/4, length boundaries, GQA, FP16/BF16, cache-hit fallback, alternating
BSND/TND requests, cancellation and final Beam outputs. Compare attention with
a no-padding reference and check logits, candidate ordering and cumulative scores.

For performance, use fresh engines in A/B/B/A order on the same source revision:
A has BSND disabled, B enables the selected shapes. Keep model, catalog, actual
tokenized inputs, cache policy, B/W/N and Decode configuration identical. Report
capture time and memory separately from warmed-up request latency; collect
profiles separately from timing. Confirm actual BSND hits in logs and zero
per-layer FIA updates for those Prefill replays. No ABBA measurements are
reported by this integration change.

## NPU XAttention output binding

The NPU `xllm` Decode path can write XAttention results directly into the
attention output buffer. This removes the intermediate allocation and copy
for supported homogeneous Beam Decode batches. Attention mathematics and
the installed `aclnnXAttention` kernel remain unchanged.

This optimization requires the optional `vllm_gr_xattention_out_ext` binding.
Installing vLLM-GR alone does not build it. Without a compatible binding,
the existing allocating XAttention API and output copy remain available.

### Requirements

Use the prepared [Ascend runtime](../getting_started/installation/ascend.md), including its
installed xllm-ops library and a source checkout providing
`test/python_test/pytorch_npu_helper.hpp`. The build compiles
`xattention_out.cpp` with that helper header; it does not compile the
upstream `RegisterOps.cpp` example. The binding targets the installed
scale-enabled XAttention signature used by the validated runtime below.

The validated build uses Python 3.12, PyTorch 2.10.0, torch-npu
2.10.0.post7, vLLM 0.30.0, vLLM-Ascend 0.30.0rc2, and CANN 9.2.0.
The build script currently requires those Torch, torch-npu, vLLM, and
vLLM-Ascend versions. Other versions require compatibility validation and
an update to the build contract before use.
The binding is not a portable binary for other vendor runtimes. Rebuild
and validate it after changing the runtime or custom operator provider.

The build adds a Python/C++ binding around the installed kernel. It does
not rebuild the XAttention kernel, replace `custom_ops_lib`, or change the
normal vLLM-GR package installation.

### Build

Run these commands inside the prepared NPU environment, from the vLLM-GR
repository root. Set `XLLM_OPS_DIR` to the source checkout used to build the
installed xllm-ops library. Keep the CANN and custom operator environment
from the Ascend installation guide loaded.

```bash
export XLLM_OPS_DIR=/path/to/the/installed/xllm-ops/source
export XATTENTION_OUT_BUILD="$PWD/build/xattention-out"

# Inspect the build plan. This command does not compile or install anything.
python tools/npu/xattention_out/build.py \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

# Build the optional binding in a fresh directory.
python tools/npu/xattention_out/build.py \
  --execute \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

export PYTHONPATH="$XATTENTION_OUT_BUILD/lib:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import vllm_gr_xattention_out_ext as m; print(m.__file__)'
```

The build checks the supported runtime versions before compilation and
records the build scripts, binding source, and output artifact in
`build-receipt.json`. It also places
`vllm_gr_xattention_out_ext.receipt.json` beside the extension. Keep
that receipt with the binary: the loader checks the declared binding ABI,
Python ABI, exact recorded runtime versions, extension and `custom_ops_lib`
identities, and both resolved ACLNN provider symbols and file identities.
Paths and SHA-256 hashes must match the artifacts recorded at build time.
These checks detect changed artifacts; they do not establish the function
signature or numerical correctness of an arbitrary provider with the same
symbol names. Validate a rebuilt binding against the installed allocating
kernel before using it with a different provider or runtime.

Receipt schema 1 and binding ABI 1 are retained. Extra historical
`source_pins` fields in an otherwise compatible receipt or binding metadata
are ignored. An absent or stale receipt falls back before attention
dispatch. A fresh output directory is required so previous logs and
artifacts remain available. No package is installed by the build command. Keep the binding
directory on `PYTHONPATH` when launching the engine so spawned workers
can import it.

To return to the allocating route, start a new process without the binding
directory on `PYTHONPATH`. Restart the engine after enabling, disabling,
or replacing the binding; binding discovery is cached per process.

### Supported path and fallback

The direct route applies to NPU `xllm` incremental Beam Decode when the
output is a contiguous ND tensor with the same shape and BF16/FP16 dtype
as the query. It must not share storage with any input. The binding writes
into the supplied output buffer, including a valid view with a nonzero
storage offset.

Mixed batches that require scatter, unsupported output layouts, and an
unavailable or incompatible addon retain the existing route. Ordinary
Prefill computation, Cascade Decode, and CUDA are unaffected. The existing
allocating `x_attention(...)` API is unchanged when no output is supplied.

Native execution failures propagate. The operation is not retried through
the allocating API after a possible partial output write.

### Validate the binding

```bash
VLLM_GR_TARGET_DEVICE=npu python -m pytest \
  tests/ops/npu/test_npu_xattention_out.py \
  tests/ops/npu/test_npu_xattention_out_binding.py -q
VLLM_GR_TARGET_DEVICE=npu VLLM_GR_TEST_XATTENTION_OUT_REQUIRED=1 \
  python -m pytest tests/kernels/test_x_attention_out_npu.py -q
```

The hardware tests compare the caller-owned output against the installed
allocating kernel, check alias/layout rejection and neighboring output
sentinels, and replay FULL graphs with changed inputs. The required-addon flag
fails on missing prerequisites instead of treating fallback as an optimization
pass.
Full-model result consistency and graph replay must be checked separately
for the intended model and request configuration. Importing the addon or
setting a graph option alone does not establish those results.
