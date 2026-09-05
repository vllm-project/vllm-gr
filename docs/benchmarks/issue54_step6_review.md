# Issue 54 — Step 6 NPU V1 implementation record

## Status and scope

Step 6 is **IN PROGRESS / NOT ACCEPTED**. Real NPU validation is deferred by
the user's environment decision, not waived. No Step 7 work is authorized by
this record. The starting commit is
`07ccb6f80c2e8d8802f9fe32f658742b5d71cb4d` (Step 5, with its separately recorded
revised acceptance contract). The hit/miss numerical discrepancy remains open;
see `issue54_step5_strategy_review.md`.

The first implementation slice is an isolated NPU persistent Beam decision
helper and CPU contract tests. It must not activate the incomplete NPU runner
factory, replace legacy execution, change CUDA behavior, or import ModelRunner
V2. CPU fake-operator results prove only the helper's tensor and ownership
contract, not xllm operator numerics, ACL capture, transfers, or runtime overlap.

## Initial inspection (2026-09-05; source-version conclusion superseded below)

The NPU backend factory in `npu_model_runner_patch.py` still returns `None`.
The existing `_prepare_inputs` wrapper calls persistent-state preparation
before the original method, then performs the legacy Beam remap afterwards.
The current test proves call order only: it does not prove that native input
preparation preserves the published device token or builds current-step
attention metadata.

The locally available upstream Ascend checkout
`/tmp/onerec-upstream-audit-ascend` is from the v0.11.0rc0 era (commit
`00ba07102212c7c7a40de427f09848f2e203c498`, 2025-09-30), not a verified
deployment dependency for this vLLM 0.22.1 project. In its
`vllm_ascend/worker/model_runner_v1.py`:

- `_prepare_inputs` starts at line 1237, takes `intermediate_tensors`, and
  returns twelve values at line 1564, unlike the four-value wrapper contract.
- Lines 1370–1371 copy `input_ids_cpu` into device `input_ids` inside preparation.
  Publishing tokens before that method alone does not protect them from being
  overwritten. This is evidence against assuming compatibility, not proof that
  the as-yet-unidentified target release has the same implementation.
- There is no independent `sample_tokens` method matching the proposed hook.

The local ACS copy is modified reference code, not the target upstream runner.
At this point the team had not checked the repository's pinned dependency
inventory and unnecessarily asked the user for the target version. The
following source audit corrects that omission: the supported repository
baseline is already specified, so source-level implementation is not blocked
on a user reply. Actual deployment-version differences still require later
environment validation.

Additional existing paths that cannot be reused unchanged:

- `run_beam_decision()` catches fused NPU errors and falls back to Torch; its
  NPU branch does not honor deferred `reorder_kv=False` and clones scores.
- `_finalize_decision()` can reorder KV internally. A persistent backend must
  own the reorder boundary explicitly to avoid applying the same mapping twice.
- Legacy constraint metadata builds Decode prefixes from Host `beam_suffix`;
  canonical sampler construction also checks validity with `.item()`. Neither
  is acceptable as the persistent NPU Decode input path.
- The legacy NPU output encoder materializes Beam tensors on the Host. The
  future persistent path must bypass it, not silently fall back to it.

## Entry CPU baseline

Command (one process; no GPU model process):

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_model_runner_patch.py \
  tests/test_gr_model_runner_state.py \
  tests/constraints/test_npu_backend_integration.py
```

Result: **43 passed, 4 skipped, 22 warnings in 18.00 s**. Three local skips
require `vllm_ascend.compilation.acl_graph`; one module-level skip requires
`torch_npu` and real hardware. These are **ENVIRONMENT NOT VALIDATED**, not
successful NPU tests. Warnings include the unavailable CUDA driver in the
CPU sandbox and the unregistered pytest timeout option. This is a focused
baseline, not complete CPU CI.

## Implemented decision-only slice

`vllm_gr/v1/worker/npu_gr_decision.py` now provides explicit Prefill bootstrap
and single-stage fused Group/FinalSelect helpers. Both retain sequence and
cumulative scores in `session.context.decision`; a successful call increments
its device-stage counter once. Non-final parent/prefix tensors are returned
for the future backend to consume. The helper does not reorder KV or advance
the session Decode cursor. FinalSelect returns no reorder request.

Structural checks cover active session, isolated B=1/slot zero, fixed width,
capacity, candidate/output shape and dtype, stage consistency, and final
operator alignment. The default operator path checks the actual Group and
FinalSelect exports, independently of xllm attention capability. It neither
catches operator failures into a Torch fallback nor materializes Beam tensors
on the Host. GPU, legacy NPU, and shared runtime files remain unchanged.

`tests/test_npu_model_runner_persistent.py` exercises that isolated contract
using explicitly injected CPU golden operators. `tests/kernels/test_persistent_gr_npu.py`
adds six real-NPU decision probes: widths 64/128/256, `(capacity, generation
stages)` of `(3, 2)` and `(5, 3)`, and narrower final selection. These probes
omit operator injection. They compare real fused results with a CPU golden,
including valid sequence contents, canonical storage identity and parent
prefix. The hardware tests intentionally copy results for comparison and
simulate the external session cursor update; they do not test KV operations,
data-residency performance, ModelRunner integration, or Graph replay.

Design qualifications for review:

- The helper is **not called by production code yet**; the NPU backend factory
  remains incomplete. This is an implementation slice, not a new fallback or
  a hidden enable switch.
- `ops=` is an explicit test-injection seam that bypasses the default NPU
  device check. Future production callers must omit it. Injection is not a
  runtime-selected CPU fallback and is not technically restricted to mocks.
- Fused operators still allocate output tensors, including sequence output.
  Reusing canonical buffers does not mean fixed operator workspace is complete.
- Full-capacity sequence input/output follows the existing caller and CPU
  reference. Real xllm capacity/stride behavior remains unvalidated until the
  new hardware probes actually run; existing smoke only used active-length
  capacity.
- Checks validate structure, not device value domains, parent bounds, validity
  masks, or fused numerical correctness. A failed operator/device copy has no
  transactional rollback guarantee. Future integration must terminate the
  affected session rather than retry it from potentially partially changed
  device state.
- This slice does not relax the original Step 6 exit gate or resolve the
  Step 5 hit/miss discrepancy.

## Earlier decision-only regression checkpoint

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_model_runner_persistent.py \
  tests/test_npu_model_runner_patch.py \
  tests/test_beam_decision.py \
  tests/test_model_runner_common.py \
  tests/test_gr_model_runner_state.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-cpu.BHPAPe/final-results.xml
```

Result: **144 passed, 5 skipped, 23 warnings in 18.66 s**. The new CPU
contract file contributes 27 passing cases with no skip. The five skips
are three existing ACL-wrapper tests and two NPU-dependent modules. The
six parameterized hardware decision cases were not executed: their module
was skipped because `torch_npu` is absent. No GPU or NPU model was launched.

Ruff check, Ruff format check, and `git diff --check` passed. Root and the
independent auditor reviewed the helper; the auditor also checked the new
hardware probe's limited scope. At this checkpoint only four new files were
present, with no changes to existing runtime files. The source hashes below
describe that earlier checkpoint, not subsequent workspace changes:

```text
npu_gr_decision.py
76cd646afb17577de2764e49b30577ec2f07a25cde17b190a703264cd9d026c4
test_npu_model_runner_persistent.py
85a0e758e328796c2b67afe7f3d4f15ee67e665b5e8dde338f404dfeab774af0
tests/kernels/test_persistent_gr_npu.py
101091489070d4cfc10c6079211eb69baeee4ed90c06b047ec1de3c5196331c1
/tmp/issue54-step6-cpu.BHPAPe/final-results.xml
ec3f86d0d78332e6151ecefe51004fd218ca08dc3dd27eedfe8d2dac0cf46361
```

```text
Step: 6 — decision-only implementation slice, incomplete
Commit: none; worktree additions over Step 5 commit 07ccb6f80c2e
Changed files: npu_gr_decision.py; test_npu_model_runner_persistent.py;
               tests/kernels/test_persistent_gr_npu.py; this review record
Behavior change: no production path change; isolated strict NPU decision helper
Tests run: command above; Ruff check/format; git diff --check
Test result: 144 passed, 5 environment skips; new CPU cases 27/27 passed
Hard counters: helper step increment/storage identity only, no E2E counters
Trace/benchmark: none; no NPU hardware execution
Known limitations: target Ascend version, full runner/backend/transport integration,
                   fixed workspace, capacity-stride hardware ABI, lifecycle and Graph
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## Corrected, pinned Ascend V1 source audit

`docs/runtime_patch_inventory.md` specifies Ascend `0.22.1rc1` and exact
commit `f528cdd561969730504bfe8b5e96b6b7e69d6c4f`;
`docs/quickstart-ascend.md` specifies the same package version. The exact
commit was fetched into a new, separate checkout without installing NPU
dependencies or modifying the old reference or GPU environment:

```bash
git -C /tmp/issue54-ascend-source.UDVZG6 fetch --depth=1 \
  https://github.com/vllm-project/vllm-ascend.git \
  f528cdd561969730504bfe8b5e96b6b7e69d6c4f
git -C /tmp/issue54-ascend-source.UDVZG6 checkout --detach FETCH_HEAD
```

The checkout is clean. Its `vllm_ascend/worker/model_runner_v1.py` SHA-256 is
`00ed6cfff6d96403462e9c9cb4a64a7ab7c7a9171a5ddd028324a80c1e86eb13`.
This actual source supersedes conclusions based only on the old v0.11 clone:

| Boundary | Source evidence | Consequence |
| --- | --- | --- |
| `_prepare_inputs` result | Returns three items at 1366–1370; caller unpacks three at 2124–2131 | The existing four-item-only wrapper is wrong for the pinned baseline |
| Device input copy | Calls inherited `_prepare_input_ids` at 1077 | Persistent Decode needs a scoped D2D input override; writing before prepare alone is insufficient |
| Attention metadata | Built in `execute_model` at 2262 | Metadata is **not** built inside `_prepare_inputs` in this baseline |
| Sampling boundary | `sample_tokens` at 2428; `_sample` call at 2477 | There is a real delayed sampling boundary for device decision/control reduction |
| Async output | Constructor at 2645; `set_async_sampled_token_ids` at 2654 | O(B) reduction must precede construction, and the second Host-history consumer must be guarded |

The inherited vLLM `_prepare_input_ids` starts at line 1691 in the installed
`vllm/v1/worker/gpu_model_runner.py`, whose SHA-256 is
`690998ae7d9353ce42d7fc43f755fdbd72097c6e1f7a364dde435743408b00f2`.
It can copy native Host/control input back to the device. Future integration
must scope its override to NPU persistent B=1 Decode and reject unsupported
PCP/DCP, speculative, grammar, routed-expert and extra timing/sync paths.
No complete runner fork or ModelRunner V2 dependency is needed for these
identified narrow boundaries; real execution remains unvalidated.

The NPU wrapper now supports the pinned three-item result and the existing
four-item extension, replacing only the logits-index element. Tail values
retain identity. Unknown Beam return arities fail before remap; ordinary
requests return the exact original object. Six new CPU cases cover these
contracts. This fixes return adaptation, **not** the persistent token-copy,
metadata or transport integration.

## Fixed lineage workspace implementation

`npu_gr_workspace.py` adds fixed request-owned storage, epoch/identity leases,
raw parent columns, FP32 cumulative-score-difference columns, final backtrace
scratch, status/active buffers, and two control storage slots. Canonical
sequence/scores remain references to the session's existing buffers. It
freezes generation length and final result width, allowing capacity larger
than actual length and a narrower result only at the final stage. A score
snapshot must precede decision mutation; history is appended, not reordered
at every stage. Final packing backtraces once and uses actual R/S ABI offsets.

Review qualifications and remaining strategy decisions:

- Per-token values use the **current GPU ABI's FP32 cumulative-score
  difference**, not a claim to recover the original sampler logprob. A CPU
  rounding counterexample explicitly proves these can differ. No token-ID
  reverse lookup is used to guess candidate identity.
- Invalid parent indices are made safe before gather and accumulate a device
  error status. The current unconnected workspace also flags inactive or
  non-finite beams. This temporary finite-only rule is **not an accepted
  reduction of the original scope**: GPU/NPU inactive-beam semantics must be
  unified before enabling runtime integration. Final transport must reject
  nonzero status; that transport is not implemented by this workspace.
- A lease can reset only once; stale, foreign, absent or released leases,
  repeated snapshots, premature final packing and malformed stage counters
  fail. Canonical object identity and tensor structure are checked. Device
  aliases bind to the actual allocation device (`npu` versus `npu:N`); a real
  `cpu:0` allocation reproduces the equivalent alias mismatch in CPU tests.
- `release()` is storage ownership bookkeeping, not proof that outstanding
  DMA has drained. Two control buffers likewise do not implement completion
  tickets or permission to overwrite in-flight slots.
- Fixed resident storage does not eliminate temporary Torch/xllm outputs.
  `device_memory_bytes` counts only workspace-owned resident allocations,
  excluding canonical aliases and short-lived operator tensors.

The hardware probe now chains real NPU decision helpers into this workspace,
packs the result, and runs the shared final-output decoder against independently
backtraced CPU parent/delta paths. Test-only per-step D2H remains explicit;
these tests cannot substantiate a transfer or performance claim.

## Latest workspace and wrapper regression

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_gr_workspace.py tests/test_npu_model_runner_persistent.py \
  tests/test_npu_model_runner_patch.py tests/test_gr_model_runner_state.py \
  tests/test_model_runner_common.py tests/test_beam_decision.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-workspace.5zhAGs/results.xml
```

Result: **175 passed, 5 environment skips, 22 warnings in 19.60 s**. Workspace
tests contribute 25 passes; decision tests 27. No GPU/NPU model was launched.
The three existing ACL tests and two NPU modules remain unvalidated, not PASS.
Seven changed/new Python files passed Ruff check and format check;
`git diff --check` passed. Root and the independent auditor reviewed the
workspace, return-contract fix and extended hardware probe.

SHA-256 for this checkpoint:

```text
results.xml: 13b5ada1351576c758913de4405aaf8951071ac8357eb41e8a2fe3e6fe22ac8a
npu_gr_workspace.py: 2322220f431fcb988f6a9e687102e1083dd2a374c24f84073908b6ac678da05e
test_npu_gr_workspace.py: b2faf126ab5d37c7692cf53ab2f06d55a287a6cb4d7e6f90786a449aeb59818a
npu_model_runner_patch.py: 8e546632493cc53c6080dc3e1de572bd3a18cb6f5b63ecb8d900976df754dc41
test_npu_model_runner_patch.py: 93a11e29816aa6fc02166a5eaad2294e47966493ae612b722db9e7c1c471f522
tests/kernels/test_persistent_gr_npu.py: e2146af2de2fd40a70ab01da553c8283254358f75a63b1173879d402bddae14e
```

```text
Step: 6 — workspace/ABI slice and pinned NPU prepare-result correction
Commit: none; Step 6 remains in progress over Step 5 commit 07ccb6f80c2e
Changed files: NPU decision/workspace modules; NPU wrapper; associated CPU
               and pending-hardware tests; this review record
Behavior change: pinned NPU prepare result now preserves upstream arity;
                 persistent helpers/workspace remain unconnected
Tests run: command above, Ruff check/format, git diff --check, pinned source audit
Test result: 175 passed, 5 environment skips; no actual NPU execution
Hard counters: CPU lineage/lease/storage contracts only; no E2E counters
Trace/benchmark: none
Known limitations: inactive semantics, NPU full backend/transport/KV integration,
                   hardware capacity ABI, Graph/trace/lifecycle acceptance
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## Input-copy integration and ACL dispatch review

The NPU adapter now exports an execute-local physical-row wrapper and a narrow
`_prepare_input_ids` wrapper from `npu_gr_inputs.py`. The common installer
installs the optional input-copy hook on the selected runner subclass; it does
not replace the inherited GPU base method. Ordinary requests and Prefill keep
their upstream input-copy path. The NPU backend factory still returns `None`:
these hooks must not be mistaken for an enabled persistent NPU backend.

For a typed persistent Decode sidecar, the execution frame temporarily exposes
W rows to native V1 preparation, then restores the existing SchedulerOutput's
logical count of one on success or exception. It retains the shared execution
context needed by the later `sample_tokens()` call. The input hook validates
request/state/stage ownership and copies the next tokens from the persistent
device tensor, bypassing the upstream Host/control-token copy. Unsupported
CP, PP, speculative, prompt-embedding, grammar, encoder and connector paths
fail explicitly rather than falling back to legacy input preparation.

Source review caught a test/implementation mismatch before enablement:
inherited `GPUModelRunner._get_cumsum_and_arange()` returns cumulative counts
without a leading zero. Ascend f528cdd passes this array at line 1077, so the
isolated B=1 frame receives `[W]`, not `[0, W]`; the latter is the separate
query-start layout assembled at lines 1047–1049. The regression now exercises
the installed upstream helper, rather than reproducing the same mock layout.
Review also identified that Prefill token views may be strided slices of the
canonical sequence history. Copying that view must not require a second
contiguous Beam-history copy or a Host round trip.

The ACL dispatch wrapper now respects explicit `force_eager` (argument index
6 in the pinned V1 contract) and model `enforce_eager`. Exact-width eligibility
requires an explicit positive Host integral unpadded token count; it cannot
infer width from a padded descriptor or convert a device scalar to Python.
Ordinary requests still return the original dispatch result unchanged. These
guards do not prove ACL capture or metadata correctness on real hardware.

This slice does not yet replace the remaining native position, sequence-length,
block-table or slot-mapping preparation; validate those device-resident inputs
and their consumption by the actual attention metadata before enabling the
factory. It also does not implement final-copy completion or control-buffer
DMA ownership. CPU frame tests are not evidence of device overlap or zero
intermediate synchronization.

### Input-copy regression checkpoint

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_gr_inputs.py \
  tests/test_npu_gr_workspace.py \
  tests/test_npu_model_runner_persistent.py \
  tests/test_npu_model_runner_patch.py \
  tests/test_model_runner_patch.py \
  tests/test_gpu_model_runner_patch.py \
  tests/test_gr_model_runner_state.py \
  tests/test_model_runner_common.py \
  tests/test_beam_decision.py \
  tests/test_engine_core_gr_abort.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-inputs.uNWL9A/results.xml
```

Result: **343 passed, 5 environment skips, 22 warnings in 20.79 s**. The new
input file contributes 59 passing CPU cases, including upstream cumulative
counts, real bootstrap-to-input strided views, and shared execute-wrapper
success/error composition. The five skips remain the same three unavailable
ACL-wrapper cases and two NPU-only modules. This is a focused CPU regression,
not complete CI or GPU/NPU execution. No model process was launched.

Ruff check passed for all six input-hook/adapter/dispatcher source and test
files. Ruff format check passed for the five files excluding the common
dispatcher, whose pre-existing execute-wrapper formatting was left untouched.
`git diff --check` passed. After the test run only Ruff formatting was applied
to `npu_gr_inputs.py`; no runtime semantics changed.

```text
npu_gr_inputs.py
d8fb8fbd347433c1a79932a9f1a8186502bffb9e4350e8b9daac186994681bce
test_npu_gr_inputs.py
e07ee7ff8ce07d9f59b98f15da8345d3918b45d8f3369bb6ad9cc5758f09b454
npu_model_runner_patch.py
b56f9b432de22c9dc395d152ce8e6f2547c33cba155ff2714062a0c46601a1f5
model_runner_patch.py
6f60859ecd19064bf1b5ce59cd6cb98aaeda2c26805f6f73051a39b98098500a
test_npu_model_runner_patch.py
8acf78eeee3411888171bda8c80b495feed2738c1dde43536de3db3d96b3d7ff
test_model_runner_patch.py
de0058dc6999194f5d927328aad6fe97850de6b3374459acf253ae286a825ebb
/tmp/issue54-step6-inputs.uNWL9A/results.xml
3a99d4ecd49930380ac9aff272e55510bc699e18949a4b5f83ad489613da42c5
```

```text
Step: 6 — NPU V1 input-copy integration slice, incomplete
Commit: none; worktree over Step 5 commit 07ccb6f80c2e
Changed files: npu_gr_inputs.py; NPU adapter; common hook installer;
               corresponding three CPU test files; this review record
Behavior change: scoped physical Decode geometry; resident token copy boundary;
                 eager/exact-width ACL safety gates; factory remains disabled
Tests run: focused command above; Ruff check/format; git diff --check
Test result: 343 passed, 5 environment skips; new input CPU cases 59/59 passed
Hard counters: fake single dispatch/copy and logical restoration only; no E2E claim
Trace/benchmark: none; no GPU or NPU hardware run
Known limitations: metadata/KV/backend/transport integration and real NPU acceptance
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## Prefix binding, native-slot isolation and metadata integration

The next NPU prepare slice is wired into `make_patched_prepare_inputs_npu`.
`npu_gr_metadata.py` provides a final-Prefill binding helper for the future
backend bootstrap and the pre/post-native-prepare metadata operations now
called by the adapter. The binding helper uses the pinned Ascend public
`MultiGroupBlockTable.block_tables` and `get_device_tensor(num_reqs)` APIs.
It returns a stable view of exactly `ceil(prefix_len / block_size)` columns,
excluding Native logical output blocks. Native request, session and request
metadata must agree on prefix length. The bootstrap caller is not wired yet;
the disabled NPU factory means this is not an E2E binding-count result.

Decode uses the existing `state.prefix_binding` tuple without re-extracting
Native block tables or querying Prefix Cache. The adapter validates request,
session, context, typed sidecar, physical geometry, canonical buffer capacity,
current step and fixed buffers before publishing metadata. Terminal/released
states, mismatched final flags, non-exact Beam widths and unsupported native
KV groups fail explicitly. The supported case remains B=1, slot zero, a
single non-hybrid/non-Mamba group without CP.

The pre/post split is necessary because the pinned original NPU prepare writes
positions and sequence lengths itself, whereas attention metadata is built
later in `execute_model`:

| Field | Authoritative value | Publication point |
| --- | --- | --- |
| `decode_step_buf` | `state.step - 1` | Before original prepare |
| `decode_step_1based_buf` | `state.step` | Before original prepare |
| `xllm_block_table_buf[0]` | `session.slot` | Before original prepare |
| context Prefix binding | Same `state.prefix_binding` tuple | Before original prepare |
| Native slot buffer | `PAD_SLOT_ID`; suffix writes use the Beam pool | Inside original prepare |
| Beam positions | Existing common Beam remap | After original prepare |
| device/CPU optimistic sequence length | `prefix_len + state.step` | After original prepare/remap |
| Native query starts | Original `[0, W]`, not W fake requests | Preserved |

### Why native slot computation is bypassed

Ascend f528cdd `model_runner_v1.py:1276` invokes native slot mapping while its
positions still describe `num_computed_tokens + arange(W)`. The common Beam
position remap occurs afterwards. Ascend `block_table.py:426` forwards to the
single-group method at line 146, which invokes the imported native kernel.
The installed upstream kernel reads the block row using `position //
block_size` without validating the effective allocated block count or row
boundary. W physical Beam queries therefore cannot safely be interpreted as
W consecutive Native tokens, especially near the model-length limit.

The active incremental branch in `beam_attn.py:285–378` writes into the Beam
suffix pool and returns before the standard paged-cache write. That makes
Native slot output unnecessary but does **not** make the unsafe computation
itself acceptable. The new prepare-local proxy intercepts only native slot
computation, fills the original slot buffer with PAD, and delegates other
operations and indexing to the original object. `finally` restores the exact
original table on success and error; no global method is patched. Repeated,
stale, missing, compressed or draft-slot calls fail rather than falling back.
Ordinary requests and Prefill retain original slot computation.

Because Native slots are now intentionally invalid, `cache_len == prefix_len`
is a required persistent-Decode invariant. The existing shared builder would
otherwise split queries into standard/mixed segments. Here `cache_len` denotes
the already-resident full Prefix KV extent; it does **not** require Prefix
Cache to be enabled or a cache hit. Bad cache lengths fail before publication.

These are NPU adapter implementation details for review, not an expansion to
B>1, V2, bundled execution, or different GPU/NPU Beam semantics. Prefix KV,
Native logical KV accounting and Beam suffix KV remain separate.

### Metadata regression checkpoint

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_gr_metadata.py tests/test_npu_gr_slot_mapping.py \
  tests/test_npu_gr_inputs.py tests/test_npu_gr_workspace.py \
  tests/test_npu_model_runner_persistent.py tests/test_npu_model_runner_patch.py \
  tests/test_model_runner_patch.py tests/test_gpu_model_runner_patch.py \
  tests/test_gr_model_runner_state.py tests/test_model_runner_common.py \
  tests/test_beam_decision.py tests/test_engine_core_gr_abort.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-metadata.pmpdAq/results.xml
```

Result: **402 passed, 5 environment skips, 23 warnings in 21.29 s**. The new
metadata file contributes 36 cases and the native-slot file 23. Tests include
a CPU reproduction of the near-capacity table-index failure, proxy exception
restoration, fixed-address/step invariants, shared Prefix binding consumption,
and actual shared `_segment_requests` / `_build_beam_mappings` calls proving
Beam-only segmentation, bounded query indices and no incremental suffix slot
mapping. The NPU prepare-wrapper composition test still mocks the position
remap and native execution; it is not real NPU attention validation.

The five skips are the same three missing-Ascend ACL tests and two NPU-only
modules. No GPU/NPU model process was launched. Ruff check/format for the six
files below and `git diff --check` passed; only Ruff formatting of
`npu_gr_metadata.py` followed the test run, without semantic changes.

```text
npu_gr_metadata.py
815af8a36b4fba8a74588136b23ce89b64eeca814d8c0fd9619c605da0c8887e
npu_gr_inputs.py
92ace660b598396b50d823b50af29a21c5a89f2380441acb7a0ae0d454c1438e
npu_model_runner_patch.py
a4bb6776abb24e8420120ada912b7a4796e28f62c20ba1bf2f9d828414de5389
test_npu_gr_metadata.py
4d7d7f34b318fe98cb7c1cdc2385283a5726f18dc4d65c820edd8cea061edb4b
test_npu_gr_slot_mapping.py
58097dfa5650944315ed50194c2c3ac6ed2025a9d70b78e62ce07d388d521e8a
test_npu_model_runner_patch.py
cad79d8fe0e195bdded1276d8e9649fefe2b3dbca1a02ae7c218d4e0b93502ff
/tmp/issue54-step6-metadata.pmpdAq/results.xml
2765ea25a5e49c5cb031873dd86a2e2f1348af78566480be881aa176eac965ff
```

```text
Step: 6 — NPU Prefix binding and metadata integration slice, incomplete
Commit: none; worktree over Step 5 commit 07ccb6f80c2e
Changed files: npu_gr_metadata.py; npu_gr_inputs.py; NPU prepare adapter;
               metadata/slot/adapter CPU tests; this review record
Behavior change: scoped native-slot bypass; pre/post preparation metadata;
                 strict full-Prefix binding contract; factory remains disabled
Tests run: focused command above; Ruff check/format; git diff --check
Test result: 402 passed, 5 environment skips; new metadata/slot cases 59/59 passed
Hard counters: CPU fake no-native-slot-call/no-rebinding tests only; no E2E counters
Trace/benchmark: none; no hardware execution or performance claim
Known limitations: backend bootstrap/sampler/KV/transport integration; inactive
                   Beam semantics; real attention/ACL padding, lifecycle and HBM
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## KV ownership and Prefill validity checkpoint

The new `npu_gr_kv.py` boundary dispatches xllm KV selection directly, or
the existing cascade device implementation with fixed per-layer scratch.
It validates the session epoch, slot, canonical capacity, device step and
bound tensor object/layout/storage before use. A non-final decision advances
the session KV cursor once; a final decision performs no reorder. An operator
exception poisons the owner and is propagated without a retry or fallback.
Release is idempotent but retains references: it is **not** device drain or
proof that memory can already be reused. The helper is not yet wired into a
complete NPU backend. Cascade indexing may allocate device temporaries; this
is not a zero-allocation or ACL Graph claim.

Prefill workspace validation now requires a structural candidate mask
separate from the active Beam mask. Legal padding and legal inactive `-inf`
scores are preserved, including their first-step cumulative-delta ABI.
This supersedes the earlier finite-only limitation **for Prefill only**.
Empty or non-prefix masks, inconsistent activity, NaN and positive infinity
accumulate device error status. NaN/positive-infinity rejection is defensive
hardening beyond GPU's count-only Prefill check, not proof of identical
abnormal-value behavior. Decode validity and safe decision sentinels remain
unresolved; loosening workspace validation alone would be insufficient.

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_gr_kv.py tests/test_npu_gr_workspace.py \
  tests/test_npu_gr_metadata.py tests/test_npu_gr_slot_mapping.py \
  tests/test_npu_gr_inputs.py tests/test_npu_model_runner_persistent.py \
  tests/test_npu_model_runner_patch.py tests/test_model_runner_patch.py \
  tests/test_gpu_model_runner_patch.py tests/test_gr_model_runner_state.py \
  tests/test_model_runner_common.py tests/test_beam_decision.py \
  tests/test_engine_core_gr_abort.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-kv.USFuFp/results.xml
```

Result: **445 passed, 5 environment skips, 22 warnings in 21.38 s**.
New KV tests contribute 33 cases. Workspace tests include valid padded
Prefill results compared word-for-word with the real GPU final packer on
CPU tensors. These are CPU contract tests, not GPU or NPU model execution.
The five skips remain two missing-torch_npu modules and three missing-Ascend
tests; none counts as hardware acceptance.

```text
/tmp/issue54-step6-kv.USFuFp/results.xml
f56c3e4941fb03d9827af05ed5d55fc2280f163009706bdc4f81798885ab2e34
npu_gr_kv.py
a56bdf9d766da83a2783d786c3878aa10114cff628ea7637eac7ecf1e8a952e7
npu_gr_workspace.py
7ffd8112b66f4b7a22113e275970c3c89fa594464b8a0c5318bb828af64e8093
test_npu_gr_kv.py
9acf0ebcea9360c155ce2b473a9ccf482f28d5538cac804f8b94c6f4724c62ac
test_npu_gr_workspace.py
c3095f445a112d0de3e9bea80b7c47e537b97b98dc793afbeee0562ad16ce67a
tests/kernels/test_persistent_gr_npu.py
4a9cae14ca1f3d620cb7e76e2160a924e4e3099af80d260ef74faa3812d70e67
```

```text
Step: 6 — KV ownership and Prefill validity slice, incomplete
Commit: none; worktree over Step 5 commit 07ccb6f80c2e
Changed files: npu_gr_kv.py; npu_gr_workspace.py; KV/workspace tests;
               NPU hardware probe bootstrap arguments; this review record
Behavior change: guarded device KV dispatch; structural Prefill validation;
                 NPU backend factory remains disabled
Tests run: full focused command above
Test result: 445 passed, 5 environment skips
Hard counters: CPU helper cursor/release checks only; no E2E counters
Trace/benchmark: none; no hardware or performance claim
Known limitations: persistent sampler/decision/transport composition,
                   deferred Decode status, device drain, real NPU regression
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## Unresolved Decode mask ABI: concrete CPU counterexample

`tests/profiling/repro_npu_masked_beam_gap.py` isolates a gap that is not
resolved by workspace validation or `-inf` padding. Run:

```bash
env USE_TF=0 OMP_NUM_THREADS=1 \
  python tests/profiling/repro_npu_masked_beam_gap.py
```

Observed exit code: **1**, with `contract_mismatch: true`. This is a
deliberately failing diagnostic, not an expected-skip or a passing parity
test. W=32, each row has exactly one legal candidate, all legal logprobs are
`-inf`, and every cumulative parent score is zero. There are W legal
candidates and no empty row. GPU's mask-aware merge contract chooses tokens
10 through 41 and parents 0 through 31; the repository's xllm-signature Group
and FinalSelect CPU references choose token 10 followed by 31 padding zeros,
all from parent zero. Group prefix counts differ as well.

Code evidence: `ops/cuda/beam_selection.py::_beam_group_topk_kernel` uses
candidate validity to exclude padding even at equal `-inf` scores, whereas
`ops/onerec_decode.py::_beam_search_group_torch` and
`_beam_search_rec_final_select_torch` sort scores without a candidate mask.
The NPU fused wrappers expose the latter mask-free signature. This proves a
reference/ABI insufficiency, **not** that the real NPU kernel has been run on
this input. Existing finite-candidate hardware probes do not prove this case.

Do not substitute finite sentinel scores: they can change legitimate score
ordering and the result ABI. Do not reject every inactive Beam and call that
GPU parity. A complete fix needs a mask-aware device selection contract
(including grouping, lineage, safe error sentinels and KV parents), either
in the fused operator or in an explicit backend device stage. The NPU factory
stays disabled until this and the remaining integration requirements are
resolved. This issue does not require V2, Host Beam D2H, or bundled execution.

### Actual GPU 0 oracle

The optional diagnostic was also run on physical GPU 0 (NVIDIA L20), without
loading a model or using GPU 1. It uses W=64 to satisfy the actual CUDA Worker
kernel gate, FP32 candidate/cumulative scores, one valid candidate per row,
and a sequence capacity of three stages:

```bash
env CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python tests/profiling/repro_npu_masked_beam_gap.py --cuda \
  > /tmp/issue54-step6-sampler.L7pIOo/masked-beam-gap-gpu0.json
```

The CUDA Worker Group produced tokens 10 through 73, parents 0 through 63,
all scores `-inf`, and status `[0]`:
`cuda_matches_mask_aware_oracle: true`. The diagnostic still exits **1**
because the xllm-signature CPU reference differs. Only Group was run on GPU;
FinalSelect in this diagnostic remains a CPU reference comparison. This
micro-level correctness observation is not NPU execution, E2E parity, a
performance benchmark, or an asynchronous overlap trace.

```text
Base: 07ccb6f80c2e8d8802f9fe32f658742b5d71cb4d (uncommitted Step 6 worktree)
tests/profiling/repro_npu_masked_beam_gap.py SHA256:
4201e17a8853b0951001efd3e1576e2d1f8dda678e28868e927526fb9b4325d7
/tmp/issue54-step6-sampler.L7pIOo/masked-beam-gap-gpu0.json SHA256:
f1429f356792e5d70328db17c5f93ee4193c09d956af5a53436dee7cf0d24183
```

### Strategy difference for review: mask-aware fused NPU ABI

The team recommends an explicit mask-aware xllm Group/FinalSelect extension
for the persistent path, preserving the old symbols for legacy A/B. This is
a proposed backend/operator contract addition to the original implementation
plan, **not an implemented or installed dependency**. Workspace and `/tmp`
inspection found wrappers, reference code and tests, but no local xllm Beam
kernel source to patch. The required external operator version is therefore
not yet established; no package upgrade or external publication was done.

The new contract must consume structural mask/counts, return deferred device
status, and write caller-owned fixed output buffers. A device row-head merge
can choose W (or terminal R) winners by cumulative score and original flat
index, treating a legal `-inf` as selectable and exhausted rows as absent.
Group then sorts winners by parent while preserving within-parent rank and
writes matching prefix counts; FinalSelect preserves global ranking. Error
rows must emit safe token/parent zero, score `-inf`, and consistent KV prefix
counts before any KV consumer. This is one generation stage, not a bundled
generation loop.

Do not silently run a second full Torch selector after xllm and describe it
as the unchanged fused path. An explicit alternate device selector is a
different strategy requiring separate review, trace and hardware validation.
Until the mask-aware decision contract exists, the isolated sampler may be
tested, but full Decode parity and NPU backend enablement remain incomplete.

## Persistent constrained sampler checkpoint

`npu_gr_sampler.py` provides a request-owned device-only selection boundary.
It calls the canonical constraint Backend directly with fixed prefix and
temperature tensors: Prefill uses one zero prefix row; Decode copies the
first one/two tokens from the session's canonical sequence into W rows.
The six-table protocol explicitly limits generation to stages 0/1/2. It does
not encode future B>1 admission by pretending that W rows are W requests.

The original `ConstraintOutput` and its raw logprobs/mask are retained, not
duplicated. Fixed wire scores replace structurally invalid entries with
`-inf`; structural counts, per-row deferred status, sampled-token scratch
and ranks stay on device. Legal `-inf` does not imply an invalid candidate.
Empty/holey candidate masks and valid NaN/+inf values produce deferred error
status without reading a tensor value on the Host. Numeric classification
uses the fixed mask scratch rather than separate per-step NaN/+inf tensors.
Underlying Backend/operator outputs and BF16 widening can still allocate;
this is not an allocation-free or Graph-readiness claim.

The owner enforces ordered single consumption, immutable tensor bindings,
workspace/canonical storage disjointness, and fail-closed poisoning after
an operator/ABI error. Release is idempotent and retains pending references;
device completion/drain must still be supplied by the future backend owner.
CPU execution requires explicit test injection; production requires NPU.

The helper is **not wired into `_sample` or the backend factory yet**. Its
W×W candidate `SamplerOutput` is internal and must be consumed by the device
decision, then replaced with O(B) control **before** Ascend `_bookkeeping_sync`
and `AsyncGPUModelRunnerOutput` consume it. Merely bypassing the old Host
constraint hook would not prove that intermediate Beam D2H was eliminated.

```bash
env USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/test_npu_gr_sampler.py tests/test_npu_gr_sampler_backend.py \
  tests/test_npu_gr_kv.py tests/test_npu_gr_workspace.py \
  tests/test_npu_gr_metadata.py tests/test_npu_gr_slot_mapping.py \
  tests/test_npu_gr_inputs.py tests/test_npu_model_runner_persistent.py \
  tests/test_npu_model_runner_patch.py tests/test_model_runner_patch.py \
  tests/test_gpu_model_runner_patch.py tests/test_gr_model_runner_state.py \
  tests/test_model_runner_common.py tests/test_beam_decision.py \
  tests/test_engine_core_gr_abort.py \
  tests/constraints/test_npu_backend_integration.py \
  tests/kernels/test_persistent_gr_npu.py -rs \
  --junitxml=/tmp/issue54-step6-sampler.L7pIOo/results.xml
```

Result at this checkpoint: **504 passed, 5 environment skips, 22 warnings in
22.02 s**. The sampler has 46 isolated cases; 13 composition cases use the
real `NPUConstraintBackend`, validated six-table representation and explicitly
injected CPU reference op. They compare independent legal-set log-softmax,
temperature 0/0.25/2, FP16/BF16/FP32, reordered canonical paths, missing
prefixes, raw-versus-wire padding, and backend closure/operator errors.
No real NPU operator, asynchronous ModelRunner integration, or ACL Graph is
covered by those CPU cases. The five environment skips are unchanged and
are not accepted hardware tests. Ruff check/format and diff whitespace
checks passed for the sampler, both test files and the diagnostic.

```text
npu_gr_sampler.py SHA256:
b909888c4f4c1089709e4642a56dbf74d8803be28d77d37d628a8ead83848f4b
test_npu_gr_sampler.py SHA256:
fa9ee5f9cfaecbb8615ab9b7696cd4d5f6587599409d184a52b6235269b4969c
/tmp/issue54-step6-sampler.L7pIOo/results.xml SHA256:
5ba672f5a440ec6125c581aea54461ba6867eb51d2ef482b8bcd634438695469
```

```text
Step: 6 — persistent constrained sampler slice, incomplete
Commit: none; worktree over Step 5 commit 07ccb6f80c2e
Changed files: npu_gr_sampler.py; isolated and real-table composition tests;
               masked decision diagnostic; this review record
Behavior change: direct device-prefix constraint sampling helper; strict owner
                 and deferred status; no factory or runtime sampler enablement
Tests run: focused command above; Ruff check/format; git diff --check;
           separate GPU 0 mask-aware Group oracle
Test result: 504 passed, 5 environment skips; GPU oracle matches masked contract;
             xllm-signature reference diagnostic still reports mismatch
Hard counters: CPU call/cursor/pointer checks only; no NPU E2E hard counters
Trace/benchmark: no async overlap or performance measurement
Known limitations: mask-aware fused ABI; full sampler/decision/transport
                   integration; device drain; real NPU hardware acceptance
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

### Final sampler normalization regression

Three more real-table composition cases cover degree=2 > W=1 at each of
the three stages and temperatures 0/0.25/2. Only one candidate is returned,
but its logprob is compared with log-softmax over **both** legal candidates
and remains negative rather than becoming zero through TopK-only
renormalization. The composition file now has 16 cases.

The entire sampler checkpoint command above was rerun unchanged except for
`--junitxml=/tmp/issue54-step6-sampler.L7pIOo/results-normalization.xml`:
**507 passed, 5 environment skips, 22 warnings in 21.98 s**. No production
code changed after the earlier 504-case run. Ruff check/format and
`git diff --check` passed again. The Step 6 exit gate remains **FAIL** for
incomplete integration and an unvalidated NPU environment.

```text
tests/test_npu_gr_sampler_backend.py SHA256:
b50fa3dcca2d9a2f20a84410ea795f21d9031eba2ce74157a9247349d0ddd9f8
/tmp/issue54-step6-sampler.L7pIOo/results-normalization.xml SHA256:
a54329dc31a463c380a4b7458eeff7d8928d432108a59a76f3003dc11dcdff04
```

## Shutdown ownership ordering and GPU validation

The ModelRunner shutdown wrapper now treats cleanup as three ordered phases:
persistent GR state and queued device ownership, the runner-owned constraint
Backend, then native ModelRunner shutdown.  A failed GR release stops the
sequence.  It leaves the constraint Backend attached and does not invoke
native shutdown, so the same runner can retry cleanup without invalidating
storage still reachable from queued work.  A successful retry releases GR
state first, closes the constraint Backend once, and calls native shutdown
once.  The synthetic no-native shutdown path follows the same ownership rule.

This closes the previously identified shutdown hole around the two-phase
`GRModelRunnerState.release()` contract.  The state contract already marks
execution closed before calling the backend drain, retains the Beam session
when that drain fails, and skips a completed backend drain if session cleanup
needs a retry.  Unit coverage now verifies both native and synthetic shutdown
failure paths and the successful retry ordering.

The saved Step 6 CPU regression selection passed **514 tests**, with the same
**5 environment skips** for missing `torch_npu` / `vllm-ascend`.  An expanded
GPU-path CPU stability selection covering scheduling, frontend delivery,
abort, terminal shutdown, late output, and Worker faults passed **666 tests**.
The CUDA kernel regression was submitted through `gpu run` and passed all
**16 tests** in `test_cuda_control_output.py` and
`test_persistent_gr_cuda.py`.

Four real OneRec-1.7B runs were then executed through `gpu run` with prefix
caching disabled, W=128, input length 1024, and a Constraint Tables document
compiled from 16,981 unique SID triples in RecIF `ad_test.parquet`.  Three
persistent ASYNC probes covered eager decode-boundary shutdown, eager advance
failure, and CUDA Graph decode-boundary shutdown.  A separate engine-driven
ASYNC eager run completed the full three-token generation and returned all
128 final sequences.  The native completion ABI v2 was built into an isolated
`/tmp` runtime; neither the shared venv nor its other editable source checkout
was modified.

The decode-boundary shutdown probe stopped with one live Scheduler request,
one frontend request, one Beam session, one occupied control slot, and an
unfinished output future.  Shutdown restored all 54,704 native KV blocks,
balanced the single persistent state, emptied both request registries and the
Beam registry, drained control/pinned storage, released every native
completion ticket, and passed all reported lifecycle gates.  The advance
fault probe also passed: it quarantined the failed transport with both control
entries retained, preserved the injected exception in the completed future,
then drained the transport and restored all ownership during shutdown.
The Graph probe captured W=128, recorded one FULL graph dispatch, restored its
initial 54,677 KV blocks, and passed the same lifecycle gates.
The full-output run completed the fixed generation contract and produced a
stable final ABI fingerprint.  Its harness-level exit gate remains FAIL by
design because no EngineCore host trace or per-owner memory attribution was
requested; the run is functional evidence, not a Step 6 performance or trace
acceptance result.

```text
/tmp/vllm-gr-step6-onerec-eager-shutdown.json SHA256:
149ae01ec10bdaf0a515b62c63ef60f4b75238e2c551cf0c95c526478f48ad8d
/tmp/vllm-gr-step6-onerec-advance-fault-shutdown.json SHA256:
3f7d8189332c66a65c57bafbb40aa93d57f7031688257eb346963cd17ddfcfad
/tmp/vllm-gr-step6-onerec-graph-shutdown.json SHA256:
2f97110afd2266b71a1c6de0e5bf1cf5bc8e140646b21bc428fe3f8f3985a4d3
/tmp/vllm-gr-step6-onerec-final-output.json SHA256:
99a3caa3dcfcbb55bbe353f4532e4d241943692533d9b15056b946feeb00f7e4
/tmp/vllm-gr-step6-ad-constraint-triples.json SHA256:
bf8654c4291a73c0ec9a622fd21e8a58ebf636f9a10128b213744444c5d786ea
```

NPU hardware validation remains explicitly deferred because this environment
has no NPU runtime.  The NPU backend factory continues to return `None` rather
than enabling an incomplete path or falling back silently.  In particular,
the mask-aware xllm decision ABI and full sampler/decision/KV/control/final-D2H
composition remain prerequisites for enabling the factory.  The CPU module
tests are design and composition evidence only, not NPU acceptance.

```text
Step: 6 — release-safety and GPU-stability checkpoint, NPU integration incomplete
Commit: none; worktree over e87dac84f648
Behavior change: shutdown stops after failed GR release and retains later-owned resources
GPU validation: 16 CUDA tests; four real OneRec runs via gpu run
CPU validation: 514 passed / 5 environment skips; expanded matrix 666 passed
NPU validation: deferred; no NPU runtime, factory remains fail-fast disabled
Exit gate: FAIL — Step 6 incomplete / NPU ENVIRONMENT NOT VALIDATED
```

## Remaining integration and exit gate

After the isolated decision and workspace contracts are verified, Step 6 still requires:

1. Version-matched NPU V1 preparation and output integration, preserving logical
   B accounting while supplying physical Beam rows; actual metadata must see
   the correct token, device step, block table and slot mapping.
2. Device-resident constraint prefixes and deferred validity status, fixed
   minimal lineage/logprob storage, and reusable workspace/control buffers.
3. Backend-owned page-locked final staging and completion handling, exactly
   one final Beam D2H, and O(B) intermediate control transport.
4. Explicit xllm/cascade attention capabilities, exact-width FULL ACL gates,
   and idempotent success/abort/error/shutdown release.
5. Real NPU eager/ACL/cascade smoke, cache-disabled correctness, cross-backend
   ABI parity, trace overlap, and HBM reuse checks.

No NPU hard counter, trace, numerical parity, memory stability, or performance
claim is established by the entry inspection or CPU fake-operator tests.
Step 6 cannot be committed as an accepted Step or marked PASS before the
remaining required checks are completed.

## 2026-09-05 explicit mask-aware NPU device stage

The proposed mask-aware xllm operator symbols were not added: the installed
xllm_ops ABI has only `beam_search_group` and
`beam_search_rec_final_select`, and no local operator source is available in
this workspace. `vllm_gr/v1/worker/npu_gr_masked_decision.py` instead
implements the alternative already allowed by the review above: an explicit
B=1 device tensor stage with fixed caller-owned outputs.

The stage performs a stable score sort followed by a stable structural-validity
sort. This makes a legal `-inf` candidate rank ahead of invalid `-inf` padding
without replacing either value with a finite sentinel. Non-final winners are
then stably grouped by parent and receive consistent prefix counts; final
winners retain global score order. Insufficient or invalid candidates produce
safe zero token/parent rows and a device status for the eventual single final
D2H check. The canonical sequence and scores remain device tensors.

CPU contract validation (not NPU hardware acceptance):

```text
146 passed
```

This includes the new legal-`-inf` counterexample, finite reference parity,
unaligned final result width, safe error rows, fixed output addresses, and the
existing NPU sampler/decision/workspace/KV contracts. NPU support and ACL Graph
behavior of the selected `torch.sort(..., stable=True, out=...)` path remain a
hardware acceptance item. The NPU factory remains disabled until the composite
backend and final async transport are connected.

## 2026-09-05 composite NPU backend software integration

`vllm_gr/v1/worker/npu_gr_backend.py` now implements the shared
`GRDeviceBackend` lifecycle and `npu_model_runner_patch.py` returns it from the
NPU factory. For a persistent final-Prefill or Decode frame, the patched NPU
sampler creates the common state before sampling and routes raw logits through
the request-owned `NPUPersistentConstraintSampler`. The backend then connects:

```text
constraint sampler -> explicit masked decision -> lineage workspace
                   -> xllm/cascade KV reorder -> alternating O(B) control
                   -> one packed final D2H -> common two-phase release
```

The decision stage copies its non-final parent prefix into the canonical Beam
buffers consumed by KV/attention. The workspace now accepts structural selected
validity and exact local scores, so a legal `-inf` remains a valid result while
its active bit is false; finite paths preserve the existing CUDA FP32-delta ABI.
No new xllm_ops symbols are assumed. The old mask-free xllm decision helper is
retained only as an isolated legacy/hardware comparison path.

The production final path owns a pinned Host staging buffer and NPU Event,
enqueues exactly one packed final copy, waits for that event, validates deferred
status, and publishes the common `gr_final_i32_f32_v1` payload. Intermediate
control remains two fixed int32 `[1,1]` device slots and is handed to the pinned
vllm-ascend async output object, which retains the source tensor through copy
completion. Backend release synchronizes NPU work before releasing sampler, KV,
workspace and session ownership; a drain failure leaves all ownership intact for
the common release retry.

The composite path currently requires B=1 and request width equal to the fixed
Beam pool width. That matches the exact-width primary path; smaller eager widths
remain a Step 8 matrix item. NPU runtime execution, stable-sort support/perf,
ACL Graph capture, pinned-copy behavior and HBM accounting remain deferred
hardware acceptance, as requested.

CPU integration validation after wiring:

```text
333 passed, 3 environment skips
full repository CPU-only regression: 1929 passed, 3121 skipped
```

The integration tests drive the real common state machine across Prefill, two
Decode decisions, KV calls, alternating control identities, final
packing/unpacking and ordered release. They also cover cached common-state
reuse with fresh backend ownership and a failed device drain that retains every
resource until a successful retry.
