# NPU test runner and acceptance suites

The NPU test entry point supports the complete suite, an incomplete candidate,
and a grouped P0 selection for fixed-beam constrained SID inference. These are
explicit selections; adding the interface does not replace model benchmarks or
change an external CI policy. CUDA keeps its existing test path. The runner
interface adds test coverage and orchestration, not production inference behavior.

## Scope and staged adoption

| Selection | Purpose | Acceptance boundary |
| --- | --- | --- |
| `full` | Default; retain the existing complete `tests/` selection with slow tests enabled | Existing test assertions and platform availability conditions remain in force |
| `p0-candidate` | Opt-in; run the 15 operator/integration representatives and two existing model smoke nodes in the checked-in manifest | All 17 mandatory nodes must execute and pass; no missing nodes or skips are accepted |
| `p0` | Opt-in; run the grouped host/runtime contracts, native representatives and B1/B4 model/graph matrix | Every frozen node must pass setup, call and teardown; an unfrozen selection is blocked |

The manifest is [npu_test_suites.json](../../tools/ci/npu_test_suites.json),
schema version 1. Its `p0-candidate` entry is marked `candidate-incomplete`.
The operator and model groups remain separate in the result evidence. P1 and
P2 selections, nightly schedules, and performance thresholds are not implemented
by this entry point. Platform markers such as `npu_test` describe execution
dependencies; they do not define P0/P1/P2 priority.

The candidate's existing model nodes cover B1/B2 with beam width 8. Its graph
node requires B2 FULL replay and intentionally checks B1 shape-miss fallback.
It does not supply the target B1/B4, per-request input length 128, beam width
128, Prefill FULL and Decode FULL matrix. Removing the B1 fallback assertion
would lose coverage rather than complete that matrix.

Direct XAttention reference cases cover B1/B2. The selected B4 attention
representative exercises Cascade, so it must not be described as B4 fused
XAttention validation. Raw BF16 constrained-TopK loader readiness has a separate
known-precision skip; the selected BF16 production wrapper widens logits to
FP32. Long contexts, online serving, repeated-request memory stability and
calibrated performance acceptance also need separate coverage.

The formal P0 selection adds the previously missing B1/B4 matrix while retaining
the existing B2/W8 shape-switch and B1 fallback checks. Its reviewed selection
has 225 exact nodes in 10 groups. Collection and full execution must validate
that selection in the prepared Ascend runtime before it is accepted; a checked-in
node list is not a report that those nodes passed. See the
[P0 matrix and evidence contract](npu-p0-matrix.md) for its coverage and limits.

P0 host-oriented groups use CPU tensors or scoped test doubles inside the same
real Ascend Python used by the device groups. Each group has a fresh pytest
process so fixture patches do not leak into later native/model execution. The
runner does not substitute a fake global runtime or infer that results from a
different dependency environment apply to this run.

## Prepare the runtime and the model

Use the matching vendor runtime and custom operators described in the
[Ascend guide](../quickstart-ascend.md). Installing the GR NPU package does not
install or upgrade that vendor runtime. Preserve paths added by CANN and the
operator environment when prepending the checkout to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
vllm-gr check-env --npu
```

For either selected model suite, set `VLLM_GR_TEST_MODEL` to a verified local
OneRec-compatible checkpoint with the required SID vocabulary. Record the
checkpoint and tokenizer identity. A smaller compatible model can establish
functional smoke coverage; it does not establish acceptance for another model
size or the production gate model. Missing model weights, operators or the
Ascend runtime are not successful mandatory-node validation.

Allocate a device through the surrounding scheduler or lease mechanism before
running the commands below. `ASCEND_RT_VISIBLE_DEVICES` controls visibility;
it does not acquire a lease. Confirm the physical-device mapping in the actual
container/runtime, especially when the host and container expose different
device sets.

## Run the NPU entry point

The repository wrapper accepts an inherited NPU device, tier and output path:

```bash
./tools/run_test.sh \
  --npu-device "$allocated_npu_device" \
  --npu-tier full \
  --npu-results-dir "$new_result_directory"
```

`full` is the default tier. Supplying `p0-candidate` is a deliberate opt-in:

```bash
./tools/run_test.sh \
  --npu-device "$allocated_npu_device" \
  --npu-tier p0-candidate \
  --npu-results-dir "$new_result_directory"
```

The existing `--device-id N` and `--device-id=N` options can also supply the
allocated device when either `--npu-tier` or `--npu-results-dir` enables the
runner. In that opt-in path the device must be one canonical nonnegative
integer and must agree with inherited visibility. If both device options are
present, their values must match. Without NPU runner options, the existing
`--device-id` behavior, including comma-separated devices, remains unchanged.

After the formal selection has been collected, reviewed and frozen, opt in to
the grouped P0 suite with:

```bash
./tools/run_test.sh \
  --npu-device "$allocated_npu_device" \
  --npu-tier p0 \
  --npu-results-dir "$new_result_directory"
```

The manifest's `frozen` guard prevents an incomplete draft from qualifying as
P0. Do not bypass it, delete mandatory nodes or relax assertions to obtain a
passing run. `p0` does not silently fall back to `p0-candidate` or `full`.

The wrapper preserves the NPU runtime/operator preparation before invoking the
test runner. Passing a device prevents its NPU path from choosing a different
card. The runner uses the supplied device; it neither acquires nor releases the
caller's lease. Keep that lease through operator preparation, pytest/model
children, any requested wheel check, and cleanup. A busy or mismapped device is
a resource failure, not permission to switch to an unleased device.

The direct runner interface, after preparing the runtime and operator
environment, is:

```bash
python tools/ci/run_npu_tests.py \
  --device "$allocated_npu_device" \
  --tier p0-candidate \
  --output-dir "$new_result_directory"
```

Use a new result directory for each attempt and retain failed attempts. Do not
change model paths, test selectors, assertions or expected counts merely to
make a candidate run pass.

The direct runner also accepts `--tier p0`. Its `--timeout SECONDS` bounds each
pytest process. The candidate and each P0 group default to 1800 seconds; `full`
retains the surrounding job's deadline unless explicitly overridden. The runner
resolves and checks the runtime once before the P0 group loop; each preflight
command is bounded to 120 seconds. A timeout is a failure, with cleanup evidence
for task-tagged children.

## Interpret collection and execution separately

Collection establishes which nodes pytest selected; it does not execute those
tests and cannot establish correctness, graph replay or performance. The
candidate and each P0 group must collect exactly their declared nodes, then
execute the same selection. A successful process exit with skipped mandatory
nodes does not satisfy either contract. Collection skips, xfail/xpass outcomes,
missing or additional nodes, and incomplete phases cannot qualify P0.

Retain the selection and manifest identity, runtime and import origins, device
identity, collection result, per-node outcomes, logs and JUnit evidence. Report
failed, errored, skipped and unrun work separately. Existing numerical and graph
assertions remain unchanged. A graph flag or capture message is not a substitute
for the model node's actual replay/fallback assertions. These smoke nodes are
not a latency benchmark. P0 stores separate event reports, JUnit, logs, command
receipts and audits under `groups/<name>/`. The top-level result aggregates
observed outcomes and phases. After a failed group, later groups remain explicit
`NOT_RUN`; their nodes are not counted as passes. Process cleanup or a changed
source tree also prevents acceptance.

## Connect an existing external Jenkins allocation

The external job retains ownership of scheduling, locking, cancellation and
artifact retention. Adapt the existing call to `tools/run_test.sh`; adding this
repository interface does not modify a live Jenkins job or GitHub branch rules.

When the job has already injected its allocated device as
`ASCEND_RT_VISIBLE_DEVICES` into the container, a conservative full-suite call
inside that container is:

```bash
./tools/run_test.sh \
  --npu-device "${ASCEND_RT_VISIBLE_DEVICES:?missing inherited NPU device}" \
  --npu-tier full \
  --npu-results-dir "$build_result_directory" || exit 1
```

Use the outer job's actual allocation variable when adapting a different
wrapper. Do not call the repository's idle-device selector again after the
external job has acquired a lease. The caller must retain a safe mapping from
its allocated physical device to the runtime's accepted visibility value.

Adopt selected suites as additional reviewed comparisons before considering any
replacement for the complete suite. Completing the P0 matrix does not install
a Jenkins trigger, make it required for every PR, or define P1/P2 schedules.
Keep the existing static hooks, OneRec system benchmark, Pangu offline accuracy,
Pangu online accuracy/API
contract and their failure policies. This pytest selection does not replace
them or turn an external result comment into a required GitHub check.
