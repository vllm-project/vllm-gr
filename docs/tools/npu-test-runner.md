# NPU test runner and candidate suites

The NPU test entry point supports a complete-suite path and an explicit,
incomplete candidate selection. It does not replace the existing model,
benchmark, or external CI policy with a shorter default gate. CUDA keeps its
existing test path.

## Scope and staged adoption

| Selection | Purpose | Acceptance boundary |
| --- | --- | --- |
| `full` | Default; retain the existing complete `tests/` selection with slow tests enabled | Existing test assertions and platform availability conditions remain in force |
| `p0-candidate` | Opt-in; run the 15 operator/integration representatives and two existing model smoke nodes in the checked-in manifest | All 17 mandatory nodes must execute and pass; no missing nodes or skips are accepted |

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

Host-oriented cases run in other dependency environments are not added to this
manifest as a universal suite. Keep their environment and results distinct.

## Prepare the runtime and the model

Use the matching vendor runtime and custom operators described in the
[Ascend guide](../quickstart-ascend.md). Installing the GR NPU package does not
install or upgrade that vendor runtime. Preserve paths added by CANN and the
operator environment when prepending the checkout to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
vllm-gr check-env --npu
```

For the candidate model group, set `VLLM_GR_TEST_MODEL` to a verified local
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

The direct runner accepts `--timeout SECONDS` for the pytest deadline. The
candidate defaults to 1800 seconds; `full` retains the surrounding job's
deadline unless explicitly overridden. Runtime preflight is bounded to 120
seconds. A timeout is a failure, with cleanup evidence for task-tagged children.

## Interpret collection and execution separately

Collection establishes which nodes pytest selected; it does not execute those
tests and cannot establish correctness, graph replay or performance. The
candidate must collect exactly its declared nodes, then execute the same
selection. A process exiting successfully after every required node skipped
does not satisfy the candidate contract.

Retain the selection and manifest identity, runtime and import origins, device
identity, collection result, per-node outcomes, logs and JUnit evidence. Report
failed, errored, skipped and unrun work separately. Existing numerical and graph
assertions remain unchanged. A graph flag or capture message is not a substitute
for the model node's actual replay/fallback assertions. These smoke nodes are
not a latency benchmark.

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

Adopt `p0-candidate` as an additional reviewed comparison first. Complete the
missing B1/B4 model/graph matrix and evaluate the collected differences before
considering any replacement for the complete suite. Keep the existing static
hooks, OneRec system benchmark, Pangu offline accuracy, Pangu online accuracy/API
contract and their failure policies. This pytest selection does not replace
them or turn an external result comment into a required GitHub check.
