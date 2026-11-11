# NPU P0 matrix and evidence contract

The formal `p0` selection extends the 17-node `p0-candidate` smoke suite with
host/runtime contracts and an explicit B1/B4 model/graph matrix. It targets
fixed-beam constrained SID inference through the public V1 API. It does not
change model math, CUDA dispatch, or external CI rules. The integration also
includes the runner and contiguous FIA block-table prerequisites, so its test
head can be validated without a local production-code overlay. Merge the
prerequisites first to keep the final matrix-only difference easy to review.
See the [runner guide](npu-test-runner.md) for prerequisites and invocation.

## Selected P0 scope

The reviewed manifest contains 225 unique nodes in 10 groups. The counts below
describe the mandatory selection for each run, not a guarantee of success. Actual
collection must agree with the exact manifest IDs, and every mandatory node
must execute before a run can qualify. An unfrozen manifest is blocked.

| Group | Nodes | Main contract |
| --- | ---: | --- |
| `patch_platform` | 6 | Initialization order, visible patch failures and platform/import isolation |
| `admission_cleanup` | 40 | Admission limits, startup errors, physical token budgets and resource release |
| `kv_metadata_graph` | 35 | KV identity/layout and graph metadata/update lifecycle |
| `constraint_core` | 15 | Constraint artifact, validity, ABI and failure cleanup contracts |
| `projection_lifecycle` | 15 | Stage projection semantics and failed weight-update handling |
| `runtime_contracts` | 19 | Supported versions, runtime resolution, import origins and rejection paths |
| `entry_contracts` | 50 | Device/tier routing, exact selection, outcome audits and process cleanup |
| `graph_observer_contracts` | 26 | Diagnostic observation/restoration and matrix assertion controls |
| `kernels` | 15 | Real NPU numerical references, KV operators, selection and backend registration |
| `model` | 4 | Existing model smoke/fallback nodes plus the new B1/B4 matrix |
| **Total** | **225** | Every selected node is mandatory |

The first eight groups exercise host contracts in the same prepared Ascend
Python as the native and model groups. Scoped mocks and CPU tensors isolate
particular boundaries; they do not establish a production CPU Beam runtime.
Each group starts a new pytest process to prevent fixture patches from leaking
into another group. The runner resolves/checks the real runtime before the
group loop and preserves the caller's vendor paths and allocated device.

## Model and graph cases

The existing nodes in
[`test_onerec_v1_npu_e2e.py`](../../tests/st/test_onerec_v1_npu_e2e.py)
remain mandatory. Their B1/B2, W8 checks include B2 FULL replay, the deliberate
B1 shape-miss fallback, repeated requests and shape reuse. The new matrix does
not replace or weaken that fallback coverage.

The new nodes in
[`test_onerec_v1_npu_p0_matrix.py`](../../tests/st/test_onerec_v1_npu_p0_matrix.py)
are:

| Node parameter | Requests | Raw prompt per request | Effective Prefill per request | Beam width | Semantic steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b1` | 1 | 127 tokens | 128 tokens | 128 | 3 |
| `b4` | 4 | 127 tokens | 128 tokens | 128 | 3 |

The public API appends one SID begin token, so raw 127 means effective L128.
The B4 inputs contain distinct prompts. Prefill totals are therefore 128 and
512 tokens; each Decode input has 128 and 512 beam rows respectively. These
cases use BF16, fixed beam width, a generated legal SID artifact and the same
input identity for eager, graph and repeated graph execution. Fused FFN is
disabled; dynamic beam is not enabled.

Each matrix node checks:

- The requested number of outputs and 128 unique legal SID triples per request,
  finite ordered scores, correct prompt/boundary tokens and finish status.
- Identical ordered output tokens between eager and graph, and between repeated
  graph calls. Scores use the existing comparison contract, `abs=1e-4` and
  `rel=1e-5`; this is not a bitwise-score equality claim.
- Actual cold Prefill inputs, positions, request identities and attention
  lengths, with no padding to a larger bucket or reuse of prior prompt KV.
- One replay of the exact Prefill bucket, two FULL Decode dispatches per graph
  request, and no added graph fallbacks. Eager execution must not dispatch FULL.
- Stable graph context across repeated calls, no unfinished requests and
  successful prefix-cache reset before proceeding.

These assertions distinguish actual graph replay from flags or capture logs.
The evidence records input/model/artifact identity, observed forward metadata
and outputs. A passed synthetic reference or host diagnostic test does not
substitute for either model node.

## Qualification and artifacts

Use the normal [runner interface](npu-test-runner.md#run-the-npu-entry-point)
with `--npu-tier p0`, a new output directory and the caller's leased device.
The checked-in manifest is the node-selection source of truth. Review the
actual collection again when changing node IDs, assertion contracts or runtime
dependencies. The `frozen` flag records selection review, not a successful run;
changing counts to hide missing nodes is not acceptance.

For each group, retain `pytest-events.json`, `junit.xml`, `pytest.log`, the
command execution receipt and `result.json` under `groups/<name>/`. The root
result aggregates node outcomes and observed setup/call/teardown counts, while
runtime and source records identify what ran. Collection alone is not a pass.
Each group's pytest temporary files are retained under `basetemp/`, including
the model's inputs, constraint artifact and serialized outputs. Use a short
`TMPDIR` for engine IPC sockets; it is independent of pytest's artifact directory.
Mandatory skips, collection skips, xfail/xpass, missing/additional nodes,
incomplete phases, source changes or cleanup failure prevent P0 acceptance.
After the first failed group, later groups are explicitly `NOT_RUN`.

Keep every failed or blocked attempt and report it separately from a successful
retry. Version/import failures, unavailable model/operators and device resource
failures must be reported as such; they cannot establish numerical correctness.
Do not infer acceptance for another checkpoint or device from these results.

## Remaining scope

Formal P0 here is a bounded correctness and replay selection. Direct fused
XAttention reference cases cover B1/B2; the selected B4 Cascade case and the
model's observed route do not by themselves prove direct fused B4 XAttention
numerical parity. The raw BF16 constrained-TopK loader's known-precision case is
also distinct from the selected production wrapper, which widens logits.

Long contexts, dynamic beam, online-service behavior, prolonged memory
stability, other model families and calibrated performance acceptance need
separate coverage. P1/P2 selections and nightly schedules are not implemented.
There is no latency SLA or performance-improvement claim from these tests.
Adding `p0` neither configures Jenkins to run it automatically nor makes it a
required GitHub check; external CI adoption remains a separate reviewed change.
