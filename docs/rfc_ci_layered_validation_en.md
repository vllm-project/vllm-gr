# RFC-001: CI layered validation and change-driven selection

Revision 6. Status: implemented L1-L5 baseline; change-driven selection remains
a proposal. This RFC updates PR #534 against the repository implementation.
It changes documentation only, not workflows, selection, or benchmark policy.
Implementation snapshot: upstream `main` commit `c3fbc64`.

## 1. Scope and implementation status

PR #613 introduced cost/scope levels and strict pytest classification. Levels
describe test workload; workflow events decide when those levels run. The old
proposal that equated L1 with local checks and L5 with releases was not adopted.

The current baseline includes:

- Explicit level markers and a parameter-rule inventory, resolved together by
  `tests/level_plugin.py` and `tests/level_policy.py`.
- PR L1/L2 selection and scheduled L3/L4/L5 selection.
- An existing documentation-only PR route.
- Independent benchmark acceptance and the `Benchmark blocking gate`.

The proposed `test_scope.js`, `run_changed_tests.sh`, and
`changed_test_selection.py` are not implemented. Their names below identify
possible follow-up deliverables, not commands that can be run today.

## 2. Current level and trigger contract

- **L1:** fast unit, contract, validation, and failure-path checks; PR scope.
- **L2:** representative device/integration correctness; PR scope.
- **L3:** broader functional matrices and model/system integration; pushes to
  `main`, including merges into `main`.
- **L4:** expensive model, accuracy, performance, and larger workload matrices;
  daily at 02:00 Asia/Shanghai (`0 18 * * *` UTC).
- **L5:** stability, soak, and stress; Monday at 03:00 Asia/Shanghai
  (`0 19 * * 0` UTC). An empty L5 selection reports that no tests ran.

`pre-commit.yml` sets `VLLM_GR_TEST_LEVEL=pr` for the full PR validation path.
`pr` means `l1 or l2`. PR validation requires `ready`, a non-draft PR, and the
workflow's event/base-branch conditions. `test-levels.yml` also supports manual
L1-L5 dispatch, defaulting to L3.

Documentation-only PRs take their existing lightweight route. The allowlist is
Markdown under `docs/` and root `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, and
`CLAUDE.md`. An arbitrary Markdown path or a file under `docs/` is not sufficient.
Mixed changes, rename source paths, and uncertain/incomplete diffs must be
handled conservatively. The base-pinned classifier and final gate revalidation
remain authoritative; see [Contributing](../CONTRIBUTING.md#pull-request-ci).

### 2.1 Classification sources and maintenance

For a function, class, or module whose specifications share one level, use one
explicit `@pytest.mark.l1` through `@pytest.mark.l5` marker. Use
`tests/level_inventory.json` when parameter values or supported runtime options
need different levels. Neither source silently overrides a contradictory level
from the other. Conflicts and invalid rules fail ordinary collection too;
`--strict-test-levels` additionally rejects unclassified items.

When migrating a function to explicit markers, remove its obsolete inventory
entry and cover every parameter instance. Rename/move operations must update
inventory keys. Strict collection proves classification consistency, not the
appropriateness of the chosen level or device correctness. See the
[classification guide](tools/README.md#maintaining-the-classification).

Do not publish inventory totals as collected-case totals. A function can expand
into many parameter instances, explicit markers need no inventory entry, and
module skips can prevent collection. Counts require a pinned revision,
environment, runtime options, and collection/error/skip summaries. Historical
counts from PR #613 are not a current coverage inventory.

### 2.2 Hardware eligibility and process isolation

`cpu_test`, `gpu_test`, and `npu_test` are separate from L1-L5. They describe
eligibility and do not automatically skip execution. L1 does not imply that a
test module can import without the accelerator runtime. Root fixtures can also
import Torch and vLLM.

`run_test_level.sh` combines a level expression with the target platform filter:
CUDA/NPU runs include host tests and tests for that accelerator; CPU selects
host tests only. An unset `VLLM_GR_TARGET_DEVICE` leaves platform selection
unfiltered. The wrapper enables slow tests and strict levels and uses `-x`.
An empty non-L5 selection fails.

With `VLLM_GR_RUN_MODEL_SMOKE=1`, the wrapper partitions the files named in
`inproc_test_selection.py`: ordinary tests run separately from tests requiring
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Any future selector must preserve this
partition and propagate failures from either subprocess.

### 2.3 Benchmarks are an independent PR gate

`Benchmark blocking gate` aggregates policy resolution, pre-commit validation,
Decode/Recall, Pangu offline/online acceptance, and the required current-revision
A/B result. Documentation-only and explicit A/B-only requests have their own
branches. A skipped required validation is not a successful run.

These shell-script acceptance paths are not selected by pytest level markers.
For example, the Pangu pytest accuracy cases can be L4 while the independent
Pangu acceptance scripts still run before PR merge. Changing an item's level
does not move, disable, or satisfy those scripts. CUDA/NPU thresholds and
branch-specific behavior come from the benchmark policy and resolver.

The scheduled CUDA L4 lane additionally prepares OpenOneRec inputs and runs
the existing Recall/performance scripts. The NPU lane does not thereby gain
coverage of CUDA-only scripts. Preserve accepted thresholds and input identity.

### 2.4 Runner availability and local full validation

`test-levels.yml` declares GPU and NPU matrix lanes on a shared self-hosted
pool without corresponding device labels. Each lane checks the actual device
tooling; a mismatch reports that the level did not run. Two declared lanes do
not guarantee execution on both platforms. The accelerator concurrency group
serializes participating jobs; it does not reserve a physical device against
external Jenkins jobs or developer processes.

The resource probe requires `VLLM_GR_ISOLATED_RUNNER=true`, set only when all
eligible devices are isolated. Model tests need the matching runtime, weights,
and data. `HF_TOKEN` is a secret needed for an uncached gated OpenOneRec dataset,
not a normal repository variable. Repository variables and secrets serve
different purposes.

Local hooks retain full slow-suite execution unless `VLLM_GR_TEST_LEVEL` is
set. When there is no capacity to run scheduled levels separately, keep local
full validation; the proposed selector must not silently replace it.

```bash
# Run in the prepared runtime. Leave VLLM_GR_TEST_LEVEL unset for full coverage.
bash tools/pre_commit/run_tests.sh

# Optional classification audit; collection does not execute the suite.
python -m pytest --collect-only --strict-test-levels --run-slow tests

# Selected levels, when separate runs are appropriate.
bash tools/ci/run_test_level.sh pr
bash tools/ci/run_test_level.sh l3
bash tools/ci/run_test_level.sh l4
bash tools/ci/run_test_level.sh l5
```

`tools/run_test.sh` additionally prepares/selects the accelerator. Its CUDA and
NPU paths differ; neither entry point is the full set of standalone benchmark
gates. See [validation entry points](tools/README.md#validation-levels).

## 3. Proposed change-driven selection

### 3.1 Preserve the baseline

The first implementation must retain full L1/L2 for every non-documentation PR.
A code change can break device execution without changing a test, so
"code-only" is not evidence that L2 is unnecessary. L1-only selection is not
approved by this RFC. Reducing L2 would require a reviewed ownership/dependency
map, independent validation of missed regressions, and a separate policy change.

Representative Beam correctness, graph/eager parity, request isolation,
completion, and cancellation assertions stay in PR scope. Changing level labels
or truncating CUDA Graph capture grids is not equivalent to eliminating
redundant setup. Startup capture, allocation, and maximum-shape behavior are
coverage even when requests do not replay every captured bucket.

### 3.2 Classify paths, then resolve actual collected tests

A proposed base-pinned path classifier could report three states:

- `docs-only`: precisely the existing documentation allowlist and safeguards.
- `code-only`: non-documentation changes with no known test/dependency changes;
  still run the full L1/L2 baseline and existing benchmark gates.
- `test-change`: changed tests or shared test infrastructure; retain the baseline
  and select additional affected specifications as described below.

The JavaScript classifier may identify changed files but must not claim to
resolve pytest levels. Only collection in the prepared target runtime knows
the actual `callspec.params`, explicit/inherited markers, and CLI options.
`resolve_level()` reads inventory policies; the pytest plugin also checks
explicit markers. Reuse both through real strict collection rather than
calling the inventory resolver alone.

Select the union of baseline L1/L2 and affected tests, not an intersection that
accidentally filters edited L3/L4/L5 tests out. If affected items include L3,
the initial proposal runs full L3 in that platform lane. For L4/L5, run the
affected collected specifications on the matching device and report that full
scheduled coverage was not run. Missing models/data must be reported and routed
to suitable infrastructure, never represented as a successful empty selection.

### 3.3 Selection completeness

Required handling before implementing the proposal:

1. Compare canonical function IDs for ownership, but retain exact collected
   parameter IDs and their resolved levels for selection. If one parameter
   changes and that cannot be identified reliably, select all instances of the
   function or the entire file. A changed L2 specification does not classify
   the rest of its function as L2.
2. Added/renamed tests use head-side IDs. Deleted tests cannot be executed;
   report deletion, update stale inventory entries, and run the baseline.
3. Shared fixtures, inherited tests, test builders, data, conftests, markers,
   inventory/policy/plugins, runtime requirements, and test/CI launchers may
   affect tests outside the edited file. A shared-infrastructure change must
   conservatively escalate affected higher levels or route to an explicit
   broader validation requirement; baseline L1/L2 alone is not proof of safety.
4. Use exact node IDs or a collection plugin for selection. `-k` is a substring
   expression, not a reliable exact-ID selector. Filter affected items from one
   full collection so paths do not append redundant whole-suite runs. Existing
   `run_test_level.sh` already supplies `tests`; adding a path is not a narrow
   replacement for that positional argument.
5. Preserve the platform filter, strict classification, slow-test inclusion,
   in-process partition, exit codes, and skip reporting on every execution path.
   Mandatory changed tests selecting zero eligible cases require a visible
   routing/validation decision, not the optional empty-L5 exception.
6. Paginate and validate the changed-file list and both sides of renames. Pin
   base/head and revalidate before the final gate. Missing policy, truncation,
   collection errors, or ambiguous dependencies require conservative fallback.

### 3.4 Follow-up deliverables and acceptance

Possible implementation files are `tools/ci/test_scope.js`,
`tools/ci/changed_test_selection.py`, and a dedicated runner if needed. Do not
encode unions of changed node IDs as `VLLM_GR_TEST_LEVEL`: the existing wrapper
accepts only `pr` and `l1` through `l5`.

Before enabling a selector, compare its selected IDs with an independent
expected set on prepared CUDA and NPU runtimes. Cover explicit/inventory
classification, mixed-level parameters, renames/deletions, shared fixtures,
dependency changes, empty sets, device mismatches, and subprocess failures.
Report passed, failed, skipped, deselected, and unavailable coverage separately.

Benchmark gates, input identity, thresholds, local full-suite defaults, and
branch protection remain unchanged. Rolling back the new selector must restore
the existing PR baseline, not remove the level plugin or scheduled validation.

## 4. Coverage-preserving performance work

Measure resource waits, environment setup, collection, engine initialization,
capture, request execution, and shutdown separately. Preserve full test inputs,
assertions, capture boundaries, and benchmark measurement semantics.

Same-configuration engine reuse is a candidate, not implemented here. Group
only tests with matching model/artifact identity, dtype, runner, startup options,
graph policy, and isolation requirements. Retain per-case results and verify
request retirement and state cleanup. Startup/shutdown, cold-start, fault
injection, and independently calibrated benchmarks may require fresh engines.
Validate performance gains and full functional coverage independently.

## 5. Open work and authoritative references

Open work: change-driven selection, dependency ownership mapping, meaningful
L5 stability coverage, evaluating non-fail-fast nightly reporting, device labels
and reservations, and measured reduction of repeated initialization. None is
established merely by this RFC or a successful collection.

Implementation references:

- [Validation and classification](tools/README.md).
- [Hook scope and documentation checks](PRE_COMMIT.md).
- [PR labels and required gates](../CONTRIBUTING.md#pull-request-ci).
- `tests/level_plugin.py`, `tests/level_policy.py`, `tests/level_inventory.json`.
- `tools/ci/run_test_level.sh`, `tools/ci/inproc_test_selection.py`.
- `tools/ci/run_resource_level.sh`, `tools/ci/resolve_runtime.py`.
- `tools/ci/docs_only.js`, `tools/pre_commit/run_tests.sh`.
- `.github/workflows/pre-commit.yml`, `.github/workflows/test-levels.yml`.
- `.github/benchmark-gate.yaml` and the existing `tests/system/` gate scripts.
