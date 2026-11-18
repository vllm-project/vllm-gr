# Development utilities and validation

Run commands from the repository root. Use the
[development navigation](../development.md#select-regression-coverage) to locate
coverage and [Pre-commit setup](../PRE_COMMIT.md) for hook commands and scope.

## Validation levels

Choose the smallest relevant checks first, then add the device/system coverage
needed for the behavior being changed. Report exactly what ran. Passing a
targeted check does not satisfy every CI gate.

| Level | Entry point | Requirements and evidence |
| --- | --- | --- |
| Documentation | Explicit Markdown, character, typo, and whitespace hooks | No accelerator; also check local links and commands |
| Other pre-commit hooks | `SKIP=run-tests pre-commit run --all-files` | Prepared hook environments; explicitly excludes pytest |
| Targeted regression | `python -m pytest -q <test-file-or-node>` | Dependencies and fixtures of the selected tests; inspect device checks |
| Kernel/integration | Selected `tests/kernels/` or `tests/integration/` files | Matching CUDA or Ascend runtime and a usable device |
| Model/system | Selected `tests/st/` files or documented probes | Device, model snapshot, configuration/artifact, and any test data |
| Local full test runner | `./tools/run_test.sh` | Available accelerator, installed runtime, model dependencies; platform behavior below |
| Wheel smoke | `tools/run_wheel_smoke_test.sh` | Build tooling; validates installed wheel metadata/imports outside the checkout |
| Benchmark acceptance | `tests/system/` scripts and CI workflow | Fixed inputs, resolved branch/device policy, and saved results |

### Documentation and selected hooks

Use the [documentation checklist](../PRE_COMMIT.md#documentation-checks) for
Markdown changes and [hook scope](../PRE_COMMIT.md#choose-the-scope) for other
edits. Explicit hook IDs avoid launching the full pytest suite.

### Regression design

Extend the existing suite for the affected behavior. For a bug fix, reproduce
it before the change when the environment permits, then verify the same case
afterward. Test externally visible results and lifecycle effects through the
real entry point where practical. Small contract tests and mocks can isolate a
boundary, but cannot establish device execution or serving correctness.

For lifecycle changes, cover error/cancellation cleanup as well as completion.
Use deterministic inputs where possible and follow the existing numerical
parity contract. Do not replace a failing assertion with a skip, retry, looser
tolerance, or weaker benchmark threshold merely to obtain a passing run.

### Targeted tests and dependencies

The root `tests/conftest.py` resets sliced-logits state through an autouse
fixture that imports Torch and vLLM. Host-oriented tests are therefore not
necessarily runnable with only pytest or a CPU package install. Some test
modules also import runtime/kernel dependencies during collection.

Slow tests are skipped unless `--run-slow` is supplied. Device/model skips have
their own conditions. Neither omitting `--run-slow` nor running `--collect-only`
turns the entire suite into a dependency-free host check. Do not bypass shared
fixtures to make a runtime test appear supported.

### Test levels and CI selection

Strict CI collection requires exactly one `l1` through `l5` level for each
collected pytest item. Ordinary local collection permits unclassified items,
but still rejects conflicting markers and invalid rules. Choose a level using
both the risk of an undetected regression and execution cost. Levels describe
CI scope; they do not imply that a test ran successfully on every device.

| Level | Scope | Typical coverage |
| --- | --- | --- |
| L1 | Every PR | Fast host-side unit, contract, validation, and failure-path checks |
| L2 | Every PR | Small, representative device and integration correctness checks |
| L3 | After merge to `main` | Broader functional matrices and model or system integration |
| L4 | Nightly | Expensive model, accuracy, performance, and larger workload matrices |
| L5 | Weekly | Long-running stability, soak, and stress tests |

Keep representative assertions for critical behavior in L1 or L2, including
beam-result consistency against an independent reference, request isolation,
completion and cancellation, and graph/eager parity where applicable. For an
expensive matrix, keep a meaningful small specification in the PR scope and
assign expanded specifications to L3/L4. Cost alone does not justify moving
all coverage for a critical behavior out of PR checks. A benchmark gate or a
mock test is not automatically equivalent to an end-to-end assertion.

#### Maintaining the classification

For a test whose collected specifications all belong to the same level, put
one explicit marker close to it: on the function, class, or module. For
example:

```python
@pytest.mark.l1
def test_rejects_invalid_request():
    ...
```

If different parameter values or runtime options need different levels, use
`tests/level_inventory.json` with the parameter-free pytest node ID. For
example, a small batch can be L2 while the rest of the matrix is L3:

```json
"tests/example.py::test_case": {
  "default": "l3",
  "rules": [{"level": "l2", "when": {"ref": "params.batch", "eq": 1}}]
}
```

The example is an entry inside the manifest's `tests` object. Rules must be
non-overlapping and refer to actual collected parameters or supported options.
Do not combine a manifest rule with an incompatible explicit marker, or apply
different level markers through function, class, module, or parameter scope:
conflicting levels fail collection. The `cpu_test`, `gpu_test`, and
`npu_test` platform markers are a separate requirement; see [Test tiers](#test-tiers).

When migrating a test from the manifest to explicit markers, remove its old
manifest entry. The entry covers every parameter instance of that function:
migrate the entire function and ensure each collected instance has exactly one
appropriate marker. If only some specifications have been migrated, keep a
complete manifest rule for the function instead of leaving other cases
unclassified or adding incompatible markers.

When adding a test, choose its level and declare it explicitly or add its
parameter rule. When renaming or moving a test with a manifest entry, move the
entry to the new node ID and remove the old one. When changing parameters,
update the rule and verify both the small PR cases and the expanded cases.
Run strict collection and the relevant selected tests, for example:

```bash
python -m pytest --collect-only --strict-test-levels tests
bash tools/ci/run_test_level.sh pr
```

Strict collection rejects unclassified items and conflicting or invalid
rules; it cannot decide whether a level is appropriate. Review the actual
assertions and selected specifications as well as collection results.

#### Running by level

Select a level with:

```bash
bash tools/ci/run_test_level.sh pr  # L1 and L2
bash tools/ci/run_test_level.sh l3
bash tools/ci/run_test_level.sh l4
bash tools/ci/run_test_level.sh l5
```

The wrapper enables slow tests and strict classification. When
`VLLM_GR_TARGET_DEVICE` is `cuda` or `npu`, it selects CPU tests and tests
for that accelerator; `cpu` selects CPU tests only. Without a target it
does not apply a platform filter. An unknown or unclassified test is an error.
An empty L5 selection reports that no stability tests ran; an empty selection
at another level is an error. Additional pytest
arguments may be supplied after the level. The full local `run-tests` hook
continues to run all tests unless `VLLM_GR_TEST_LEVEL` is explicitly set.

The pre-commit CI job sets `VLLM_GR_TEST_LEVEL=pr` to run L1/L2. The separate
level workflow selects L3 on pushes to `main`, L4 nightly, and L5 weekly; it
also supports manual level selection. These level runs require a GPU runner.
The shared self-hosted pool has no GPU/NPU labels, so it can assign an NPU
runner: in that case the workflow reports that the level did not run and skips
the test steps. A scheduled run therefore does not guarantee GPU execution or
coverage of both platforms. The existing benchmark gates remain separate.
Use runners with isolated accelerator access. The level workflow does not
terminate processes belonging to other jobs. L3/L4 model tests require the
model and runtime dependencies in their fixtures. The L4 GPU job also prepares
the gated OpenOneRec dataset and runs the full OneRec recall and performance
scripts; a cache miss requires the `HF_TOKEN` secret. The Pangu pytest cases
cover the corresponding model scenarios; the existing PR A/B
gate remains the entry point for its PR-only comparison. The PR L1/L2 job and
manual L2 workflow include the device resource probe only when the repository
variable `VLLM_GR_ISOLATED_RUNNER` is exactly `true`. Set it only when the
physical devices of every runner eligible for these jobs are isolated from
other work: the probe discovers and locks devices, and its two-device scenario
runs only when two are available.
Otherwise the job reports that the probe did not run. The wrapper also reports
any skipped device scenarios. `tools/run_test.sh` remains available to run the
probe locally on a suitably isolated host.

After locating coverage, run the selected files or pytest node IDs in the
installed development runtime. For example, for initialization changes:

```bash
python -m pytest -q tests/test_patch_initialization.py
```

Packaging tests intentionally have a separate invocation that excludes the
runtime conftest, matching the wheel smoke script:

```bash
python -m pytest --confcutdir=tests/packaging -q tests/packaging/test_install_metadata.py
```

That command requires Git, pytest, packaging, and the build tools declared in
the `dev` extra, including pip, setuptools, setuptools-scm, and wheel.
Build isolation alone does not install
these tools into the test environment. It does not establish inference support.
Check the [coverage map](../development.md#select-regression-coverage)
for the tests associated with other code areas.

The configuration declaration and AST-hook tests also have a standalone entry
point. They use pytest and regex without importing the inference runtime:

```bash
python -m pytest --confcutdir=tests/config -q tests/config
```

This invocation covers only the declaration and hook boundary. Run the existing
`tests/test_gr_config*.py` and related runtime configuration tests normally, with
their shared fixtures, in the prepared development environment.

### Test tiers

Every test declares the platforms it runs on with a tier marker registered in
`pyproject.toml`; collection fails and lists the modules when one is missing.
As in upstream vLLM, the markers only describe the environment and never skip
a test, so each host selects its tier with `-m`:

```bash
python -m pytest tests -m 'cpu_test and not gpu_test and not npu_test'  # CPU host
python -m pytest tests -m 'gpu_test or (not cpu_test and not npu_test)'  # CUDA host
python -m pytest tests -m 'npu_test or (not cpu_test and not gpu_test)'  # NPU host
```

| Marker | Use when the test |
| --- | --- |
| `cpu_test` | passes with no accelerator visible |
| `gpu_test` | needs a CUDA device |
| `npu_test` | needs the Ascend runtime (`torch_npu`, vLLM-Ascend, xllm-ops) |
| `model_test` | also needs model weights; combine it with a tier |

A device marker on a test, class, or `pytest.param` overrides the module's
`cpu_test`; the expressions above implement that. A test marked both
`gpu_test` and `npu_test` runs once on each kind of accelerator host. Do not
use `-m 'not cpu_test'` on an accelerator host: it drops the device tests that
live in `cpu_test` modules. `tests/kernels/` and `tests/st/` default
undeclared tests to `gpu_test`, so NPU and host-side tests there must declare
their tier.

Choose the tier from what the test executes, not from its file name; many
`test_npu_*.py` modules run on any machine. On a CUDA host, hide the GPUs and
check which tests pass:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q -rs tests/test_example.py
```

On an Ascend host, run that check where the Ascend runtime is not installed,
such as a CPU-only container, because a test can depend on `torch_npu`
imports as well as on visible devices. Mark device tests the same way on
both platforms, and keep the availability guard next to the marker:

```python
@pytest.mark.gpu_test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_path() -> None: ...


@pytest.mark.npu_test
@pytest.mark.skipif(not _npu_is_available(), reason="NPU is not available")
def test_npu_path() -> None: ...
```

When only some parameters need a device, mark those parameters, for example
`pytest.param("cuda", marks=pytest.mark.gpu_test)`.

### Device and model tests

Select a device allocated for the run and use the matching installed runtime.
The following examples use device 0; change it to the assigned device. Run
CUDA and NPU examples in their respective environments.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/kernels/test_rec_constrained_topk_cuda.py
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest -q tests/constraints/test_npu_backend_integration.py
```

For OneRec system tests, `VLLM_GR_TEST_MODEL` can point to a local directory
containing `config.json`, tokenizer files, and weights. `HF_HUB_OFFLINE=1`
prevents Hugging Face fallback downloads; it does not make other setup scripts
offline. An example CUDA system run is:

```bash
VLLM_GR_TEST_MODEL=/path/to/OneRec-1.7B \
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 USE_TF=0 \
python -m pytest --run-slow -q tests/st/test_onerec_v1_scheduler_st.py
```

For NPU end-to-end coverage, use
`tests/st/test_onerec_v1_npu_e2e.py` with `--run-slow`, the Ascend runtime,
xllm-ops, and `ASCEND_RT_VISIBLE_DEVICES`. Pangu tests use `PANGU1B_MODEL_PATH`
or `PANGU_MODEL_PATH`; see the [Pangu guide](../../examples/offline_inference/pangu1b/README.md).
Missing prerequisites may cause skips or failures depending on the test.

Use the [runner isolation probe](../modelrunner_v2_multibatch.md#reproduce-the-execution-regression)
for V1/V2 batch-switch and graph lifecycle changes. Launch runner/configuration
comparisons in fresh processes. Record actual replay/fallback counters and
clean shutdown, not only returned candidates.

### Local test runner

```bash
./tools/run_test.sh [--wheel] [--device-id N]
```

The script performs these steps:

1. Verifies that device-index parsing honors `CUDA_VISIBLE_DEVICES` /
   `ASCEND_RT_VISIBLE_DEVICES` (pure logic, no device).
2. Sources `tools/resource.sh` and selects a device:
    - `--device-id N` (comma-separated list accepted) uses those device ids
      directly and skips the resource auto-detection step, for a preallocated
      device.
    - Otherwise runs `tests/development/test_resource.sh` (allocation/locking
      probes) and calls `get_gpu` / `get_npu`. Auto-selection can wait for an idle
      device and is not an exclusive reservation for the suite.
    - Selection respects the runtime visibility variables: when
      `CUDA_VISIBLE_DEVICES` (or `ASCEND_RT_VISIBLE_DEVICES`) is set, only the
      listed indices are considered (see [Resource helper](#resource-helper)).
3. On CUDA, runs `pre-commit run --all-files` with the selected device, including
   the `run-tests` hook.
4. On NPU, probes/builds xllm-ops through `tools/install_xllm_ops.sh`, loads the
   custom operator environment, sets spawn-related variables, and directly runs
   `tools/pre_commit/run_tests.sh`. This branch does not run the other pre-commit
   hooks; run them separately when needed.

Both device paths reach the slow suite described in
[pre-commit scope](../PRE_COMMIT.md#choose-the-scope). The NPU installer may
clone/build dependencies when they are unavailable, so
prepare the vendor image and operator library before using this as a test gate.
This runner is not the full set of standalone CI benchmarks.

For a locally cached OneRec model:

```bash
VLLM_GR_TEST_MODEL=/path/to/OneRec-1.7B \
HF_HUB_OFFLINE=1 \
./tools/run_test.sh
```

Pass `--wheel` to append the wheel smoke test after successful test execution.
Use `--device-id` for a preallocated device; the runner otherwise picks an idle
device within the runtime-visible set.

### Packaging

```bash
VLLM_GR_TARGET_DEVICE=cpu ./tools/run_wheel_smoke_test.sh --python python3
```

Use `cuda` or `npu` for the other metadata targets. The script builds a wheel,
installs it with `--no-deps` into a temporary directory, checks import origin
and dependency metadata outside the source checkout, and runs packaging tests.
It does not validate accelerator inference. Its build helper installs build
tools and replaces `build/`, `dist/`, and generated package metadata; preserve
any build artifacts needed for comparison before running it.

### Benchmark acceptance

Use [benchmark setup](../benchmarks/README.md) and the relevant gate guide before
invoking `tests/system/run_benchmark_and_verify.sh`,
`tests/system/run_beam_v1_recall.sh`, or the Beam performance/A/B scripts.
These require additional datasets, catalogs, model paths, and policy settings.
Pangu acceptance has separate offline/online scripts.

Policies are resolved through `.github/benchmark-gate.yaml`, `tools/ci/`, and the
workflow; target branch and device matter. See the
[Video-100 policy](../benchmarks/parent_score_gate.md),
[Recall baseline](../benchmarks/beam_v1_recall_gate.md), and
[A/B utilities](../../tools/AB_Tests/README.md). Preserve the accepted thresholds
and input identity. Historical results only apply to their recorded revision
and configuration.

Keep functional, graph replay, accuracy, and performance evidence distinct.
For device results retain the code revision/import origin, runtime versions,
device, model/tokenizer/artifact identity, runner and graph settings, exact
command, and output location. Report skips and unavailable coverage explicitly.

## Resource helper

Source `tools/resource.sh` from the root. It delegates to `resource-npu.sh` when
`npu-smi` is present. Device selection does not reserve the returned device, and
it honors the runtime visibility variables: when `CUDA_VISIBLE_DEVICES` (or
`ASCEND_RT_VISIBLE_DEVICES` for NPU) is set, `get_gpu`/`get_gpus` (respectively
`get_npu`/`get_npus`) only consider the listed indices and never return a device
outside that set.

| Function | Arguments and result |
| --- | --- |
| `get_gpu` | `[memory_mb] [timeout_seconds]`; one available device ID |
| `get_gpus` | `[count] [memory_mb] [timeout_seconds]`; comma-separated IDs |
| `lock_gpu` / `lock_gpus` | Device ID(s); returns PID(s) of CUDA memory-holding processes or NPU sleep stubs |
| `unlock_gpu` / `unlock_gpus` | PID(s) returned by the matching lock call; stops those processes |

On CUDA, memory-holding locks provide an advisory signal to cooperating callers;
they are not an atomic scheduler reservation. On NPU, `lock_npu` and `lock_npus`
are non-reserving stubs that only start background `sleep` processes. They do not
allocate NPU memory, so `get_npu` and `get_npus` can select the same device again,
even when all callers use these helpers. Shared NPU runs require an external
scheduler or reservation mechanism that holds the device for the entire workload.
`ASCEND_RT_VISIBLE_DEVICES` controls visibility; it does not reserve a device.

Timeout 0 means no deadline. CUDA selection defaults to 16 MiB; single NPU
selection defaults to 3000 MiB. Supply explicit thresholds when sharing devices
across different environments. For example, select a CUDA device within 30 seconds:

```bash
source ./tools/resource.sh
device_id=$(get_gpu 18 30) || exit 1
CUDA_VISIBLE_DEVICES="$device_id" python -m pytest -q tests/kernels/test_rec_constrained_topk_cuda.py
```

Callers own the cleanup of processes returned by the lock helpers. Do not use
device-wide process cleanup to reclaim another developer's work. The CI cleanup
path validates CUDA device mapping before acting; consult
`tools/ci/check_cuda_device_mapping.py` and `tools/ci/gpu_process_guard.py` when
changing that behavior.

`scripts/kill_vllm_on_device.sh` sends SIGTERM to matching EngineCore processes
on the selected GPUs and then requires those GPUs to have no compute processes.
The name and device checks do not establish which job owns a process. Use it
only on devices allocated to the cleanup job, not on shared devices where it
could terminate another user's work.

## GitHub Actions tester

`tools/run_act.sh` invokes `act` with Docker's NVIDIA runtime, selects and locks
a GPU, maps the local cache, and supplies `HF_TOKEN` when it is already set in
the environment. It requires Docker, `act`, the relevant image, and workflow
inputs. Its current container setup is CUDA-specific; it is not an NPU runner.

```bash
./tools/run_act.sh
```

Local emulation does not prove GitHub event, label, artifact, or hosted-runner
behavior. Use the [CI workflow description](../../CONTRIBUTING.md#pull-request-ci)
when interpreting the result.

## Export branch

`tools/export_changes.sh` is a history-export utility, not a validation step.
With no arguments it checks out/resets `export-branch` from `upstream/main` and
applies the committed difference, excluding `trc`. With both remote branch and
commit message arguments, it also commits and force-pushes to `origin` before
returning to the original ref. It requires a clean tracked worktree and does
not include uncommitted work in the exported difference.

Use it only for an intended history export after reviewing the script and target
branch. Ordinary documentation or test work does not require this utility.
