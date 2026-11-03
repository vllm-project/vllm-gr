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

That command requires pytest, setuptools, and packaging, but does not establish
inference support. Check the [coverage map](../development.md#select-regression-coverage)
for the tests associated with other code areas.

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

### Local full test runner

```bash
./tools/run_test.sh
```

The current script performs these steps:

1. Runs `tests/development/test_resource.sh`, which probes allocation/locking.
2. Sources `tools/resource.sh` and selects an available accelerator. Detection
   uses the presence of `npu-smi`; otherwise it follows the CUDA path. Selection
   can wait for an idle device and is not an exclusive reservation for the suite.
3. On CUDA, runs `pre-commit run --all-files` with the selected device, including
   the `run-tests` hook.
4. On NPU, probes/builds xllm-ops through `tools/install_xllm_ops.sh`, loads the
   custom operator environment, sets spawn-related variables, and directly runs
   `tools/pre_commit/run_tests.sh`. This branch does not run the other pre-commit
   hooks; run them separately when needed.

Both paths reach the full slow suite described in
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
The runner chooses its own device; for a specific preallocated device, use the
targeted commands or direct test script with the corresponding visibility
variable and prepared environment.

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
`npu-smi` is present. Device selection does not reserve the returned device.

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
