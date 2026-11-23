# Running tests locally

Select the test level and device for your change before running the suite locally.

## Prerequisites

Prepare the development environment for [CUDA](../../getting_started/installation/gpu.md)
or [Ascend](../../getting_started/installation/npu.md). Identify the devices
assigned to your job.

## Run the test suite

Use this command from the repository root to select device 0.

```bash
./tools/run_test.sh --device-id 0
```

For multiple assigned devices, pass a comma-separated list such as
`--device-id 0,2`. Add `--wheel` to run the wheel smoke test after the checks
succeed. For OneRec tests that use a local checkpoint, set
`VLLM_GR_TEST_MODEL` to the snapshot directory. See the
[OneRec CUDA example](../../user_guide/onerec_quickstart.md#5-validate-and-troubleshoot)
or the [Ascend system tests](../../../tests/st/test_onerec_v1_npu_e2e.py).

## CUDA and Ascend checks

| Platform | What the runner executes |
| --- | --- |
| CUDA | Runs `pre-commit run --all-files` on the selected devices, including the test hook. |
| Ascend NPU | Checks the vendor runtime, probes and builds xllm-ops as needed, loads the custom operators, then runs the test hook directly. Run other [pre-commit checks](../pre_commit_setup.md#choose-the-scope) separately when needed. |

### Ascend test runtime

For local NPU validation, use the [Ascend vendor image](../../getting_started/installation/npu.md#prerequisites)
and build xllm-ops for its CANN version. If the job activates Conda, check
that `CONDA_PREFIX`, `PATH` and the CANN environment select that runtime.
An environment built for vLLM 0.22 needs the current vendor stack before
it can run the tests for vLLM 0.30.0.

From the repository root, check the selected Python against the versions
declared by the checkout before installing the GR developer extras. The
resolver reads package metadata without allocating an NPU.

```bash
python3 tools/ci/resolve_runtime.py resolve --device npu \
  --output /tmp/vllm-gr-npu-runtime.json
VLLM_GR_TARGET_DEVICE=npu python3 -m pip install -e ".[dev]" --no-build-isolation
```

Run the suite on your assigned devices as described above. The runner also
checks the vendor runtime before its resource tests and operator setup.
The xllm-ops installer checks the Python bindings for the required operators,
including `rec_constrained_topk`, before and after a build. The runner loads
the vendor's `bin/set_env.bash` when present. The fallback requires a single
`set_env.bash` elsewhere in the vendor directory.

## Select devices and test levels

On shared machines, pass the IDs allocated to your job. The runner uses
`--device-id` to set `CUDA_VISIBLE_DEVICES` or `ASCEND_RT_VISIBLE_DEVICES`
for the tests, replacing any inherited value. Without that option, the runner
first runs resource probes, then waits for an idle device from the visible
set. A probe on multi-GPU hosts can temporarily select GPU 1. Automatic
selection gives no exclusive reservation for the suite.

The [optional resource helpers](../../../tools/devices/resource.sh) are for
single-host development. Use the scheduler or reservation mechanism to
allocate devices on shared hosts. Callers own cleanup of processes started
by the lock helpers.

An unset visibility mask allows all enumerated devices, and an empty mask
exposes none. Malformed masks stop selection. The helpers intersect valid
masks with the available devices and skip devices whose telemetry is missing
or invalid. CUDA locks verify the physical device UUID. Ascend locks use
sleep processes as advisory markers.

By default the test hook runs the full pytest suite with `--run-slow`. Set
`VLLM_GR_TEST_LEVEL=pr` to select L1 and L2. For a CUDA level run, set
`VLLM_GR_TARGET_DEVICE=cuda` to select CPU and CUDA tests within those levels.
The Ascend runner sets its target automatically. See
[running by level](test_levels.md#running-by-level) for other selections.

## CUDA process inspection and cleanup

Run the read-only process check on GPUs allocated to your job. From the
repository root, replace `0` with the numeric device index assigned to you.

```bash
(
  set -e
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
  CI_CUDA_MAPPING_FILE=$(mktemp)
  export CI_CUDA_MAPPING_FILE
  trap 'rm -f -- "$CI_CUDA_MAPPING_FILE"' EXIT
  python tools/ci/check_cuda_device_mapping.py --output "$CI_CUDA_MAPPING_FILE"
  python tools/ci/gpu_process_guard.py check
)
```

The [cleanup wrapper](../../../scripts/kill_vllm_on_device.sh) sends SIGTERM
to matching EngineCore processes and fails if compute processes remain.
Confirm job ownership and replace the final command with
`bash scripts/kill_vllm_on_device.sh --yes` only when you intend to clean those
devices. The wrapper requires `--yes` before querying devices or signaling processes.
