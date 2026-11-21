# Pytest resource gate

The L2 resource probe runs three separately reported scenarios: single-device
selection/locking, two-device selection/locking, and CUDA physical device 1
under a single-device visibility mask. It calls the production Bash allocator;
CUDA visible-ordinal translation has separate L1 coverage.

```bash
VLLM_GR_ISOLATED_RUNNER=true python3 -m pytest --strict-test-levels -v -rs \
  tests/development/test_resource_probe.py
```

Only an isolated runner may discover or lock devices. An unavailable device or
unsupported mask scenario is explicitly skipped, not reported as passed. The
masked scenario never expands an existing runner allocation. Each scenario has
a 90-second subprocess timeout; EXIT traps release recorded locks and the parent
terminates the owned process group on timeout, failure, or cancellation.

Set `VLLM_GR_RESOURCE_REPORT_DIR` to retain per-scenario logs outside pytest's
temporary directory. JUnit reporting uses pytest's standard `--junitxml` option.
Both `tests/development/test_resource.sh` and `tools/ci/run_resource_level.sh`
remain compatibility entrypoints. The former now also requires the explicit
isolated-runner opt-in. The latter retains its nonisolated-runner warning.

Validate the host contracts with `python -m pytest tests/ci/test_resource_pytest.py`.
Actual lock acquisition and POSIX process-group cleanup require a Linux runner;
host simulations do not establish device behavior.
