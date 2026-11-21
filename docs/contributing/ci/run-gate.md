# Unified pytest gate entrypoint

```bash
python -m tools.ci.run_gate --suite functional --hardware cuda:H100 \
  --output gate-results/run-001
```

Repeat `--suite` to run several suites in one command. Initially `unit` selects
L1/L2 tests, `resource` runs isolated allocator scenarios, and `functional` runs
the Beam drivers. Hardware selection uses the shared hardware policy; it does
not make CUDA-only drivers eligible for Ascend. Model preparation remains lazy.
Use an empty output directory for every invocation. The runner refuses reuse.

Each suite writes `pytest.log` and `junit.xml`. The top-level
`summary.json` records the checkout commit, target, unique run ID, requested
suites and their outcomes. A report is successful only if all collected tests
completed setup, call and teardown successfully. Unexpected skips, xfails,
unexpected passes, empty selections, timeouts and cancellation cannot pass.
The default suite deadline is four hours (`--timeout` seconds); the child pytest
timeout is disabled so the enclosing process-group deadline controls cleanup.

`resource` still requires `VLLM_GR_ISOLATED_RUNNER=true`. General-purpose test
suites may contain intentional skips; such runs are diagnostic failures until
their selected tests are eligible. Do not weaken report validation to hide them.
Use `--cards N` to require exactly N visible devices and select the matching unit or resource cases.

This entrypoint depends on the hardware selection and functional gate changes.
It does not yet replace existing workflow orchestration. Benchmark adapters
are added separately to preserve their policy, provenance and measurements.
