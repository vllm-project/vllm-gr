# Functional pytest adapters

This change depends on the model-input fixtures and resource process helper.
Apply those changes before collecting these tests.

The existing Beam driver CLIs remain unchanged. Pytest adds seven separately
reported CUDA cases for checkpoint reload, trie reload, pause/drain and reload
memory, including the existing graph variants. Every driver keeps its own
assertions, tiny-model preparation and result format. A successful child exit
also requires a fresh `result.json` with boolean `passed: true`.

```bash
VLLM_GR_FUNCTIONAL_GATE=1 python -m pytest --run-slow --strict-test-levels \
  tests/st/test_beam_functional_gate.py
```

Set `VLLM_GR_FUNCTIONAL_REPORT_DIR` to an empty output directory to retain
driver logs, generated models and JSON reports. Each driver has a one-hour
deadline and owns a process group for cancellation cleanup. CUDA-specific
drivers are not advertised as NPU coverage. Without explicit opt-in these
expensive tests skip; the future strict gate runner must reject such skips.

Adapter tests use simulated results and do not establish real accuracy,
reload behavior, graph replay, device memory behavior or performance parity.
