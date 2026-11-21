# Functional pytest adapters

This change depends on the model-input fixtures and resource process helper.
Apply those changes before collecting these tests.

Pangu's legacy offline and online shell gates now select the existing model
pytest cases. They still resolve and enforce their original accuracy policy;
`PANGU1B_MEASURE_ONLY=1` requests validated measurements without applying a
second threshold. Invalid/nonfinite metrics and missing results still fail.
Offline prompts follow `PANGU1B_NUM_PROMPTS` (default 8), with a 1800-second
subprocess deadline configurable through `PANGU1B_OFFLINE_TIMEOUT`.
The online class retains its five health/API/accuracy checks and fixture-owned
server cleanup. Model fixtures prepare inputs on the executing runner.

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
