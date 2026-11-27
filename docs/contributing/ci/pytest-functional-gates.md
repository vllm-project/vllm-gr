# Functional pytest adapters

The adapter uses the resource gate's shared process helper for cleanup.

The existing Beam driver CLIs remain unchanged. Pytest adds seven separately
reported CUDA cases for checkpoint reload, trie reload, pause/drain and reload
memory, including the existing graph variants. Every driver keeps its own
assertions, tiny-model preparation and result format. A successful child exit
also requires a fresh `result.json` with boolean `passed: true`.

```bash
VLLM_GR_FUNCTIONAL_GATE=1 python -m pytest --run-slow --strict-test-levels \
  tests/e2e/test_beam_functional_gate.py
```

Pytest's temporary directory contains driver logs, generated models and the
driver's result files. Each driver has a one-hour
deadline and owns a process group for cancellation cleanup. CUDA-specific
drivers are not advertised as NPU coverage. Without explicit opt-in these
expensive tests skip. The unified entrypoint enables them explicitly.

Adapter tests use simulated results and do not establish real accuracy,
reload behavior, graph replay, device memory behavior or performance parity.
