# Model-test input fixtures

Pangu pytest tests use session fixtures to prepare inputs on the executing runner.
Only selected tests download weights; collection and unrelated host tests do not.
Install the `dev` extra in the prepared CUDA/Ascend environment, then run:

```bash
python -m pytest --run-slow tests/st/test_pangu1b_accuracy_st.py \
  tests/st/test_pangu1b_online_st.py -m model_test
```

Fixtures are registered in `tests/conftest.py`:

- `pangu_model` returns validated Pangu weights; `gsm8k_mini` returns the bundled
  dataset with its reviewed canonical JSON identity.
- `onerec_model` returns OneRec weights; `onerec_dataset` returns `benchmark_data`.
- `onerec_pid_constraints` and `onerec_iid_constraints` return directories with
  `catalog.json` and `triples.json`, using the existing converter/compiler tools.
  These full-mapping inputs do not replace the separate video task catalog and
  require the compiler's prepared runtime dependencies.

Model/data fixtures record file hashes and source identity in JUnit suite
properties. Local overrides are labeled `local`, not claimed to match a Hub SHA.
Explicit overrides remain supported:

- `PANGU1B_MODEL_PATH`, then `PANGU_MODEL_PATH`, for Pangu weights.
- `PANGU1B_DATASET_PATH` for a custom GSM8K-format question/answer JSON list.
- `VLLM_GR_TEST_MODEL` for OneRec weights.
- `VLLM_GR_TEST_DATASET` for the snapshot root containing `benchmark_data/`.

Invalid explicit paths fail without fallback. Standard HF blob-cache snapshots
can be local overrides. Missing weights trigger preparation; download failures
are errors. The existing runtime guard remains: missing vLLM skips local Pangu
runs, while `PANGU1B_REQUIRE_RUNTIME=1` makes this an error in strict gates.
Legacy shell entrypoints and workflow download/cache steps remain supported.

OneRec recipes key the cache on mapping bytes, converter/compiler source,
parameters, and builder version. Their output stops at tokenizer-independent
triples; a tokenizer-dependent Constraint Table must also include tokenizer
identity. The online fixture owns server startup/cleanup; setup time is outside
inference measurement. Fixture tests simulate model execution, not accuracy.

```bash
python -m pytest -q --strict-test-levels tests/ci/test_artifact_preparation.py \
  tests/ci/test_derived_artifacts.py tests/ci/test_artifact_fixtures.py
```
