# Model-test input fixtures

OneRec pytest tests use session fixtures to prepare inputs on the executing runner.
Only selected tests download weights; collection and unrelated host tests do not.
Install the `dev` extra in the prepared CUDA/Ascend environment and request
the fixture from a test:

```python
def test_model_config(onerec_model):
    assert (onerec_model / "config.json").is_file()
```

Fixtures are registered in `tests/conftest.py`:

- `onerec_model` returns OneRec weights; `onerec_dataset` returns `benchmark_data`.
- `onerec_pid_constraints` and `onerec_iid_constraints` return directories with
  `catalog.json` and `triples.json`, using the existing converter/compiler tools.
  These full-mapping inputs do not replace the separate video task catalog and
  require the compiler's prepared runtime dependencies.

Fixtures return prepared paths. The cache keys include the pinned revision
or derived input identity. Use these environment variables for local inputs:

- `VLLM_GR_TEST_MODEL` for OneRec weights.
- `VLLM_GR_TEST_DATASET` for the snapshot root containing `benchmark_data/`.

Invalid explicit paths fail without fallback. Standard HF blob-cache snapshots
can be local overrides. Missing weights trigger preparation; download failures
are errors.

OneRec recipes key the cache on mapping bytes, converter/compiler source,
parameters, and builder version. Their output stops at tokenizer-independent
triples; a tokenizer-dependent Constraint Table must also include tokenizer
identity. Input setup runs before inference measurement.

```bash
python -m pytest -q --strict-test-levels tests/ci/test_artifact_preparation.py \
  tests/ci/test_derived_artifacts.py
```
