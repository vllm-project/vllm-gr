# OneRec Precision Test Scripts — Overview & Usage

This document describes the OneRec precision-testing files and scripts used to
gate Beam V1 recommendation accuracy in this repository. It is written in a
revision-agnostic way; it covers both the per-task runners and the two
top-level harnesses.

At a glance: run the five-task OneRec canary through the Beam V1 ingress paths,
then compare the results against a versioned "constrained golden" baseline.
Blocking metrics must not regress one-sidedly against the baseline, and the run
configuration must match the golden's minimal configuration contract.

## Data flow

```text
+-----------------------------------------------------------------+
| tools/prepare_decode_data.py     benchmark_data + sid2pid/iid.json |
|   produces: prepared_dir/manifest.json + per-domain catalog/       |
|             triples/(optional) constraint-table safetensors        |
+-----------------------------------------------------------------+
        | manifest.json shared by runner / online serve
        v
+-----------------------------------------------------------------+
| run_recommendation_regression.py   per task/domain (--tasks), resumable |
|   constraint mode trie | triples | artifact; execution offline|online|
|   produces: decode/<task>.jsonl + <task>.summary.json              |
+-----------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------------+
| gate_constrained_golden.py   --golden + --candidate-dir            |
|   blocking gate (versioned golden)                                 |
+-------------------------------------------------------------------+
        ^
        +-------- orchestrated by run_onerec_precision_diff.sh ------+
```

CI (`.github/workflows/pre-commit.yml`) invokes
`run_onerec_precision_diff.sh` (offline, five-task canary); a
constrained-golden policy miss hard-fails the job, and diagnostics are uploaded
as an artifact.

---

## 1. Data preparation

### `benchmarks/open_one_rec/tools/prepare_decode_data.py`

Materializes the OpenOneRec `sid2pid.json` / `sid2iid.json` logical universes
into the three forms decode_graph consumes, and verifies all three describe the
same SID set:

- `catalog.json` — Trie catalog (`tools/convert_catalog.py process_json`)
- `constraint_triples.json` — triples document (`tools/build_constraint_triples.py
  process_catalog`)
- `constraint_table.safetensors` (with `--build-artifacts`) — precompiled
  schema-v1 constraint-table artifact, avoids runtime compilation; bound to the
  tokenizer digest / vocab size and rebuilt automatically when the tokenizer
  changes.

Two domains: `video` (mapping `sid2pid.json`; serves tasks video/ad/label_cond/
interactive) and `product` (mapping `sid2iid.json`; serves product). Everything
is recorded in `manifest.json` (`tasks -> domain` plus per-domain paths, sha256s,
`logical_universe_sha256`, and artifact metadata).

Usage:

```bash
python -m benchmarks.open_one_rec.tools.prepare_decode_data \
  --dataset-path <OpenOneRec benchmark_data root> \
  --output-dir <prepared_dir> \
  --model <model path or ID> \
  --build-artifacts \
  --artifact-backend cuda   # cuda | npu | cpu
  [--revision <sha>] [--force]
```

Note: when the local model directory exists, the manifest logic re-validates the
current tokenizer digest against the recorded artifact; `--force` rebuilds
everything; the `.safetensors` files are gitignored.

---

## 2. Core regression logic (no vLLM dependency, CPU-testable)

### `benchmarks/open_one_rec/regression.py`

Pure functions plus a streaming accumulator, shared by the runners and the
comparator:

- `RECOMMENDATION_TASKS` = `video, ad, product, label_cond, interactive`;
  `TASK_SIZES` / `TASK_MAPPING_FILES` (product uses `sid2iid.json`, the rest
  `sid2pid.json`).
- `sha256_file / prompt_digest` — file/prompt fingerprints.
- `load_sample_manifest` — validates a versioned sample list (row-range check,
  duplicate check, optional parquet `source_sha256` pinning).
- `validate_generations` — every beam text must match
  `<|sid_begin|><s_a_N><s_b_N><s_c_N><|sid_end|>`; reports beam_count /
  empty / invalid_sid / duplicate counts.
- `RecommendationMetricAccumulator` — streaming pass@k / position1_pass@k /
  recall@k (k=1,32) for both SID and PID (via `utils_by_pid` mapping);
  `evaluation_mode` in sid / pid / both.
- `compare_jsonl` — per-sample comparison of two result files (top1 / top-k
  ranking, top-k set, per-sample metrics, prompt fingerprint, aggregate metrics).
- `validate_golden_identity` — a reused golden must match the candidate on
  model / revision / source_sha256 / beam_width / temperature / evaluation_mode;
  execution / implementation_revision / execution_variant / constraint /
  batch_size / seed are intentionally ignored (they are the subject of the
  paired differential).
- `comparison_passed` — `exact` (top1 + per-sample metrics identical) or
  `no-regression` (aggregate metrics must not drop beyond absolute/relative
  tolerance).
- `write_jsonl / iter_jsonl / _index_records` — result-file IO.

### `benchmarks/open_one_rec/tasks/recommendation/utils_by_pid.py`

PID extraction and metric helpers. The notable change is that
`apply_sid_to_pid_strategy("most_popular_after_downsampling")` now resolves ties
deterministically in mapping-file order, so repeated accuracy evaluations are
stable across runs and branches (required for golden reproducibility).

### `benchmarks/open_one_rec/open_one_rec_loader.py`

Data loader imported from OpenOneRec. Notable capabilities:

- `load_data(..., sample_ids=...)`: load by explicit row IDs — order-preserving
  and never reads/writes the random-sample cache. This is the reproducibility
  contract used by CI manifests; `sample_ids` and `sample_size` are mutually
  exclusive.
- `_make_metadata_serializable`: drops `answer` from per-sample metadata (it is
  promoted to top-level `ground_truth`) and injects `row_index`.
- `custom_input_len`: trims/pads prompts to an exact token count.

---

## 3. Runner

### `benchmarks/open_one_rec/run_recommendation_regression.py`

Per-task, resumable runner with offline and online ingress. Highlights:

- `_resolve_constraint`: resolves the three constraint modes from the prepared
  manifest; `artifact` mode produces the canonical Beam V1 config
  (`vllm_gr_config` with `execution_mode=v1`, graph, worker_decision, and the
  constraint-table artifact).
- `_engine_is_beam_v1`: inspects the engine's `vllm_config` to decide between
  `llm.beam_search_v1` (one session per call, `ignore_eos=True`,
  `<|sid_begin|>/<|sid_end|>`) and the legacy `llm.beam_search`.
- The offline collector decodes each beam from its last five token IDs to
  produce a standalone SID text (independent of `add_special_tokens` policy and
  of itemic tokens already present in the user history).
- Online goes through `/v1/completions` and sends the already-rendered chat
  transcript verbatim as `prompt` (avoids applying the chat template twice);
  SID sentinels are passed via `vllm_xargs.begin_token/end_token`.
- Supports `--resume` (validates run_identity/prompt fingerprints before
  appending), `--strict` (fails on any contract-violating sample), and
  `--max-model-len` (escape hatch for long OneRec prompts beyond the model's
  default 8192 context).
- Records `max_model_len` plus a `VLLM_*` config-env snapshot (`config_env`) in
  `run_identity` and the summary, so a run/golden can be traced back to its
  capture configuration (provenance; the gate only hard-fails on the core
  measurement fields).

Usage:

```bash
python -m benchmarks.open_one_rec.run_recommendation_regression \
  --task video --model <model> --dataset-path <data root> \
  --prepared-manifest <prepared_dir>/manifest.json \
  --output-dir <out> \
  --constraint-mode artifact --execution offline \
  --beam-width 32 --temperature 0 --strict
# online: --execution online --host 127.0.0.1 --port 7012 --endpoint /v1/completions
# one offline process per constraint domain (reuses the engine across tasks):
#   --tasks video,ad,label_cond,interactive
```

Outputs: `<task>.jsonl` (one record per sample: generations, token IDs,
cumulative logprobs, `run_identity`, `contract`, `metrics`) plus
`<task>.summary.json` (aggregate metrics and protocol fingerprint).

## 4. Comparison / gate

### `benchmarks/open_one_rec/compare_recommendation_regression.py`

Per-sample comparison of a baseline and a candidate `<task>.jsonl` (diagnostic).
First `validate_golden_identity` (identity mismatch → exit code 2), then
`compare_jsonl` and a policy verdict:

- `exact`: top1 and per-sample metrics must be identical;
- `no-regression`: aggregate metrics (repeat `--metric`) must not drop beyond
  `--absolute-tolerance` / `--relative-tolerance`.

Usage:

```bash
python -m benchmarks.open_one_rec.compare_recommendation_regression \
  --baseline <golden_dir>/<task>.jsonl \
  --candidate <decode_dir>/<task>.jsonl \
  --output <task>.diff.json \
  --policy no-regression \
  --metric pass@1 --metric pass@32 --metric recall@32 ... \
  [--allow-differences] [--absolute-tolerance 0.005] [--relative-tolerance 0.1]
```

### `benchmarks/open_one_rec/gate_constrained_golden.py`

The blocking gate itself.

- `CAPTURE_POLICY` — the unified five-task gate policy. Blocking metrics are
  `pass@1`/`pass@32`, `recall@32`, `pid_pass@1`/`pid_pass@32`, `pid_recall@32`;
  the `position1_*` / `recall@1` metrics are diagnostic (reported, not
  blocking). The aggregate tolerance band
  (`comparison=candidate_greater_than_or_equal`, global
  `blocking_absolute_tolerance` 0.02, per-metric `pass@1`/`pid_pass@1` 0.03) is
  only a fallback for goldens captured without per-sample metrics; the
  per-sample paired tests below are the primary verdict.
- `capture_constrained_golden(summaries)` — builds a golden from one clean run's
  summaries; rejects mixed protocol fields and nonzero `invalid_samples`. It
  records the full run configuration (including `max_model_len`, the `VLLM_*`
  `config_env` snapshot and `implementation_revision`) and, when per-sample
  metrics are supplied, freezes them so the gate can run the paired tests.
- `evaluate_constrained_golden` — compares candidate summaries against the
  golden:
    - **Configuration contract** (hard failure, regardless of metrics): the
    minimal set of fields that define *what* the run measures must match —
    `sample_manifest_sha256`, `beam_width`, `temperature` (protocol) plus the
    per-task `source_sha256` and `constraint` domain.
    - **Recording only** (provenance, not gated): the remaining runtime
    parameters — `max_model_len`, the `VLLM_*` `config_env`, `execution` /
    `execution_variant`, `seed`, and code/model revisions — are recorded so one
    golden stays reusable across a wide parameter range and across the code
    under test.
    - **Metric verdict**: when the golden and candidate both carry per-sample
    metrics, the primary comparison is a one-sided paired sign test per metric
    *family* (`pass` / `position1_pass` / `recall` and their `pid_` variants).
    Each sample is reduced to a family scalar — binary families to the
    first-hit depth (the count of satisfied `metric@k`), continuous families to
    the sum across `k` — and the candidate's regression count is tested against
    the no-change null. Every family is judged against the same per-family
    threshold (`per_sample.fdr_q`, 0.05). No multiple-testing correction is
    applied: the six families come from the same samples and the `pid_`
    variants re-score the SID ones, so they are strongly dependent and a
    Benjamini-Hochberg step would only divide the threshold by the family
    count. Pairing on the same sample ids cancels benign, bidirectional
    configuration noise while a one-sided regression stands out; family-level
    aggregation keeps a regression that only moves the middle of the `k` range
    visible instead of testing every nested `metric@k`. The report carries a
    per-case three-way label (`better`/`worse`/`same`) for every sample and
    family. Goldens without per-sample metrics fall back to the aggregate
    tolerance band; a candidate that omits per-sample data while the golden
    froze it is an error.
  The report's `passed` field is the CI gate result.

Usage (two modes):

```bash
# gate mode (CI)
python -m benchmarks.open_one_rec.gate_constrained_golden \
  --golden <golden>.json \
  --candidate-dir <decode output dir> \
  --output <report>.json \
  --evaluation-mode both --task video --task product ...

# capture mode (recapturing a golden)
python -m benchmarks.open_one_rec.gate_constrained_golden \
  --capture \
  --candidate-dir <decode output dir> \
  --output <new_golden>.json \
  --capture-description "..." --capture-model OpenOneRec/OneRec-1.7B
```

`--candidate-dir` must contain `<task>.summary.json` per task.

---

## 5. Manifests (data + goldens)

### `benchmarks/open_one_rec/manifests/ci_precision_v1.json`

Critical ("easy-to-flip") canary list (version 1): 100 rows for each of
the five tasks (500 total). Each row flips
at least one metric family between the four real execution paths
(`v1` / `nograph` / `batch` / `batch+nograph`); rows are drawn
stratified-random within each flip-count level (seeded) so no single `sid`
range is favoured. Per-task `source_sha256` pins the parquet. All reproducible
runs and the golden's protocol `sample_manifest_sha256` reference it.

### `benchmarks/open_one_rec/manifests/ci_precision_v1_constrained_golden_offline.json`

Offline baseline (`execution=offline`, `execution_variant=offline-v1-graph`); it
freezes the per-sample metrics the paired tests consume. Both ingress paths
execute the same canonical engine and match on every blocking metric, so an
online baseline (`execution=online`, `execution_variant=graph-worker`) captured
from the same manifest can gate the other ingress too.

---

## 6. Harnesses (shell, top-level entry points)

### `tests/system/run_onerec_precision_diff.sh`

Main flow: prepare (auto unless `--skip-prepare`) → run decode per domain
(video / product) in offline or online mode → constrained-golden gate
(policy miss exits 1).

- `--decode-execution offline|online` (default offline):
    - offline: the runner boots the canonical Beam V1 engine itself
    (execution_mode=v1 + startup graph) and drives `llm.beam_search_v1` — the
    maintained GPU path.
    - online: the script derives a canonical GR config YAML from the prepared
    manifest and serves it with `vllm-gr serve --vllm-gr-config`; the runner
    goes over HTTP. Loopback proxy exclusions are set automatically.
- Without an explicit `--constrained-golden`, the matching offline/online
  golden is picked from `--decode-execution`, so a bare `--decode-execution
  online` run cannot gate against — or, on capture, overwrite — the offline
  golden.
- `--capture-golden`: recaptures the golden from this run's decode results
  (for engine/protocol changes).
- `PYTHON` may be overridden to point at a venv with the repo-pinned vLLM build.

Usage:

```bash
./tests/system/run_onerec_precision_diff.sh \
  --model OpenOneRec/OneRec-1.7B \
  --revision cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b \
  --dataset-path <data root> \
  --prepared-dir <prepared_dir> \
  --output-dir <out> \
  --decode-execution offline \
  --beam-width 32 --temperature 0 --overwrite
# recapture: add --capture-golden --constrained-golden <new golden path>
```

Output layout: `<out>/decode/` (decode results),
`<out>/diffs/constrained-golden.json` (gate report), `<out>/logs/`.

### `tests/system/run_onerec_full_regression.sh`

Runs the full-data regression without golden generation/gating: prepare → per
task `run_recommendation_regression` (offline/online, trie/triples/artifact).
With `--baseline-dir`, each task is additionally diffed via
`compare_recommendation_regression` (exact/no-regression). Suited to
development-time validation, not the CI gate.

Usage:

```bash
./tests/system/run_onerec_full_regression.sh \
  --model <model> --dataset-path <data root> \
  --prepared-dir <prepared_dir> --output-dir <out> \
  --constraint-mode artifact --execution offline \
  --sample-manifest manifests/ci_precision_v1.json \
  [--baseline-dir <golden_dir> --comparison-policy no-regression]
```

---

## 7. Accuracy test script

### `benchmarks/open_one_rec/one_rec_acc_test.py`

Lightweight offline/online accuracy test: with `--endpoint` it drives the HTTP
API, otherwise a local `GRLLM.beam_search`; supports `--gr`, `--n` (beam width),
`--catalog-path`/`--constraint-backend`, and `--beam-max-width`. Notable
behavior: the Completion API receives the already-rendered chat text verbatim as
`prompt`, while the Chat API receives structured messages; `run_api_inference`
validates the returned choice count; offline outputs are normalized into
`MockRequestOutput`; `_validate_recommendation_outputs` enforces the
standalone-SID output contract (any missing beam, empty, or invalid SID raises).
Main use: single-task debugging; covered by `tests/test_onerec_acc.py`.

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model <model> --dataset-path <data root> --task-type video \
  --num-prompts 100 --gr --use-beam-search --n 32 --constraint-backend constraint_table \
  --catalog-path <prepared>/video/constraint_triples.json
```

---

## 8. CI changes

### `.github/workflows/pre-commit.yml`

Runs the official-protocol five-task precision differential:
`run_onerec_precision_diff.sh` offline against
`ci_precision_v1_constrained_golden_offline.json`. A constrained-golden policy
miss fails the job. Results are uploaded as the `onerec-precision-<run_id>`
artifact (14-day retention).

### `.gitignore`

Adds `.sentry-native/`, `shellcheck-v0.11.0/`, `OpenOneRec/`, `generated_data/`
(local tooling and CI byproducts).

---

## 9. Unit tests

Three CPU-only test files (no GPU needed):

- `tests/test_onerec_acc.py` — Completion/Chat API payload and response
  parsing, transport failures, output-contract validation.
- `tests/test_onerec_regression.py` — `_online_batch` / `_offline_batch`
  (a V1 engine must be driven through `beam_search_v1` with `ignore_eos`),
  sample-manifest validation, manifest/parquet fingerprints, PID tie
  determinism, streaming metrics, compare/identity/policy, and prepare-data
  build/consistency.
- `tests/test_onerec_constrained_golden.py` — gate tolerances, metric filtering,
  runtime-identity ignoring, capture contracts (protocol consistency,
  `invalid_samples`), and the configuration contract (the core measurement
  fields are hard-gated while runtime parameters are recorded but not gated).

Run from a repo venv:

```bash
PYTHONPATH=<repo> python -m pytest \
  tests/test_onerec_acc.py tests/test_onerec_regression.py tests/test_onerec_constrained_golden.py
```

---

## Quick debug commands

```bash
# 1) Prepare data only (validate/reuse artifacts)
python -m benchmarks.open_one_rec.tools.prepare_decode_data \
  --dataset-path ~/<OpenOneRec>/benchmark_data --output-dir <prepared> \
  --model <model> --build-artifacts

# 2) Run one task's decode and emit a summary
python -m benchmarks.open_one_rec.run_recommendation_regression \
  --task video --model <model> --dataset-path <data root> \
  --prepared-manifest <prepared>/manifest.json --output-dir <out> \
  --constraint-mode artifact --execution offline --sample-manifest manifests/ci_precision_v1.json \
  --strict --overwrite

# 3) Gate-only comparison (no re-inference)
python -m benchmarks.open_one_rec.gate_constrained_golden \
  --golden manifests/ci_precision_v1_constrained_golden_offline.json \
  --candidate-dir <out>/decode --evaluation-mode both --task video --task product

# 4) End to end (CI-equivalent; add --decode-execution online for the HTTP path)
bash tests/system/run_onerec_precision_diff.sh \
  --model OpenOneRec/OneRec-1.7B --revision <sha> \
  --dataset-path <data root> --prepared-dir <prepared> --output-dir <out> \
  --sample-manifest manifests/ci_precision_v1.json \
  --beam-width 32 --temperature 0 --overwrite
```
