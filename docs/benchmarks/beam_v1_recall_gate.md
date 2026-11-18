# Beam V1 five-task Recall gate

The GPU CI gate for `main` and `decode_graph` uses
[v1_recall_b4_baseline.json](../../benchmarks/beam_search/v1_recall_b4_baseline.json)
as its accepted accuracy reference. It measures OneRec-1.7B with ModelRunner V1,
Graph execution, batch size 4, beam width 128, temperature 0.6, and 1000 fixed
samples per task. Four tasks share the SID-to-PID engine; product uses a separate
SID-to-IID engine.

## Accepted vLLM 0.30.0 calibration

On 2026-11-02, the upgrade owner accepted the upgraded version's accuracy after
validation and authorized replacing the pre-upgrade reference. The new baseline
uses three actual A100 measurements recorded in the checked-in accuracy
reference linked above. That JSON contains the calibration values and source
metadata, so readers do not need access to historical CI artifacts to inspect
the accepted reference.
Each SID floor remains the minimum of that task's calibration measurements;
the stored JSON preserves their full precision.

| Task | Previous SID Recall@32 floor | Accepted v0.30.0 floor |
| --- | ---: | ---: |
| video | 3.725000% | 3.752103% |
| ad | 7.046865% | 7.055754% |
| product | 6.653889% | 6.640278% |
| label_cond | 2.443016% | 2.414683% |
| interactive | 20.846230% | 21.032897% |

The calibration used an A100 PCIe 40 GB, CUDA 13.0, PyTorch 2.13.0+cu130,
vLLM 0.30.0 and Python 3.11.12. The
[baseline JSON](../../benchmarks/beam_search/v1_recall_b4_baseline.json)
records the exact model snapshot, tested commit, CI artifact and checksum.

The original run failed the previous `label_cond` floor. That historical result
is retained; the accepted measurements now define the reference for future
regressions. The three rounds contain cold/warm product variation and are all
retained. They are an empirical calibration, not a statistical confidence bound.
The PID reference is the first calibration measurement and remains diagnostic.

## Blocking behavior and input identity

CI starts with one measurement of all five tasks. If a task falls below its floor,
two fresh confirmation runs are added. The gate fails only when the same task is
below its floor in all three runs; equality passes. A normal passing run still
runs once. Calibration count does not set the number of regular CI runs.

The summarizer reads `source_ci.calibration_repeats` and requires an integer of
at least three, with exactly that many finite SID measurements for every task.
Legacy baselines without this field retain their ten-measurement contract.
Do not duplicate a measurement to fill a calibration list.

Sample IDs, dataset hashes, model revision, constraint artifact and tokenizer
identities, graph profile, and engine policy remain checked. Capture and B4
replay validation remain blocking. The rebaseline changes no inference code,
evaluation formula, dataset, graph execution requirement, or failure mode.

The A100 calibration does not establish identical absolute metrics on L20 or
NPU. The separate [Video-100 gate](parent_score_gate.md), Pangu offline/online
accuracy thresholds, and performance thresholds keep their existing policies.
The GPU workflow reads this Recall baseline from its checked-out candidate; the
Video-100/Pangu YAML policy and resolver come from the same tested commit,
as described in the [CI policy guide](../ci/benchmark-policy-approvals.md).
On `main` and `decode_graph`, NPU runs use the Video-100 benchmark.

## Local validation

```bash
python -m pytest -q tests/test_v1_recall_summary.py
SKIP=run-tests pre-commit run --all-files
```

For an accelerator run with the fixed model snapshot and prepared data:

```bash
V1_RECALL_BASELINE_FILE="$PWD/benchmarks/beam_search/v1_recall_b4_baseline.json" \
  bash tests/system/run_beam_v1_recall.sh
```

The checked-in tests cover equality at the floor, screening and confirmation,
true regression failure, calibration counts, input identity, and graph replay.
