# Beam V1 five-task Recall gate

The GPU CI gate for `main` and manual `workflow_dispatch` runs on
`decode_graph` use
[v1_recall_b4_baseline.json](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/beam_search/v1_recall_b4_baseline.json)
as its accepted accuracy reference. It measures OneRec-1.7B with ModelRunner V1,
Graph execution, batch size 4, beam width 128, temperature 0.6, and 1000 fixed
samples per task. Four tasks share the SID-to-PID engine; product uses a separate
SID-to-IID engine.

## Accepted baseline

The baseline JSON stores full-precision task floors, calibration measurements,
runtime and input identities, and source artifact provenance. Each SID floor is
the minimum of that task's accepted calibration measurements. The PID reference
remains diagnostic. Calibration is empirical, not a statistical confidence bound;
do not replace it with a current failing run merely to make CI pass.

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
replay validation remain blocking.

The A100 calibration does not establish identical absolute metrics on L20 or
NPU. The separate [Video-100 gate](parent_score_gate.md), Pangu offline/online
accuracy thresholds, and performance thresholds keep their existing policies.
The workflow reads both the Recall baseline and Video-100/Pangu policy from
the checked-out test revision; see the [CI policy guide](../../contributing/ci/benchmark-policy-approvals.md).
On `main`, NPU runs retain the Video-100 benchmark instead of this GPU-only gate.

## Local validation

```bash
python -m pytest -q tests/ci/test_v1_recall_summary.py
SKIP=run-tests pre-commit run --all-files
```

For an accelerator run with the fixed model snapshot and prepared data:

```bash
V1_RECALL_BASELINE_FILE="$PWD/benchmarks/beam_search/v1_recall_b4_baseline.json" \
  bash tests/system/run_beam_v1_recall.sh
```

The checked-in tests cover equality at the floor, screening and confirmation,
true regression failure, calibration counts, input identity, and graph replay.
