# OneRec Accuracy Benchmark

## Scope

This guide covers the five OpenOneRec recommendation tasks: `video`, `ad`,
`product`, `label_cond`, and `interactive`. Throughput-only experiments belong
in the [throughput guide](throughput.md); do not use prompt padding or truncation
when reporting accuracy.

The benchmark implementation is based on the OpenOneRec code imported at
commit `ae668a6a30bd71bde7a3f459fe0ac958182ce1d2`. For reproducible comparisons,
pin the model revision, dataset files, sample manifest, and prepared constraint
manifest as well.

## Official-compatible protocol

Use the following settings for accuracy baselines and constrained candidates:

| Setting | Value |
| --- | --- |
| Model | `OpenOneRec/OneRec-1.7B` at a pinned revision |
| Chat template | `qwen3_soft_switch.jinja2`, thinking disabled |
| Beam width / returned sequences | 32 |
| Temperature | 0 |
| SID protocol | append <code>&lt;&vert;sid_begin&vert;&gt;</code>, decode 3 semantic tokens, then <code>&lt;&vert;sid_end&vert;&gt;</code> |
| Evaluation cutoffs | 1 and 32, using the first-ranked predictions |
| Metrics | `pass@k`, `position1_pass@k`, and `recall@k` |
| Recommendation tasks | `video`, `ad`, `product`, `label_cond`, `interactive` |

The generic imported recommendation task dictionary contains
`num_beams=128`, `temperature=0.6`, `top_p=0.95`, and `top_k=50`. Those are task
generation defaults, not the official paired accuracy oracle used here. Do not
silently inherit them for a reported accuracy comparison; pass BW32/T0
explicitly.

The application-level `max_new_tokens=5` is represented as the two SID
sentinels plus exactly three scored semantic-ID tokens. Sentinel scores are
excluded on both sides.

## Prerequisites

- Install `vllm-gr` and its development dependencies.
- Download the OneRec-1.7B model and the OpenOneRec-RecIF `benchmark_data`
  directory.
- Use enough accelerator memory for beam width 32.

The expected dataset layout is:

```text
benchmark_data/
├── ad/ad_test.parquet
├── interactive/interactive_test.parquet
├── label_cond/label_cond_test.parquet
├── product/product_test.parquet
├── video/video_test.parquet
├── sid2iid.json
└── sid2pid.json
```

## Prepare the constraint data

Build full logical universes instead of deriving a catalog from the selected
evaluation rows:

```bash
python -m benchmarks.open_one_rec.tools.prepare_decode_data \
  --dataset-path /path/to/OpenOneRec-RecIF/benchmark_data \
  --output-dir ./generated_full
```

This creates two domains:

- `video`, `ad`, `label_cond`, and `interactive` use the complete
  `sid2pid.json` universe.
- `product` uses the complete `sid2iid.json` universe.

Each domain is emitted as both a Trie catalog and Constraint Tables triples.
The generated `manifest.json` pins the source mapping, representation digests,
counts, and logical SID universe. Preparation fails if the Trie and triples
representations are not logically equivalent to the source mapping.

To precompile tokenizer-bound artifacts too, add:

```bash
  --model /path/to/OneRec-1.7B --build-artifacts
```

## Run the constrained dataset offline

The recommended local accuracy path is offline, one active request at a time:

```bash
./tests/system/run_onerec_full_regression.sh \
  --model /path/to/OneRec-1.7B \
  --dataset-path /path/to/OpenOneRec-RecIF/benchmark_data \
  --prepared-dir ./generated_full \
  --output-dir ./results/constrained-offline \
  --constraint-mode triples \
  --execution offline \
  --beam-width 32 \
  --temperature 0 \
  --resume
```

Without `--sample-manifest`, this runs the complete test split for every
selected task. Use `--tasks video,product` to select tasks. For a quick,
reproducible canary, add:

```bash
  --sample-manifest benchmarks/open_one_rec/manifests/ci_precision_v1.json
```

The fixed CI manifest contains 100 stable row IDs per task. It is a regression
canary, not a replacement for the complete official dataset when publishing
accuracy.

## Run the five-task gate and native diagnostic

For changes to Beam search, attention, logits, constraints, or scheduling, run
the paired differential:

```bash
./tests/system/run_onerec_precision_diff.sh \
  --model /path/to/OneRec-1.7B \
  --revision <immutable-model-revision> \
  --dataset-path /path/to/OpenOneRec-RecIF/benchmark_data \
  --prepared-dir ./generated_full \
  --output-dir ./results/precision-diff \
  --constrained-golden \
    benchmarks/open_one_rec/manifests/ci_precision_v1_constrained_golden.json \
  --beam-width 32 \
  --temperature 0 \
  --overwrite
```

The blocking baseline is the versioned, post-fix Constraint Tables golden in
`ci_precision_v1_constrained_golden.json`. It pins the protocol, sample and
prepared-manifest digests, source datasets, constraint digests, and all five
tasks' aggregate metrics. The candidate runs through the local serving path,
so the gate covers the production transport and graph worker. The six primary
SID/PID metrics (`pass@1`, `pass@32`, and `recall@32`) must not regress;
position metrics and `recall@1` remain diagnostic at this 100-sample scale.

The same command also runs unmodified `vllm.LLM.beam_search`, with all
`vllm-gr` plugins disabled and no catalog loaded. That paired native result is
a diagnostic reference rather than the blocking oracle. Use
`run_onerec_full_regression.sh --execution offline` when only the offline
implementation is in scope.

Existing native outputs may be reused with `--golden-dir` only when their
summary identity matches the same model revision, dataset and sample manifest,
beam width, temperature, semantic depth, and evaluator contract.

## Measured full-dataset precision

Validation run on the complete `benchmark_data` test sets (129,249 samples
across the five recommendation tasks), OneRec-1.7B, beam width 32,
`max_new_tokens=5` (two SID sentinels + three scored semantic-ID tokens),
serial beam sessions, offline execution. `pid_recall@32` is the item-level
recall obtained by mapping generated SIDs through `sid2pid.json`; it is the
metric comparable with the official OneRec-1.7B publication (Recall@32).

| Task | Samples | vllm-gr | Native vLLM | Official |
| --- | --- | --- | --- | --- |
| `interactive` | 1,000 | 0.1954 | 0.1938 | 0.1941 |
| `label_cond` | 34,891 | 0.0184 | 0.0184 | 0.0184 |
| `ad` | 27,677 | 0.0699 | 0.0697 | 0.0707 |
| `product` | 27,910 | 0.0353 | 0.0352 | 0.0360 |
| `video` | 38,781 | 0.0271 | 0.0270 | 0.0272 |

Findings:

- **vllm-gr is precision-equivalent to native vLLM.** The paired tasks differ
  by at most 0.0016 (`interactive`); `product` matches to within 0.000003.
  Native baselines ran unmodified `vllm.LLM.beam_search` with all `vllm-gr`
  plugins disabled, under the same prompts, beam width, and evaluator.
- **The residual gap to the official numbers is protocol noise, not
  implementation loss.** Native vLLM shows the same ~1% deviation from the
  published values, and the deviation is not one-sided across tasks. These
  runs inherited the task-config default `temperature=0.6`; beam-search
  ranking is invariant under uniform temperature rescaling of the cumulative
  log-probabilities, so the produced ranking matches the BW32/T0 protocol.
  Pass `--temperature 0` for exact BW32/T0 oracles.
- **Grouped beam decode is 12–17.5x faster than the native offline loop**
  (`ad`: 28 ms vs. 349 ms per sample; worst case `interactive`: 49 ms vs.
  864 ms per sample) with identical precision.
- The metrics above were reproduced bit-identically across three code
  baselines: the pre-rework precision branch, the reworked
  `origin/precision` baseline, and a rebase onto
  `upstream/decode_graph` (including the batch-detokenize change in #333).

Reproduce the vllm-gr side with:

```bash
python benchmarks/open_one_rec/one_rec_acc_test.py \
  --task-type <task> --use-beam-search --gr --n 32 \
  --dataset-path ./data --max-concurrency 1
```

## Accuracy gate policy

On `decode_graph`, the legacy BW128/T0.6 video-100 benchmark is a performance
and execution smoke test only. Its historical fixed accuracy floors remain in
place for `main`, while `decode_graph` uses the following five-task BW32/T0
constrained-golden comparison as its accuracy gate.

The five-task blocking oracle is the fixed constrained golden produced after
the beam-lifecycle precision fixes. A candidate must match all
pinned protocol and data identities, produce all 100 rows per task, report
zero invalid samples, and meet or exceed every task's golden value for:

- `pass@1`, `pass@32`, and `recall@32`;
- `pid_pass@1`, `pid_pass@32`, and `pid_recall@32`.

The lower-signal `position1_*`, `recall@1`, and `pid_recall@1` metrics are
still recorded in the gate report but are non-blocking for the 100-row canary.
A one-row boundary change moves a pass metric by 0.01, which is too coarse for
those small position hit counts. Increase the canary size before promoting
them to blocking metrics.

The constrained golden is version-controlled and must never be regenerated
from the candidate inside CI. Updating it requires an explicit review showing
the old and new five-task results and preserving the pinned model, manifests,
dataset hashes, constraint digests, BW32, and T0 protocol.

### Known BF16 native-versus-constrained residual

The post-fix 100-row comparison has one native/constrained boundary difference
in `ad`: SID `position1_pass@32` is 0.08 versus 0.07, and PID is 0.07 versus
0.06. At the same time constrained `pass@32` improves from 0.26 to 0.27 and
SID/PID recall does not regress. This residual is separate from the corrected
constraint-scoring bug.

The differing sample is row 14739. Full-vocabulary scoring makes the prefill
score agree with native within 2.4e-7, but the common top sequence diverges at
the second semantic decode step by about 0.05 log-probability. The cause is the
attention execution shape: CUSTOM cascade attention evaluates the shared
prefix and beam suffix in separate FlashAttention calls and merges their LSEs,
whereas native uses one dense/paged attention call. BF16 reductions are not
associative, so the changed reduction order produces small hidden-state
differences that accumulate across layers and can move a candidate across the
rank-32 beam boundary.

The following alternatives were isolated and ruled out for this sample:

- sliced versus full-vocabulary normalization;
- CUDA Graph replay (graph on/off was bit-identical);
- omission from the constraint catalog;
- beam candidate-width truncation;
- the local attention-metadata worktree changes.

Increasing internal beam width is not a correctness fix: width 64 recovers the
target and improves this canary, while width 128 moves it to rank 33. Beam
search optimizes cumulative model score, so fixed-cutoff task accuracy is not
monotonic in search width. Native comparisons therefore remain diagnostic and
are emitted under `diffs/native/`; they do not determine the CI exit status.

## CI layers

CI intentionally keeps two complementary checks:

1. The legacy Video-100 job uses fixed absolute floors from
   `.github/benchmark-gate.yaml` as a fast health check for offline and online
   execution. These floors are not the official oracle.
2. Changes targeting `decode_graph` run the five-task constrained-golden gate
   at beam width 32 and temperature 0. The paired native run is retained in
   the uploaded diagnostics but is non-blocking.

The workflow uploads native records, candidate records, summaries, server
logs, the constrained-golden report, and per-task native diffs for diagnosis.

## Interpreting results

Use aggregate metrics to accept or reject accuracy. Use the other fields to
find the cause of a failure:

- `top1_mismatches`: top-ranked output changed; diagnostic by itself.
- `ranked_top32_mismatches`: ordering within the evaluated cutoff changed;
  diagnostic.
- `per_sample_metric_mismatches`: which samples changed their metric result.
- `aggregate_metrics`: paired baseline, candidate, and delta used by the gate.
- `invalid_samples`: candidate output-contract failures; must be zero.

Do not compare two independently sampled prompt sets or compare a constrained
candidate with a reduced catalog. Either mistake can make an accuracy delta
look like an implementation regression.
