# Beam V1 B=4 fixed-sample Recall gate calibration

The gate uses five fixed 1,000-sample OneRec tasks, beam width 128 and SID
Recall@32. PID Recall@32 remains in the report for diagnosis. The calibration
source is the ten-repeat CI run (workflow run `38889274665`, artifact
`12049173929`, commit `0c0df7957eb727bbd427d8a885dc30d1929372dd`).
Its exact task scores are stored in `v1_recall_b4_baseline.json`.

## CI calibration and local comparison

Each cell is a percentage; SD is a sample standard deviation in percentage
points. All runs use the same fixed sample IDs and task evaluator.

| Task | CI n=10 SID min | CI mean | CI median | CI max | CI SD (pp) | Local n=20 SID (all equal) | Local minus CI min (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video | 3.72500% | 3.73548% | 3.72500% | 3.76722% | 0.01789 | 3.69294% | -0.03206 |
| ad | 7.04687% | 7.09501% | 7.04687% | 7.23508% | 0.08076 | 6.99234% | -0.05452 |
| product | 6.65389% | 6.66289% | 6.65389% | 6.68389% | 0.01449 | 6.68179% | +0.02790 |
| label_cond | 2.44302% | 2.46152% | 2.46468% | 2.46468% | 0.00682 | 2.49302% | +0.05000 |
| interactive | 20.84623% | 20.92177% | 20.84623% | 21.11345% | 0.12228 | 21.02694% | +0.18071 |

| Task | CI n=10 PID min | CI mean | CI median | CI max | CI SD (pp) | Local n=20 PID (all equal) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| video | 3.64944% | 3.66270% | 3.64944% | 3.70056% | 0.02208 | 3.61599% |
| ad | 6.62857% | 6.67171% | 6.62857% | 6.79179% | 0.07125 | 6.54905% |
| product | 3.82698% | 3.87165% | 3.89079% | 3.89079% | 0.03082 | 3.94679% |
| label_cond | 2.40635% | 2.41068% | 2.40635% | 2.42635% | 0.00734 | 2.47635% |
| interactive | 19.72889% | 19.79711% | 19.72889% | 19.97111% | 0.11051 | 19.77210% |

The CI ten-repeat wall times were 782–870 s (median 808.5 s). The local
20-repeat wall times were 950–1,677 s (median 985.5 s); the local run took
22,001 s in total. The local complete result is at
`/workspace/zhanghan/PROJECT/vllm-gr/results/onerec-v1/manual-gate20-20261016-173427/b4_repeat_summary.json`
with exit code 0. The earlier `local-gate20-20261016-172341` attempt exited
143 and is excluded.

These are **not controlled cross-machine comparisons**. The local run used
commit `16401cf11e872fe7cf5ad955568d9d6054806f42` with 27 prefill graph
buckets; the CI calibration used 132 buckets. The task input digests and sample
IDs matched, but the graph configuration did not. The five local scores cannot
be pooled into the CI calibration, nor can the differences be attributed only
to hardware. In fact, the local video and ad scores are below the CI historical
minimum on every local repeat; a single common cross-machine floor would
misclassify this already observed configuration change.

## Gate decision

For each task, use its own historical minimum over the ten CI repeats. A PR
normally runs once. If any task is strictly below its historical minimum, run
two more complete, fresh-engine repeats. Fail only when the **same task** is
strictly below its historical minimum in all three runs. Other tasks cannot
offset it. Explicitly requested diagnostic runs above three use the first
three for the gate and report all remaining scores.

This is a rank-based rule: if the ten baseline scores and three new scores for
a task are independent and exchangeable under an unchanged, stable process,
the probability that all three new scores rank below every baseline score is
`1 / C(13, 3) = 1/286 ~= 0.35%`. The strict inequality makes ties
conservative. By the union bound, five tasks have at most `5/286 ~= 1.75%`
unconditional false-failure probability per PR under those assumptions. This
is **not** a guarantee conditional on one already fixed baseline or across
many future PRs. A changed GPU/software stack or correlated runs invalidates
the exchangeability assumption; recalibrate on the intended runner if either
changes.

The ten scores have only two or three distinct levels per task. We therefore
avoid a normal-theory `mean - k * SD` cutoff. This gate detects persistent
below-history shifts on the chosen 1,000 samples; it cannot guarantee a
0.05-percentage-point effect bound on full test sets or detect a regression
confined to a higher score mode. Review the summary and periodically repeat
the longer calibration when changing the runner or graph profile.

Statistical references: [NIST on process stability](https://itl.nist.gov/div898/handbook/ppc/section4/ppc45.htm),
[NIST on nonnormal data](https://itl.nist.gov/div898/handbook/ppc/section4/ppc47.htm),
[NIST on order statistics](https://itl.nist.gov/div898/handbook/prc/section2/prc262.htm),
and [NIST on familywise error control](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
