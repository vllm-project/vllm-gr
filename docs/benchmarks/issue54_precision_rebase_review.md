# Issue 54 decode_graph rebase and OneRec precision review

Date: 2026-09-06

## Rebase boundary

`scheduler_async` was rebased from its original fork point onto the newest
locally available `upstream/decode_graph` revision:

- base: `5310824226d33f9db67c059f716556c06d131e26`
- base subject: `perf(beam_search): batch-detokenize best beams in finalize (#333)`
- pre-rebase backup: `backup/scheduler_async-before-decode-graph-rebase-20260906`
- pre-rebase tip: `c6904b393493b977d77e072d6e57e5a67fbde578`

Refreshing the remote ref was attempted first, but the configured HTTPS
remote required credentials unavailable in this environment. Consequently,
"newest" here means the newest local remote-tracking revision, not a freshly
authenticated fetch.

The upstream geometry-cache change and the persistent Step 5 implementation
both modify attention and model-runner bookkeeping. Git produced a partially
combined result that referenced members which the persistent runner does not
create (`_shared_sampling_cache`, `_req_ids_key`, `_logit_idx_tuple`) and an
attention eviction method which was absent. Commit `291f480` restores the
tested persistent implementations in those conflict areas. The upstream
batch-detokenize finalization remains integrated.

## Precision-fix source

The sibling checkout `/data/zhanghan/vllm-gr` contains the precision repair at
`20aa33243b70ce7a5564d1a2b28f01fdbe659c95` (`fix(onerec): align beam decode
precision and add regressions`). It is directly based on the same
`5310824` decode_graph revision.

The repair contains changes for several distinct execution paths:

| Change in `20aa332` | Persistent scheduler disposition |
| --- | --- |
| Preserve parent cumulative scores across per-step Beam requests | Already covered structurally. Persistent CUDA bootstraps `decision.sequence` and `decision.beam_scores` from the prefill device decision and advances them in one resident session. There is no per-step request reconstruction. |
| Project recomputed cached suffix coordinates through the refreshed block table | Applies to the removed per-step/recompute path. Persistent decode uses the incremental branch and its dedicated Beam KV pool. |
| Reject the old Beam-request path when prefix caching is disabled | Does not apply. Persistent decode binds native prefix KV once and was explicitly validated with prefix caching disabled. |
| Return generated text without slicing by Python prompt character length | Ported. `GRLLM.beam_search(decode_prompt_in_text=False)` returns the generated tail; the benchmark restores the prompt-injected SID begin marker before evaluation. |
| Make SID-to-PID ties deterministic | Ported. Mapping order now breaks equal downsampled-count ties. |
| Treat empty transported FlatLogprobs as no candidates | Ported for standard and constrained extraction. |
| Add reproducible manifests, native golden generation, and streaming regression runners | Used as the reference procedure. The current Step 9 runner remains the execution harness for the persistent path; wholesale copying would reintroduce assumptions about the old scheduler and online path. |

## Constraint data

The earlier smoke test used a reduced 16,981-SID triples document. That input
is useful for kernel and lifecycle checks but changes the candidate universe,
so it is not a precision oracle.

The full regression uses representations generated from the complete RecIF
mappings:

| Domain | Source entries | Catalog SHA256 | Triples SHA256 |
| --- | ---: | --- | --- |
| video (`sid2pid.json`) | 1,519,078 | `472b0d1f8bde08ffbe0d75f300b1eab7f721e7a96136e09b413f7e6362ffa87b` | `fba93a0f741cf11247a831dfbf6c20e0e4b33b765481da17d46cd72b48a5c6be` |
| product (`sid2iid.json`) | 1,293,437 | `05e9c05433e8bd94ca47d4a7ba9d9bcc6411c3a07a1b07c7a2536c56769bf1fe` | `a401a749b6d3737dae97a61dffce5011f033ed35fcf3367ae6f759158b10a44d` |

## Validation evidence

The full-domain Video-100 canary used the same cached 100 records, model,
BW32/T0 evaluator, and triples input in both checkouts. The old precision
branch ran with its required prefix cache; the persistent branch ran with
prefix caching disabled and Beam Graph enabled. All reported accuracy metrics
were exactly equal. The main item-level values were:

| Implementation | PID pass@1 | PID pass@32 | PID recall@32 | Generation time |
| --- | ---: | ---: | ---: | ---: |
| `precision@20aa332` | 0.06 | 0.21 | 0.0301111111 | 22.01 s |
| rebased persistent scheduler | 0.06 | 0.21 | 0.0301111111 | 22.29 s |

The earlier reduced-domain comparison was also exactly equal between the two
implementations (PID pass@32 0.01 and PID recall@32 0.001). This isolates the
low reduced-domain score to the experimental candidate universe rather than a
persistent scheduling regression.

The first full pass exposed one more legacy benchmark presentation bug. The
generated tail contains the three semantic SID tokens and `<|sid_end|>`, while
`<|sid_begin|>` belongs to the prompt augmentation and is intentionally absent
when `decode_prompt_in_text=False`. The old evaluator only strips the end
marker when it sees a begin marker, so every exact-SID metric was reported as
zero even though PID mapping and item metrics were valid. The benchmark now
restores the prompt-injected begin marker before evaluation. A repeated
Video-100 canary produced identical SID and PID metrics (`pass@1=0.06`,
`pass@32=0.21`, `recall@32=0.0301111111`), confirming that the fix changes
framing only.

CPU regression after the rebase and applicable precision fixes, with CUDA
hidden from pytest so all GPU work remains under `gpu run` ownership:

```text
1676 passed, 3121 skipped, 17 warnings in 46.14s
```

## Five-task full-dataset result

The final run processed 130,259 samples with BW32/T0, persistent asynchronous
scheduling, prefix caching disabled, Beam Graph enabled, and the complete
domain constraint tables. The five task sizes themselves sum to 130,259; the
129,249 total in the sibling precision guide is an arithmetic typo.

| Task | Samples | SID pass@1 | SID pass@32 | SID recall@32 | PID pass@1 | PID pass@32 | PID recall@32 | Recorded PID recall@32 | Delta | Mean ms/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video | 38,781 | 0.057502 | 0.179779 | 0.028920 | 0.056290 | 0.175266 | 0.028031 | 0.0271 | +0.000931 | 46.338 |
| ad | 27,677 | 0.019439 | 0.216714 | 0.075335 | 0.016729 | 0.202804 | 0.069996 | 0.0699 | +0.000096 | 28.067 |
| product | 27,910 | 0.037693 | 0.258115 | 0.067120 | 0.013687 | 0.157327 | 0.035343 | 0.0353 | +0.000043 | 29.440 |
| label_cond | 34,891 | 0.007710 | 0.046746 | 0.019830 | 0.007394 | 0.045685 | 0.019301 | 0.0184 | +0.000901 | 24.069 |
| interactive | 1,000 | 0.068000 | 0.359000 | 0.219722 | 0.065000 | 0.345000 | 0.209358 | 0.1954 | +0.013958 | 52.766 |

### Native vLLM and published OneRec comparison

The OpenOneRec OneRec-1.7B model card publishes only Recall@32 for these five
tasks. The native vLLM values below come from the completed full-split
`vllm.LLM.beam_search` runs in the sibling checkout, with all vllm-gr plugins
disabled. The raw native JSON files are under `/tmp/onerec_native_<task>/`.

| Task | Persistent PID pass@1 | Native PID pass@1 | Persistent PID pass@32 | Native PID pass@32 | Persistent PID recall@32 | Native PID recall@32 | Published Recall@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video | 0.056290 | 0.048787 | 0.175266 | 0.169877 | 0.028031 | 0.027009 | 0.0272 |
| ad | 0.016729 | 0.016512 | 0.202804 | 0.202515 | 0.069996 | 0.069695 | 0.0707 |
| product | 0.013687 | 0.013579 | 0.157327 | 0.156503 | 0.035343 | 0.035247 | 0.0360 |
| label_cond | 0.007394 | 0.006334 | 0.045685 | 0.043220 | 0.019301 | 0.018398 | 0.0184 |
| interactive | 0.065000 | 0.067000 | 0.345000 | 0.314000 | 0.209358 | 0.193785 | 0.1941 |

The observed Recall@32 deltas are:

| Task | Persistent - native | Persistent - published | Relative to published |
| --- | ---: | ---: | ---: |
| video | +0.001022 | +0.000831 | +3.055% |
| ad | +0.000301 | -0.000704 | -0.995% |
| product | +0.000096 | -0.000657 | -1.825% |
| label_cond | +0.000903 | +0.000901 | +4.897% |
| interactive | +0.015573 | +0.015258 | +7.861% |

The persistent constrained result is above native vLLM on PID Recall@32 for
all five tasks. Against the published model-card values, it is above on video,
label_cond, and interactive, and below by 0.000704 on ad and 0.000657 on
product. Those two gaps are smaller than 0.001 absolute. PID pass@32 is also
above native on all five tasks. PID pass@1 is above native on four tasks and
is 0.002 lower on interactive (0.065 versus 0.067).

This is an observed full-dataset comparison, not a strict zero-tolerance
paired oracle. The saved native jobs inherited the imported task temperature
of 0.6; the persistent jobs explicitly used temperature 0. The model, complete
test splits, beam width 32, SID-to-PID evaluator, and cutoff are otherwise the
same. A future strict paired gate should regenerate native outputs at T0 from
the same immutable input manifest. The published values have four-decimal
precision and do not include pass metrics or raw predictions.

Official source (accessed 2026-09-06):
<https://huggingface.co/OpenOneRec/OneRec-1.7B>

Native result SHA256 values, in task order interactive, label_cond, ad,
product, and video:

```text
7ebae5e56a2706eeae2d217e71580817951582a094a2129c7131bbc71c522b45
188375835d8a4282bc74c9a9de0e23f60189293fc73f0fefd9ad7a4e6b190852
dcdead07582a3c341e8a1e9264fccc4315f15b008c26e7ee03e77fd17904ac96
79413c95641f68090d46051938cb6f6725aaf854aa64d077541f640b005a9e7c
931cfa5092c083dd046e4bf6acde52dd2f1edae5fa86edddd88ca419af76ac22
```

No task regressed against the recorded full-dataset PID recall@32 values. A
first full pass before the SID framing repair produced exactly the same PID
metrics for every task, providing an independent repeat of the item-level
result. Every persistent request reported one native scheduler request, no
decode request objects, no prefix-cache query, no intermediate Beam-state
D2H, one final result D2H, and matched GR-state create/free counts.
