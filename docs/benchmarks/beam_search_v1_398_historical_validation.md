# Historical Beam Search V1 validation for #398

This is the pre-rebase validation record. It predates the Core admission reuse
and batched result-validation changes. Its timer includes lazy result
materialization, unlike the current call/return-only ABBA. See the
[current validation record](beam_search_v1_398_validation.md) for the latest
measurements, full-suite results and CI gates.

This records the earlier GPU validation and public-output optimization for
[PR #404](https://github.com/JiusiServe/vllm-gr/pull/404).
It does not establish completion of every acceptance gate in
[#398](https://github.com/JiusiServe/vllm-gr/issues/398).

## Revisions and scope

- Parent: `a94830c86f7f7abf8b8766de6660810479170ce5`.
- Integration before output-validation optimization:
  `200a11e4ca07e777d1ff7341ce6eaef976b3f1b0`.
- The final experiment used that integration plus the integer fast path,
  typed terminal-result conversion, and optional strict output validation
  included in this PR. Tuple annotations and import formatting were applied
  after measurement without changing validation behavior.
- Both revisions return complete public results from `beam_search_v1()`.
  The integration additionally waits for Worker retirement proof before
  publishing a successful terminal result.

At this historical revision, local tests passed **364 cases** covering the receive boundary,
public output, strict/default validation, client, scheduler, and contracts:

```bash
python3 -m pytest -q \
  tests/test_beam_search_v1_result_validation.py \
  tests/test_gr_async_scheduler.py \
  tests/contract/test_beam_batch_contracts.py \
  tests/test_beam_search_v1_output_processor.py \
  tests/test_beam_search_v1_client.py \
  tests/test_beam_search_v1_entrypoint.py \
  tests/test_offline_beam_output_boundary.py
```

CI-pinned Ruff 0.14.10 check/format and isort checks passed on the modified
Python files. Targeted Python 3.12 mypy checks used `--follow-imports skip`;
they do not replace all configured CI type checks.

## Public-interface performance

Environment: NVIDIA L20X, OneRec-1.7B BF16, Python 3.12,
vLLM 0.22.1, PyTorch 2.11.0+cu130, and FlashInfer 0.6.11.post2.
The model snapshot is `cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b`.
All GPU experiments reserved the device through `canhazgpu run`.

Protocol:

- Single fixed token-ID prompt, `P=1024`, `W=128`, effective `N=3`, `B=1`,
  temperature zero, and ignored EOS. Prefill and Decode graphs enabled.
- The timer covers `beam_search_v1()` and materialization of tokens and text
  for all 128 returned beams. Prefix cache is reset outside the timer before
  each request.
- Record the first request separately, then run 10 warmups and 100 measured
  requests per condition.
- The three current-version conditions alternate in blocks of 10 requests
  in the same engine; block order reverses on alternating rounds. The
  previous validator condition restores only the preceding `_sequences`
  implementation and retains the integer fast path.
- The parent runs in a separate engine on the same physical GPU. Input,
  model, library versions, tokenizer, and constraint-table digests match.
- Diagnostic Worker hooks fix both fused Q/K RMSNorm reductions to
  `XBLOCK=16`, `R0_BLOCK=128`, `num_warps=4`, with
  `TORCHINDUCTOR_COMPILE_THREADS=1`. This control is not a production change.

All latency values below are milliseconds:

| Condition | Public mean | P50 | P95 | Output conversion mean |
| --- | ---: | ---: | ---: | ---: |
| Parent | 17.747 | 17.694 | 18.244 | 0.256 |
| Previous validator with integer fast path | 18.932 | 18.827 | 19.491 | 0.375 |
| Default light validation | 18.775 | 18.665 | 19.227 | 0.241 |
| Strict validation | 18.932 | 18.873 | 19.334 | 0.386 |

Light validation reduces the same-engine public mean by **0.157 ms (0.83%)**
and output conversion by **0.134 ms (35.8%)**. Full tokens, cumulative scores,
and text hashes match across all **400 measured requests**.
The [measurement data](beam_search_v1_398_measurements.json) includes the
individual public-latency samples, summaries, source identities, and digests.

The parent comparison still shows a gap. Separate engine starts introduce
variation, so this single round does not establish a fixed regression
percentage. Earlier three-round measurements also showed a gap, while a
later controlled single round reversed its direction. The same-engine
comparison isolates this PR's validation optimization more directly.

## Numerical differences and earlier coverage

Before the output-validation changes, cross-process differences first
appeared in Prefill logits, before Beam selection. Generated kernels and
controlled runs traced the observed differences to different Inductor/Triton
Q/K RMSNorm reduction configurations. The first layer and the remaining
27 layers use separate fused reductions; controlling only one was
insufficient.

With both reductions fixed, parent and integration each produced 111
identical complete outputs. Two additional captured requests had identical
Prefill, Decode 1, and Decode 2 logits, with maximum absolute difference zero.
This explains the observed fixed-input difference; it does not establish
general accuracy or require bitwise equality across arbitrary compiler
configurations.

Earlier validation at `200a11e`, before the output-validation optimization:

| Check | Recorded result | Coverage limit |
| --- | --- | --- |
| Focused and kernel tests | 474 focused + 34 kernel cases passed | Earlier integration source |
| Real-model startup widths | W=1,4,16,64,127,128,129,255,256; offline/online 18/18 passed | Representative widths, not every W=1..256 |
| Original-length Video-100 | V1 and legacy: SID pass@1=0.06, pass@32=0.22, recall@32=0.03153968 | Offline local comparison; full online accuracy flow not rerun |
| Video-100 ranked candidates | All 12,800 ranked sequences matched; maximum score difference about 3.8e-6 | Same 100 prompts |
| Six fresh process starts | torch.compile 20.65-21.35 s; first request 29.8-30.5 s | Earlier large cold-compile variation not reproduced or explained |

These test groups overlap and should not be added into a single total.
The Video-100 metrics exceed the configured `decode_graph_gpu` accuracy
floors (0.05 / 0.21 / 0.030); they are not evidence that the entire CI
benchmark workflow or its default GPU thresholds passed.

## Follow-up recorded at that revision

At that revision, broader startup-width/reference coverage, the full online
accuracy flow, historical cold-compile variation, and final-revision CI
evidence remained unresolved. The public parent
comparison above is not a replacement for the issue's separate exact
merged-#393 comparison requirement.

Retirement optimization was then deferred to a subsequent change. Earlier EngineCore
tracing measured about 0.388 ms of additional result-ready-to-emission wait
for release confirmation. Any optimization must retain matching receipts,
owner/generation fencing, cancellation and failure cleanup, delayed-consumer
handling, and proof of release before resource reuse. This PR keeps that
lifecycle behavior intact.
