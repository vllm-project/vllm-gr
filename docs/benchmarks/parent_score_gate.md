# Video-100 accuracy gate

This policy accommodates corrected parent cumulative scores. PRs targeting
`main` or `decode_graph` share the same policy profile. On those branches, NPU
CI runs the Video-100 gate; GPU CI runs the separate five-task
[Beam V1 Recall gate](beam_v1_recall_gate.md).

## Effective checks

The source of truth is [benchmark-gate.yaml](../../.github/benchmark-gate.yaml).
The [benchmark script](../../tests/system/run_benchmark_and_verify.sh) checks
offline and online accuracy separately against the same minimum thresholds:

| Target branch / runner | pass@1 minimum | pass@32 minimum | recall@32 minimum |
| --- | ---: | ---: | ---: |
| `main` or `decode_graph` / any GPU or NPU | 0.05 | 0.21 | 0.030 |
| Other branches / any GPU or NPU | 0.09 | 0.29 | 0.04 |

Every metric must meet its minimum; equality passes. An accuracy value below
its minimum fails with exit code `86`. For example, the reported offline
`pass@1 = 0.06` fails the old `0.09` floor but meets the shared `0.05` floor.
Passing that one metric does not establish that the whole CI run passes.

The gate runs the video task with 100 prompts, beam width 128 and concurrency 1.
Performance thresholds and failure modes, online execution checks, and the
separate Pangu policy are unchanged by this adjustment.

## Baseline evidence

The before commit is `bac1451557f108cc8bdf81b5b37160fc43690b3f`; the runtime fix is
`0f5018516f8205cc1f5a5e9cd6f74bb1216565f1`. The results below use OneRec-1.7B,
the video task, 100 samples, beam width 128 and temperature 0.6.

| Device / run | pass@1 | pass@32 | recall@32 |
| --- | ---: | ---: | ---: |
| L20 before, offline | 0.13 | 0.30 | 0.04440079365 |
| L20 fixed, offline | 0.06 | 0.23 | 0.03267857143 |
| L20 fixed, frontend NumPy reference | 0.06 | 0.23 | 0.03267857143 |
| L20 before, online | 0.11 | 0.31 | 0.04437301587 |
| L20 fixed, online | 0.06 | 0.23 | 0.03365079365 |
| A100 reported failing CI, offline | 0.06 | 0.22 | 0.03153968254 |

The A100 offline row comes from the reported CI failure. L20 runs provide the
before/after and frontend-reference comparison for the score correction.
A100 online measurements are not available: applying the same floors online
is a baseline assumption, not evidence that online and offline results are
identical. Input and software fingerprints for the reported A100 run are not
available, so the table does not establish a controlled cross-device comparison.

The floors leave one Top-1 hit, one Top-32 hit and about 0.00154 absolute recall
below the reported A100 result. These margins are a regression budget, not a
statistical confidence interval.

All GPU and NPU models share these gate floors, not identical measured results. The
fixed L20 offline and online results and the reported A100 offline result all
meet them. Other GPU models and NPUs have not been measured here; applying these floors
to them is a shared policy decision. The A100 online assumption above remains
unchanged.

The L20 fixed worker and frontend reference have identical Top-1 on all 100
samples and exactly equal cumulative scores for shared candidates. Both return
128 catalog-valid candidates with finite scores per sample. Their full ranked
lists are not identical: mean Top-32 overlap is 31.97/32; 46 of 12,800 ranked
positions have different scores. This supports cumulative-score correctness,
not exact beam-list equivalence or correctness of the shared constraint scorer.

## Scope and policy selection

The workflow retains its existing device detection: `npu-smi` selects NPU;
otherwise the benchmark device is GPU. No GPU model lookup is required.
PR runs use `github.base_ref` as the target branch; push and manual runs use
`github.ref_name`. Runs targeting `main` or `decode_graph` select a logical policy key in
the existing board-override schema:

| Device | Policy selector |
| --- | --- |
| Any GPU | `decode_graph_gpu` |
| NPU `Ascend910_9362` | `decode_graph_Ascend910_9362` |
| Other or unidentified NPU | `decode_graph_npu` |

All three profiles reuse the same accuracy floors. The `Ascend910_9362` profile
also reuses its existing performance overrides: TTFT 280, online elapsed 20 s,
and offline elapsed 25 s. Other devices keep the default performance thresholds.

The resolver applies the `decode_graph_npu` accuracy profile to every NPU board,
regardless of target branch. It first applies the physical board's overrides,
so performance thresholds and the reported board ID remain unchanged. GPU
selection remains branch-scoped. Pangu settings and the blocking accuracy mode
remain unchanged.

For compatibility with older resolvers, the YAML explicitly assigns these
accuracy floors to `Ascend910_9362`, `Ascend_9362`, `Ascend910B1` and
`unknown`. `Ascend_9362` is a possible alias that has not been verified against
a 910C runtime. Older resolvers only recognize exact IDs; other NPU IDs need
the updated resolver or an explicit logical profile to receive these floors.
The strict bootstrap policy and policies without the NPU profile retain their
own accuracy floors.

An independent Jenkins pipeline must load the updated policy and resolver.
Updating this repository does not update that external pipeline automatically.

## Rollout order

GitHub Actions uses the resolver from the PR target's pinned `base.sha`. A
changed candidate policy can be used only after the required approval for the
current PR head. The independent Jenkins pipeline may use a different policy
source; confirm its effective source and thresholds in its own logs.

After updating the runtime PR, check the **Resolve benchmark policy** job's
summary for the source ref, the device's policy selector above, and the effective
floors. If the selector is missing or the old floors remain, check that the run
includes the workflow selector and that its base contains the policy update.

## Validation without accelerator inference

```bash
python -m pytest -q tests/test_benchmark_branch_policy.py
python tools/ci/resolve_benchmark_policy.py \
  --policy-file .github/benchmark-gate.yaml --validate-only
```

These checks exercise the workflow's GPU/NPU and branch selection and verify
the resolved thresholds. They do not replace accelerator measurements.
