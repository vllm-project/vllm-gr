# Video-100 accuracy gate

This policy accommodates corrected parent cumulative scores. PRs targeting
`main` and manual `workflow_dispatch` runs on `decode_graph` use this policy
profile. Automatic PR validation remains restricted to `main`. NPU
CI runs the Video-100 gate; GPU CI runs the separate five-task
[Beam V1 Recall gate](beam_v1_recall_gate.md).

## Effective checks

The source of truth is [benchmark-gate.yaml](https://github.com/JiusiServe/vllm-gr/blob/main/.github/benchmark-gate.yaml).
The [benchmark script](https://github.com/JiusiServe/vllm-gr/blob/main/tests/system/run_benchmark_and_verify.sh) checks
offline and online accuracy separately against the same minimum thresholds:

| Target branch / runner | pass@1 minimum | pass@32 minimum | recall@32 minimum |
| --- | ---: | ---: | ---: |
| `main` or manual `decode_graph` / any GPU | 0.05 | 0.21 | 0.030 |
| Other branch/event combinations / any GPU | 0.09 | 0.29 | 0.04 |
| Any branch / any NPU | 0.05 | 0.21 | 0.030 |

Every metric must meet its minimum; equality passes. An accuracy value below
its minimum fails with exit code `86`. For example, the reported offline
`pass@1 = 0.06` fails the old `0.09` floor but meets the shared `0.05` floor.
Passing that one metric does not establish that the whole CI run passes.

The gate runs the video task with 100 prompts, beam width 128 and concurrency 1.
Performance thresholds and failure modes, online execution checks, and the
separate Pangu policy are unchanged by this adjustment.

## Interpretation

These floors are shared acceptance policy, not a claim that devices or online
and offline execution produce identical scores. The regression budget is not a
statistical confidence interval. Evaluate all required metrics on the selected
workload; one passing metric or a historical comparison does not establish a
complete CI pass.

## Scope and policy selection

The workflow retains its existing device detection: `npu-smi` selects NPU;
otherwise the benchmark device is GPU. No GPU model lookup is required.
PR runs use `github.base_ref` as the target branch; push and manual runs use
`github.ref_name`. Runs targeting `main` select a logical policy key in
the existing board-override schema. The historical selector names remain stable
for external callers; they do not select the target branch:

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

## Policy revision

GitHub Actions reads the policy and resolver from the same fixed `github.sha`
checkout as the tests. For a PR this is the test merge, so both the base's
threshold updates and the PR's policy edits are included. Policy, resolver, and
test changes can land together through ordinary PR review. See the
[CI policy guide](../../contributing/ci/benchmark-policy-approvals.md).
The independent Jenkins pipeline may use a different policy source; confirm
its effective source and thresholds in its own logs.

After updating the runtime PR, check the **Resolve benchmark policy** job's
summary for the source ref, the device's policy selector above, and the effective
floors. If the selector is missing or the old floors remain, check that the run
includes the workflow selector and that its tested commit contains the policy
update.

## Validation without accelerator inference

```bash
python -m pytest -q tests/ci/test_benchmark_branch_policy.py
python tools/ci/resolve_benchmark_policy.py \
  --policy-file .github/benchmark-gate.yaml --validate-only
```

These checks exercise the workflow's GPU/NPU and branch selection and verify
the resolved thresholds. They do not replace accelerator measurements.
