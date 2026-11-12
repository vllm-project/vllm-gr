# Troubleshooting

Start with the failing boundary: environment, startup, admission, execution or
output validation. Keep the original exception and the resolved configuration
when reporting a problem.

| Symptom | Check | Next action |
| --- | --- | --- |
| Runtime version rejected | Active interpreter and installed vLLM version | Follow the [version policy](../configuration/dependency_version_policy.md); GR 0.30.0 requires vLLM 0.30.0 |
| `vllm` does not recognize `--gr` | CLI dispatch integration in the active environment | Apply the [GR dispatch patch](../cli/README.md#enable-gr-dispatch) |
| NPU operator or vendor import fails | Prepared Ascend runtime and xLLM operator subset | Follow [Ascend setup](installation/ascend.md), keeping the original loader error |
| Beam width rejected | Engine startup capacity and request width schedule | Set the matching startup widths and restart; see [Configuration](../configuration/README.md#beam-settings) |
| Constraint digest mismatch | Artifact and tokenizer snapshot identities | Rebuild the table from the same immutable model/tokenizer snapshot |
| Table request length rejected | Three effective SID generation steps, excluding boundary markers | Use the [catalog example](quickstart.md#4-why-these-values-are-fixed) |
| Chunked Prefill rejected | GR Beam startup requirements | Disable chunked Prefill and budget for the complete remaining prompt |
| HTTP prompt list runs as separate sessions | `gr.beam_batch_sizes`, native capacity and equal generation budgets | Inspect `Beam batch declined`; see [batching](quickstart.md#batching-several-prompts-in-one-http-request) |
| Graph enabled but replay missing | Selected runner, configured capture geometry and actual dispatch | Check [graph diagnostics](../features/cuda-graphs.md#compare-actual-execution); enabled capture is not proof of replay |
| Fewer candidates returned | Valid paths and final result width | Check [Beam result semantics](../features/beam-search.md#result); outputs are not padded to fabricate valid candidates |

## Collect useful evidence

Record the failing command, commit, Python and runtime versions, accelerator,
model snapshot, resolved GR configuration and complete exception. For a device
run also include runner selection, graph mode and whether the failure occurs
during startup or a request. Replace private paths and credentials before sharing.

For reproducible correctness and performance investigations, use the
[validation guide](../contributing/tools/README.md#validation-levels). Report
quality, graph replay and latency independently; success in one does not establish
the others. Open a [bug report](https://github.com/JiusiServe/vllm-gr/issues/new/choose)
with the smallest reproduction and the relevant evidence.
