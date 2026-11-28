# OneRec-1.7B

OneRec uses Beam search to rank candidate item sequences for recommendation.
vLLM-GR supports OneRec on
[CUDA](../getting_started/installation/gpu.md) and
[Ascend NPU](../getting_started/installation/npu.md).

## Inputs and outputs

The input represents a user's interaction history and the recommendation
task, formatted for the checkpoint and tokenizer. Use the released
[benchmark messages](../benchmarking/README.md#onerec-benchmark-dataset) as
the starting point for that format; the quickstart's built-in prompt is an
inference smoke test, not a complete business-data example.

The output is a ranked list of semantic IDs (SIDs), such as
`<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>`.
These token sequences are not business item IDs. Your application needs the
SID-to-item mapping associated with the checkpoint and item collection to
resolve them to products or content. A SID can map to more than one item;
the application must choose how to rank or expand those matches. The
[benchmark's SID/PID evaluation](../benchmarking/accuracy.md#example-output)
illustrates this distinction.

For your own data, prepare compatible history prompts, item SIDs, and their
business-ID mapping before serving. A constraint catalog limits which SIDs
may be generated; it does not train new item meanings or create the
SID-to-item mapping. Loading an arbitrary catalog alone does not establish
recommendation quality for a new domain.

## Download the model

Use the [OpenOneRec/OneRec-1.7B checkpoint](https://huggingface.co/OpenOneRec/OneRec-1.7B)
with its tokenizer. Follow the
[download steps](../getting_started/quickstart.md#download-onerec)
to set `ONEREC_MODEL`.

## Model configuration

Use the checkpoint's tokenizer when preparing prompts and compiling the
[constraint table](../configuration/constraint_tables.md). Keep the raw
catalog and business-ID mapping with the model deployment. The
[Beam settings](../configuration/README.md#beam-settings) define search and
result widths; the [feature matrix](../user_guide/feature_compatibility.md)
records device and execution limits.

## Offline inference

Run a recommendation prompt with the
[OneRec quickstart](../getting_started/quickstart.md#onerec). The Python entry
point is `GRLLM.beam_search()`. Use the model configuration above when
moving from the smoke test to catalog-constrained inference.

## Online serving

Send recommendation prompts through the
[HTTP Completions API](../serving/completions.md) or use the experimental
[gRPC API](../serving/grpc.md) for tokenized prompts.

## Evaluation

Follow the [OneRec accuracy benchmark](../benchmarking/accuracy.md) to score
recommendations with task data, or the
[throughput benchmark](../benchmarking/throughput.md) to measure inference
performance.
