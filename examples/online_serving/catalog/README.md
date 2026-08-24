# Using a Constraint Table to constrain recommendation results

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Building the Artifact](#building-the-artifact)
- [Invoking the Server](#invoking-the-server)
- [Invoking the Client](#invoking-the-client)

## Overview

This example compiles an item catalog into the immutable six-table Artifact
loaded by each Worker before it becomes ready. Limiting the available items
increases accuracy and reduces the decoding search space.

## Installation

- Refer to [quickstart.md](../../../docs/quickstart.md) for installation.

## Building the Artifact

- Follow [Open One Rec Benchmark Dataset](../../../docs/benchmarks/README.md#open-one-rec-benchmark-dataset) for One Rec Benchmark Dataset retrieval.

- Use `benchmarks/open_one_rec/tools/convert_catalog.py` to convert OpenOneRec sid2pid.json file to vllm-gr catalog format.
- For full usage/options, see [OpenOneRec tools README](../../../benchmarks/open_one_rec/tools/README.md).
- OpenOneRec's list of acceptable SIDs is in `<OpenOneRecFolder>/sid2pid.json`.

Here's an example where `--dataset-path` must point to the directory containing `sid2pid.json`.

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
    --dataset-path ./data \
    --output-path ${MyFolder}/catalog.json

python tools/build_constraint_triples.py \
    --catalog ${MyFolder}/catalog.json \
    --output ${MyFolder}/constraint_triples.json \
    --begin '<|sid_begin|>' \
    --end '<|sid_end|>'

python tools/build_constraint_table.py \
    --triples ${MyFolder}/constraint_triples.json \
    --model OpenOneRec/OneRec-1.7B \
    --output ${MyFolder}/constraint_table.safetensors
```

Copy the printed Artifact and tokenizer digests into
`${MyFolder}/vllm_gr_config.yaml`:

```yaml
schema_version: 1
beam:
  max_width: 128
  max_decode_steps: 3
  graph_enabled: true
constraint_table:
  enabled: true
  backend: npu
  path: /absolute/path/to/constraint_table.safetensors
  format: constraint_table_v1
  artifact_digest: sha256:...
  tokenizer_digest: sha256:...
  max_top_k: 128
attention_backend: CUSTOM
```

Use `backend: cpu` to keep the six tables on Host while the model logits stay
on their accelerator device.

Relative `constraint_table.path` values in this YAML file are resolved against
the directory containing the YAML file. The normalized absolute path is then
sent to Workers; all Workers must be able to access that path.

`constraint_table.max_top_k` must cover `beam.max_width`. The NPU backend
supports `max_top_k <= 256`; the CPU backend has no framework-defined fixed
upper limit.

## Invoking the Server

- Pass the single vLLM-GR configuration entry to the server.

```bash
 vllm-gr serve OpenOneRec/OneRec-1.7B \
    --vllm-gr-config ${MyFolder}/vllm_gr_config.yaml \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 16384 \
    --scheduling-policy priority

```

## Invoking the Client

- Client execution does not change
- see beam_search example [README.md](../beam_search/README.md#run-client-on-another-terminal) for client single prompt execution.
- see benchmark documentation [README.md](../../../docs/benchmarks/README.md) for available benchmarks for online serving.

## Result verification

A utility was added to check if client results are all within the catalog

```bash
bash examples/online_serving/beam_search/client.sh > result.json
python examples/online_serving/catalog/verify_catalog.py result.json ${MyFolder}/catalog.json
```

Expected results:

```text
Loading results from ../vllm/trc/test/result.json...
Loading catalog from OpenOneRec/benchmark_data/catalog.json...
Catalog loaded with 1519078 items.
First 10 items in catalog: ['<|sid_begin|><s_a_7060><s_b_18><s_c_1748><|sid_end|>', '<|sid_begin|><s_a_522><s_b_2833><s_c_5018><|sid_end|>', '<|sid_begin|><s_a_5368><s_b_7001><s_c_5852><|sid_end|>', '<|sid_begin|><s_a_6118><s_b_1302><s_c_4114><|sid_end|>', '<|sid_begin|><s_a_4990><s_b_5835><s_c_5629><|sid_end|>', '<|sid_begin|><s_a_3492><s_b_2549><s_c_384><|sid_end|>', '<|sid_begin|><s_a_1280><s_b_6846><s_c_7175><|sid_end|>', '<|sid_begin|><s_a_1848><s_b_3131><s_c_3721><|sid_end|>', '<|sid_begin|><s_a_5001><s_b_2547><s_c_7816><|sid_end|>', '<|sid_begin|><s_a_4616><s_b_4751><s_c_2473><|sid_end|>']
Verifying 512 generated choices...

SUCCESS: All generated items exist in the catalog.
```
