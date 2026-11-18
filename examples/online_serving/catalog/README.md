# Using catalog to constrain recommendation results

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting the Catalog](#getting-the-catalog)
- [Build the V1 Constraint Artifact](#build-the-v1-constraint-artifact)
- [Prepare the Canonical Configuration](#prepare-the-canonical-configuration)
- [Invoking the Server](#invoking-the-server)
- [Invoking the Client](#invoking-the-client)
- [Result Verification](#result-verification)

## Overview

Constraining generation to the catalog prevents out-of-catalog recommendations and can reduce the search space.
Accuracy and latency should be evaluated for the selected model and catalog.

## Installation

- Refer to [quickstart.md](../../../docs/quickstart.md) for CUDA installation or
  [quickstart-ascend.md](../../../docs/quickstart-ascend.md) for Ascend installation.

## Getting the Catalog

- Follow [Open One Rec Benchmark Dataset](../../../docs/benchmarks/README.md#open-one-rec-benchmark-dataset) for One Rec Benchmark Dataset retrieval.
- Use `benchmarks/open_one_rec/tools/convert_catalog.py` to convert the OpenOneRec
  `sid2pid.json` file to the SID catalog format.
- The resulting JSON list is the input to the triples converter and the expected
  catalog used later by `verify_catalog.py`. It is not a canonical V1 serving
  configuration.
- For full usage/options, see [OpenOneRec tools README](../../../benchmarks/open_one_rec/tools/README.md).
- OpenOneRec's list of acceptable SIDs is in `<OpenOneRecFolder>/sid2pid.json`.

Here's an example where `--dataset-path` must point to the directory containing `sid2pid.json`.

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
    --dataset-path ./data \
    --output-path ${MyFolder}/catalog.json
```

## Build the V1 Constraint Artifact

V1 uses a Constraint Table artifact, not a JSON list passed directly to the
server. Convert the SID catalog to a triples document, then compile it with the
same model snapshot that the server will use:

```bash
export ONEREC_REVISION=cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b

python tools/build_constraint_triples.py \
    --catalog ${MyFolder}/catalog.json \
    --output ${MyFolder}/constraint_triples.json \
    --begin '<|sid_begin|>' \
    --end '<|sid_end|>'

python tools/build_constraint_table.py \
    --triples ${MyFolder}/constraint_triples.json \
    --model OpenOneRec/OneRec-1.7B \
    --revision "$ONEREC_REVISION" \
    --output ${MyFolder}/constraint_table.safetensors \
    --trust-remote-code
```

Record the printed `artifact_digest` and `tokenizer_digest`. Both values are
required by the canonical configuration.

## Prepare the Canonical Configuration

Create `vllm_gr_config.yaml` from the engine configuration in the
[offline Beam Search V1 example](../../offline_inference/beam_search/README.md#engine-configuration).
Set `constraint_table.path` to the absolute artifact path and replace both
digest placeholders with the values printed by the build command. Use
`constraint_table.backend: cuda` on GPU or `npu` on Ascend.

The example uses `beam.max_width: 128`, `beam.max_decode_steps: 3`, and
`beam.execution_mode: v1`. HTTP requests must use `n: 128`.

## Invoking the Server

Start the server with the canonical configuration:

```bash
vllm serve OpenOneRec/OneRec-1.7B --gr \
    --revision "$ONEREC_REVISION" \
    --vllm-gr-config ${MyFolder}/vllm_gr_config.yaml \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 128 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --async-scheduling \
    --no-enable-chunked-prefill
```

Do not pass the legacy `--catalog-path` serving option with this canonical V1
configuration. The old frontend catalog injection flow is retired.

## Invoking the Client

- The request schema is unchanged. Set `n` to `128` and use the configured SID
  boundary tokens.
- `examples/online_serving/beam_search/client.sh` sends a Chat Completions
  request to `/v1/chat/completions`. It uses the full model name
  `OpenOneRec/OneRec-1.7B`, `n: 128`, and the configured SID boundary tokens.
- See benchmark documentation
  [README.md](../../../docs/benchmarks/README.md) for available benchmarks for
  online serving.

## Result Verification

A utility checks whether client results are all within the SID catalog:

Run the client from its expected directory, but retain `result.json` in the
caller's working directory:

```bash
(cd examples/online_serving/beam_search && bash client.sh -b 128) > "${MyFolder}/result.json"
python examples/online_serving/catalog/verify_catalog.py "${MyFolder}/result.json" "${MyFolder}/catalog.json"
```

`client.sh` sends a Chat Completions response. `verify_catalog.py` reads
`choices[].message.content`, so do not substitute a `/v1/completions` response
without changing the model name and result extraction.

Representative results; catalog size, item order, and generated choices can
vary:

```text
Loading results from ${MyFolder}/result.json...
Loading catalog from ${MyFolder}/catalog.json...
Catalog loaded with <catalog-size> items.
Verifying 128 generated choices...

SUCCESS: All generated items exist in the catalog.
```
