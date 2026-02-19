# Using catalog to constrain recommendation results

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting the Catalog](#getting-the-catalog)
- [Invoking the Server](#invoking-the-server)
- [Invoking the Client](#invoking-the-client)

## Overview
This example demonstrates using an items catalog to eliminate non-existing items given by the recommendation system.
Limiting the available items increases accuracy and enables reducing the search space to get the same accuracy but with less latency.

## Installation
- Refer to [quickstart.md](../../../docs/quickstart.md) for installation.

## Getting the Catalog
- Follow [Open One Rec Benchmark Dataset](../../../docs/benchmarks/README.md#open-one-rec-benchmark-dataset) for One Rec Benchmark Dataset retrieval.

- Use `convert_catalog.py` script to convert OpenOneRec catalog to vllm-gr catalog format.
- OpenOneRec catalog is in `<OpenOneRecFolder>/benchmark_data/sid2pid.json`
```bash
python convert_catalog.py ${OpenOneRecFolder}/benchmark_data/sid2pid.json catalog.json
```


## Invoking the Server
- Add the catalog attribute to server invokation
```bash
 vllm serve --gr \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --catalog-path ${MyFolder}/catalog.json \
     OpenOneRec/OneRec-8B-pro
```
## Invoking the Client
- Client execution is not change
- see beam_search example [README.md](../beam_search/README.md#run-client-on-another-terminal) for client single prompt execution.
- see benchmark documentation [README.md](../../../docs/benchmarks/README.md) for available benchmarks for online serving.