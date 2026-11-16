# Using a catalog to constrain recommendation results

Use a compiled Constraint Table to limit OneRec Beam Search V1 candidates to
items in a catalog.

## Getting the catalog

Download [OpenOneRec-RecIF](../../../docs/benchmarks/dataset.md#get-dataset)
and use the [catalog tools](../../../benchmarks/open_one_rec/tools/README.md)
to convert the dataset's `sid2pid.json` into a SID catalog. Follow the
[Constraint Table instructions](../../offline_inference/beam_search/README.md#constraint-table-artifact)
to build the artifact with the same model and tokenizer snapshot that the
server will use.

## Invoking the server

Follow the [online Beam Search V1 guide](../beam_search/README.md#prepare-the-canonical-configuration)
to configure the artifact and start the HTTP server. Set the artifact path and
digests in the canonical `--vllm-gr-config` file. The legacy `--catalog-path`
server option is retired.

## Invoking the client

[Send a sample Beam request](../beam_search/README.md#send-a-request) to verify
the server setup. Use the [OneRec benchmark guides](../../../docs/benchmarks/README.md)
to measure serving performance or evaluate recommendation quality.
