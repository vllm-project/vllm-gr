# Prepare a Constraint Table for OneRec serving

The **raw catalog** lists five-token OneRec SIDs, including their begin and
end markers. It lets the verifier check the generated SIDs against the source
data. **Constraint triples** describe the three inner components and are the
input to offline `quickstart.py --catalog`. The **Constraint Table** is a
compiled safetensors artifact that HTTP and gRPC servers load with a YAML
configuration. Build it with the tokenizer from the model you will serve.

## Prepare a serving configuration

Set `ONEREC_MODEL` to a [local OneRec snapshot](../../README.md#run-onerec-offline).
In Bash at the repository root, choose an output directory and a device
backend. On Ascend set `ONEREC_BACKEND=npu`.

```bash
export ONEREC_WORKDIR="$PWD/results/onerec-serving"
export ONEREC_BACKEND="${ONEREC_BACKEND:-cuda}"
export ONEREC_VERIFY_CATALOG="$PWD/examples/online_serving/catalog/sample_catalog.json"
python -m examples.online_serving.catalog.prepare_serving_config \
  --model "$ONEREC_MODEL" --constraint-backend "$ONEREC_BACKEND" \
  --workdir "$ONEREC_WORKDIR"
```

The command uses the two-SID sample by default. It prints the paths of the
compiled table and `$ONEREC_WORKDIR/vllm_gr_config.yaml`. The YAML includes
the table's absolute path and both artifact and tokenizer digests. Use it to
[serve over HTTP](../beam_search/README.md#start-the-server) or
[serve over gRPC](../grpc/README.md#serve-onerec-over-grpc).

To use other inputs, change the command above as follows. Keep
`ONEREC_VERIFY_CATALOG` for the client terminal when verifying a response.

| Source | Argument to add | `ONEREC_VERIFY_CATALOG` for verification |
| --- | --- | --- |
| Built-in sample | Omit the input argument | The sample path exported above |
| Your raw SID catalog | `--raw-catalog /path/to/catalog.json` | That same raw catalog path |
| Previously generated triples | `--triples /path/to/constraint_triples.json` | The original raw catalog used to produce those triples |

The preparer can compile triples on their own. To verify a response you also
need the matching raw catalog. For the [offline example](../../offline_inference/beam_search/README.md#run-with-a-constraint-table),
this is `catalog.json` beside `constraint_triples.json`.

The following sections give the individual commands for converting a dataset,
compiling a table, and writing the same server configuration manually.

## Set the model and dataset paths

Use the default work directory below for a fresh setup. To reuse the
[two-SID offline example](../../offline_inference/beam_search/README.md#run-with-a-constraint-table),
set `ONEREC_WORKDIR="$PWD/results/onerec-example"` first. Skip directly to
[compiling the table](#compile-a-persistent-table-for-serving) only if you know
that the existing triples were generated from the exact raw catalog you intend
to serve and verify.

To switch from the sample to the full dataset, set `ONEREC_DATA` to the local
dataset directory, choose a fresh `ONEREC_WORKDIR`, and start at
[Build a catalog from the OneRec dataset](#build-a-catalog-from-the-onerec-dataset).
Regenerate both the catalog and its triples before compiling the table.
For a different raw catalog you already have, place it at
`$ONEREC_WORKDIR/catalog.json` and start at
[Extract constraint triples from the catalog](#extract-constraint-triples-from-the-catalog).

Use the local snapshot for `ONEREC_MODEL` and set `ONEREC_BACKEND` to
`npu` on Ascend. See the
[download instructions](../../../docs/user_guide/onerec_quickstart.md#1-check-the-environment)
if you need the model or mapping.

```bash
export ONEREC_MODEL="${ONEREC_MODEL:-/absolute/path/to/OneRec-1.7B}"
export ONEREC_DATA="${ONEREC_DATA:-/absolute/path/to/OpenOneRec-RecIF}"
export ONEREC_WORKDIR="${ONEREC_WORKDIR:-$PWD/results/onerec-serving}"
export ONEREC_BACKEND="${ONEREC_BACKEND:-cuda}"
export ONEREC_VERIFY_CATALOG="$ONEREC_WORKDIR/catalog.json"
mkdir -p "$ONEREC_WORKDIR"
```

## Compile the Constraint Table

### Build a catalog from the OneRec dataset

Use the dataset's `benchmark_data/sid2pid.json` to generate `catalog.json`
for the triples converter. If you already have a catalog of OneRec SIDs with
their boundary markers, save it as `$ONEREC_WORKDIR/catalog.json` and start
with the next step. See the
[converter documentation](../../../benchmarks/open_one_rec/tools/README.md)
for conversion options and input formats.

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA/benchmark_data" \
  --output-path "$ONEREC_WORKDIR/catalog.json"
```

### Extract constraint triples from the catalog

Convert `catalog.json` to `constraint_triples.json`. For offline inference,
stop here and pass the triples file to
[`quickstart.py --catalog`](../../offline_inference/beam_search/README.md#run-with-a-constraint-table).
The offline script builds a temporary table with the inference model's tokenizer.

```bash
python tools/constraints/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/catalog.json" \
  --output "$ONEREC_WORKDIR/constraint_triples.json" \
  --begin '<|sid_begin|>' --end '<|sid_end|>'
```

### Compile a persistent table for serving

Compile `constraint_triples.json` with the tokenizer from the serving model
snapshot. This writes `constraint_table.safetensors` for HTTP or gRPC serving
and saves the two digests for the server configuration in `digests.txt`.

```bash
python tools/constraints/build_constraint_table.py \
  --triples "$ONEREC_WORKDIR/constraint_triples.json" \
  --model "$ONEREC_MODEL" \
  --output "$ONEREC_WORKDIR/constraint_table.safetensors" \
  --trust-remote-code > "$ONEREC_WORKDIR/digests.txt" &&
  cat "$ONEREC_WORKDIR/digests.txt"
```

## Write the server configuration

Generate the server YAML using the printed digest values.

```bash
ARTIFACT_DIGEST=$(awk '$1 == "artifact_digest:" {print $2}' "$ONEREC_WORKDIR/digests.txt")
TOKENIZER_DIGEST=$(awk '$1 == "tokenizer_digest:" {print $2}' "$ONEREC_WORKDIR/digests.txt")
cat > "$ONEREC_WORKDIR/vllm_gr_config.yaml" <<EOF
attention-config:
  backend: CUSTOM
gr:
  beam_graph_enabled: true
  worker_decision: true
  beam_width: [128, 128, 128]
  candidate_width: [128, 128, 128]
  max_decode_steps: 3
  constraint_mode: table
  constraint_table:
    backend: $ONEREC_BACKEND
    max_top_k: 128
    path: "$ONEREC_WORKDIR/constraint_table.safetensors"
    format: constraint_table_v1
    digests:
      artifact: "${ARTIFACT_DIGEST:?Build the Constraint Table first}"
      tokenizer: "${TOKENIZER_DIGEST:?Build the Constraint Table first}"
EOF
```

## Change the beam width

Set the request width to the maximum value in `gr.beam_width`. The first
entry of `gr.beam_width` controls how many beams Prefill keeps, so the first
entry of `gr.candidate_width` must equal it. For example, with a retained
schedule of `[4, 8, 4]`, send request width `8` and set the initial
candidate width to `4`.

Set `gr.constraint_table.max_top_k` and `--max-logprobs` to at least the
largest configured candidate width. In these server examples, set
`--max-num-seqs` to at least the largest configured retained width.

For a CUDA beam width of 256, change these fields in the generated YAML.
CUDA Constraint Table sampling supports up to 256 candidates.

```yaml
gr:
  beam_width: [256, 256, 256]
  candidate_width: [256, 256, 256]
  constraint_table:
    max_top_k: 256
```

With this configuration, use `--max-logprobs 256` or higher and
`--max-num-seqs 256` or higher in the example server command. Restart the
server, then request width 256 with HTTP `-b 256` or gRPC `--width 256`.

A constrained request can return fewer beams if the catalog has too few
reachable SIDs.

## Verify a saved response

From the repository root, verify the saved response. The verifier accepts
OneRec three-layer SIDs with begin and end markers.

```bash
python examples/online_serving/catalog/verify_catalog.py \
  "$ONEREC_WORKDIR/result.json" "$ONEREC_VERIFY_CATALOG"
```

The verifier reads `choices[].text` from a Completions response. It also
accepts saved Chat Completions responses with SIDs in
`choices[].message.content`. Server errors, empty choices, incomplete SIDs
and SIDs outside the catalog cause a nonzero exit.

The verifier reports the actual number of valid choices. For two valid
choices it prints `SUCCESS: All 2 generated SIDs exist in the catalog.`
To also check the count, add `--expected-count 2` for the two-SID example
or `--expected-count 128` for a catalog and prompt known to return 128 choices.
