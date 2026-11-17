# Prepare a Constraint Table for OneRec serving

Use a OneRec SID catalog to build the constraint triples used by offline
inference and the persistent Constraint Table used by online serving. Run
the commands from the repository root in Bash.

## Set the model and dataset paths

Use the default output directory for a fresh serving setup. To reuse the
[two-SID offline example](../../offline_inference/beam_search/README.md#run-with-a-constraint-table)
in a new shell, set `ONEREC_WORKDIR` to `$PWD/results/onerec-example` first.
If that directory already contains `constraint_triples.json`, start at
[Compile a persistent table for serving](#compile-a-persistent-table-for-serving)
after the setup below. Otherwise, place your catalog in the output directory
or generate one from the OneRec mapping.

For serving, use the local snapshot for `ONEREC_MODEL` and set `ONEREC_BACKEND`
to `npu` on Ascend. Set `ONEREC_DATA` if you are generating a catalog from the
dataset. See the
[download instructions](../../../docs/onerec_v1_async_quickstart.md#1-check-the-environment)
if you need the model or mapping.

```bash
export ONEREC_MODEL="${ONEREC_MODEL:-/absolute/path/to/OneRec-1.7B}"
export ONEREC_DATA="${ONEREC_DATA:-/absolute/path/to/OpenOneRec-RecIF}"
export ONEREC_WORKDIR="${ONEREC_WORKDIR:-$PWD/results/onerec-serving}"
export ONEREC_BACKEND="${ONEREC_BACKEND:-cuda}"
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
python tools/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/catalog.json" \
  --output "$ONEREC_WORKDIR/constraint_triples.json" \
  --begin '<|sid_begin|>' --end '<|sid_end|>'
```

### Compile a persistent table for serving

Compile `constraint_triples.json` with the tokenizer from the serving model
snapshot. This writes `constraint_table.safetensors` for HTTP or gRPC serving
and saves the two digests for the server configuration in `digests.txt`.

```bash
python tools/build_constraint_table.py \
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

Set the request width to match the first entry of `gr.beam_width`. The first
candidate width also matches the request width because Prefill expands one
prompt into that many beams. Set `gr.constraint_table.max_top_k` to cover
the largest configured candidate count. Set
`--max-logprobs` to at least the candidate width and `--max-num-seqs` to at
least the beam width for these server examples.

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
  "$ONEREC_WORKDIR/result.json" "$ONEREC_WORKDIR/catalog.json"
```

The verifier reads `choices[].text` from a Completions response. It also
accepts saved Chat Completions responses with SIDs in
`choices[].message.content`. Server errors, empty choices, incomplete SIDs
and SIDs outside the catalog cause a nonzero exit.

The verifier reports the actual number of valid choices. For two valid
choices it prints `SUCCESS: All 2 generated SIDs exist in the catalog.`
To also check the count, add `--expected-count 2` for the two-SID example
or `--expected-count 128` for a catalog and prompt known to return 128 choices.
