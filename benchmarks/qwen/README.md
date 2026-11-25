# Qwen offline Beam search benchmark

This directory contains the Qwen benchmark runner, workload generator, shell
entry point and usage guide. Model selection is configurable; Qwen3-0.6B
is the verified example below.

This benchmark runs the offline `GRLLM.beam_search` API directly from a
vllm-gr checkout. It does not depend on gr-inference or an API server.

## Quick start

```bash
MODEL=/path/to/Qwen3-0.6B \
TOKEN_ID_START=1000 TOKEN_ID_END=2000 WORKLOAD_SEED=0 \
CUDA_VISIBLE_DEVICES=0 \
CONTEXT_LEN=128 BEAM_WIDTH=128 BEAM_STEPS=3 \
REQUESTS=1 WARMUP_RUNS=1 REPEAT=5 \
benchmarks/qwen/run_beam_search_benchmark.sh
```

By default, the launcher creates or reuses `.venv` and installs this
checkout in editable mode if needed. Set `INSTALL_MODE=skip` to require an
existing environment that imports this checkout or `INSTALL_MODE=always`
to reinstall it.

Automatic workload generation reads `vocab_size` from the local
`MODEL/config.json`. Set `WORKLOAD_JSONL` to reuse existing input IDs
with either a local checkpoint or a model ID.

`BEAM_STEPS=3` means three generated Beam stages: the Prefill produces the
first Beam result and two Decode stages produce the next two results.
`DECODE_STEPS` is accepted by the shell wrapper as a compatibility alias.

Results are written under `results/beam_search/`, which is ignored by git.
The result JSON includes per-run wall latency, request latency, QPS, generated
tokens/s, Beam candidates/s, output token IDs/scores, runtime versions, and
the exact vllm-gr commit.

## Timeline

Enable a gzip-compressed PyTorch CPU/CUDA timeline that can be opened directly
in Perfetto:

```bash
MODEL=/path/to/Qwen3-0.6B \
TOKEN_ID_START=1000 TOKEN_ID_END=2000 WORKLOAD_SEED=0 \
REPEAT=5 TIMELINE=1 \
benchmarks/qwen/run_beam_search_benchmark.sh
```

By default, `TIMELINE_REPEAT` follows `REPEAT`, so the trace above contains
five complete `beam_search` runs. The profile runs execute after ordinary
measurement and are excluded from the reported latency/QPS samples. Useful
optional settings are:

```bash
TIMELINE_REQUESTS=1
TIMELINE_REPEAT=5
TIMELINE_RECORD_SHAPES=1
TIMELINE_WITH_STACK=1
TIMELINE_WITH_MEMORY=1
TIMELINE_OUTPUT=/tmp/vllm-gr-beam-v1.json.gz
```

Shape, stack, and memory collection are disabled by default because they add
substantial trace volume and profiler overhead.

## Constraint table

This Qwen benchmark requires an enabled canonical CUDA constraint table.
For a real recommendation benchmark, pass a triples document produced by the
vllm-gr constraint tooling:

```bash
MODEL=/path/to/model \
TOKEN_ID_START=1000 TOKEN_ID_END=2000 \
CONSTRAINT_TRIPLES=/path/to/constraint_triples.json \
benchmarks/qwen/run_beam_search_benchmark.sh
```

When `CONSTRAINT_TRIPLES` is omitted, the benchmark creates a synthetic
three-level table so an ordinary Qwen model can exercise the V1 path. The
default `SYNTHETIC_TOKEN_LAYOUT=spread` selects tokens across the shared
`[TOKEN_ID_START, TOKEN_ID_END)` interval. The same bounds apply to input
IDs and output constraint tokens. The synthetic table needs at least
`3 * BEAM_WIDTH` distinct non-special tokenizer IDs in this interval; an
insufficient interval raises an error rather than selecting outside it. Synthetic constraints
are for performance/debugging only and are not a substitute for the business
SID catalog.

## Direct Python invocation

Generate a workload independently so it can be inspected and reused across
multiple benchmark invocations:

```bash
.venv/bin/python benchmarks/qwen/make_workload.py \
  --model-dir /path/to/Qwen3-0.6B \
  --context-len 128 \
  --requests 1 \
  --token-id-start 1000 --token-id-end 2000 --seed 0 \
  --output-jsonl results/beam_search/workload.jsonl
```

Then pass that exact JSONL file to the benchmark:

```bash
.venv/bin/python benchmarks/qwen/benchmark.py \
  --model /path/to/Qwen3-0.6B \
  --output-json results/beam_search/manual.json \
  --workload-jsonl results/beam_search/workload.jsonl \
  --token-id-start 1000 --token-id-end 2000 \
  --context-len 128 \
  --beam-width 128 \
  --beam-steps 3 \
  --requests 1 \
  --warmup-runs 1 \
  --repeat 5
```

The generator reads `vocab_size` from the local model's `config.json` and
requires `0 <= token_id_start < token_id_end <= vocab_size`. The range is
half-open: start is included and end is excluded. Choose the interval for
the shared input/output vocabulary; `[1000, 2000)` above is an example, not a business SID
catalog. Each request samples exactly `context_len` IDs uniformly with
replacement using seed `seed + request_index`. A fixed seed and configuration
produce reproducible workloads; increasing the request count preserves the
existing rows. Length does not have to be a multiple of the interval size.

This ID-based mode does not encode/decode text and does not load a tokenizer.
All IDs must come from the supplied interval; the generator cannot determine
whether that interval represents valid business SID sequences. The constraint
table defines legal output paths within that interval. Provided real triples
are compiled and checked against the same bounds before the engine starts;
out-of-range tokens cause an error, not silent filtering.

Set `WORKLOAD_JSONL=/path/to/existing.jsonl` in the shell wrapper to skip
generation and reuse exactly the same input file, including an externally
tokenized workload. `TOKEN_ID_START` and `TOKEN_ID_END` are always required,
including for reused workloads, whose input IDs are checked against them;
`WORKLOAD_SEED` defaults to zero. The direct benchmark also requires
`--workload-jsonl`. No workload is silently regenerated by the Python benchmark.
Constraint triples and the canonical artifact default beside the result JSON.

Run the shell entry point with `--help`, or the Python entry point with
`--help`, for all supported settings.

## Reading the result JSON

`input_token_lengths` lists the lengths of the token ID arrays actually sent,
in request order. `context_len` reports their common length, or `null` for
mixed lengths. `declared_context_len` records the CLI setting.

Each measured round records the original request latencies and interpolates
p50 and p95 linearly across its sorted samples. The top-level percentile
medians are medians across measured rounds. Warmup and profiler runs are
excluded.
