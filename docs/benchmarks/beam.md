# OneRec Beam Latency Benchmark

`benchmarks.open_one_rec.beam_latency` is the reproducible local benchmark for
the OneRec beam-search path. It measures the same fixed request at normalized
input lengths, with a prefix-cache miss immediately followed by a hit for each
measured pair. `compare` runs eager and CUDA-graph modes in separate child
processes and compares both latency and complete output-token digests.

## Prerequisites

- vLLM `0.22.1` and the vllm-gr checkout to evaluate.
- An exclusive NVIDIA GPU. On the shared development host, reserve one with
  `gpu run`.
- A OneRec model and the production constraint-table catalog. Do not use a
  synthetic catalog for a report intended to represent the issue benchmark.
- A fixed prompt text file. The prompt is left-truncated or newline-padded so
  every request has the exact model input length after the serving layer's
  begin marker is appended.

Build the production catalog and Constraint Tables triples if they are not
already available:

```bash
python -m benchmarks.open_one_rec.tools.convert_catalog \
  --dataset-path /path/to/OpenOneRec/data \
  --output-path /path/to/video_catalog.json

python tools/build_constraint_triples.py \
  --catalog /path/to/video_catalog.json \
  --output /path/to/video_constraint_triples.json \
  --begin "<|sid_begin|>" \
  --end "<|sid_end|>"
```

For a task-specific reduced catalog, use
`benchmarks.open_one_rec.tools.create_task_catalog` instead. The benchmark only
requires the resulting triples file.

## Run the benchmark

From the vllm-gr checkout:

```bash
gpu run --gpus 1 --timeout 1h -- \
  python -m benchmarks.open_one_rec.beam_latency \
  --mode compare \
  --model /path/to/OneRec-1.7B \
  --catalog /path/to/video_constraint_triples.json \
  --prompt /path/to/fixed_prompt.txt \
  --output-json results/beam_latency.json \
  --output-markdown results/beam_latency.md
```

The default contract is input lengths `1024 2048 5120 8192 10240`, beam width
`128`, five measured miss/hit pairs per length, and two warmup pairs. Reduce
`--input-lengths`, `--warmups`, or `--runs` for a smoke test; keep the defaults
for issue-quality results. A single mode can be run with `--mode eager` or
`--mode graph`.

The command writes one JSON file containing:

- exact model, catalog, prompt, vLLM, torch, CUDA, driver, GPU, repository SHA,
  and dirty-tree metadata;
- the resolved GR beam configuration and benchmark contract;
- raw per-request timing samples for prefix-cache miss and hit;
- prefill, decode-engine, end-to-end, p50/p90/p95, and output-stability data;
- eager-versus-graph deltas and the percentage improvement for every length.

The optional Markdown file contains a compact issue-ready table with hardware,
vLLM, vllm-gr commit, tree state, and output-parity columns.

When `--output-json` is supplied in `compare` mode, the eager and graph child
results are retained beside the combined result as `<stem>.eager.json` and
`<stem>.graph.json`. These files make reruns auditable without repeating the
GPU experiment.

## Timing and correctness contract

The request timing includes the `beam_search` call and a CUDA synchronization.
The prefill sample is the first timed engine step; decode-engine time is the
sum of subsequent timed engine steps. The runner asserts one request, exactly
128 beams by default, the expected sequence length, the configured end marker,
and the expected engine-step token sequence.

Output digests must match between eager and graph mode and remain stable across
the measured repetitions before interpreting a latency difference. If either
check fails, report the run as a correctness failure rather than as a
performance win.

The benchmark is intentionally separate from the pytest suite because model
startup, graph capture, and GPU timing are not suitable for a normal unit-test
job. The pure normalization, validation, digest, and comparison helpers are
covered by `tests/test_beam_latency_benchmark.py`.
