# OpenOneRec Tools

## Dataset prerequisite

Before using these tools, download the OpenOneRec dataset as described in
[the dataset guide](../../../docs/benchmarking/README.md#download-the-dataset).
For a custom dataset layout, set `ONEREC_DATA` to the directory containing the
task parquet files and catalog mappings.

## create_task_catalog.py

Generate a reduced SID catalog JSON for a specific task for vllm-gr catalog format.

Output format:

```json
[
  ["<|sid_begin|>", "<s_a_...>", "<s_b_...>", "<s_c_...>", "<|sid_end|>"],
  ...
]
```

### Quick start

From repository root:

```bash
export ONEREC_DATA="$PWD/data/benchmark_data"
python benchmarks/open_one_rec/tools/create_task_catalog.py \
  --task-type interactive \
  --dataset-path "$ONEREC_DATA" \
  --output-path interactive_catalog.json
```

### CLI flags

- `--task-type`, `-t` (required): task type, e.g. `interactive`, `video`, `label_cond`, `product`, `ad`.
- `--dataset-path`, `-d`: data root directory (default: `./data/benchmark_data`).
- `--output-path`, `-o`: output JSON path (default: `<task_type>_test_sid_catalog.json`).
- `--log-level`: `DEBUG|INFO|WARNING|ERROR` (default: `INFO`).

### Notes

- The tool reads answers from `metadata["answer"]` (falls back to top-level `answer` if present).
- The output is deduplicated (unique SID sequences only).
- The tool searches `<dataset-path>/<task>/<task>_test.parquet` and
  `<dataset-path>/<task>_test.parquet`.

## convert_catalog.py

Convert OpenOneRec `sid2pid.json` keys into SID token sequences for vllm-gr catalog format.

### Quick start

From repository root:

```bash
export ONEREC_DATA="$PWD/data/benchmark_data"
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA" \
  --output-path catalog.json
```

### CLI flags

- `--dataset-path`, `-d`: data root directory containing `sid2pid.json` (default: `./data/benchmark_data`).
- `--output-path`, `-o`: output catalog JSON path (default: `catalog.json`).
- `--log-level`: `DEBUG|INFO|WARNING|ERROR` (default: `INFO`).

### Notes

- The script reads `sid2pid.json` at `<dataset-path>/sid2pid.json`. For
  OpenOneRec-RecIF, set the dataset path to its `benchmark_data` subdirectory.
- The script uses only JSON keys from the input map.
- Each key is decoded using base `8192` into `<s_a_*>`, `<s_b_*>`, `<s_c_*>`.
- Output format is compatible with catalog files consumed by vllm-gr.
