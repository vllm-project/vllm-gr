# Rust BeamProc Coordinator

Rust BeamProc is an optional sidecar that performs beam-search coordination
outside the Python API server. The API server manages the sidecar lifecycle,
configuration, IPC endpoints, readiness, and shutdown.

Do not start a second `beam-proc` process manually when using the managed
integration described below.

## Prerequisites

- A working `vllm-gr` installation
- A supported model and accelerator environment
- A free HTTP port for the API server
- Rust 1.88.0 or newer and Cargo for source or editable installations

Prebuilt wheels contain the compiled BeamProc executable. Rust and Cargo are
not required on the target machine when installing such a wheel.

## Install from a wheel

Install the wheel normally:

```bash
python -m pip install /path/to/vllm_gr-*.whl
```

The API server automatically locates the packaged executable. Do not set
`VLLM_GR_BEAM_PROC_BINARY` unless a different executable must be tested.

## Build from source

Source and editable installations normally build BeamProc during package
installation. It can also be built explicitly:

```bash
cargo build --release --locked \
  --manifest-path rust/Cargo.toml \
  --bin beam-proc
```

The resulting executable is:

```text
rust/target/release/beam-proc
```

## Start the API server with Rust BeamProc

Enable BeamProc before starting `vllm-gr`:

```bash
export VLLM_GR_BEAM_PROC_ENABLED=1
export VLLM_GR_BEAM_PROC_NUM_WORKERS=4
```

For a source checkout, explicitly select the locally built executable:

```bash
export VLLM_GR_BEAM_PROC_BINARY="$PWD/rust/target/release/beam-proc"
```

Start the server on an unused port. The catalog argument is required for
catalog-constrained recommendation and can be omitted for other tasks:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-gr serve \
  OpenOneRec/OneRec-1.7B \
  --port 8000 \
  --max-logprobs 128 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority \
  --attention-backend CUSTOM \
  --catalog-path /path/to/video_catalog.json \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The catalog is tokenized and exported to the generated BeamProc configuration
before the sidecar starts.

When no explicit configuration is provided, the API server:

1. Creates private IPC endpoints in a temporary directory.
2. Writes a temporary BeamProc configuration.
3. Starts the Rust executable.
4. Waits for the sidecar readiness file.
5. Connects the API server and engine-core channels.
6. Stops the sidecar when the API server exits.

## Send a request

Use the same OpenAI-compatible endpoint as the Python implementation:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "OpenOneRec/OneRec-1.7B",
    "messages": [
      {
        "role": "user",
        "content": "Your prompt"
      }
    ],
    "use_beam_search": true,
    "n": 128,
    "max_tokens": 5,
    "temperature": 0.0
  }'
```

The client API does not change when Rust BeamProc is enabled.

## Run the benchmark helper

The helper builds the Rust executable when necessary, starts the server, runs
warmup, executes the accuracy benchmark, and stops the server:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=/path/to/OpenOneRec/benchmark_data \
PORT=8000 \
TASK_TYPE=interactive \
NUM_PROMPTS=1000 \
BEAM_WIDTH=128 \
MAX_LOGPROBS=128 \
CONCURRENCY_LIMIT=1 \
VLLM_GR_BEAM_PROC_NUM_WORKERS=4 \
./scripts/run_beamproc_mode.sh rust
```

Choose a GPU and port that are not being used by another server or CI job.
Results are written to `results/acc_test_results` by default. Override the
location with `RESULT_DIR`.

## Compare Rust and Python

Use identical benchmark parameters for both modes:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8000 NUM_PROMPTS=1000 \
DATASET_PATH=/path/to/OpenOneRec/benchmark_data \
BEAM_WIDTH=128 MAX_LOGPROBS=128 CONCURRENCY_LIMIT=1 \
RESULT_DIR=results/rust_bw128 \
./scripts/run_beamproc_mode.sh rust

CUDA_VISIBLE_DEVICES=0 PORT=8000 NUM_PROMPTS=1000 \
DATASET_PATH=/path/to/OpenOneRec/benchmark_data \
BEAM_WIDTH=128 MAX_LOGPROBS=128 CONCURRENCY_LIMIT=1 \
RESULT_DIR=results/python_bw128 \
./scripts/run_beamproc_mode.sh python
```

Run the modes sequentially on the same GPU. Confirm that every request succeeds
before using elapsed time or latency as performance evidence.

## Logging

Enable Rust logs while diagnosing startup or protocol problems:

```bash
export VLLM_GR_BEAM_PROC_LOG_LEVEL=info
export VLLM_GR_BEAM_PROC_LOG_FILE=/tmp/vllm-gr-beam-proc.log
```

Disable Rust logging for performance measurements:

```bash
export VLLM_GR_BEAM_PROC_LOG_LEVEL=off
```

The API server log reports the BeamProc PID, generated configuration path, and
IPC addresses after startup.

## Configuration variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `VLLM_GR_BEAM_PROC_ENABLED` | Enables the Rust coordinator | Disabled |
| `VLLM_GR_BEAM_PROC_BINARY` | Overrides the packaged or source executable | Auto-detected |
| `VLLM_GR_BEAM_PROC_NUM_WORKERS` | Sets the Rust worker count | `1` |
| `VLLM_GR_BEAM_PROC_LOG_LEVEL` | Sets the Rust log level | `info` |
| `VLLM_GR_BEAM_PROC_LOG_FILE` | Sets the Rust log destination | Generated temporary path |
| `VLLM_GR_BEAM_PROC_CONFIG` | Uses an explicit advanced configuration | Generated automatically |

The benchmark helper changes the worker default to `4` and disables Rust logs
unless those values are overridden.

## Troubleshooting

### Binary not found

Build the release binary and set its absolute path:

```bash
cargo build --release --locked \
  --manifest-path rust/Cargo.toml \
  --bin beam-proc
export VLLM_GR_BEAM_PROC_BINARY="$PWD/rust/target/release/beam-proc"
```

### GPU memory error

Stop other processes using the selected GPU or choose another free GPU. Do not
reduce benchmark server parameters when comparing results with a fixed
baseline.

### Address or port already in use

Choose another API server port. With generated configuration, BeamProc uses
private IPC endpoints and does not require manually assigned ZMQ ports.

### Confirm which implementation is active

Rust mode requires `VLLM_GR_BEAM_PROC_ENABLED=1`. The server log should contain
a `Started beam-proc` message. Python mode can be selected explicitly with:

```bash
DATASET_PATH=/path/to/OpenOneRec/benchmark_data \
./scripts/run_beamproc_mode.sh python
```
