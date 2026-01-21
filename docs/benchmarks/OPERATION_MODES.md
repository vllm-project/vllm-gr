# OneRec Benchmark - Operation Modes

This guide explains the two modes of operation for running the OneRec benchmark.

## Overview

OneRec supports two distinct operation modes:

1. **API Server Mode** - Run vLLM API server, then benchmark against it via OpenAI-compatible API
2. **Throughput Mode** - Direct throughput testing without running an API server

| Mode | Use Case | Latency | Throughput | Setup Complexity |
|------|----------|---------|-----------|------------------|
| **API Server** | Production-like testing, distributed setup | Higher | Lower | High |
| **Throughput** | Performance testing, local testing | Lower | Higher | Low |

## Mode 1: API Server Mode

### Overview

In this mode, you:
1. Start a vLLM API server
2. Run the benchmark client that sends requests to the server via OpenAI-compatible API

**Advantages**:
- ✅ Simulates production deployment
- ✅ Can distribute server and client on different machines
- ✅ Supports load balancing and scaling
- ✅ Realistic latency measurements

**Disadvantages**:
- ❌ Network overhead
- ❌ More complex setup
- ❌ Requires running two processes

### Architecture

```
┌─────────────────────────────┐
│     vLLM API Server         │
│  (Port: 10001, Host: ...)   │
│  Model: OpenOneRec/OneRec   │
└──────────────┬──────────────┘
               │ HTTP/REST API
               │
┌──────────────┴──────────────┐
│   OneRec Benchmark Client   │
│  (OpenAI-compatible client) │
└─────────────────────────────┘
```

### Step 1: Start vLLM API Server

#### Command Format

```bash
python -m benchmarks.open_one_rec.one_rec_main bench throughput\
  --model <MODEL_ID> \
  --host <HOST> \
  --port <PORT> \
  [OPTIONAL_FLAGS]
```

#### Example: Local Server

```bash
# Terminal 1: Start vLLM server locally
python -m vllm_gr.entrypoints.openai.api_server \
  --model OpenOneRec/OneRec-1.7B \
  --host localhost \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

**Output**:
```
INFO 01-20 10:30:45 api_server.py:42] Started serving model OpenOneRec/OneRec-1.7B
INFO 01-20 10:30:45 api_server.py:43] Listening on 127.0.0.1:8000
```

#### Example: Network Server (Remote Access)

```bash
# Terminal 1: Start vLLM server on specific host/port
python -m vllm.entrypoints.openai.api_server \
  --model OpenOneRec/OneRec-1.7B \
  --host 10.10.10.174 \
  --port 10001 \
  --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

#### Server Configuration Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | Required | Model ID or path |
| `--host` | str | `localhost` | Server hostname/IP |
| `--port` | int | `8000` | Server port |
| `--gpu-memory-utilization` | float | `0.9` | GPU memory percentage |
| `--max-num-seqs` | int | `256` | Max sequences in batch |
| `--max-num-batched-tokens` | int | `51200` | Max tokens per batch |
| `--enable-chunked-prefill` | bool | False | Enable chunked prefill |
| `--dtype` | str | `auto` | Data type (float16, bfloat16) |
| `--tensor-parallel-size` | int | `1` | Number of GPUs for tensor parallelism |

#### Verify Server is Running

```bash
# Check if server is responding
curl http://localhost:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"OpenOneRec/OneRec-1.7B","object":"model"}]}
```

### Step 2: Run Benchmark Against Server

#### Command Format

```bash
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model <MODEL_ID> \
  --dataset-path <DATA_DIR> \
  --endpoint http://<HOST>:<PORT>/v1 \
  --backend openai-chat \
  --task-types <TASKS> \
  --num-prompts <NUM> \
  --use-beam-search \
  --n <BEAMS> \
  [OPTIONAL_FLAGS]
```

#### Example: Benchmark Against Local Server

```bash
# Terminal 2: Run benchmark
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --endpoint http://localhost:8000/v1 \
  --backend openai-chat \
  --task-types video \
  --num-prompts 100 \
  --use-beam-search \
  --n 4
```

#### Example: Benchmark Against Remote Server

```bash
# Terminal 2: Run benchmark against remote server
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path /path_to/benchmark_data \
  --endpoint http://10.10.10.174:10001/v1 \
  --backend openai-chat \
  --task-types video,interactive \
  --num-prompts 500 \
  --use-beam-search \
  --n 256 \
  --save-result \
  --result-dir ./results
```

#### API Mode Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--endpoint` | str | Required | OpenAI API endpoint (e.g., http://localhost:8000/v1) |
| `--backend` | str | Required | Must be `openai-chat` |
| `--use-beam-search` | bool | False | Enable beam search |
| `--n` | int | 1 | Number of beams |
| `--save-result` | bool | False | Save results to file |
| `--result-dir` | str | `./results` | Directory for results |

### Step 3: Save and Analyze Results

```bash
# Results are saved as JSON files
ls -la results/

# Analyze results
python -c "
import json
with open('results/openai-chat-infqps-OneRec-1.7B-*.json') as f:
    data = json.load(f)
    print(f'Total requests: {data[\"total_requests\"]}')
    print(f'Throughput: {data[\"effective_total_throughput\"]} tokens/s')
    print(f'Latency: {data[\"latency_p99\"]} ms (p99)')
"
```

### Complete Example Workflow: API Mode

```bash
# Terminal 1: Start server
python -m vllm.entrypoints.openai.api_server \
  --model OpenOneRec/OneRec-1.7B \
  --host localhost \
  --port 8000 \
  --gpu-memory-utilization 0.8
# Wait for: "Started serving model..."

# Terminal 2: Run benchmark
sleep 5  # Wait for server startup
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --endpoint http://localhost:8000/v1 \
  --backend openai-chat \
  --task-types rec_reason,item_understand \
  --num-prompts 200 \
  --use-beam-search \
  --n 8 \
  --save-result \
  --result-dir ./results

# Results will be in ./results/
```

---

## Mode 2: Throughput Mode

### Overview

In this mode, you run the benchmark directly against the model without an API server.

**Advantages**:
- ✅ Simpler setup (single command)
- ✅ No network overhead
- ✅ Lower latency
- ✅ Higher throughput
- ✅ Easy local testing

**Disadvantages**:
- ❌ Model must be on same machine
- ❌ Limited to single GPU (unless using tensor parallelism)
- ❌ Cannot simulate distributed setup

### Architecture

```
┌────────────────────────────────┐
│  OneRec Benchmark + vLLM       │
│  ┌──────────────────────────┐  │
│  │   Model (OpenOneRec)     │  │
│  └──────────────────────────┘  │
│  Direct inference (no HTTP)    │
└────────────────────────────────┘
```

### Basic Usage

#### Command Format

```bash
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model <MODEL_ID> \
  --dataset-path <DATA_DIR> \
  --task-types <TASKS> \
  --num-prompts <NUM> \
  [OPTIONAL_FLAGS]
```

#### Example: Simple Throughput Test

```bash
# Direct throughput test (no server)
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100
```

#### Example: Throughput with Beam Search

```bash
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-types interactive,video \
  --num-prompts 200 \
  --use-beam-search \
  --n 8
```

#### Example: Advanced Throughput Test

```bash
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path /data/users/sefi/from_git/vllm_013/dataset/open_one/benchmark_data \
  --task-types rec_reason,item_understand,ad,product,label_cond,video,interactive,label_pred \
  --num-prompts 500 \
  --use-beam-search \
  --n 4 \
  --save-result \
  --result-dir ./results \
  --save-detailed
```

#### Throughput Mode Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | Required | Model ID or path (e.g., OpenOneRec/OneRec-1.7B) |
| `--dataset-path` | str | Required | Path to benchmark data |
| `--dataset-name` | str | Required | Must be `onerec` |
| `--task-types` | str | All 8 | Comma-separated task types |
| `--num-prompts` | int | 1000 | Number of prompts to sample |
| `--use-beam-search` | bool | False | Enable beam search |
| `--n` | int | 1 | Number of beams |
| `--max-tokens` | int | 128 | Max tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature |
| `--top-p` | float | 0.95 | Nucleus sampling parameter |
| `--save-result` | bool | False | Save results to JSON |
| `--result-dir` | str | `./results` | Results directory |
| `--save-detailed` | bool | False | Save detailed per-request results |
| `--gpu-memory-utilization` | float | 0.9 | GPU memory utilization |
| `--tensor-parallel-size` | int | 1 | Number of GPUs for parallelism |

### Multi-GPU Throughput Testing

For larger models or higher throughput, use tensor parallelism:

```bash
# Using 4 GPUs
python -m benchmarks.open_one_rec.one_rec_main bench throughput\
  --dataset-name onerec \
  --model OpenOneRec/OneRec-8B \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 500 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### Complete Example Workflow: Throughput Mode

```bash
# Single command - everything in one process
python -m benchmarks.open_one_rec.one_rec_main bench throughput
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-types rec_reason,item_understand,video \
  --num-prompts 300 \
  --use-beam-search \
  --n 4 \
  --save-result \
  --result-dir ./results

# Results available immediately in ./results/
```

---

## Mode Comparison

### Side-by-Side Comparison

```
┌──────────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect                   │ API Server Mode     │ Throughput Mode     │
├──────────────────────────┼─────────────────────┼─────────────────────┤
│ Setup                    │ 2 terminals/procs   │ 1 command           │
│ Network overhead         │ Yes (HTTP)          │ No                  │
│ Latency                  │ Higher (+HTTP)      │ Lower               │
│ Throughput               │ Lower               │ Higher              │
│ Distributed testing      │ Yes                 │ No                  │
│ Production-like          │ Yes                 │ No                  │
│ Easy local testing       │ No                  │ Yes                 │
│ Model location           │ Any machine         │ Same machine        │
│ GPU scaling              │ Per-server          │ Tensor parallelism  │
└──────────────────────────┴─────────────────────┴─────────────────────┘
```

### Decision Matrix

**Use API Server Mode if**:
- ✅ Testing production-like deployment
- ✅ Running server and benchmark on different machines
- ✅ Measuring real-world network latency
- ✅ Testing load balancing/scaling
- ✅ Multiple clients hitting same server

**Use Throughput Mode if**:
- ✅ Quick local performance testing
- ✅ Maximizing throughput measurements
- ✅ Development and debugging
- ✅ Simple setup preferred
- ✅ Single benchmark run needed

---

## Advanced: Using launch.json

You can also run both modes through VS Code's launch configuration.

### launch.json Configuration

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        // API Server Mode
        {
            "name": "API Server",
            "type": "debugpy",
            "request": "launch",
            "program": "-m",
            "args": [
                "vllm.entrypoints.openai.api_server",
                "--model", "OpenOneRec/OneRec-1.7B",
                "--host", "localhost",
                "--port", "8000",
                "--gpu-memory-utilization", "0.8"
            ],
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": "/data/models/huggingface"
            }
        },
        // API Benchmark
        {
            "name": "Benchmark against API Server",
            "type": "debugpy",
            "request": "launch",
            "program": "-m",
            "args": [
                "benchmarks.open_one_rec.one_rec_main bench throughput",
                "--dataset-name", "onerec",
                "--model", "OpenOneRec/OneRec-1.7B",
                "--dataset-path", "./data",
                "--endpoint", "http://localhost:8000/v1",
                "--backend", "openai-chat",
                "--task-types", "video",
                "--num-prompts", "100",
                "--use-beam-search",
                "--n", "4",
                "--save-result",
                "--result-dir", "./results"
            ],
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
                "HF_HOME": "/data/models/huggingface"
            }
        },
        // Throughput Mode
        {
            "name": "Throughput Test",
            "type": "debugpy",
            "request": "launch",
            "program": "-m",
            "args": [
                "benchmarks.open_one_rec.one_rec_main bench throughput",
                "--dataset-name", "onerec",
                "--model", "OpenOneRec/OneRec-1.7B",
                "--dataset-path", "./data",
                "--task-types", "video,interactive",
                "--num-prompts", "200",
                "--use-beam-search",
                "--n", "8",
                "--save-result",
                "--result-dir", "./results"
            ],
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": "/data/models/huggingface"
            }
        }
    ]
}
```

### Running from launch.json

1. Press `Ctrl+Shift+D` to open Debug view
2. Select configuration from dropdown
3. Press `F5` to run

---

## Monitoring Results

### API Server Mode Results

```bash
# Results file location
ls -la results/openai-chat-infqps-*.json

# View results
cat results/openai-chat-infqps-OneRec-1.7B-20260120-103000.json | jq '.'

# Expected metrics:
# {
#   "effective_total_throughput": 150.5,  # tokens/sec
#   "total_requests": 100,
#   "latency_p50": 234.5,                 # ms
#   "latency_p99": 456.7
# }
```

### Throughput Mode Results

```bash
# Results file location
ls -la results/throughput-*.json

# View results
cat results/throughput-OneRec-1.7B-20260120-103000.json | jq '.results'

# Expected metrics:
# {
#   "total_throughput": 450.2,           # tokens/sec
#   "effective_throughput": 425.8,
#   "num_requests": 200
# }
```

### Detailed Results

If using `--save-detailed`, you get per-request metrics:

```bash
cat results/openai-chat-infqps-OneRec-1.7B-*-detailed.json | jq '.[] | {request_id, latency, tokens}'
```

---

## Troubleshooting

### API Server Mode Issues

#### Server won't start
```bash
# Check port is free
lsof -i :8000

# Use different port
--port 8001
```

#### Benchmark can't connect
```bash
# Verify endpoint is correct
curl http://localhost:8000/v1/models

# Check firewall
telnet localhost 8000
```

### Throughput Mode Issues

#### Out of memory
```bash
# Reduce batch size
--num-prompts 50

# Use smaller model
--model OpenOneRec/OneRec-1.7B
```

#### Slow performance
```bash
# Check GPU utilization
nvidia-smi dmon -s p

# Enable chunked prefill
# (requires modifying model config)
```

---

## Summary

| Task | Command |
|------|---------|
| Start API server | `python -m vllm.entrypoints.openai.api_server --model ... --host ... --port ...` |
| Run API mode benchmark | `python -m vllm.benchmarks.throughput --endpoint http://... --backend openai-chat ...` |
| Run throughput mode | `python -m vllm.benchmarks.throughput --model ... --dataset-path ...` |
| Monitor results | `cat results/*.json \| jq '.'` |

For more details, see:
- [README.md](README.md) - Quick start
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
