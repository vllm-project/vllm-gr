# Using vLLM Client for Evaluation

Guide for integrating vLLM OpenAI-compatible client for LLM-based evaluation in OneRec.

## Overview

Instead of using external LLM APIs (Gemini, DeepSeek), you can now use vLLM for evaluation tasks. This allows you to:

- ✅ Use your own models for evaluation
- ✅ Keep evaluation local (no external API calls)
- ✅ Faster evaluation (no network latency)
- ✅ Full control over evaluation parameters
- ✅ Support for any model that vLLM supports

## Setup

### 1. Install Required Dependencies

```bash
pip install openai  # For OpenAI-compatible client
pip install vllm    # For vLLM inference
```

### 2. Start vLLM API Server

The evaluation client communicates with vLLM via OpenAI-compatible API.

```bash
# Terminal 1: Start vLLM evaluation server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-72B-Chat \
  --host localhost \
  --port 8001 \
  --gpu-memory-utilization 0.5

# Verify server is running
curl http://localhost:8001/v1/models
```

**Important**: Use a different port than your main benchmark server if running both simultaneously.

### 3. Configure Evaluation in Task Config

Update the task configuration to enable vLLM-based evaluation:

```python
# In benchmarks/open_one_rec/configs/config.py

REC_REASON_CONFIG = {
    "name": "rec_reason",
    # ... other config ...
    "evaluation_config": {
        "llm_eval_enabled": True,  # Enable LLM evaluation
        "vllm_endpoint": "http://localhost:8001/v1",  # vLLM endpoint
        "vllm_api_key": "not-needed",                 # Not needed for local vLLM
        "vllm_async": False,                          # Use sync client
        "llm_max_workers": 4,                         # Parallel evaluation threads
        "llm_max_samples": 500,                       # Max samples to evaluate
    },
}
```

## Configuration Options

### Evaluation Config Parameters

```python
{
    # Enable/disable LLM evaluation
    "llm_eval_enabled": True,
    
    # vLLM Endpoint Configuration
    "vllm_endpoint": "http://localhost:8001/v1",  # Default: localhost:8000/v1
    "vllm_api_key": "not-needed",                 # API key (not needed for local)
    "vllm_async": False,                          # Use async client
    
    # Evaluation Parameters
    "llm_max_workers": 4,                         # Parallel workers
    "llm_max_samples": 500,                       # Max samples (None = all)
    
    # Generation Parameters (optional)
    "llm_temperature": 0.7,
    "llm_top_p": 0.95,
    "llm_max_tokens": 1000,
}
```

### Endpoint Configuration

#### Local vLLM Server

For local evaluation:

```python
"evaluation_config": {
    "llm_eval_enabled": True,
    "vllm_endpoint": "http://localhost:8001/v1",
    "vllm_api_key": "not-needed",
    "vllm_async": False,
}
```

#### Remote vLLM Server

For remote server (e.g., on different machine):

```python
"evaluation_config": {
    "llm_eval_enabled": True,
    "vllm_endpoint": "http://10.10.10.174:8001/v1",
    "vllm_api_key": "not-needed",
    "vllm_async": False,
}
```

#### With Authentication

If your vLLM server requires authentication:

```python
"evaluation_config": {
    "llm_eval_enabled": True,
    "vllm_endpoint": "http://10.10.10.174:8001/v1",
    "vllm_api_key": "your-api-key",
    "vllm_async": False,
}
```

## Usage Workflow

### Step 1: Start Evaluation Server (in one terminal)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-72B-Chat \
  --host localhost \
  --port 8001 \
  --gpu-memory-utilization 0.5 \
  --enable-chunked-prefill
```

### Step 2: Run Benchmark with Evaluation (in another terminal)

```bash
python -m vllm.benchmarks.throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --data-dir ./data \
  --task-types rec_reason \
  --num-prompts 100 \
  --save-result \
  --result-dir ./results
```

The benchmark will automatically use vLLM for evaluation if configured.

## Code Integration

### Using vLLM Client Directly

If you want to use the vLLM client in your code:

```python
from benchmarks.open_one_rec.evaluators.reco_reason import RecoReasonEvaluator
from benchmarks.open_one_rec.configs.config import REC_REASON_CONFIG

# Create evaluator with vLLM configuration
evaluator = RecoReasonEvaluator(
    task_config=REC_REASON_CONFIG,
    # ... other params ...
)

# The evaluator will automatically create vLLM client
# if llm_eval_enabled is True in task_config
```

### Creating vLLM Client Manually

```python
from openai import OpenAI

# Create client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

# Use it like OpenAI client
response = client.chat.completions.create(
    model="Qwen/Qwen-72B-Chat",
    messages=[
        {"role": "user", "content": "Evaluate this recommendation..."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Async Client Usage

For async evaluation:

```python
from openai import AsyncOpenAI
import asyncio

async def evaluate_async():
    client = AsyncOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )
    
    response = await client.chat.completions.create(
        model="Qwen/Qwen-72B-Chat",
        messages=[
            {"role": "user", "content": "Evaluate..."}
        ]
    )
    
    return response.choices[0].message.content

# Run async
result = asyncio.run(evaluate_async())
```

## Advanced Configuration

### Multi-GPU Evaluation Server

For faster evaluation with multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-72B-Chat \
  --host localhost \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 256
```

### Smaller Model for Faster Evaluation

For quicker evaluation with smaller model:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-7B-Chat \
  --host localhost \
  --port 8001 \
  --gpu-memory-utilization 0.5
```

### Quantized Model

For reduced memory usage:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-72B-Chat \
  --quantization awq \
  --host localhost \
  --port 8001 \
  --gpu-memory-utilization 0.6
```

## Monitoring Evaluation

### Check Evaluation Status

During evaluation, monitor the vLLM server:

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Check vLLM server logs
tail -f /tmp/vllm_*.log
```

### View Evaluation Results

```bash
# Results are saved with evaluation metrics
cat results/onerec_results_*.json | jq '.evaluation_metrics'

# Expected output:
# {
#   "llm_eval_num_samples": 100,
#   "llm_eval_score": 0.82,
#   "llm_eval_avg_confidence": 0.76
# }
```

## Troubleshooting

### Issue: Connection refused

```
ConnectionError: Failed to connect to http://localhost:8001/v1
```

**Solution**: Ensure vLLM server is running and port is correct

```bash
# Check if server is running
curl http://localhost:8001/v1/models

# If not running, start it
python -m vllm.entrypoints.openai.api_server --port 8001 ...
```

### Issue: Model not found

```
ValueError: Model not found in vLLM server
```

**Solution**: Verify model is available and server has it loaded

```bash
# Check available models
curl http://localhost:8001/v1/models | jq '.data[].id'

# If model not loaded, restart server with correct model
```

### Issue: Out of memory during evaluation

```
RuntimeError: CUDA out of memory
```

**Solution**: Use smaller model or reduce batch size

```bash
# Reduce memory usage
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-7B-Chat \
  --gpu-memory-utilization 0.4

# Or use quantization
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-72B-Chat \
  --quantization awq
```

### Issue: Slow evaluation

**Solution**: Increase parallel workers and batch size

```python
# In config:
"evaluation_config": {
    "llm_max_workers": 8,  # Increase from 4
    # ... other params ...
}

# On server:
python -m vllm.entrypoints.openai.api_server \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384
```

## Complete Example

### Full Workflow with vLLM Evaluation

```bash
# Terminal 1: Start benchmark model server
python -m vllm.entrypoints.openai.api_server \
  --model OpenOneRec/OneRec-1.7B \
  --host localhost \
  --port 8000 \
  --gpu-memory-utilization 0.5

# Terminal 2: Start evaluation model server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-7B-Chat \
  --host localhost \
  --port 8001 \
  --gpu-memory-utilization 0.5

# Terminal 3: Run benchmark with vLLM evaluation
sleep 5  # Wait for servers to start
python -m vllm.benchmarks.throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --data-dir ./data \
  --endpoint http://localhost:8000/v1 \
  --backend openai-chat \
  --task-types rec_reason \
  --num-prompts 100 \
  --save-result \
  --result-dir ./results
```

Results with vLLM evaluation metrics will be saved to `./results/`.

## Comparison: vLLM vs External APIs

| Aspect | vLLM | Gemini | DeepSeek |
|--------|------|--------|----------|
| **Cost** | Free | Paid | Paid |
| **Latency** | Low (local) | High (API) | High (API) |
| **Model Choice** | Any vLLM model | Fixed | Fixed |
| **Customization** | Full | Limited | Limited |
| **Setup** | More complex | Simple | Simple |
| **Privacy** | Full (local) | Shared | Shared |
| **Throughput** | High | Limited by API | Limited by API |

## Related Documentation

- [README.md](README.md) - Quick start guide
- [OPERATION_MODES.md](OPERATION_MODES.md) - Running benchmarks
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
