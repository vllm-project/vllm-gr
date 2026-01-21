# OneRec Troubleshooting Guide

Comprehensive troubleshooting guide for common issues in OneRec benchmark.

## Quick Diagnostics

Run this script to diagnose your setup:

```python
#!/usr/bin/env python
"""OneRec setup diagnostics."""

import sys
import importlib.util
from pathlib import Path

def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if module can be imported."""
    pkg = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✓ {pkg}")
        return True
    except ImportError as e:
        print(f"✗ {pkg}: {e}")
        return False

def check_file(path: str) -> bool:
    """Check if file exists."""
    if Path(path).exists():
        print(f"✓ {path}")
        return True
    else:
        print(f"✗ {path} (missing)")
        return False

print("=" * 60)
print("OneRec Setup Diagnostics")
print("=" * 60)

# Check Python version
print(f"\nPython: {sys.version}")

# Check dependencies
print("\n--- Dependencies ---")
checks = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("pandas", "Pandas"),
    ("pyarrow", "PyArrow"),
    ("vllm", "vLLM"),
]

for module, name in checks:
    check_import(module, name)

# Check OneRec modules
print("\n--- OneRec Modules ---")
onerec_modules = [
    "benchmarks.open_one_rec",
    "benchmarks.open_one_rec.open_one_rec_dataset",
    "benchmarks.open_one_rec.open_one_rec_loader",
    "benchmarks.open_one_rec.configs.config",
]

for module in onerec_modules:
    check_import(module)

# Check data files
print("\n--- Data Files ---")
data_dir = Path("./data")
if data_dir.exists():
    print(f"✓ Data directory: {data_dir}")
    tasks = [
        "rec_reason", "item_understand", "ad", "product",
        "label_cond", "video", "interactive", "label_pred"
    ]
    for task in tasks:
        parquet_file = data_dir / task / f"{task}_test.parquet"
        if parquet_file.exists():
            print(f"  ✓ {task}")
        else:
            print(f"  ✗ {task} (missing)")
else:
    print(f"✗ Data directory not found: {data_dir}")

# Check models
print("\n--- Model Access ---")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen/qwen-72b-instruct",
        trust_remote_code=True
    )
    print("✓ Model accessible (qwen/qwen-72b-instruct)")
except Exception as e:
    print(f"⚠ Model download failed: {e}")
    print("  (OK if running on air-gapped system)")

print("\n" + "=" * 60)
print("Diagnostics complete")
print("=" * 60)
```

Run it:
```bash
python diagnose.py
```

## Common Issues & Solutions

### 1. Import Errors

#### Issue: `ModuleNotFoundError: No module named 'benchmarks'`

```
ModuleNotFoundError: No module named 'benchmarks'
```

**Cause**: Python path doesn't include vllm-gr directory

**Solution**:
```bash
# Method 1: Install package
cd vllm-gr
pip install -e .

# Method 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/data/users/sefi/from_git/vllm_013/vllm-gr"

# Method 3: Run from project root
cd /data/users/sefi/from_git/vllm_013/vllm-gr
python -m benchmarks.open_one_rec.one_rec_main
```

#### Issue: `ModuleNotFoundError: No module named 'vllm'`

```
ModuleNotFoundError: No module named 'vllm'
```

**Solution**: Install vLLM
```bash
pip install vllm
# Or with GPU support
pip install vllm[cuda]  # Specify CUDA version
```

#### Issue: `ImportError: cannot import name 'SampleRequest'`

**Solution**: Update vLLM version
```bash
pip install --upgrade vllm
```

### 2. Data-Related Issues

#### Issue: `FileNotFoundError: Data file not found`

```
FileNotFoundError: Data file not found: ./data/rec_reason/rec_reason_test.parquet
```

**Cause**: Data files not in expected location

**Solution**:
```bash
# Check data structure
ls -R ./data/

# Expected output:
# ./data/
# ├── rec_reason/
# │   └── rec_reason_test.parquet
# ├── item_understand/
# │   └── item_understand_test.parquet
# ...

# If missing, create sample data
python -c "
from benchmarks.open_one_rec.tests.create_sample_data import create_all_samples
create_all_samples('./data')
"
```

#### Issue: `ValueError: metadata missing 'answer' field`

```
ValueError: metadata missing 'answer' field
```

**Cause**: Parquet data doesn't have required "answer" in metadata

**Solution**:
```python
import pandas as pd
import json

def fix_metadata(parquet_path: str):
    df = pd.read_parquet(parquet_path)

    # Ensure all rows have 'answer'
    def fix_row(meta_str):
        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        if "answer" not in meta:
            meta["answer"] = "N/A"
        return json.dumps(meta)

    df["metadata"] = df["metadata"].apply(fix_row)
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Fixed: {parquet_path}")

# Fix all data files
import os
for task in ["rec_reason", "item_understand", ...]:
    path = f"./data/{task}/{task}_test.parquet"
    if os.path.exists(path):
        fix_metadata(path)
```

#### Issue: `ParquetInvalidError: File is not a Parquet file`

```
ParquetInvalidError: File is not a Parquet file
```

**Cause**: File is corrupted or wrong format

**Solution**:
```bash
# Check file type
file ./data/rec_reason/rec_reason_test.parquet

# If not Parquet, convert from source
python -c "
import pandas as pd
# Convert CSV to Parquet
df = pd.read_csv('./data/rec_reason/rec_reason_test.csv')
df.to_parquet('./data/rec_reason/rec_reason_test.parquet', index=False)
"
```

### 3. Model & Tokenizer Issues

#### Issue: `OSError: Can't load model`

```
OSError: Can't load model 'qwen/qwen-72b-instruct'. Model not found on huggingface.co
```

**Cause**: Model not accessible or wrong model ID

**Solution**:
```bash
# Verify model exists
python -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('Qwen/Qwen-72B-Chat')
    print('✓ Model found')
except Exception as e:
    print(f'✗ Model not found: {e}')
"

# Use correct model ID
# Common models:
# - meta-llama/Llama-2-70b-chat-hf
# - Qwen/Qwen-72B-Chat
# - mistralai/Mixtral-8x7B-Instruct-v0.1
```

#### Issue: `RuntimeError: CUDA out of memory`

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause**: Model too large for GPU

**Solution**:
```bash
# Method 1: Reduce batch size
python -m vllm.benchmarks.throughput \
  --num-prompts 100 \
  --max-tokens 64

# Method 2: Use smaller model
--model meta-llama/Llama-2-13b-chat-hf

# Method 3: Enable quantization
--quantization awq  # or gptq, squeezellm

# Method 4: Use multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Method 5: Check GPU memory
python -c "
import torch
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

#### Issue: `ValueError: Trust remote code is not enabled`

```
ValueError: Trust remote code is not enabled for ... but found remote code
```

**Solution**:
```bash
# Add trust_remote_code flag
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen-72B-Chat',
    trust_remote_code=True
)
"
```

### 4. Generation & Inference Issues

#### Issue: `ValueError: Temperature must be > 0`

**Cause**: Invalid generation parameters

**Solution**:
```bash
# Use valid temperature
--temperature 0.7  # Should be > 0

# If want deterministic, use:
--temperature 0.01  # Very small, close to greedy
```

#### Issue: `RuntimeError: Tensor sizes don't match`

**Cause**: Tokenizer mismatch

**Solution**:
```bash
# Verify tokenizer matches model
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-72B-Chat', trust_remote_code=True)
print(f'Vocab size: {len(tokenizer)}')
"

# Ensure model and tokenizer are compatible
# They should come from same source
```

### 5. Evaluation Issues

#### Issue: `KeyError: 'logprobs' not in output`

**Cause**: Model doesn't return logprobs

**Solution**:
```bash
# Use OpenAI-compatible endpoint
--endpoint http://localhost:8000/v1

# vLLM server supports logprobs
vllm serve Qwen/Qwen-72B-Chat --gpu-memory-utilization 0.8
```

#### Issue: `ValueError: BERTScore model not found`

```
ValueError: Model bert-base-chinese not found
```

**Solution**:
```bash
# Download manually
python -c "
from bert_score import score
score(['test'], ['test'], lang='zh', model_type='bert', num_layers=9)
"

# Or change metric
# In config.py, use different model_type
```

### 6. Command-line Issues

#### Issue: `error: unrecognized arguments: --task-types`

**Cause**: Argument not recognized by vLLM

**Solution**:
```bash
# Make sure one_rec_patch is loaded
# Either:
# 1. Run through benchmark entry point
python benchmarks/open_one_rec/one_rec_main.py --task-types rec_reason

# 2. Or import patch before running
python -c "
import benchmarks.open_one_rec.one_rec_patch
from vllm.entrypoints.cli.main import main
# Then run normally
"
```

#### Issue: `error: argument --dataset-name: invalid choice: 'onerec'`

**Cause**: OneRec dataset not registered

**Solution**:
```bash
# Ensure patch is applied before parsing args
# Check that __init__.py imports the patch

# File: benchmarks/open_one_rec/__init__.py
# Should contain:
# from . import one_rec_patch
```

### 7. Performance Issues

#### Issue: Benchmark running very slowly

**Cause**: Many reasons - check logs

**Solution**:
```bash
# Profile the benchmark
python -m cProfile -s cumtime \
  -m benchmarks.open_one_rec.one_rec_main bench throughput \ \
  --dataset-name onerec \
  --model qwen/qwen-72b \
  --dataset-path ./data \
  --num-prompts 10

# Check GPU utilization
nvidia-smi dmon -s p  # In separate terminal

# Reduce sample size for testing
--num-prompts 10 --max-tokens 64
```

#### Issue: Memory leak - memory grows over time

**Solution**:
```python
# Add cleanup in code
import gc
import torch

for i in range(num_iterations):
    # Your code here
    gc.collect()
    torch.cuda.empty_cache()
```

### 8. Configuration Issues

#### Issue: `KeyError: 'generation_config'` in task config

**Cause**: Task config missing required section

**Solution**:
```python
# Verify config structure
from benchmarks.open_one_rec.configs.config import REC_REASON_CONFIG

required_keys = [
    "name", "description", "size",
    "data_fields", "prompt_config",
    "generation_config", "evaluation_config"
]

for key in required_keys:
    assert key in REC_REASON_CONFIG, f"Missing: {key}"

print("✓ Config valid")
```

#### Issue: Custom config not applied

**Solution**:
```python
# Modify before dataset creation
from benchmarks.open_one_rec.configs.config import REC_REASON_CONFIG

# Store original
original_temp = REC_REASON_CONFIG["generation_config"]["temperature"]

# Modify
REC_REASON_CONFIG["generation_config"]["temperature"] = 0.5

# Create dataset - will use modified config
from benchmarks.open_one_rec import OneRecDataset
dataset = OneRecDataset(...)

# Restore if needed
REC_REASON_CONFIG["generation_config"]["temperature"] = original_temp
```

## Debug Mode

### Enable Verbose Logging

```bash
export LOGLEVEL=DEBUG
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model qwen/qwen-72b \
  --dataset-path ./data \
  -v  # Verbose flag
```

### Python Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from benchmarks.open_one_rec import OneRecDataset

dataset = OneRecDataset(...)  # Will print debug info
```

### Test Individual Components

```python
# Test data loading
print("Testing data loading...")
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader
from benchmarks.open_one_rec.configs.config import REC_REASON_CONFIG

loader = BaseLoader(
    task_config=REC_REASON_CONFIG,
    data_dir="./data",
    tokenizer=None
)

# Test dataset
print("Testing dataset...")
from benchmarks.open_one_rec import OneRecDataset
dataset = OneRecDataset(
    task_types=["rec_reason"],
    model_path="qwen/qwen-72b-instruct",
    dataset_path="./data",
    sample_size=10
)
print(f"Loaded {len(dataset.data)} samples")

# Test sampling
print("Testing sampling...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-72b-instruct", trust_remote_code=True)
requests = dataset.sample(tokenizer, num_requests=5)
print(f"Sampled {len(requests)} requests")
```

## Getting Help

### Information to Provide

When reporting issues, include:

1. **Environment**:
   ```bash
   python --version
   pip list | grep -E "vllm|torch|transformers"
   nvidia-smi
   ```

2. **Error Message**: Full traceback

3. **Reproduction Steps**: Minimal code to reproduce

4. **Logs**: Output with `LOGLEVEL=DEBUG`

### Useful Commands for Debugging

```bash
# Check file permissions
ls -la ./data/rec_reason/

# Validate data
python -c "
import pandas as pd
df = pd.read_parquet('./data/rec_reason/rec_reason_test.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'First row:\\n{df.iloc[0]}')
"

# Check GPU
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits

# Check available models
huggingface-cli repo_list
```

## Related Documentation

- [README.md](README.md) - Quick start
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [DATA_FORMAT.md](DATA_FORMAT.md) - Data format
- [API_REFERENCE.md](API_REFERENCE.md) - Python API
