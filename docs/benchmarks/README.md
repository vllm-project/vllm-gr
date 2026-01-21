# OneRec Benchmark Documentation

A comprehensive generative recommendation benchmark for evaluating large language models on multiple recommendation and understanding tasks.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Operation Modes](#operation-modes)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Task Types](#task-types)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Operation Modes

OneRec supports two operation modes with different tradeoffs:

**📘 [OPERATION_MODES.md](OPERATION_MODES.md)** - Comprehensive guide including:
- **Mode 1: Throughput Mode** - Simple local benchmarking (1 command)
- **Mode 2: API Server Mode** - Production-like testing (2 terminals)
- Detailed command reference for both modes
- When to use each mode
- launch.json configuration examples
- Results monitoring and analysis

## Overview

OneRec is a benchmark suite designed to evaluate generative recommendation models across 8 distinct task types:

| Task Type | Category | Description | Sample Size |
|-----------|----------|-------------|------------|
| `rec_reason` | Recommendation | Generate reasons for item recommendations | 2000 |
| `item_understand` | Understanding | Generate captions for video items | 500 |
| `ad` | Recommendation | Predict ad engagement | 500 |
| `product` | Recommendation | Product recommendation reasoning | 1000 |
| `label_cond` | Recommendation | Conditional label prediction | 800 |
| `video` | Recommendation | Video recommendation scoring | 1500 |
| `interactive` | Recommendation | Interactive engagement prediction | 600 |
| `label_pred` | Classification | User engagement classification | 346190 |

### Key Features

- **Multi-task evaluation**: Benchmark models across diverse recommendation scenarios
- **Flexible task selection**: Run all or specific task types
- **Chat template support**: Custom prompt formatting with Jinja2 templates
- **Thinking mode**: Optional extended reasoning for capable models (e.g., Qwen3)
- **Comprehensive metrics**: Task-specific evaluation (AUC, F1, BERTScore, WIP scoring)
- **Beam search support**: Configurable beam search parameters for generation

## Quick Start

OneRec supports two operation modes. Choose based on your needs:

### Mode 1: Throughput Mode (Simple, Local)

For quick local benchmarking without an API server:

```bash
# Single command - everything in one process
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100
```

**Best for**: Quick testing, development, maximizing throughput

### Mode 2: API Server Mode (Production-like)

For production-like testing with separate server and client:

```bash
# Terminal 1: Start vLLM API server
  vllm serve \
  --gr \
  --model OpenOneRec/OneRec-1.7B

# Terminal 2: Run benchmark against server
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint http://localhost:8000/v1 \
  --model OpenOneRec/OneRec-1.7B \
  --backend openai-chat \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100
```

**Best for**: Production-like testing, distributed deployment, load testing

### With Beam Search

Both modes support beam search:

```bash
# Throughput mode with beam search
python -m  benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100 \
  --use-beam-search \
  --n 8 \
  --output-json results/throughput  


# API mode with beam search (requires API server)
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint http://localhost:8000/v1 \
  --model OpenOneRec/OneRec-1.7B \
  --backend openai-chat \
  --dataset-name onerec \
  --dataset-path ./data \  
  --task-types video \
  --num-prompts 100 \
  --use-beam-search \
  --n 8 \
  --save-result \
  --result-dir results/serve \  
  --save-detailed  
```

### Operation Modes Comparison

| Aspect | Throughput Mode | API Server Mode |
|--------|---|---|
| Setup | 1 command | 2 terminals |
| Latency | Lower | Higher (+HTTP) |
| Throughput | Higher | Lower |
| Production-like | No | Yes |
| Network overhead | None | Yes (HTTP) |
| Use case | Local testing | Deployment simulation |

**See [OPERATION_MODES.md](OPERATION_MODES.md) for detailed mode documentation.**

## Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU inference)
- 32GB+ RAM (recommended)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Installation**

```bash
pip install -e .
```

4. **Verify installation**

```bash
python -c "from benchmarks.open_one_rec import OneRecDataset; print('✓ Installation successful')"
```


##  Open One Rec Benchmark Dataset
### Get Dataset
Use your huggingface username to approve this link before you download:

https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF

```bash
mkdir -p OpenOneRec  
cd OpenOneRec
git lfs install  
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
cd -
```

### Data Structure

OneRec expects data in **Parquet format** with the following schema:

```
├── data/
│   ├── rec_reason/
│   │   └── rec_reason_test.parquet
│   ├── item_understand/
│   │   └── item_understand_test.parquet
│   └── ...
```

### Parquet Schema

Each Parquet file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `messages` | str/list | Chat messages in JSON format or list of dicts |
| `metadata` | str/dict | JSON metadata containing answer and context |

## Usage

### Command Line Arguments

#### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-name` | str | - | Must be `onerec` for OneRec benchmark |
| `--model` | str | - | Model ID or path to load |
| `--dataset-path` | str | None | Path to benchmark data directory |
| `--num-prompts` | int | 1000 | Number of prompts to sample |
| `--disable-shuffle` | bool | False | Disable shuffling of samples |

#### OneRec Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task-types` | str | All 8 tasks | Comma-separated list of task types to run |
| `--use-beam-search` | bool | False | Enable beam search during generation |
| `--n` | int | 8 | Number of beams for beam search |

#### Generation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | int | 128 | Maximum tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top-p` | float | 0.95 | Nucleus sampling parameter |


## Configuration

### Task Configuration Files

Each task has a dedicated configuration file in `configs/config.py` with the following structure:

```python
TASK_CONFIG = {
    "name": "task_name",
    "source": "data source",
    "splits": ["train", "test"],
    "size": 1000,
    "sample_size": 100,
    "description": "Task description",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,
        "custom_chat_template": "template.jinja2",
    },
    "generation_config": {
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "evaluation_config": {
        "metrics": ["accuracy", "f1"],
    },
}
```

### Custom Configuration

To use custom settings, modify the configuration in your benchmark launch script:

```python
from benchmarks.open_one_rec import OneRecDataset

dataset = OneRecDataset(
    task_types=["rec_reason", "item_understand"],
    model_path="qwen/qwen-72b-instruct",
    dataset_path="./data",
    tokenizer=tokenizer,
    enable_thinking=True,  # Enable thinking mode
)
```

## API Server Interactions
### Message Format

Messages follow the OpenAI chat format:

```json
[
  {
    "role": "system",
    "content": "You are a helpful recommendation assistant."
  },
  {
    "role": "user",
    "content": "Recommend a video based on user history..."
  }
]
```

Or with multi-modal content:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this video"},
      {"type": "image_url", "image_url": {"url": "..."}}
    ]
  }
]
```

### Metadata Format

```json
{
  "answer": "Expected model output",
  "user_id": 12345,
  "item_id": 67890,
  "context": "Additional context",
  "ground_truth": "Expected answer"
}
```

### Example Data Row

```python
{
    "messages": '[{"role": "user", "content": "Recommend a video"}]',
    "metadata": '{"answer": "Video ID: ABC123", "user_id": 1001, "item_id": 501}'
}
```

### Creating Sample Data

```python
import pandas as pd
import json

def create_sample_data(output_path: str, num_samples: int = 100):
    """Create sample OneRec data."""
    records = []

    for i in range(num_samples):
        record = {
            "messages": json.dumps([
                {"role": "user", "content": f"Sample prompt {i}"}
            ]),
            "metadata": json.dumps({
                "answer": f"Sample answer {i}",
                "user_id": i,
                "item_id": i * 10,
            })
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"✓ Created sample data: {output_path}")

# Usage
create_sample_data("./data/rec_reason/rec_reason_test.parquet", 100)
```

## Task Types

### 1. Recommendation Reason (`rec_reason`)

Generate natural language explanations for why a specific item is recommended.

- **Category**: Recommendation
- **Sample Size**: 2000
- **Output**: Recommendation reason text
- **Metrics**: BLEU, ROUGE, BERTScore

```python
# Example input
{
    "user_history": ["Video A", "Video B"],
    "item": "Video C",
    "task": "Generate recommendation reason"
}

# Expected output
"This video combines elements from your favorite categories..."
```

### 2. Item Understanding (`item_understand`)

Generate descriptive captions for video items based on understanding.

- **Category**: Understanding
- **Sample Size**: 500
- **Output**: Video caption/description
- **Metrics**: F1, BERTScore, WIP scoring
- **Features**: Thinking mode support

```python
# Example input
{
    "item_features": {...},
    "context": "Video understanding"
}

# Expected output
"A product review video demonstrating key features..."
```

### 3. Ad Engagement (`ad`)

Predict and explain ad engagement for user interactions.

- **Category**: Recommendation
- **Sample Size**: 500
- **Output**: Engagement prediction
- **Metrics**: AUC, accuracy

### 4. Product Recommendation (`product`)

Generate product recommendations with reasoning.

- **Category**: Recommendation
- **Sample Size**: 1000
- **Output**: Product ID + recommendation reason
- **Metrics**: Precision@K, NDCG

### 5. Label Conditional (`label_cond`)

Predict item labels based on conditional information.

- **Category**: Recommendation
- **Sample Size**: 800
- **Output**: Predicted label
- **Metrics**: Accuracy, F1

### 6. Video Recommendation (`video`)

Score and rank videos for recommendation.

- **Category**: Recommendation
- **Sample Size**: 1500
- **Output**: Video ranking/scores
- **Metrics**: NDCG, MAP

### 7. Interactive (`interactive`)

Predict engagement in interactive scenarios.

- **Category**: Recommendation
- **Sample Size**: 600
- **Output**: Engagement prediction
- **Metrics**: AUC, F1

### 8. Label Prediction (`label_pred`)

Binary classification of user engagement labels.

- **Category**: Classification
- **Sample Size**: 346190
- **Output**: Binary prediction (yes/no)
- **Metrics**: AUC, weighted AUC
- **Special**: Uses logprobs-based classification

```python
# Example
# Expected output: "是" (yes) or "否" (no)
# Extracted from model logprobs
```

## Evaluation

### Evaluation Workflow

1. **Load Data**: Benchmark loads samples from Parquet files
2. **Generate Responses**: Model generates outputs using configured parameters
3. **Extract Metrics**: Task-specific evaluators compute metrics
4. **Save Results**: Results saved to JSON files

### Evaluator Types

| Evaluator | Tasks | Metrics |
|-----------|-------|---------|
| `RecommendationEvaluator` | rec_reason, ad, product, label_cond, video, interactive | BLEU, ROUGE, BERTScore |
| `ItemUnderstandEvaluator` | item_understand | F1, BERTScore, WIP |
| `RecoReasonEvaluator` | rec_reason | ROUGE, customized metrics |
| `LabelPredEvaluator` | label_pred | AUC, weighted AUC |

### Running Evaluation

```bash
# Evaluation runs automatically after generation
# Results saved to output-json

python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --num-prompts 100 \
  --output-json results/throughput  
```

### Result Format

```json
{
  "model": "qwen/qwen-72b",
  "task_type": "rec_reason",
  "metrics": {
    "bleu": 0.45,
    "rouge_1": 0.52,
    "bertscore_f1": 0.68,
    "average": 0.55
  },
  "samples": 1000,
  "timestamp": "2026-01-20T10:30:00"
}
```

## Troubleshooting

### Common Issues

#### 1. Data Not Found

```
FileNotFoundError: Data file not found: ./data/rec_reason/rec_reason_test.parquet
```

**Solution**: Ensure data directory structure matches task names:
```bash
ls -R ./data/
# Output should show:
# ./data/rec_reason/rec_reason_test.parquet
# ./data/item_understand/item_understand_test.parquet
# ...
```

#### 2. Tokenizer Not Found

```
RuntimeError: Failed to load tokenizer from /path/to/model
```

**Solution**: Verify model path and check tokenizer compatibility:
```bash
python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('your-model-id', trust_remote_code=True)"
```

#### 3. Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**: Reduce batch size and max tokens:
```bash
--num-prompts 100 \
--max-tokens 64 \
--dtype float16
```

#### 4. Beam Search Not Working

```
extra_body parameters ignored
```

**Solution**: Use vLLM serving endpoint instead of local inference:
```bash
# Start server
vllm serve qwen/qwen-72b-instruct --gpu-memory-utilization 0.9

# Use endpoint
--endpoint http://localhost:8000/v1
```

### Debug Mode

Enable verbose logging:

```bash
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types rec_reason,item_understand,label_pred \
  --num-prompts 100 \
  --use-beam-search \
  --n 8 \
  --output-json results/throughput
  -v  # Verbose logging
```

### Validate Setup

```bash
# Test imports
python -c "
from benchmarks.open_one_rec import OneRecDataset
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader
print('✓ All imports successful')
"

# Test data loading
python -c "
from benchmarks.open_one_rec import OneRecDataset
dataset = OneRecDataset(
    task_types=['rec_reason'],
    model_path='qwen/qwen-72b-instruct',
    dataset_path='./data'
)
print(f'✓ Loaded {len(dataset.data)} samples')
"
```

## API Reference

### OneRecDataset

```python
from benchmarks.open_one_rec import OneRecDataset

class OneRecDataset(BenchmarkDataset):
    """
    OneRec benchmark dataset.

    Args:
        task_types: List of task types to load (default: all 8)
        model_path: Path to model for tokenizer
        dataset_path: Path to data directory
        tokenizer: Optional tokenizer instance
        disable_shuffle: Disable data shuffling
        sample_size: Number of samples per task
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        task_types: list[str] = None,
        model_path: str = None,
        dataset_path: str = None,
        tokenizer = None,
        **kwargs
    ):
        pass

    def load_data(self) -> None:
        """Load data from disk."""
        pass

    def sample(
        self,
        tokenizer,
        num_requests: int,
        **kwargs
    ) -> list:
        """Sample requests from dataset."""
        pass
```

### BaseLoader

```python
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader

class BaseLoader:
    """
    Base data loader for OneRec tasks.

    Handles:
    - Parquet file loading
    - Message format conversion
    - Chat template application
    - Metadata processing
    """

    def load_data(
        self,
        split: str = "test",
        sample_size: int = None
    ) -> dict:
        """Load and process data."""
        pass
```

### Task Registry

```python
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY

# Access task configuration
rec_reason_config = TASK_REGISTRY["rec_reason"].config

# List all registered tasks
print(TASK_REGISTRY.keys())
# Output: ['rec_reason', 'item_understand', 'ad', ...]
```

## Contributing

To add a new task type:

1. **Create evaluator**: `benchmarks/open_one_rec/evaluators/my_task.py`
2. **Add configuration**: Add entry to `TASK_REGISTRY` in `configs/config.py`
3. **Register task**: Add task name to `TASK_TYPES` list

## License

This benchmark is built upon [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec), which is licensed under Apache 2.0.

## Contact & Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/JiusiServe/vllm-gr/issues).
