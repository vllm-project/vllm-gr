# OneRec Configuration Guide

Complete reference for configuring OneRec benchmark tasks and generation parameters.

## Table of Contents

- [Task Configurations](#task-configurations)
- [Generation Parameters](#generation-parameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Custom Configuration](#custom-configuration)
- [Configuration Examples](#configuration-examples)

## Task Configurations

All task configurations are defined in `configs/config.py`. Each task has the following structure:

### Configuration Schema

```python
TASK_CONFIG = {
    # Task identification
    "name": str,                    # Unique task identifier
    "source": str,                  # Data source
    "description": str,             # Human-readable description

    # Dataset information
    "splits": list[str],            # Available splits (train, test, val)
    "size": int,                    # Total dataset size
    "sample_size": int,             # Default sample size

    # Data structure
    "data_fields": {
        "messages_field": str,      # Column name for messages
        "metadata_field": str,       # Column name for metadata
    },

    # Prompt handling
    "prompt_config": {
        "enable_thinking": bool,              # Enable reasoning mode
        "custom_chat_template": str,          # Path to Jinja2 template
    },

    # Generation settings
    "generation_config": {
        "max_new_tokens": int,                # Maximum generation length
        "temperature": float,                 # Sampling temperature
        "top_p": float,                       # Nucleus sampling
        "top_k": int,                         # Top-K sampling
        "do_sample": bool,                    # Use sampling vs greedy
        "num_return_sequences": int,          # Number of candidates
        "repetition_penalty": float,          # Penalize repetition
        "return_logprobs": bool,              # Return token logprobs
        "logprobs": int,                      # Number of logprobs
        "target_tokens": list[str],           # Tokens for classification
    },

    # Evaluation configuration
    "evaluation_config": {
        "metrics": list[str],                 # Metrics to compute
        "bertscore_model_type": str,          # BERTScore model
        "bertscore_num_layers": int,          # BERTScore layers
        "bertscore_lang": str,                # BERTScore language
        "wip_enabled": bool,                  # WIP evaluation enabled
        "wip_judge_model": str,               # Judge LLM
        "wip_max_workers": int,               # Concurrent workers
        "wip_core_threshold": int,            # Core importance
        "wip_max_samples": int,               # Sample limit
    },

    # Task metadata
    "task_type": str,                         # Type: generation/classification
}
```

### Task Configurations

#### 1. Label Prediction (`label_pred`)

Classification task for predicting user engagement.

```python
LABEL_PRED_CONFIG = {
    "name": "label_pred",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 346190,
    "sample_size": 346190,
    "description": "Predict user engagement with video content (yes/no classification)",

    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },

    "prompt_config": {
        "enable_thinking": False,
        "custom_chat_template": "qwen3_soft_switch.jinja2",
    },

    "generation_config": {
        "max_new_tokens": 1,
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "do_sample": True,
        "num_return_sequences": 1,
        "return_logprobs": True,
        "logprobs": 10000,  # Ensure target tokens are included
        "target_tokens": ["是", "否"],  # Binary classification
        "max_new_thinking_tokens": 1000,
    },

    "evaluation_config": {
        "metrics": ["auc", "weighted_auc"],
    },

    "task_type": "logprobs_classification",
}
```

**Key Settings**:
- `max_new_tokens: 1` - Output single token classification
- `return_logprobs: True` - Extract probability from logprobs
- `target_tokens` - Classes for classification
- `logprobs: 10000` - Return top logprobs to ensure coverage

#### 2. Item Understanding (`item_understand`)

Caption generation for video items.

```python
ITEM_UNDERSTAND_CONFIG = {
    "name": "item_understand",
    "size": 500,
    "sample_size": 500,
    "description": "Video SID to Caption generation task",

    "prompt_config": {
        "enable_thinking": False,
        "custom_chat_template": "qwen3_soft_switch.jinja2",
    },

    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 128,
        "temperature": 0.01,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "do_sample": False,
        "num_return_thinking_sequences": 1,
        "max_new_thinking_tokens": 1000,
    },

    "evaluation_config": {
        "metrics": [
            "macro_wip_double_weighted_f1",
            "micro_wip_double_weighted_f1",
        ],
        "bertscore_model_type": "bert-base-chinese",
        "bertscore_num_layers": 9,
        "bertscore_lang": "zh",
        "wip_enabled": True,
        "wip_judge_model": "gemini",
        "wip_max_workers": 1,
        "wip_core_threshold": 5,
        "wip_max_samples": 500,
    },
}
```

**Key Settings**:
- `temperature: 0.01` - More deterministic generation
- `do_sample: False` - Use greedy decoding
- `wip_enabled: True` - Use WIP evaluation
- `bertscore_lang: "zh"` - Chinese language support

#### 3. Recommendation Reason (`rec_reason`)

Natural language explanation for recommendations.

```python
REC_REASON_CONFIG = {
    "name": "rec_reason",
    "size": 2000,
    "sample_size": 2000,
    "description": "Generate reasons for item recommendations",

    "generation_config": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.05,
    },

    "evaluation_config": {
        "metrics": ["bleu", "rouge_1", "rouge_2", "rouge_l", "bertscore_f1"],
        "bertscore_lang": "zh",
    },
}
```

#### 4-7. Other Tasks

Similar configurations for:
- `AD_CONFIG` - Ad engagement prediction
- `PRODUCT_CONFIG` - Product recommendation
- `LABEL_COND_CONFIG` - Conditional label prediction
- `VIDEO_CONFIG` - Video recommendation
- `INTERACTIVE_CONFIG` - Interactive engagement

## Generation Parameters

### Common Generation Parameters

```python
# Sampling control
{
    "temperature": 0.8,          # 0 = greedy, 1.0 = uniform, > 1 = more random
    "top_k": 50,                 # Consider only top-K tokens
    "top_p": 0.95,               # Nucleus sampling threshold
    "do_sample": True,           # Use sampling vs greedy
    "repetition_penalty": 1.0,   # Penalize repeated tokens (> 1.0)
}

# Output control
{
    "max_new_tokens": 128,           # Maximum generation length
    "num_return_sequences": 1,       # Number of candidates
    "min_length": 1,                 # Minimum generation length
    "length_penalty": 1.0,           # Length normalization
}

# Logprobs (for classification)
{
    "return_logprobs": True,    # Return token probabilities
    "logprobs": 10000,          # Return top-10000 logprobs
    "target_tokens": ["是", "否"], # Tokens to extract probabilities for
}
```

### Preset Configurations

#### Greedy Decoding (Deterministic)

```python
{
    "temperature": 0.01,
    "top_p": 1.0,
    "do_sample": False,
    "max_new_tokens": 128,
}
```

**Use case**: Classification, factual tasks requiring consistency

#### Standard Sampling

```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 256,
}
```

**Use case**: Generation tasks, creative content

#### High Diversity

```python
{
    "temperature": 1.2,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 512,
}
```

**Use case**: Brainstorming, exploring solution space

#### Beam Search

```python
{
    "num_beams": 4,
    "temperature": 1.0,     # Ignored when using beams
    "do_sample": False,
    "max_new_tokens": 256,
    "length_penalty": 1.0,
}
```

**Use case**: Quality-critical generation

## Evaluation Metrics

### Supported Metrics

| Metric | Task Types | Range | Notes |
|--------|-----------|-------|-------|
| **BLEU** | Generation | 0-1 | N-gram overlap (1-4) |
| **ROUGE** | Generation | 0-1 | Recall-oriented metrics (ROUGE-1/2/L) |
| **BERTScore** | Generation | 0-1 | Contextual similarity |
| **F1** | Classification | 0-1 | Harmonic mean of precision/recall |
| **AUC** | Classification | 0-1 | Area under ROC curve |
| **Weighted AUC** | Classification | 0-1 | Class-weighted AUC |
| **WIP** | Complex | Custom | Weighted Information Points (LLM-based) |
| **Accuracy** | Classification | 0-1 | Exact match rate |

### Metric Computation

#### BLEU

```python
# Configuration
{
    "metrics": ["bleu"],
    "bleu_weights": [0.25, 0.25, 0.25, 0.25],  # BLEU-1 to BLEU-4
}

# Output range: 0-1 (higher is better)
# Measures N-gram overlap between generated and reference
```

#### ROUGE

```python
{
    "metrics": ["rouge_1", "rouge_2", "rouge_l"],
}

# ROUGE-1: Unigram overlap
# ROUGE-2: Bigram overlap  
# ROUGE-L: Longest common subsequence
```

#### BERTScore

```python
{
    "metrics": ["bertscore_f1"],
    "bertscore_model_type": "bert-base-chinese",
    "bertscore_num_layers": 9,
    "bertscore_lang": "zh",
}

# Uses contextual embeddings from BERT
# Better correlation with human judgment
```

#### WIP Scoring

```python
{
    "wip_enabled": True,
    "wip_judge_model": "gemini",      # Judge LLM: gemini/deepseek/claude
    "wip_max_workers": 4,              # Parallel evaluation
    "wip_core_threshold": 5,           # Importance threshold (1-5)
    "wip_max_samples": 1000,           # Sample limit
}

# Uses LLM-based evaluation
# Weighted by information importance
# Slower but more flexible
```

## Custom Configuration

### Modifying Existing Tasks

```python
# Method 1: Update configuration directly
import sys
sys.path.insert(0, './benchmarks')

from open_one_rec.configs.config import REC_REASON_CONFIG

# Modify configuration
REC_REASON_CONFIG["generation_config"]["temperature"] = 0.5
REC_REASON_CONFIG["evaluation_config"]["metrics"] = ["bleu", "rouge_1"]
```

### Creating Custom Task Configuration

```python
MY_CUSTOM_CONFIG = {
    "name": "my_custom_task",
    "size": 1000,
    "sample_size": 100,
    "description": "My custom task",

    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },

    "prompt_config": {
        "enable_thinking": False,
        "custom_chat_template": "my_template.jinja2",
    },

    "generation_config": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    },

    "evaluation_config": {
        "metrics": ["bleu", "rouge_1", "bertscore_f1"],
    },
}

# Register in task registry
from open_one_rec.open_one_rec_loader import TASK_REGISTRY, TaskRegistration
from open_one_rec.evaluators.recommendation import RecommendationEvaluator

TASK_REGISTRY["my_custom_task"] = TaskRegistration(
    name="my_custom_task",
    config=MY_CUSTOM_CONFIG,
    evaluator_class=RecommendationEvaluator,
    category="custom",
)
```

### Runtime Configuration Override

```python
# Using command line
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types rec_reason \
  --num-prompts 100 \
  --output-json results/throughput  
  --n 8 # Override beam count

# In Python
from benchmarks.open_one_rec.configs.config import LABEL_PRED_CONFIG

LABEL_PRED_CONFIG["generation_config"].update({
    "temperature": 0.5,
    "max_new_tokens": 64,
})
```

## Configuration Examples

### Example 1: Fast Baseline

For quick benchmarking with reduced quality:

```python
# Command line
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types rec_reason \
  --num-prompts 100 \
  --output-json results/throughput  
  --max-tokens 64 \
  --temperature 0.01

# Results: Fast (~5 min), deterministic, lower quality
```

### Example 2: High-Quality Evaluation

For comprehensive benchmarking:

```python
# Command line
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-8B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types rec_reason \
  --use-beam-search \
  --n 256 \
  --num-prompts 500 \
  --max-tokens 256

# Results: Slower yet higher quality, beam search
```

### Example 3: Multi-task with Thinking

For models supporting thinking mode:

```python
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model Qwen/Qwen3-32B-Instruct \
  --dataset-path ./data \
  --task-types rec_reason,item_understand,label_pred \
  --num-prompts 200

# Python code to enable thinking:
import sys
sys.path.insert(0, './benchmarks')

from open_one_rec.configs.config import REC_REASON_CONFIG
REC_REASON_CONFIG["prompt_config"]["enable_thinking"] = True
```

### Example 4: Classification-Focused

For evaluating classification tasks:

```python
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model OpenOneRec/OneRec-1.7B \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types label_pred,ad,label_cond \
  --num-prompts 100 \
  --output-json results/throughput  

# These tasks use:
# - Low max_new_tokens (1-10)
# - Logprobs extraction
# - Classification metrics (AUC, accuracy)
```

### Example 5: Custom Generation Settings

Creating a config with custom parameters:

```python
# config.py
CUSTOM_GENERATION_CONFIG = {
    "name": "rec_reason_custom",
    "size": 2000,
    "sample_size": 200,
    "description": "Rec reason with custom generation",

    "prompt_config": {
        "enable_thinking": True,
        "custom_chat_template": "qwen3_soft_switch.jinja2",
    },

    "generation_config": {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "max_new_thinking_tokens": 2000,
    },

    "evaluation_config": {
        "metrics": ["bleu", "rouge_1", "rouge_l", "bertscore_f1"],
        "bertscore_lang": "zh",
    },
}
```

## Best Practices

1. **Start with defaults** - Use provided configurations as baselines
2. **Profile performance** - Benchmark with different settings
3. **Match task type** - Use appropriate metrics for task category
4. **Consider hardware** - Adjust max_tokens based on GPU memory
5. **Enable thinking carefully** - Only for models that support it
6. **Validate results** - Check metric ranges and distributions
7. **Document changes** - Comment custom configurations
