# OneRec API Reference

Complete Python API documentation for OneRec benchmark components.

## Table of Contents

- [OneRecDataset](#onerecdata)
- [BaseLoader](#baseloader)
- [TaskRegistry](#taskregistry)
- [Evaluators](#evaluators)
- [Utilities](#utilities)

## OneRecDataset

Main dataset class for loading and sampling OneRec data.

### Class Definition

```python
from benchmarks.open_one_rec import OneRecDataset
from vllm.benchmarks.datasets import BenchmarkDataset

class OneRecDataset(BenchmarkDataset):
    """
    Custom dataset for OneRec benchmark with multiple task types.
    Loads data using DataLoaderWrapper with task_name, split, and sample_size.
    
    Inherits from: vllm.benchmarks.datasets.BenchmarkDataset
    """
```

### Constructor

```python
def __init__(
    self,
    task_types: list[str] | None = None,
    model_path: str | None = None,
    dataset_path: str | None = None,
    tokenizer: TokenizerLike | None = None,
    disable_shuffle: bool = False,
    sample_size: int | None = None,
    random_seed: int = 0,
    **kwargs
) -> None:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_types` | list[str] | All 8 tasks | Task types to load (e.g., ["rec_reason", "item_understand"]) |
| `model_path` | str | None | Model ID/path for tokenizer loading |
| `dataset_path` | str | None | Path to data directory containing task subdirectories |
| `tokenizer` | TokenizerLike | None | Tokenizer instance (loaded from model_path if not provided) |
| `disable_shuffle` | bool | False | Whether to disable data shuffling |
| `sample_size` | int | None | Override sample size from task config |
| `random_seed` | int | 0 | Seed for reproducibility |

#### Returns

- **None**: Initializes dataset, automatically loads data

#### Raises

- **ValueError**: If model_path not provided and tokenizer None
- **FileNotFoundError**: If data files not found at dataset_path

#### Example

```python
# Basic usage
dataset = OneRecDataset(
    model_path="qwen/qwen-72b-instruct",
    dataset_path="./data"
)

# Specific tasks
dataset = OneRecDataset(
    task_types=["rec_reason", "item_understand"],
    model_path="meta-llama/llama-2-70b-chat-hf",
    dataset_path="./data",
    disable_shuffle=True,
    sample_size=100
)

# With custom tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-72b-instruct")
dataset = OneRecDataset(
    task_types=["label_pred"],
    tokenizer=tokenizer,
    dataset_path="./data"
)
```

### Methods

#### load_data

```python
def load_data(self) -> None:
    """
    Load data for all specified task types.
    
    Processes:
    - Loads Parquet files for each task
    - Converts to standardized format
    - Shuffles if enabled
    
    Raises:
        FileNotFoundError: If data files missing
        ValueError: If data format invalid
    """
```

**Example**:
```python
dataset = OneRecDataset(model_path="...", dataset_path="./data")
dataset.load_data()  # Called automatically in __init__
```

#### sample

```python
def sample(
    self,
    tokenizer: TokenizerLike,
    num_requests: int,
    request_id_prefix: str = "",
    **kwargs
) -> list[SampleRequest]:
    """
    Sample requests from loaded data.
    
    Args:
        tokenizer: Tokenizer for encoding
        num_requests: Number of samples to return
        request_id_prefix: Prefix for request IDs
        **kwargs: Additional arguments (no_oversample, etc.)
    
    Returns:
        List of SampleRequest objects with:
        - prompt: Formatted prompt text
        - prompt_len: Token count of prompt
        - expected_output_len: Expected generation length
        - request_id: Unique identifier (format: "{task}_{id}")
    
    Example:
        requests = dataset.sample(tokenizer, num_requests=100)
        for req in requests:
            print(f"ID: {req.request_id}, prompt_len: {req.prompt_len}")
    """
```

**Example**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-72b-instruct")
requests = dataset.sample(tokenizer=tokenizer, num_requests=500)

# Access request properties
for i, req in enumerate(requests[:3]):
    print(f"Request {i}:")
    print(f"  ID: {req.request_id}")
    print(f"  Prompt length: {req.prompt_len} tokens")
    print(f"  Expected output: {req.expected_output_len} tokens")
    print(f"  Prompt: {req.prompt[:100]}...")
```

### Properties

```python
# Number of loaded samples
len(dataset.data)  # int

# Sample data structure
dataset.data  # list of dicts:
# [{
#     "prompt": str,
#     "expected_output_len": int,
#     "ground_truth": str,
#     "task_type": str,
#     "id": str
# }, ...]

# Loaded tasks
dataset.task_types  # list[str]

# Dataset metadata
dataset.dataset_path  # str
dataset.random_seed  # int
```

## BaseLoader

Base class for loading task-specific data.

### Class Definition

```python
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader

class BaseLoader(ABC):
    """
    Base loader for OneRec task data.
    
    Handles:
    - Parquet file loading and parsing
    - Message format conversion
    - Chat template application
    - Metadata processing
    - Sample caching
    """
```

### Constructor

```python
def __init__(
    self,
    task_config: dict[str, Any],
    data_dir: str | None = None,
    tokenizer: Any | None = None,
    enable_thinking: bool | None = None
) -> None:
    """
    Initialize base loader.
    
    Args:
        task_config: Task configuration dictionary
        data_dir: Data directory path
        tokenizer: Tokenizer instance (optional, required for chat templates)
        enable_thinking: Enable thinking mode for supported models
    
    Raises:
        ValueError: If tokenizer not provided but required
    """
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_config` | dict | Required | Task configuration from TASK_REGISTRY |
| `data_dir` | str | None | Path to data directory |
| `tokenizer` | Any | None | Tokenizer instance (required for chat templates) |
| `enable_thinking` | bool | None | Enable thinking mode (override config) |

#### Example

```python
from benchmarks.open_one_rec.configs.config import REC_REASON_CONFIG
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-72b-instruct")

loader = BaseLoader(
    task_config=REC_REASON_CONFIG,
    data_dir="./data",
    tokenizer=tokenizer,
    enable_thinking=False
)
```

### Methods

#### load_data

```python
def load_data(
    self,
    split: str = "test",
    sample_size: int | None = None
) -> dict[str, dict[str, Any]]:
    """
    Load and process data.
    
    Args:
        split: Dataset split (default: "test")
        sample_size: Override sample size
    
    Returns:
        Dictionary of processed samples:
        {
            sample_id: {
                "prompt": "formatted prompt",
                "ground_truth": "expected output",
                "metadata": {
                    "row_index": int,
                    ... other fields ...
                }
            }
        }
    
    Raises:
        FileNotFoundError: If data file not found
        ValueError: If data format invalid
    """
```

**Example**:
```python
data = loader.load_data(split="test", sample_size=100)
print(f"Loaded {len(data)} samples")

for sample_id, sample in list(data.items())[:1]:
    print(f"Sample ID: {sample_id}")
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Ground truth: {sample['ground_truth']}")
    print(f"Metadata: {sample['metadata']}")
```

#### load_data (DataLoaderWrapper convenience method)

```python
# Using DataLoaderWrapper for easier task management
from benchmarks.open_one_rec.open_one_rec_loader import DataLoaderWrapper

wrapper = DataLoaderWrapper(
    model_path="qwen/qwen-72b-instruct",
    benchmark_version="v1.0",
    data_dir="./data"
)

# Load specific task
data = wrapper.load_data(
    task_name="rec_reason",
    split="test",
    sample_size=100
)
```

## TaskRegistry

Access and manage task configurations.

### Module-level Access

```python
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY, TaskRegistration

# Registry structure
TASK_REGISTRY: dict[str, TaskRegistration]
# {
#     "rec_reason": TaskRegistration(...),
#     "item_understand": TaskRegistration(...),
#     ...
# }
```

### TaskRegistration Class

```python
from dataclasses import dataclass

@dataclass
class TaskRegistration:
    """Task registration information"""
    
    name: str                  # Task name (e.g., "rec_reason")
    config: dict[str, Any]    # Task configuration
    evaluator_class: type     # Evaluator class
    category: str             # Category: "recommendation", "caption", etc.
```

### Usage Examples

#### List All Tasks

```python
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY

# Get all task names
all_tasks = list(TASK_REGISTRY.keys())
print(all_tasks)
# Output: ['rec_reason', 'item_understand', 'ad', 'product', 
#          'label_cond', 'video', 'interactive', 'label_pred']

# Get tasks by category
recommendation_tasks = [
    name for name, reg in TASK_REGISTRY.items()
    if reg.category == "recommendation"
]
print(recommendation_tasks)
# Output: ['rec_reason', 'ad', 'product', 'label_cond', 'video', 'interactive']
```

#### Access Task Configuration

```python
# Get specific task config
rec_reason_reg = TASK_REGISTRY["rec_reason"]
print(rec_reason_reg.name)           # "rec_reason"
print(rec_reason_reg.config["size"]) # 2000
print(rec_reason_reg.evaluator_class) # RecommendationEvaluator

# Modify configuration temporarily
config_copy = rec_reason_reg.config.copy()
config_copy["generation_config"]["temperature"] = 0.5
```

#### Validate Task Registration

```python
def is_valid_task(task_name: str) -> bool:
    """Check if task is registered."""
    return task_name in TASK_REGISTRY

# Usage
if is_valid_task("rec_reason"):
    print("Valid task")
else:
    print("Unknown task")
```

#### Register Custom Task

```python
from benchmarks.open_one_rec.evaluators.recommendation import RecommendationEvaluator

# Custom task config
CUSTOM_CONFIG = {
    "name": "my_task",
    "size": 1000,
    "sample_size": 100,
    # ... other config fields ...
}

# Register
TASK_REGISTRY["my_task"] = TaskRegistration(
    name="my_task",
    config=CUSTOM_CONFIG,
    evaluator_class=RecommendationEvaluator,
    category="custom"
)

# Now available in benchmark
assert "my_task" in TASK_REGISTRY
```

## Evaluators

Task-specific evaluation classes.

### Base Evaluator Interface

```python
from benchmarks.open_one_rec.evaluators.base_evaluator import BaseEvaluator

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        pass
```

### RecommendationEvaluator

```python
from benchmarks.open_one_rec.evaluators.recommendation import RecommendationEvaluator

evaluator = RecommendationEvaluator()

# Compute metrics
metrics = evaluator.evaluate(
    predictions=["Video ABC recommended", "Video XYZ suitable"],
    references=["Recommend ABC", "Recommend XYZ"]
)

# Returned metrics
# {
#     "bleu": 0.45,
#     "rouge_1": 0.52,
#     "rouge_2": 0.38,
#     "rouge_l": 0.50,
#     "bertscore_f1": 0.68,
# }
```

### LabelPredEvaluator

```python
from benchmarks.open_one_rec.evaluators.label_pred import LabelPredEvaluator

evaluator = LabelPredEvaluator()

# Compute classification metrics
metrics = evaluator.evaluate(
    predictions=["是", "否", "是"],
    references=["是", "否", "否"]
)

# Returned metrics
# {
#     "accuracy": 0.667,
#     "precision": 0.667,
#     "recall": 0.667,
#     "f1": 0.667,
#     "auc": 0.75,
# }
```

### ItemUnderstandEvaluator

```python
from benchmarks.open_one_rec.evaluators.item_understand import ItemUnderstandEvaluator

evaluator = ItemUnderstandEvaluator()

metrics = evaluator.evaluate(
    predictions=["A video about cooking techniques"],
    references=["Professional chef demonstrating pasta making"],
    enable_wip=True  # Enable WIP scoring
)

# Returned metrics
# {
#     "bertscore_f1": 0.72,
#     "wip_score": 0.85,
# }
```

### RecoReasonEvaluator

```python
from benchmarks.open_one_rec.evaluators.reco_reason import RecoReasonEvaluator

evaluator = RecoReasonEvaluator()

metrics = evaluator.evaluate(
    predictions=["Recommended because matches your interests"],
    references=["Similar to videos you watched"]
)
```

## Utilities

Utility functions and helpers.

### utils.py

```python
from benchmarks.open_one_rec.utils import *
```

#### Common Functions

```python
# Load configuration
from benchmarks.open_one_rec.utils import load_config

config = load_config("rec_reason")
print(config["size"])

# Parse task types
def parse_task_types(task_string: str) -> list[str]:
    """Parse comma-separated task types."""
    return task_string.split(",")

# Validate task type
def validate_task_type(task_name: str) -> bool:
    """Check if task is valid."""
    from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY
    return task_name in TASK_REGISTRY
```

### utils_by_pid.py

Utilities for process-specific operations.

```python
from benchmarks.open_one_rec.utils_by_pid import *

# Get process metrics
pid = get_current_pid()
memory_usage = get_memory_usage(pid)
gpu_usage = get_gpu_usage(pid)
```

## Complete Example Workflow

```python
from benchmarks.open_one_rec import OneRecDataset
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY
from transformers import AutoTokenizer

# 1. Initialize dataset
print("Step 1: Initialize dataset...")
dataset = OneRecDataset(
    task_types=["rec_reason", "item_understand"],
    model_path="qwen/qwen-72b-instruct",
    dataset_path="./data",
    sample_size=100
)
print(f"Loaded {len(dataset.data)} total samples")

# 2. Get tokenizer
print("\nStep 2: Get tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "qwen/qwen-72b-instruct",
    trust_remote_code=True
)

# 3. Sample requests
print("\nStep 3: Sample requests...")
requests = dataset.sample(
    tokenizer=tokenizer,
    num_requests=50
)
print(f"Sampled {len(requests)} requests")

# 4. Inspect samples
print("\nStep 4: Inspect samples...")
for i, req in enumerate(requests[:2]):
    print(f"\nRequest {i}:")
    print(f"  ID: {req.request_id}")
    print(f"  Prompt length: {req.prompt_len} tokens")
    print(f"  Expected output: {req.expected_output_len} tokens")

# 5. Access task registry
print("\nStep 5: Access task info...")
for task_name in ["rec_reason", "item_understand"]:
    reg = TASK_REGISTRY[task_name]
    print(f"\nTask: {task_name}")
    print(f"  Category: {reg.category}")
    print(f"  Size: {reg.config['size']}")
    print(f"  Evaluator: {reg.evaluator_class.__name__}")

print("\n✓ Workflow complete")
```

## Error Handling

### Common Exceptions

```python
# FileNotFoundError
try:
    dataset = OneRecDataset(
        model_path="...",
        dataset_path="./nonexistent"
    )
except FileNotFoundError as e:
    print(f"Data not found: {e}")

# ValueError
try:
    dataset = OneRecDataset(
        task_types=["nonexistent_task"],
        model_path="...",
        dataset_path="./data"
    )
except ValueError as e:
    print(f"Invalid configuration: {e}")

# RuntimeError
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("invalid_model")
except (OSError, ValueError) as e:
    print(f"Model not found: {e}")
```

## Type Hints

Useful type definitions for IDE support:

```python
from typing import Optional, List, Dict, Any
from benchmarks.open_one_rec import OneRecDataset
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader, TaskRegistration
from vllm.benchmarks.datasets import SampleRequest

def process_dataset(
    task_types: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    num_samples: int = 100
) -> List[SampleRequest]:
    """Type-hinted function."""
    
    dataset: OneRecDataset = OneRecDataset(
        task_types=task_types,
        model_path=model_path
    )
    
    requests: List[SampleRequest] = dataset.sample(
        tokenizer=None,
        num_requests=num_samples
    )
    
    return requests
```

---

For more information, see:
- [README.md](README.md) - Quick start and usage guide
- [CONFIGURATION.md](CONFIGURATION.md) - Detailed configuration reference
- [DATA_FORMAT.md](DATA_FORMAT.md) - Data format specification
