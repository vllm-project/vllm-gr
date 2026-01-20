# OneRec Data Format Specification

Detailed specification of data formats required for OneRec benchmark.

## Table of Contents

- [Directory Structure](#directory-structure)
- [File Format](#file-format)
- [Schema Definition](#schema-definition)
- [Data Examples](#data-examples)
- [Data Validation](#data-validation)
- [Creating Sample Data](#creating-sample-data)

## Directory Structure

OneRec expects data organized by task type in Parquet format:

```
data/
├── rec_reason/
│   └── rec_reason_test.parquet
├── item_understand/
│   └── item_understand_test.parquet
├── ad/
│   └── ad_test.parquet
├── product/
│   └── product_test.parquet
├── label_cond/
│   └── label_cond_test.parquet
├── video/
│   └── video_test.parquet
├── interactive/
│   └── interactive_test.parquet
└── label_pred/
    └── label_pred_test.parquet
```

### Directory Naming Rules

- Directory name must match task name exactly
- File name format: `{task_name}_test.parquet`
- Optional splits: `{task_name}_train.parquet`, `{task_name}_val.parquet`

## File Format

### Parquet Format

All data must be stored in Apache Parquet format with the following advantages:

✅ **Compression**: Reduces storage by 80-90%  
✅ **Schema preservation**: Column types preserved  
✅ **Efficient read**: Only load required columns  
✅ **Wide ecosystem**: Supported by pandas, polars, spark, etc.

### Schema

Each Parquet file must contain exactly 2 columns:

| Column | Type | Description |
|--------|------|-------------|
| `messages` | string or list | Chat messages in JSON format |
| `metadata` | string or dict | Task metadata with ground truth |

## Schema Definition

### Messages Column

#### Format Option 1: JSON String

```json
[
  {
    "role": "system",
    "content": "You are a helpful recommendation assistant."
  },
  {
    "role": "user",
    "content": "Based on user's history, recommend a video."
  },
  {
    "role": "assistant",
    "content": "I would recommend video ABC123 because..."
  }
]
```

#### Format Option 2: List of Dicts (Python)

```python
[
    {"role": "system", "content": "You are a helpful recommendation assistant."},
    {"role": "user", "content": "Based on user's history, recommend a video."},
    {"role": "assistant", "content": "I would recommend video ABC123 because..."}
]
```

#### Field Definition

- **role**: One of `"system"`, `"user"`, `"assistant"`
- **content**: Message text (string) or list of content items

#### Multi-modal Content

For supporting images or other modalities:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this video"},
      {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
  }
]
```

**Content types**:
- `text`: Plain text message
- `image_url`: Image from URL
- `image`: Base64 encoded image
- `video`: Video reference
- `audio`: Audio reference

### Metadata Column

#### Format Option 1: JSON String

```json
{
  "answer": "Expected model output",
  "user_id": 12345,
  "item_id": 67890,
  "context": "Additional context information",
  "timestamp": "2026-01-20T10:30:00",
  "source": "recommendation_system",
  "tags": ["recommendation", "video"]
}
```

#### Format Option 2: Dict (Python)

```python
{
    "answer": "Expected model output",
    "user_id": 12345,
    "item_id": 67890,
    "context": "Additional context information",
    "timestamp": "2026-01-20T10:30:00",
    "source": "recommendation_system",
    "tags": ["recommendation", "video"]
}
```

#### Required Fields

- **answer**: ✅ **REQUIRED** - Expected model output (ground truth)
- All other fields: Optional

#### Common Optional Fields

```json
{
  "user_id": "unique user identifier",
  "item_id": "unique item identifier",
  "context": "contextual information",
  "timestamp": "ISO format timestamp",
  "source": "data source",
  "tags": ["category", "tags"],
  "confidence": 0.95,
  "difficulty": "easy|medium|hard",
  "category": "task category",
  "version": "data version",
  "evaluator_id": "who created this sample"
}
```

## Data Examples

### Example 1: Recommendation Reason Task

```python
import pandas as pd
import json

# Sample data
records = [
    {
        "messages": json.dumps([
            {
                "role": "system",
                "content": "You are a recommendation expert for video content."
            },
            {
                "role": "user",
                "content": "User watched: [cooking show, travel vlog]. Recommend similar video."
            }
        ]),
        "metadata": json.dumps({
            "answer": "I recommend travel cooking vlog series because it combines both interests",
            "user_id": 1001,
            "item_id": 5001,
            "context": "user preference: travel and cooking",
            "confidence": 0.9
        })
    }
]

df = pd.DataFrame(records)
df.to_parquet("rec_reason_test.parquet", index=False)
```

### Example 2: Item Understanding Task

```python
records = [
    {
        "messages": json.dumps([
            {
                "role": "system",
                "content": "You are a video understanding expert."
            },
            {
                "role": "user",
                "content": "Generate a caption for video SID: abc123_xyz789"
            }
        ]),
        "metadata": json.dumps({
            "answer": "A professional chef demonstrates advanced pasta-making techniques in Italian kitchen",
            "user_id": 1002,
            "item_id": 5002,
            "video_duration": 420,  # seconds
            "video_category": "cooking"
        })
    }
]

df = pd.DataFrame(records)
df.to_parquet("item_understand_test.parquet", index=False)
```

### Example 3: Label Prediction Task (Classification)

```python
records = [
    {
        "messages": json.dumps([
            {
                "role": "system",
                "content": "You are a video engagement predictor."
            },
            {
                "role": "user",
                "content": "Will user engage with this cooking tutorial? Answer yes or no."
            }
        ]),
        "metadata": json.dumps({
            "answer": "是",  # yes in Chinese (Chinese character required)
            "user_id": 1003,
            "item_id": 5003,
            "user_level": "expert",
            "watch_history": 150,  # videos watched
            "engagement_score": 0.85
        })
    }
]

df = pd.DataFrame(records)
df.to_parquet("label_pred_test.parquet", index=False)
```

### Example 4: Ad Task

```python
records = [
    {
        "messages": json.dumps([
            {
                "role": "user",
                "content": "Predict if user will click this product ad: Fashion brand X"
            }
        ]),
        "metadata": json.dumps({
            "answer": "Will click",
            "user_id": 1004,
            "item_id": 5004,
            "ad_category": "fashion",
            "ctr": 0.045,  # click-through rate
            "bid_price": 2.5
        })
    }
]

df = pd.DataFrame(records)
df.to_parquet("ad_test.parquet", index=False)
```

### Example 5: Multi-modal Content

```python
records = [
    {
        "messages": json.dumps([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this product"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/product.jpg"}}
                ]
            }
        ]),
        "metadata": json.dumps({
            "answer": "Premium wireless headphones with noise cancellation and 30-hour battery life",
            "item_id": 5005,
            "product_type": "electronics",
            "price": 299.99
        })
    }
]

df = pd.DataFrame(records)
df.to_parquet("product_test.parquet", index=False)
```

## Data Validation

### Validation Checklist

Run this validation script to check data integrity:

```python
import pandas as pd
import json
from pathlib import Path

def validate_onerec_data(parquet_path: str) -> bool:
    """Validate OneRec data file."""
    
    try:
        # Load parquet
        df = pd.read_parquet(parquet_path)
        print(f"✓ File loaded successfully")
        
        # Check columns
        if set(df.columns) != {"messages", "metadata"}:
            print(f"✗ Invalid columns: {set(df.columns)}")
            print(f"  Expected: {{'messages', 'metadata'}}")
            return False
        print(f"✓ Columns correct")
        
        # Check row count
        if len(df) == 0:
            print(f"✗ Empty dataframe")
            return False
        print(f"✓ Has {len(df)} rows")
        
        # Validate each row
        invalid_rows = []
        for idx, row in df.iterrows():
            # Check messages
            try:
                messages = row["messages"]
                if isinstance(messages, str):
                    json.loads(messages)
                elif not isinstance(messages, list):
                    invalid_rows.append((idx, "messages must be string (JSON) or list"))
            except json.JSONDecodeError:
                invalid_rows.append((idx, "messages JSON invalid"))
            
            # Check metadata
            try:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    meta_dict = json.loads(metadata)
                elif isinstance(metadata, dict):
                    meta_dict = metadata
                else:
                    invalid_rows.append((idx, "metadata must be string (JSON) or dict"))
                    continue
                
                # Check for answer field
                if "answer" not in meta_dict:
                    invalid_rows.append((idx, "metadata missing 'answer' field"))
                elif not meta_dict["answer"]:
                    invalid_rows.append((idx, "metadata 'answer' is empty"))
                    
            except json.JSONDecodeError:
                invalid_rows.append((idx, "metadata JSON invalid"))
        
        if invalid_rows:
            print(f"✗ Found {len(invalid_rows)} invalid rows:")
            for idx, error in invalid_rows[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
            if len(invalid_rows) > 5:
                print(f"  ... and {len(invalid_rows) - 5} more")
            return False
        
        print(f"✓ All rows valid")
        
        # Check data types
        print(f"\nData types:")
        print(f"  messages: {df['messages'].dtype}")
        print(f"  metadata: {df['metadata'].dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

# Usage
validate_onerec_data("rec_reason_test.parquet")
```

### Batch Validation

```python
def validate_all_tasks(data_dir: str) -> dict:
    """Validate all task data files."""
    
    tasks = [
        "rec_reason",
        "item_understand",
        "ad",
        "product",
        "label_cond",
        "video",
        "interactive",
        "label_pred"
    ]
    
    results = {}
    for task in tasks:
        task_dir = Path(data_dir) / task
        parquet_file = task_dir / f"{task}_test.parquet"
        
        if parquet_file.exists():
            print(f"\n{'='*50}")
            print(f"Validating: {task}")
            print(f"{'='*50}")
            results[task] = validate_onerec_data(str(parquet_file))
        else:
            print(f"\n✗ Missing: {parquet_file}")
            results[task] = False
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    valid = sum(1 for v in results.values() if v)
    print(f"Valid: {valid}/{len(tasks)}")
    
    return results

# Usage
validate_all_tasks("./data")
```

## Creating Sample Data

### Method 1: Using Pandas

```python
import pandas as pd
import json
import random

def create_sample_data(output_path: str, task_name: str, num_samples: int = 100):
    """Create sample data for testing."""
    
    # Task-specific templates
    templates = {
        "rec_reason": {
            "message": "Recommend a video based on user's preference: {preference}",
            "answer": "I recommend video {video_id} because it matches your interests in {category}"
        },
        "item_understand": {
            "message": "Generate caption for video: {video_id}",
            "answer": "A {category} video featuring {topic} with duration {duration} seconds"
        },
        "label_pred": {
            "message": "Will user engage with this content? Answer: 是 or 否",
            "answer": random.choice(["是", "否"])
        }
    }
    
    template = templates.get(task_name, templates["rec_reason"])
    
    records = []
    for i in range(num_samples):
        record = {
            "messages": json.dumps([
                {"role": "user", "content": template["message"].format(
                    preference=f"preference_{i}",
                    video_id=f"vid_{i}",
                    category=f"category_{i}",
                    topic=f"topic_{i}",
                    duration=random.randint(60, 600)
                )}
            ]),
            "metadata": json.dumps({
                "answer": template["answer"] if isinstance(template["answer"], str) else template["answer"],
                "user_id": 1000 + i,
                "item_id": 5000 + i,
                "sample_index": i
            })
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"✓ Created {output_path} with {num_samples} samples")

# Create all task data
import os
os.makedirs("./data", exist_ok=True)

tasks = [
    "rec_reason", "item_understand", "ad", "product",
    "label_cond", "video", "interactive", "label_pred"
]

for task in tasks:
    os.makedirs(f"./data/{task}", exist_ok=True)
    create_sample_data(f"./data/{task}/{task}_test.parquet", task, 100)
```

### Method 2: Using Polars (Faster)

```python
import polars as pl
import json
import random

def create_sample_data_polars(output_path: str, task_name: str, num_samples: int = 100):
    """Create sample data using Polars (faster)."""
    
    messages_list = [
        json.dumps([{"role": "user", "content": f"Sample prompt {i}"}])
        for i in range(num_samples)
    ]
    
    metadata_list = [
        json.dumps({
            "answer": f"Sample answer {i}",
            "user_id": 1000 + i,
            "item_id": 5000 + i
        })
        for i in range(num_samples)
    ]
    
    df = pl.DataFrame({
        "messages": messages_list,
        "metadata": metadata_list
    })
    
    df.write_parquet(output_path)
    print(f"✓ Created {output_path} with {num_samples} samples")

# Usage
create_sample_data_polars("./data/rec_reason/rec_reason_test.parquet", "rec_reason", 1000)
```

### Method 3: From CSV/JSON

```python
import pandas as pd
import json

def create_from_csv(csv_path: str, output_parquet_path: str):
    """Convert CSV to OneRec Parquet format."""
    
    df = pd.read_csv(csv_path)
    
    # Rename columns if needed
    if "prompt" in df.columns:
        df["messages"] = df["prompt"].apply(
            lambda x: json.dumps([{"role": "user", "content": x}])
        )
    
    if "expected_output" in df.columns:
        df["metadata"] = df["expected_output"].apply(
            lambda x: json.dumps({"answer": x})
        )
    
    # Keep only required columns
    df = df[["messages", "metadata"]]
    
    df.to_parquet(output_parquet_path, index=False)
    print(f"✓ Converted {csv_path} to {output_parquet_path}")

# Usage
create_from_csv("data.csv", "rec_reason_test.parquet")
```

### Method 4: Download from Source

```bash
# Download from OpenOneRec repository
git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
cd OpenOneRec

# Convert to required format
python scripts/convert_format.py --output-dir ./data
```

## Data Statistics

### Expected Data Sizes

| Task | Size | File Size | Duration |
|------|------|-----------|----------|
| rec_reason | 2000 | ~50 MB | 2 min |
| item_understand | 500 | ~15 MB | 0.5 min |
| ad | 500 | ~12 MB | 0.5 min |
| product | 1000 | ~30 MB | 1 min |
| label_cond | 800 | ~20 MB | 0.8 min |
| video | 1500 | ~40 MB | 1.5 min |
| interactive | 600 | ~18 MB | 0.6 min |
| label_pred | 346190 | ~2 GB | ~20 min |

**Total**: ~2.3 GB, ~30 min to process all tasks

## Data Privacy

- All sample data is synthetic or from public sources
- No personal information included
- Safe for sharing and benchmarking
- Complies with data privacy regulations

## Troubleshooting

### Issue: "Data file not found"

```
FileNotFoundError: Data file not found: ./data/rec_reason/rec_reason_test.parquet
```

**Solution**: Check directory structure:
```bash
find ./data -name "*.parquet" | sort
```

### Issue: "Invalid column names"

```
ValueError: Expected columns {messages, metadata}, got {prompt, answer}
```

**Solution**: Rename columns in your data:
```python
df = df.rename(columns={"prompt": "messages", "answer": "metadata"})
```

### Issue: "Metadata missing answer field"

```
ValueError: metadata missing 'answer' field
```

**Solution**: Ensure all metadata has "answer":
```python
df["metadata"] = df["metadata"].apply(
    lambda x: {**json.loads(x) if isinstance(x, str) else x, "answer": "..."}
)
```

---

For more information, see [README.md](README.md) and [CONFIGURATION.md](CONFIGURATION.md).
