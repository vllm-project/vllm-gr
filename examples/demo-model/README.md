# Demo HSTU Model for vllm-GR

Minimal HSTU model configuration for testing and development.

## Purpose

The `demo-model` directory contains a `config.json` that defines a small
HSTU model architecture. **No weight files are included**.

To run this directory intentionally, the caller must now opt in to
dummy/random weight loading explicitly (for example,
`allow_missing_weights=True`). This keeps real model directories in
strict fail-fast mode while preserving a clear demo path.

Use cases:

- Validating the inference pipeline without trained weights
- Development and debugging
- CI / smoke testing

For verifying real trained weights, use `load_converted_model.py` instead.

## Directory Structure

```text
demo-model/
├── config.json    # Model architecture + feature configuration
└── README.md
```

## Usage

```python
from vllm_gr import GRLLM, GRPrompt

llm = GRLLM(
    model="examples/demo-model",
    max_seq_len=64,
    max_candidates=50,
    allow_missing_weights=True,
)

prompt: GRPrompt = {
    "user_id": "demo_user",
    "user_history": [
        {"item_id": 1, "timestamp": 1609459200},
        {"item_id": 2, "timestamp": 1609462800},
    ],
    "candidates": [
        {"item_candidate_id": 100, "query_time": 1609470000},
        {"item_candidate_id": 200, "query_time": 1609470000},
    ],
}

outputs = llm.recommend(prompts=[prompt])
```

See `examples/simple_inference.py` for a complete runnable example.

## config.json Fields

The config follows the `HSTUConfig` schema (a HuggingFace `PretrainedConfig`
subclass). Key sections:

| Section | Fields |
| --------- | -------- |
| Architecture | `hstu_num_heads`, `hstu_attn_num_layers`, `hstu_transducer_embedding_dim`, ... |
| Features | `hstu_uih_feature_names`, `hstu_candidate_feature_names`, `merge_uih_candidate_feature_mapping` |
| Embeddings | `embedding_tables` (list of name / dim / num_embeddings / feature_names) |
| Tasks | `task_names` (e.g. `["is_click"]`) |

For this minimal demo config, the candidate side only keeps the core fields
needed by `simple_inference.py`:

- `item_candidate_id`
- `query_time`

It intentionally does not declare candidate weight/watchtime features.

## Using Trained Weights

To run with real weights, first convert a DLRM_v3 checkpoint:

```bash
python scripts/convert_gr_checkpoint.py \
    --input-path /path/to/training_checkpoint \
    --output-dir /path/to/converted_model \
    --dataset movielens-1m
```

The converted directory will contain `config.json` + `model.safetensors`.
Then point GRLLM at that directory:

```python
llm = GRLLM(model="/path/to/converted_model")
```

## Notes

- Scores from random weights are **not meaningful** — the model runs a
  real DlrmHSTU forward pass but without learned parameters.
- Sparse (embedding) weights live on CPU; dense weights live on GPU.
  This matches the production weight-loading strategy.
