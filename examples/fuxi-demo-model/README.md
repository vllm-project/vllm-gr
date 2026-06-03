# FuXi Demo Model for vllm-GR

Minimal FuXi-alpha model configuration for testing and development.

## Purpose

The `fuxi-demo-model` directory contains a `config.json` that defines a small
FuXi model architecture. **No weight files are included**.

To run this directory intentionally, pass `allow_missing_weights=True` so vLLM
uses the dummy/random weight loader.

Use cases:

- Validating the FuXi inference pipeline without trained weights
- Validating the FuXi inference pipeline without trained weights
- Development and debugging
- CI / smoke testing

## Usage

```python
from vllm_gr import GRLLM

llm = GRLLM(
    model="examples/fuxi-demo-model",
    max_seq_len=64,
    max_candidates=50,
    allow_missing_weights=True,
)

prompt = {
    "user_id": "demo_user",
    "model_type": "fuxi",
    "user_history": [
        {"item_id": 1, "timestamp": 1609459200},
        {"item_id": 2, "timestamp": 1609462800},
    ],
    "candidates": [
        {"item_id": 100},
        {"item_id": 200},
    ],
}

outputs = llm.recommend(prompts=[prompt])
```

See `examples/fuxi_simple_inference.py` for a complete runnable example.

## Notes

- Scores from random weights are **not meaningful**.
- Prefix-cache is disabled for FuXi Phase 1 (`enable_prefix_caching=False`).
- Candidates do not enter the encoder; scoring uses
  `last_history_hidden · candidate_embedding`.
