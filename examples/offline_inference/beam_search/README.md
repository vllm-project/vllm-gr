# Offline Beam Search Demo

This directory contains a standalone example demonstrating how to run beam search in **offline mode** using the `vllm-gr` framework for generative recommendation models.

## Installation

Follow the instructions in  
[`docs/quickstart.md`](../../../docs/quickstart.md)  
to install the required dependencies and set up your environment.

## Running the Example

This example shows how to run offline beam search on the **OneRec‑1.7B** model using `vllm-gr`.

```bash
python offline_beam_search.py \
    --model OpenOneRec/OneRec-1.7B \
    --beam_width 128 \
    --compare latency
```

Key arguments:

- `--model`: Name or path of the model to load (required).
- `--beam_width`: Beam width used during beam search (default: 128).
- `--compare`: Comparison mode against vanilla vLLM. Options: none, latency, output, both (default: none).

## Expected result

The script prints the latency of running the predefined input through vllm-gr, and—if comparison is enabled—also reports the latency of vanilla vLLM.

Example:

```bash
python offline_beam_search.py --model OpenOneRec/OneRec-1.7B --compare latency
...
...
Vanilla vLLM with BW=128 took 4.882346 seconds
vLLM-gr with BW=128 took 0.627150 seconds
```

## RECIF

Convert the original training checkpoint once. The default output is a sibling
directory named `<checkpoint>_hf`:

```bash
python tools/convert_recif_checkpoint.py \
    --input /path/to/recif-checkpoint \
    --output /path/to/recif-checkpoint-hf
```

Conversion uses PyTorch's restricted `weights_only=True` loader. Legacy pickle
checkpoints can be loaded with `--allow-unsafe-pickle`, but that opt-in may
execute arbitrary code and must only be used for a trusted checkpoint.

RECIF then uses the same standard token-prompt `GRLLM.beam_search()` entry point
as other models:

```bash
python offline_recif_beam_search.py \
    --model_path /path/to/recif-checkpoint-hf \
    --history 598080194427,628177754964,755993681678 \
    --beam 64 \
    --attention-backend CUSTOM
```

Omit `--attention-backend CUSTOM` (or pass `AUTO`) to use vLLM's default
attention backend.

RECIF requires `logprobs_mode="processed_logprobs"`. Model initialization fails
fast for other modes. The online path uses the grouped beam fork/update flow
after prefill, retaining the parent KV cache between hierarchy levels.

SID history conversion is application logic. The example converts it to a
standard vLLM `prompt_token_ids` payload before calling `beam_search()`.

For online inference, start a tokenizer-free server:

```bash
vllm-gr serve /path/to/recif-checkpoint-hf \
    --skip-tokenizer-init \
    --max-logprobs 8 \
    --max-num-seqs 64 \
    --logprobs-mode processed_logprobs
```

Then call the standard `POST /v1/completions` API. `prompt` is the token-ID
sequence produced by the example's `recif_utils.build_prefix(history_sids)`;
`n` is the requested result count and is not fixed to 64.

```json
{
  "model": "/path/to/recif-checkpoint",
  "prompt": [0, 1, 8194, 16387, 0],
  "use_beam_search": true,
  "n": 64,
  "max_tokens": 3,
  "return_token_ids": true
}
```

RECIF serving leaves `choices[].text` empty and returns the three generated
IDs in `choices[].token_ids`. Convert them to application SIDs with the same
`generated_tokens_to_sid()` helper used by the offline example.
