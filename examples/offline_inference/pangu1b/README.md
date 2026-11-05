# Pangu Embedded 1B Offline Inference

Run openPangu-Embedded-1B with `vllm-gr`.

## Download the model

```bash
pip install modelscope
modelscope download --model FreedomIntelligence/openPangu-Embedded-1B-V1.1 --local_dir ./openPangu-Embedded-1B-V1.1
export PANGU1B_MODEL_PATH="$(pwd)/openPangu-Embedded-1B-V1.1"
```

You can read more about this model on [ModelScope](https://modelscope.cn/models/FreedomIntelligence/openPangu-Embedded-1B-V1.1).

## Run the example

The next command runs the example with the model you downloaded:

```bash
python examples/offline_inference/pangu1b/pangu_generate.py \
    --model "$PANGU1B_MODEL_PATH" \
    --num-prompts 8
```

On Ascend NPU, the example automatically uses KV cache block size 128, as
required by the native vLLM-Ascend attention backend. It can be overridden
when validating another backend:

```bash
PANGU1B_BLOCK_SIZE=128 \
python examples/offline_inference/pangu1b/pangu_generate.py \
    --model "$PANGU1B_MODEL_PATH" \
    --num-prompts 8
```

## Online serving

Start a local OpenAI-compatible server:

```bash
vllm-gr serve "$PANGU1B_MODEL_PATH" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tokenizer-mode slow \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching
```

Add `--block-size 128` to the serving command when running on Ascend NPU.

Then send a chat completion request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$PANGU1B_MODEL_PATH"'",
      "messages": [{"role": "user", "content": "What is 25 + 17?"}],
      "max_tokens": 64
    }'
```

## Prepare public weights for constrained SID Beam search

The SID-extended Pangu checkpoint used by this project is not publicly
available. To make SID execution reproducible from public assets,
[`prepare_pangu_sid.py`](../../../tools/model_conversion/prepare_pangu_sid.py)
prepares a separate checkpoint from the public openPangu weights. Ordinary
text inference with public Pangu weights is already supported above; this
step adds the tokenizer symbols and weight rows needed by the existing
catalog-constrained Beam path.

This is an **untrained SID checkpoint for execution smoke tests**. Its new
rows are initialized randomly with a fixed seed. It does not train a
recommendation model, establish recommendation accuracy, demonstrate a
speedup, or introduce a new Pangu network implementation. This workflow uses
the existing Constraint Table catalog path and does not require catalog-free
Beam search changes.

### Download and prepare a public snapshot

Run the following Bash commands from the repository root. Preparation runs
on CPU and requires `torch`, `safetensors`, `transformers`, `sentencepiece`,
and `protobuf`; `huggingface_hub` supplies the download CLI. Once the
snapshot is downloaded, the preparation command runs offline without an
accelerator or a `vllm-gr` installation.

In an existing `vllm-gr` accelerator environment, reuse its installed
`torch`, `transformers`, and `safetensors`, including any vendor-provided
wheels. Install only missing helper packages:

```bash
python -m pip install sentencepiece protobuf huggingface_hub
```

Alternatively, in a separate environment dedicated to CPU preparation,
install all preparation dependencies:

```bash
python -m pip install torch safetensors transformers sentencepiece protobuf huggingface_hub
```

Then download the pinned public snapshot and prepare the new checkpoint:

```bash
export PANGU_PUBLIC_REVISION=146d3a854e2930b5e3dfbf9dc1187a6c78f8decd
export PANGU_PUBLIC_MODEL_PATH="$(pwd)/models/openPangu-public-${PANGU_PUBLIC_REVISION}"
export PANGU_SID_MODEL_PATH="$(pwd)/models/openPangu-public-sid-seed20261105"

hf download openpangu/openPangu-Embedded-1B-V1.1 \
    --revision "$PANGU_PUBLIC_REVISION" \
    --local-dir "$PANGU_PUBLIC_MODEL_PATH"

python tools/model_conversion/prepare_pangu_sid.py \
    --source "$PANGU_PUBLIC_MODEL_PATH" \
    --source-revision "$PANGU_PUBLIC_REVISION" \
    --output "$PANGU_SID_MODEL_PATH" \
    --seed 20261105
```

Use an untouched public snapshot as the source and an output directory that
does not already exist. The tool reads the source without overwriting it
and publishes the output directory only after CPU validation succeeds.

The prepared checkpoint adds `<s_a_0>` through `<s_a_8191>`,
`<s_b_0>` through `<s_b_8191>`, `<s_c_0>` through `<s_c_8191>`, and the
`<|sid_begin|>` / `<|sid_end|>` sentinels. Existing token IDs, the original
embedding rows, and all other model tensors are preserved. The tied
`lm_head.weight` is materialized from the extended embedding; the public
checkpoint's zero output-head placeholder is not retained. Only the two
sentinels are special tokens, so decoding with `skip_special_tokens=True`
keeps the SID symbols.

The output also contains `sid_token_ids.json`,
`conversion_tensor_inventory.json`, and `conversion_manifest.json` with
the token layout, tensor checksums, initialization seed, source revision,
and validation results. `--source-revision` is a caller-supplied declaration:
the tool records SHA-256 hashes of the actual local source files but does
not authenticate them against that Hugging Face revision
(`revision_verified: false`). Keep the prepared weights and these local
reports outside version control.

### Build a matching catalog and run Beam search

Inference requires a working `vllm-gr` accelerator environment; follow the
[CUDA quickstart](../../../docs/quickstart.md) or
[Ascend quickstart](../../../docs/quickstart-ascend.md). Run these commands
from the repository root, using the prepared model path above.

Every catalog token must match an atomic token in the prepared tokenizer.
The following synthetic catalog contains all 8 x 8 x 8 combinations of
`<s_a_0>` through `<s_a_7>`, `<s_b_0>` through `<s_b_7>`, and `<s_c_0>`
through `<s_c_7>`: 512 legal triples, enough for a final Beam width of 32.
It is a small functional fixture rather than a recommendation dataset.

```bash
python - <<'PY'
import itertools
import json
from pathlib import Path

work = Path("artifacts/pangu-public-sid")
work.mkdir(parents=True, exist_ok=True)
catalog = [
    ["<|sid_begin|>", f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>", "<|sid_end|>"]
    for a, b, c in itertools.product(range(8), repeat=3)
]
(work / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
(work / "prompt.txt").write_text("Return a semantic ID for this request.\n", encoding="utf-8")
PY

python tools/build_constraint_triples.py \
    --catalog artifacts/pangu-public-sid/catalog.json \
    --output artifacts/pangu-public-sid/constraint_triples.json \
    --begin '<|sid_begin|>' \
    --end '<|sid_end|>'

python examples/offline_inference/beam_search/offline_beam_search_v1.py \
    --model "$PANGU_SID_MODEL_PATH" \
    --catalog artifacts/pangu-public-sid/constraint_triples.json \
    --prompt artifacts/pangu-public-sid/prompt.txt \
    --constraint-backend cuda \
    --input-len 128 \
    --generated-steps 3 \
    --beam-width 32 \
    --batch-size 1 \
    --max-model-len 2048 \
    --print-beams 5
```

On Ascend NPU, replace `--constraint-backend cuda` with
`--constraint-backend npu --execution eager`.

The Beam example accepts the generated triples document, whose rows have
`a`, `b`, and `c` token-string fields; it compiles the device Constraint
Table automatically. Pass this document to `--catalog`, rather than the
raw list in `catalog.json`. A successful run prints beams from the legal
catalog, while scores and ordering reflect the untrained SID rows. See the
[Beam example reference](../beam_search/README.md) for runtime options.
