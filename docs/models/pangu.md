# openPangu Embedded 1B

Use the public openPangu-Embedded-1B-V1.1 checkpoint for text generation
and GSM8K-mini evaluation.
Run the commands from the repository root in Bash.

## Download the model

```bash
pip install modelscope
modelscope download --model FreedomIntelligence/openPangu-Embedded-1B-V1.1 --local_dir ./openPangu-Embedded-1B-V1.1
export PANGU1B_MODEL_PATH="$(pwd)/openPangu-Embedded-1B-V1.1"
```

You can read more about this model on [ModelScope](https://modelscope.cn/models/FreedomIntelligence/openPangu-Embedded-1B-V1.1).

## Online serving

### Start the server

Start a local OpenAI-compatible server.

```bash
vllm serve "$PANGU1B_MODEL_PATH" --gr \
    --config recipes/Pangu/pangu1b.yaml \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --tokenizer-mode slow \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching
```

Both entrypoints use the [Pangu recipe](../../recipes/Pangu/pangu1b.yaml)
to select native sampling and attention.
Add `--block-size 128` to the serving command when running on Ascend NPU.

### Send a chat request

Send a request from a second terminal.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$PANGU1B_MODEL_PATH"'",
      "messages": [{"role": "user", "content": "What is 25 + 17?"}],
      "max_tokens": 64
    }'
```

## Run the offline accuracy evaluation

The runner evaluates the downloaded model on
[`gsm8k_mini.json`](../../tests/resources/pangu1b/gsm8k_mini.json) by default.

```bash
python tests/st/pangu_generate.py \
    --model "$PANGU1B_MODEL_PATH" \
    --num-prompts 8
```

The runner reports the number of correct answers and the accuracy on the
selected prompts. It defaults to KV cache block size 128 on Ascend. Set
`PANGU1B_BLOCK_SIZE` to override that value when testing a different backend.

<a id="prepare-public-weights-for-constrained-sid-beam-search"></a>

For a catalog-constrained execution check, follow the [public-weight SID example](../examples/pangu_sid.md).
