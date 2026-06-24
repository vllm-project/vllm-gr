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
