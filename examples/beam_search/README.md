# Installation
refer to [quickstart.md](../../docs/quickstart.md) for installation

# running example
- This example demonstrate using vllm-gr executing recomenandation on OneRec-1.7B model

- Run vllm-gr server
```bash
vllm serve --gr \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    OpenOneRec/OneRec-1.7B
```

run client (on another terminal)
- the client execute a predefined 120 items prompt and ask for item recommendations
- the server would reply with 512 suggested items

```bash
client.sh [-b beam_width ]
```
# Expected result
- sorted recommendations in json format with fields:
    - content: recommended token
- total execution time

```json
{"id":"chatcmpl-a779679c38e57685","object":"chat.completion","created":1770297709,"model":"OpenOneRec/OneRec-1.7B","choices":
{"index":0,"message":{"role":"assistant","content":"<|sid_begin|><s_a_4113><s_b_7330><s_c_2009><|sid_end|>","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}...{"index":511...}}
total time: 1.610658s
```
