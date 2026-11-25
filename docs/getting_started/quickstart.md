# Quickstart

Run Beam search offline with OneRec-1.7B or serve the same model over HTTP.
Offline inference searches the full vocabulary using the model and tokenizer.

## Prerequisites

- A [CUDA](installation/gpu.md), [ROCm](installation/rocm.md) or [Ascend](installation/npu.md) installation.
  On ROCm, run the CUDA commands below.
- A local OneRec-1.7B snapshot, or the Hugging Face CLI to download it.

## Offline inference

Run these commands from the repository root in the installed environment.

If you have a local OneRec-1.7B snapshot, set `ONEREC_MODEL` to its directory.
Otherwise, download it with the Hugging Face CLI:

```bash
export ONEREC_MODEL="$HOME/.cache/vllm-gr/models/OneRec-1.7B"
hf download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
```

### CUDA

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

### Ascend

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --constraint-backend npu --execution eager
```

### Expected output

```text
request=0 returned_beams=128
```

The default prompt is `Recommend the next video.` The script returns 128
candidates and prints the top five. Each row includes the candidate tokens
(`beam_tokens`) and their `cumulative_logprob`. To restrict the results to
items in your catalog, [run with a constraint table](../../examples/offline_inference/beam_search/README.md#run-with-a-constraint-table).

## Online serving

To serve OneRec over HTTP using your model snapshot:

1. [Prepare the catalog and server configuration](../../examples/online_serving/beam_search/README.md#prepare-a-persistent-service).
2. [Start the server](../../examples/online_serving/beam_search/README.md#start-the-server) and wait for `/health` to return HTTP 200.
3. In a second terminal, [send a completions request and verify the returned SIDs](../../examples/online_serving/beam_search/README.md#send-a-request-and-verify-the-result).

For gRPC clients, follow the [gRPC serving example](../../examples/online_serving/grpc/README.md).
