# Examples

## Run OneRec offline

Install the [CUDA runtime](../docs/getting_started/installation/cuda.md) or
[Ascend runtime](../docs/getting_started/installation/ascend.md). Run these commands from the
repository root in Bash. The first run needs only a local OneRec-1.7B model
snapshot with its tokenizer.

If you already have the snapshot, set `ONEREC_MODEL` to its directory.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
```

To download the snapshot with the Hugging Face CLI, use a directory outside
the checkout.

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

With the defaults, `request=0 returned_beams=128` means the first request
returned 128 candidates. The script shows the five highest-ranked candidates
with their `beam_tokens` and `cumulative_logprob`. It uses
`Recommend the next video.` by default. Pass `--prompt /path/to/prompt.txt`
to use a UTF-8 file at its natural token length.

Full-vocabulary search can return candidates outside your catalog.
[Try a small catalog](offline_inference/beam_search/README.md#run-with-a-constraint-table)
to restrict the output to known items.

- [Serve OneRec over HTTP and verify a response](online_serving/beam_search/README.md)
- [Serve OneRec over gRPC](online_serving/grpc/README.md)
- [Measure offline Beam Search latency](offline_inference/beam_search/README.md#measure-latency-without-profiling)
- [Measure offline or online throughput](../docs/usage/benchmarks/throughput.md)
- [Run offline Pangu inference](../docs/contributing/tools/pangu1b.md)

## Run Sidecar hot updates

Copy the [configuration template](sidecar/config.py). Set each task's HDFS
address and a cache directory that its target instances can read. Set each
gRPC instance's management address to use its `gr.admin_port`. Run
`vllm-gr-sidecar --config /path/to/config.py`. See the
[deployment guide](../docs/features/hot-update.md#user-startup-and-publication-guide)
for checkpoint rolling updates and online Trie switching.

## Choose a ModelRunner

Direct offline Beam Search, HTTP serving, and gRPC serving default to
ModelRunner V2 on CUDA. To select V1, set the variable below before startup.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

Use `VLLM_USE_V2_MODEL_RUNNER=1`, an empty value, or an unset variable to
select the CUDA example default. Restart the process after changing the
setting. The Ascend examples and gRPC tiny-model smoke test use V1. The
library, general CLI, and offline thinking modes default to V1 when the
variable is unset.

Both runners use the same Beam Search API and `gr` configuration.
CUDA Prefill and Decode capture policies differ by runner. See the
[V2 integration guide](../docs/features/cuda-graphs.md#modelrunner-v2-multi-batch-execution) and
[Prefill capture guide](../docs/features/cuda-graphs.md#prefill-capture-configuration).
