# Intel XPU experimental backend

The `xpu` provider is independent of CUDA and Ascend. Its first implementation
connects the existing persistent GR session protocol to native XPU ModelRunner
V2, native FlashAttention Prefill, Torch XPU candidate selection and shared-prefix
SDPA Decode. Beam token histories, scores and suffix KV stay on XPU between
stages. Native placeholders are never used as Beam model inputs.

## Runtime and installation

Use an isolated Python 3.12 environment and the official vLLM `0.30.0+xpu`
wheel. The XPU provider uses the same `0.30.0` public version contract as CUDA
and Ascend; it does not add compatibility branches for newer native runtimes.
The wheel selects matching Torch XPU and vllm-xpu-kernels dependencies.
GR XPU installation requires `VLLM_GR_TARGET_DEVICE=xpu`; automatic CUDA/NPU
detection remains unchanged.

```bash
python3.12 -m venv .venv/xpu-030
.venv/xpu-030/bin/python -m pip install 'vllm==0.30.0+xpu' \
  --extra-index-url https://wheels.vllm.ai/ced6857afa0ea7b2e3f0846a62e1394e90f15607/xpu \
  --extra-index-url https://wheels.vllm.ai/xpu \
  --extra-index-url https://download.pytorch.org/whl/xpu
VLLM_GR_TARGET_DEVICE=xpu .venv/xpu-030/bin/python -m pip install \
  --no-build-isolation -e .
```

The commit-specific index identifies the official v0.30.0 release revision.
If the host Python lacks `ensurepip`, use `virtualenv` to create the environment
instead of modifying shared system packages. The GR XPU requirements do not
replace vendor-managed Torch/vLLM. On a shared machine, select an authorized
idle device with `ZE_AFFINITY_MASK`.
The XPU requirements cap `datasets` below 5.2 to retain the shared
`pyarrow==23.0.0` data-loading baseline.

## OneRec smoke test

Pass either a complete snapshot or its Hugging Face `models--...` cache root.
The example resolves `refs/main`, or requires exactly one snapshot.

```bash
PYTHONPATH=. ZE_AFFINITY_MASK=0 VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 USE_TF=0 \
.venv/xpu-030/bin/python examples/offline_inference/beam_search/xpu_beam_search.py \
  --model /absolute/path/to/OneRec-1.7B \
  --beam-width 4 \
  --output results/xpu-onerec.json
```

The example runs two Beam sessions and checks the candidate count and SID
boundary tokens. It then teacher-forces the returned tails through native XPU
Prefill and compares cumulative log probabilities (BF16 tolerance 0.15).
The prompt spans multiple cache blocks to exercise prefix-cache reuse.
Unconstrained output is not guaranteed to contain three legal SID components
or a catalog entry; the begin/end markers are output boundaries, not filters.

## Supported contract

Use `GRLLM` with `attention_config={"backend": "CUSTOM"}`,
`enforce_eager=True`, `async_scheduling=True`, and
`enable_chunked_prefill=False`. Set `gr.beam_graph_enabled=False`,
`gr.prefill_graph_enabled=False`, `gr.worker_decision=True`, and equal retained
widths in the `gr.beam_width` list. Constraints must be disabled.
Select native ModelRunner V2 with `VLLM_USE_V2_MODEL_RUNNER=1` before startup;
native XPU ModelRunner V1 is not a GR Beam execution path.

The initial backend supports one prompt per session, fixed-width Beam,
FP16/BF16 text decoders, unquantized KV, and TP/PP/DP/DCP/PCP=1. Ordinary
generation retains the native XPU runner. Beam Decode requires ordinary causal
attention without sliding windows, ALiBi, sinks or soft-capping.

Canonical constraint tables, dynamic widths, atomic multi-prompt execution,
graph capture/replay, and quantized models are not supported. HTTP/gRPC,
multi-process workers and hot updates are not validated by this example.
Functional and teacher-forced score checks establish neither catalog accuracy
nor benchmark performance.

## Code ownership

XPU provider policy lives in `vllm_gr/platforms/xpu/provider.py`, persistent
Beam execution in `vllm_gr/v1/worker/xpu/runner.py`, and XPU attention in
`vllm_gr/v1/attention/backends/beam_attn_xpu.py`.

The existing runtime changes are limited to these platform entry points:

| Shared file | XPU extension |
| --- | --- |
| `platforms/beam_requests.py` | Admit XPU and delegate its restrictions to the XPU provider. |
| `v1/attention/selector_patch.py` | Let a provider own the CUSTOM attention implementation. |

The shared scheduler, dispatch protocol, token buffer, prefix hash memoization
and public Beam output processor retain their vLLM 0.30 implementations.
XPU execution-state cleanup belongs to the XPU adapter; the native XPU runner
and worker release model tensors and their allocator during shutdown.
Native 0.30 Prefill staging requires lists. The XPU adapter copies GR token
arrays at its own worker boundary without changing the caller's arrays or
patching the shared native staging class.
