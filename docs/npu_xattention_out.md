# Optional XAttention output binding

The NPU `xllm` Decode path can write XAttention results directly into the
attention output buffer. This removes the intermediate allocation and copy
for supported homogeneous Beam Decode batches. Attention mathematics and
the installed `aclnnXAttention` kernel remain unchanged.

This optimization requires the optional `vllm_gr_xattention_out_ext` binding.
Installing vLLM-GR alone does not build it. Without a compatible binding,
the existing allocating XAttention API and output copy remain available.

## Requirements

Use the prepared [Ascend runtime](quickstart-ascend.md), including its
installed xllm-ops library and the matching source checkout. The binding
targets the scale-enabled XAttention signature recorded in
`tools/npu/xattention_out/source_pins.json`; a different source signature is
rejected rather than inferred from an operator name.

The validated build uses Python 3.12, PyTorch 2.10.0, torch-npu
2.10.0.post7, vLLM 0.30.0, vLLM-Ascend 0.30.0rc2, and CANN 9.2.0.
The binding is not a portable binary for other vendor runtimes. Rebuild
and validate it after changing the runtime or custom operator provider.

The build adds a Python/C++ binding around the installed kernel. It does
not rebuild the XAttention kernel, replace `custom_ops_lib`, or change the
normal vLLM-GR package installation.

## Build

Run these commands inside the prepared NPU environment, from the vLLM-GR
repository root. Set `XLLM_OPS_DIR` to the source checkout used to build the
installed xllm-ops library. Keep the CANN and custom operator environment
from the Ascend installation guide loaded.

```bash
export XLLM_OPS_DIR=/path/to/the/installed/xllm-ops/source
export XATTENTION_OUT_BUILD="$PWD/build/xattention-out"

# Inspect the build plan. This command does not compile or install anything.
python tools/npu/xattention_out/build.py \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

# Build the optional binding in a fresh directory.
python tools/npu/xattention_out/build.py \
  --execute \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

export PYTHONPATH="$XATTENTION_OUT_BUILD/lib:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import vllm_gr_xattention_out_ext as m; print(m.__file__)'
```

The build checks its source and runtime compatibility before compilation
and records the inputs and output artifact in `build-receipt.json`. It also
places `vllm_gr_xattention_out_ext.receipt.json` beside the extension. Keep
that receipt with the binary: the loader checks its ABI, Python/runtime,
extension, original binding, and resolved ACLNN provider identities once
before enabling the direct route. An absent or stale receipt falls back
before attention dispatch. A
fresh output directory is required so previous logs and artifacts remain
available. No package is installed by the build command. Keep the binding
directory on `PYTHONPATH` when launching the engine so spawned workers
can import it.

To return to the allocating route, start a new process without the binding
directory on `PYTHONPATH`. Restart the engine after enabling, disabling,
or replacing the binding; binding discovery is cached per process.

## Supported path and fallback

The direct route applies to NPU `xllm` incremental Beam Decode when the
output is a contiguous ND tensor with the same shape and BF16/FP16 dtype
as the query. It must not share storage with any input. The binding writes
into the supplied output buffer, including a valid view with a nonzero
storage offset.

Mixed batches that require scatter, unsupported output layouts, and an
unavailable or incompatible addon retain the existing route. Ordinary
Prefill computation, Cascade Decode, and CUDA are unaffected. The existing
allocating `x_attention(...)` API is unchanged when no output is supplied.

Native execution failures propagate. The operation is not retried through
the allocating API after a possible partial output write.

## Validate the binding

```bash
VLLM_GR_TARGET_DEVICE=npu python -m pytest \
  tests/test_npu_xattention_out.py \
  tests/test_npu_xattention_out_binding.py -q
VLLM_GR_TARGET_DEVICE=npu VLLM_GR_TEST_XATTENTION_OUT_REQUIRED=1 \
  python -m pytest tests/kernels/test_x_attention_out_npu.py -q
```

The hardware tests compare the caller-owned output against the installed
allocating kernel, check alias/layout rejection and neighboring output
sentinels, and replay FULL graphs with changed inputs. The required-addon flag
fails on missing prerequisites instead of treating fallback as an optimization
pass.
Full-model result consistency and graph replay must be checked separately
for the intended model and request configuration. Importing the addon or
setting a graph option alone does not establish those results.
