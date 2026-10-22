# Fused Gate/Up projection and SwiGLU on Ascend NPU

This optional AscendC operator combines the Gate/Up projection and SwiGLU of
an eligible Qwen MLP. The existing `down_proj` remains unchanged. Fusion is
disabled by default and does not change the CUDA path.

The operator accepts contiguous BF16 NPU tensors `x[M, 1024]` and
`weight[6144, 1024]` in base ND storage. The first 3072 weight rows are Gate,
and the remaining 3072 are Up. It returns BF16 `[M, 3072]`. No parameter,
weight layout, or weight storage is replaced or repacked.

The computation uses FP32 matrix accumulation, rounds Gate and Up separately
to BF16, evaluates SwiGLU in FP32, and rounds its output to BF16. This preserves
the intended BF16 intermediate boundaries; it does not promise bitwise
agreement with every native GEMM algorithm on arbitrary shapes.

## Supported configuration

The validated environment is Linux on **Ascend910B4**, CANN **9.0.0**,
PyTorch and torch-npu **2.10.0**, and vLLM **0.22.1**, with the matching
vllm-ascend installation described in the
[Ascend quickstart](../../../docs/quickstart-ascend.md).

The adapter requires all of the following:

- The Qwen2 MLP implementation also used by the tested Qwen3-0.6B model,
  with hidden size 1024, intermediate size 3072, BF16 weights, and TP=1.
- Unquantized, bias-free projections, no LoRA, contiguous base ND Gate/Up
  weights, and no Gate/Up or activation hooks.
- Inactive MLP weight prefetch with a known no-op state.
- Graph execution: `enforce_eager=False`, a graph mode other than `NONE`,
  and GR Prefill graph execution enabled.

Only AscendC is provided by this feature. Other hardware, model geometries,
dtypes, and standalone eager configurations retain their native MLP. A
missing extension or another unmet installation requirement also leaves the
native model callable and produces a startup warning when fusion is requested.

`M` below is the actual number of rows passed to the MLP. For a captured
Prefill it is the padded bucket size, rather than necessarily the unpadded
prompt length. For beam Decode it is usually `batch_size * beam_width`.

| Actual M | Uncaptured calls within an eligible graph configuration | Graph capture/replay |
| --- | --- | --- |
| 128 | Fused | Native MLP |
| 512, 4096, 16384 | Fused | Fused |
| 1024 | Native MLP | Native MLP |
| Every other value | Native MLP | Native MLP |

The route is selected inside an opaque runtime operator. Compilation cannot
freeze the warmup capture-state query; replay executes the route chosen at
capture. A compile-cache marker distinguishes the fused configuration from
the native configuration. Normal warmup calls use the uncaptured column above.

M128 capture stays native because its fused graph replay regressed in the
measured microbenchmark. M1024 stays native because the custom GEMM and the
native M1024 GEMM can round near BF16 midpoints differently. The latter is a
whole-MLP fallback, not a numerical repair of the custom kernel.

## Build

First complete the [Ascend installation](../../../docs/quickstart-ascend.md).
The local build also needs CMake, a host C++ compiler, setuptools, and the
AscendC tools/templates provided by the installed CANN SDK. Use the same
Python environment for building and running the model.

Run these commands from the repository root. Adjust the CANN path to your
installation:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export GR_FFN_PYTHON=python
bash csrc/npu/fused_ffn/build.sh
source csrc/npu/fused_ffn/set_env.sh
python -c 'import torch, torch_npu, gr_ffn_ops; gr_ffn_ops.preflight(); print(gr_ffn_ops.__file__)'
```

The build compiles the local CANN operator package and the `gr_ffn_ops` Python
extension. It uses the generated CPack staging directory directly: it does
not execute the `.run` package installer, modify global OPP, or rebuild vLLM.
No dependency is downloaded by `build.sh`.

This standalone package has no repository `version.info` and requires
`CHECK_COMPATIBLE=OFF`; enabling that template option is rejected. This does
not establish compatibility with other CANN versions. Rebuild and validate
the extension after changing CANN, PyTorch, or torch-npu, or after changing
its native sources.

The Python extension declares both binding headers as build dependencies, so
changing either header triggers recompilation with `build_ext --inplace`.
After updating this package, rebuild the matching Python extension too.

The optional build controls are `GR_FFN_PYTHON` (default `python3`),
`GR_FFN_BUILD_JOBS` (default `4`), and `GR_FFN_BUILD_DIR` (default the local
`build` directory). Use an absolute path for a custom build directory and
export it before both building and
sourcing `set_env.sh`. If the local CPack layout differs, set
`GR_FFN_VENDOR_DIR` to the staged `packages/vendors/gr_ffn` directory before
sourcing it.

Source `set_env.sh` in every new runtime shell. It adds the local extension
to `PYTHONPATH`, adds the local vendor directory to `ASCEND_CUSTOM_OPP_PATH`,
and sets the absolute `GR_FFN_OPAPI_LIBRARY` path. The binding resolves its
operator only from that dedicated `libgr_ffn_opapi.so`; it does not substitute
an existing vendor's `libcust_opapi.so`.

Before replacing any MLP, the adapter calls `gr_ffn_ops.preflight()` to load
this library and resolve its two ACLNN entry points. This check does not launch
a device operator. A loading or symbol-resolution error retains the native
MLP and leaves the fusion compilation marker unpublished. An older extension
without `preflight()` also retains the native path until it is rebuilt.

## Run the existing offline Python example

Fusion is enabled with `VLLM_GR_NPU_FUSED_FFN=1`. Unset it or set it to `0`
before starting a fresh engine to select the native baseline. The enabled
and disabled runs should use identical model, graph, input, and memory
configuration.

From the repository root, prepare the shell and replace the two input paths.
The catalog must be a compatible constraint-triples JSON file; it is not
generated by the FFN build. Choose an available physical NPU for
`ASCEND_RT_VISIBLE_DEVICES`.

```bash
set -o pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source csrc/npu/fused_ffn/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_NZ=0
export USE_TF=0
export VLLM_GR_NPU_FUSED_FFN=1
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1
MODEL=/path/to/Qwen3-0.6B-OneRecVocab
CATALOG=/path/to/video_constraint_triples.json
mkdir -p results/res_log results/benchmark results/res_profile
```

Batch size 1:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" \
  --catalog "$CATALOG" \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --constraint-backend npu --execution graph \
  --batch-size 1 --beam-width 128 \
  --input-len 128 --generated-steps 3 \
  --print-beams 128 \
  --warmup-runs 5 --repeat 30 \
  --output-json results/benchmark/ffn_on_b1_l128_w128.json \
  2>&1 | tee results/res_log/ffn_on_b1_l128_w128.log
```

Batch size 4:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" \
  --catalog "$CATALOG" \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --constraint-backend npu --execution graph \
  --batch-size 4 --beam-width 128 \
  --input-len 128 --generated-steps 3 \
  --print-beams 128 \
  --warmup-runs 5 --repeat 30 \
  --output-json results/benchmark/ffn_on_b4_l128_w128.json \
  2>&1 | tee results/res_log/ffn_on_b4_l128_w128.log
```

With these inputs, the bs=1 M128 captured path remains native and provides a
fallback control. The bs=4 Decode path has M512 and can use fusion. Merely
setting the enable flag does not prove a kernel was executed.

The example enables native prefix caching and uses a distinct cache salt per
row and run to measure cold Prefill. Other callers that reuse their salt can
compute only an uncached tail, changing Prefill M and the FFN route. Inspect
the actual Prefill bucket and kernels, and use the same prefix-cache
configuration and hit/miss state for both benchmark sides. `--input-len`
includes the BEGIN token appended by the V1 input processor.

For a native comparison, set `VLLM_GR_NPU_FUSED_FFN=0`, rerun each command in
a fresh process, and use different output paths such as `ffn_off_...`.
The JSON contains latency samples and graph counters; printed beams include
the generated sequences and scores. Compare full beam outputs separately
from latency, and keep profiling outside timed benchmark runs.

To collect a profile, replace `--warmup-runs`, `--repeat`, and `--output-json`
with, for example:

```bash
  --profile-dir results/res_profile/ffn_on_b4_l128_w128 \
  --profile-warmup-runs 1 --profile-runs 5
```

For long-input validation, use `--input-len 1024` or `4096`. The tested bs=4,
length4096 setup also used `VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=16384` and
`--max-model-len 16512`, so the total Prefill bucket and attention capacity
cover four requests of that length. Use the same capacity settings for the
enabled and disabled comparison. M1024 itself continues to use native FFN.

## Confirm execution and interpret results

At startup, look for `NPU fused FFN backend=ascendc prepared for ...`.
This confirms eligible layers were bound after native weight loading and
before compilation/warmup. A warning explains why the native MLP was kept
when installation or eligibility checks failed.

Use a worker profile to confirm actual `GrGateUpSwiGlu` calls at the expected
M, unchanged native Down calls, and graph replay. On fallback shapes, expect
native Gate/Up and SwiGLU calls instead. The example's existing graph counters
validate beam graph execution; they are not FFN kernel-hit counters.

Performance depends on the row count and graph configuration. Report both
P50 and tail latency against a fresh native baseline; an observed P50
improvement does not imply every percentile improves. Numerical and
whole-model comparisons on the tested prompts do not establish quality on
an arbitrary application dataset. The PR's validation section records the
measured environment, cases, and limitations for its submitted revision.

See [SOURCES.md](SOURCES.md) for the original source paths, retained notices,
and the distinction between the Apache and CANN license files.
