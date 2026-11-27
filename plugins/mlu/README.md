# Experimental vLLM-GR MLU provider (RFC #603)

This subproject builds the independent `vllm-gr-mlu` distribution. It provides
an API-v2 provider and canonical single-Trie constraint execution. **It does
not enable GR Beam V1 inference.** The GR plugin rejects MLU runtime startup
before applying any shared patches until a base MLU vLLM ABI is validated.

## Build and test the components

From the repository root, with the core's build dependencies installed:

```bash
VLLM_GR_TARGET_DEVICE=mlu python -m build --wheel --outdir /tmp/gr-mlu-wheels .
python -m build --wheel --outdir /tmp/gr-mlu-wheels plugins/mlu
```

Install the resulting core and plugin wheels together in the selected vendor
base image. The plugin has no runtime dependencies in its wheel metadata, so
installing it cannot resolve an ordinary CUDA `vllm-gr` wheel. Install the MLU
target of core explicitly; changing `VLLM_GR_TARGET_DEVICE` when installing an
already built wheel does not change its dependencies. The base image owns
vLLM, Torch, `torch_mlu`, the driver and the SDK.

The provider is discoverable as `vllm_gr.platform_backends:mlu` without
importing Torch, vLLM or device libraries.

## Independent regression gates

Run the host gate in a prepared Python environment with Torch, NumPy, the
shared runtime dependencies, `build`, and the core's build and pytest
requirements. Use the exact review commit and a new output directory:

```bash
python plugins/mlu/tools/validate.py host --ref HEAD --output /tmp/gr-mlu-host
```

The gate clones the specified revision to a separate checkout, runs the host
contracts, builds both wheels without fetching dependencies, and checks the
installed packages from outside the source tree. `receipt.json` records the
commit, dependency versions, commands, and SHA-256 of the wheels. JUnit XML
and per-step logs are retained alongside it.

Run the device gate in the vendor image with an MLU device attached:

```bash
python plugins/mlu/tools/validate.py device --ref HEAD --output /tmp/gr-mlu-device
```

This gate verifies `cnmon`, then runs the standalone MLU constraint tests on
the installed wheels. An unavailable device or skipped test fails the gate.
Neither gate is an end-to-end Beam V1 acceptance test.

The constraint component accepts contiguous full-vocabulary FP16 or BF16
logits and `constraint_table_v1`, scores in FP32 over the complete vocabulary,
and returns int32 IDs and a distinct validity mask. The implementation does
not explicitly copy logits, prefixes or candidates to Host. P0 still needs to
verify that the vendor runtime dispatches these operators without host fallback.
The configured limits of 128 rows and 128
candidates per row are implementation limits, not validated MLU capacities.
Stable sort makes local exact ties deterministic, but cross-backend parity at
the local Top-K cutoff is not required. The legacy Top-K adapter is only a
compatibility component; it does not enable the legacy Beam API.

## Review sequence

1. **Packaging and discovery (P1):** review the MLU core wheel, entry-point
   object, import isolation, and errors for a missing or incompatible factory.
   No built-in provider has been replaced.
2. **Canonical constraint component (P1 / part of P2):** review artifact
   validation before upload, readiness before the loader returns the backend,
   and CPU reference comparisons. The public configuration spelling on current
   main is `gr.constraint_mode: table` with `gr.constraint_table.backend: mlu`.
   It is not an inference recipe yet.
3. **Native ABI gate (P0, open):** current main's built-in vLLM pin is `0.30.0`;
   the RFC describes an older `0.22.1` baseline. The public
   [Cambricon candidate](https://github.com/Cambricon/vllm-mlu/tree/dc984838c63aef3de46bbadedd82d9848a19c5d2)
   is on `v0.11.2-dev` and exposes native `mlu`, `MLUModelRunner`,
   `execute_model` and `sample_tokens`; this is not proof of compatibility with
   current main. Confirm the exact base vLLM commit and vendor distributions
   before selecting Runner hooks. Run native eager OneRec Prefill/Decode with
   GR's general plugin excluded (allow only the vendor plugins via
   `VLLM_PLUGINS`). Record MLU model and memory, driver/SDK, Python/compiler
   ABI, Torch/`torch_mlu`, model and tokenizer revisions. Verify stream/event
   completion, host staging, paged KV layout and fixed-logit operators.
4. **Beam runtime (P2, open after P0):** use current main's engine plan and
   Worker binding for the native Runner, CUSTOM Attention, stage dispatch and
   resource ownership. Check `B=1`, fixed `W`, `N=1/2/3`, no CUDA/NPU
   imports, Prefix Cache hits/misses, device Beam selection and KV reorder,
   matching two-slot receipts, retirement, cancel/error/reuse and shutdown.
   Validate the offline V1 entry point before enabling online V1. Batching,
   quantization, multi-device and graph execution remain deferred.

The development host has Python 3.11.15, Torch `2.7.0a0+git8802e90.aml` and
`torch_mlu` `1.27.135+torch2.7.1`, but no installed base vLLM. `torch_mlu`
warns that the Torch Git revision differs from its expected build, and
`cnmon info` reports `CNDEV_ERROR_NO_DRIVER`. This is a host observation,
not a supported runtime matrix.
