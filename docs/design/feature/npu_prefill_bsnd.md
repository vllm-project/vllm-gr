# Ascend BSND Prefill graphs

Ascend can replay Prefill for a batch of complete prompts using BSND
(batch, sequence, head, head dimension) layout. Each request gets a fixed
region along the sequence dimension, so replay only updates input buffers and
reuses the fused attention parameters captured for every layer.

![Two four-prompt batches each total 512 tokens. A 129-token row exceeds the 128 slots per row in the smaller BSND capture.](../../assets/npu_bsnd_layout.svg)

## Capture selection

A capture shape `(T, B)` reserves `T` token slots for exactly `B` requests, so
each request gets `T / B` slots. The runner selects the smallest ready BSND
capture that satisfies both conditions:

- `B` equals the batch's request count. A `(512, 4)` capture cannot serve two
  requests.
- `T / B` is at least the longest scheduled prompt length
  (`num_prompt_tokens`) in the batch.

`B` may be 1, 2 or 4; `T` must be positive, divisible by `B` and present in the
runner's Prefill buckets.

## Replay and fallback

1. **Place prompts.** Write complete prompts in native input-batch order and
   pad each region on the right. A padding slot maps to token slot `-1` and
   skips the KV write.
2. **Replay.** Fused attention reads Q/K/V for this Prefill, materializing
   contiguous tensors when the inputs are not contiguous. The causal mask
   protects real-token results, and the model writes real K/V to the paged
   cache for Decode.
3. **Compact.** Gather the real hidden states in the order expected by the
   logits indices.

Per-call input tensors can change between requests. The runner copies their
contents into the input buffers bound at capture. If one of those captured
buffers is replaced, replay raises a `RuntimeError` with `recapture is required`.

| Dispatch condition | Action |
| --- | --- |
| All prompts fully scheduled, zero computed tokens, batch shape matches a ready BSND capture | Replay BSND and compact the real outputs. |
| A prefix-cache hit introduces history, no shape fits, or the batch carries `intermediate_tensors`, `inputs_embeds` or extra model kwargs | Skip BSND and try the TND graph or native Prefill. |
| A required per-call tensor is missing, its dimensions are incompatible, or token counts do not fit the capture | Fall through to TND. |
| The captured-buffer identity check fails, or the capture itself fails on an eligible backend | Propagate the error. |

Every attention group must use the Beam/CUSTOM metadata builder for Beam
Prefill capture; native or mixed-backend models skip these captures. BSND
rejects KV sharing, sliding windows, softcap, sinks, ALiBi and FP8 KV. FIA
workspace and log-sum-exp buffers are cached per BSND shape in the capture
metadata (`bsnd_resources`), so their addresses are baked into the graph and
reused on every replay.

!!! note "Enabling BSND raises the bar for TND"

    When a BSND spec is configured, a TND capture failure is also fatal: BSND
    graphs are captured first, and the TND capture loop re-raises while a BSND
    spec is active.

## Configuration

```yaml
gr:
  prefill_graph_enabled: true
  npu_bsnd_capture_specs: "512,4;1024,4"
```

`gr.npu_bsnd_capture_specs` is a semicolon-separated string. An empty string
or an absent key disables BSND; YAML `null` is rejected. Shapes are parsed and
validated when the Prefill graph runner initializes, so invalid shapes fail
initialization only when `gr.prefill_graph_enabled` is true.

## Performance

BSND trades per-layer parameter updates for padding. The longest prompt in the
batch determines how much sequence space every request needs: four prompts of
length 128 fit a 512-slot capture, but moving one token from the second prompt
to the first keeps the total at 512 while requiring a 1024-slot capture.

## Validation

The [BSND tests](../../../tests/v1/worker/test_npu_prefill_bsnd.py) cover shape
selection, row order, compaction, fallback, buffer identity and the decline
path; padding writes are checked indirectly through the compaction destination
and the slot buffer. Run `python tools/pre_commit/mypy.py 3.12 tests/v1/worker/test_npu_prefill_bsnd.py`
for the host regression file.

## Source map

| File | Responsibility |
| --- | --- |
| [Ascend Prefill graph runner](../../../vllm_gr/v1/worker/npu/npu_prefill_graph_runner.py) | Parses shapes, captures buffers and compacts replay output. |
| [NPU ModelRunner patch](../../../vllm_gr/v1/worker/npu/npu_model_runner_patch.py) | Selects BSND at dispatch and falls back to TND or native Prefill. |
| [Packaged configuration](../../../vllm_gr/config/default_config.yaml) | Defines Prefill buckets and BSND settings. |
| [Ascend Decode execution](npu_decode_execution.md) | Describes how beams consume the prompt pages after Prefill. |
