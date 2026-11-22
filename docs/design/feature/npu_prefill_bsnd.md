# Ascend BSND Prefill graphs

<a id="overview"></a>

Ascend can replay Prefill for a batch of complete prompts using BSND
(batch, sequence, head, head dimension) layout. Each request has a fixed
region along the sequence dimension. Replay updates the input buffers and
reuses the fused attention parameters captured for every layer.

<a id="table-of-contents"></a>

## Architecture

### Capture shape selection

A pair `T,B` reserves `T` token slots for exactly `B` requests. Each
request gets `T / B` slots. The runner takes the smallest ready BSND capture
with the actual request count and enough space for the longest processed
prompt, including tokens added by the input processor.

![Two four-prompt batches each total 512 tokens. A 129-token row exceeds the 128 slots per row in the smaller BSND capture.](../../assets/npu_bsnd_layout.svg)

### Replay and fallback

1. Place complete prompts in native input-batch order and pad each region on the right. A padding slot maps to `-1` and skips the KV write.
2. Replay the graph. Fused attention reads contiguous Q/K/V for this Prefill. The causal mask protects real-token results. The model writes real K/V to the paged cache for Decode.
3. Compact the real hidden states in the order expected by the logits indices.

| Dispatch condition | Action |
| --- | --- |
| All prompts are fully scheduled, have zero computed tokens and fit a ready BSND shape | Replay BSND and compact the real outputs. |
| A prefix-cache hit introduces history, or no shape fits | Try the existing TND graph or native Prefill. |
| BSND capture or replay fails on an eligible backend | Propagate the error. |

Every attention group must use the Beam/CUSTOM metadata builder for Beam
Prefill capture. Native or mixed-backend models skip these captures. BSND
rejects KV sharing, sliding windows, softcap, sinks, ALiBi and FP8 KV.
Workspace and log-sum-exp buffers belong to the capture and are reused on
the serial compute stream.

## Configuration

The following startup settings enable the two shapes in the example.

```yaml
gr:
  prefill_graph_enabled: true
  npu_bsnd_capture_specs: "512,4;1024,4"
```

`B` can be 1, 2 or 4. `T` must be positive, divisible by `B` and present
in the runner's Prefill buckets. An empty or unset list disables BSND.
Invalid shapes fail initialization.

## Performance

BSND trades per-layer parameter updates for padding. The longest prompt in
the batch determines how much sequence space every request needs. For
example, four prompts of length 128 fit a 512-slot capture. Moving one token
from the second prompt to the first keeps the total at 512 but requires a
larger capture.

## Validation

The [BSND tests](../../../tests/v1/worker/test_npu_prefill_bsnd.py) cover
shape selection, row order, padding, compaction and fallback. Run the
repository type checker against the host regression file from the checkout root.

```bash
python tools/pre_commit/mypy.py 3.12 tests/v1/worker/test_npu_prefill_bsnd.py
```

For a hardware comparison, confirm that BSND replay occurred and compare
the real-token outputs with unpadded Prefill.

## Related files

- [Ascend Prefill graph runner](../../../vllm_gr/v1/worker/npu/npu_prefill_graph_runner.py) parses shapes, captures buffers and compacts replay output.
- [Packaged configuration](../../../vllm_gr/config/default_config.yaml) defines the Prefill buckets and BSND settings.
- [Ascend Decode attention](npu_decode_execution.md) describes how beams consume the prompt pages after Prefill.
