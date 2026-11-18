# Historical CUDA vocabulary slicing validation

The [canonical vocabulary slicing PR](https://github.com/JiusiServe/vllm-gr/pull/405)
investigated the numerical effects of changing projection shape. The CUDA
implementation uses the original full-vocabulary projection and collective
path for TP greater than one. The measured comparisons below use TP=1.

## Measurement

> Measured on NVIDIA L20 (48 GB) / date not recorded / PyTorch 2.11+cu130,
> vLLM 0.22.1. Not a performance guarantee.

The run used the [OneRec-1.7B model snapshot](https://huggingface.co/OpenOneRec/OneRec-1.7B/tree/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b)
in BF16 with tensor parallelism set to 1.
100 synthetic OneRec text prompts use the existing OneRecDataset and tokenizer.
Real gated RecIF data was unavailable. Input=1024 tokens including SID begin;
SID=3; beam=128; canonical CUDA tables; async_scheduling=True.
Each independent case uses 10 warmups and 100 sequential cache-miss measurements.
Prefix-cache reset is outside timing; decoding retains normal KV reuse.

Both sides use the same modified code. Full overrides the processor-local
window before profiling/capture. Slice uses offset=151669,size=24576.
E2E includes text encoding and all output beam text materialization, excluding
loading, input preparation, cache reset and serialization. CUDA events run in
a separate 10-warmup/100-measurement diagnostic pass. Forward excludes projection.
Forward events measure GPU-stream elapsed spans, including launch bubbles
especially in eager; they are not an aggregate of individual kernel durations.

SID=3 has three actual stages: prefill generates SID1, D0 SID2, and D1 SID3.
There is no D2 invocation in this API configuration.

| Stage | Full logits shape | Sliced logits shape |
| --- | --- | --- |
| Prefill | [1, 176384] | [1, 24576] |
| D0 | [128, 176384] | [128, 24576] |
| D1 | [128, 176384] | [128, 24576] |

Each Graph case records 200 beam graph replays per 100 measured requests;
eager records zero. Both execute CUDA constraint TopK and worker beam decisions.

## Numerical correctness

**Default BF16 does not pass exact A/B parity.** At identical hidden states,
prefill projection matches exactly; D0/D1 projection differs by up to 0.125.
This identifies a discrepancy at projection before constraint TopK, independent
of input/tokenizer differences.

Disabling torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
in BOTH runs makes sampled same-hidden projections exact and all 100 requests'
ordered token IDs, parent IDs and cumulative scores exactly equal for all 128
beams in both eager and Graph.

This setting also changes the full-projection baseline output. It is a separate
numerical configuration, not preservation of default BF16 behavior. Production
code does not silently change the global PyTorch flag. These observations apply
to this GPU/model/workload, not a universal bitwise guarantee across hardware.

## Default BF16 reduction

| Mode | Projection | E2E p50 / p90 / p99 ms | Forward mean/request ms | Projection mean/request ms |
| --- | --- | ---: | ---: | ---: |
| eager | full | 76.068 / 90.857 / 96.083 | 56.431 | 4.059 |
| eager | slice | 68.584 / 84.578 / 92.924 | 58.140 | 0.588 |
| graph | full | 64.023 / 67.186 / 69.429 | 47.862 | 4.029 |
| graph | slice | 61.064 / 66.471 / 69.543 | 48.072 | 0.535 |

| Mode | Exact tokens / parents / scores (requests) | Top-1 matches | Mean candidate overlap | Max shared score difference |
| --- | --- | ---: | ---: | ---: |
| eager | 0/100 / 0/100 / 0/100 | 91/100 | 121.13/128 | 0.263605118 |
| graph | 0/100 / 0/100 / 0/100 | 95/100 | 121.44/128 | 0.252901077 |

Memory is the **post-warmup measurement peak**, including KV-cache storage, not the startup peak. Graph pool is reserved bytes across private allocator pools in the memory snapshot, not only the beam decode pool.

| Mode | Projection | Peak allocated MiB | Peak reserved MiB | Graph-pool reserved MiB |
| --- | --- | ---: | ---: | ---: |
| eager | full | 40813.81 | 41104.00 | 0.00 |
| eager | slice | 40823.47 | 40960.00 | 0.00 |
| graph | full | 41158.26 | 41972.00 | 1240.00 |
| graph | slice | 41490.14 | 41898.00 | 1240.00 |

## BF16 reduced-precision reduction disabled on both sides

| Mode | Projection | E2E p50 / p90 / p99 ms | Forward mean/request ms | Projection mean/request ms |
| --- | --- | ---: | ---: | ---: |
| eager | full | 66.153 / 80.198 / 87.113 | 55.905 | 3.455 |
| eager | slice | 69.069 / 82.655 / 92.077 | 56.463 | 0.577 |
| graph | full | 65.115 / 67.904 / 70.849 | 48.099 | 3.439 |
| graph | slice | 60.820 / 65.169 / 66.921 | 47.963 | 0.544 |

| Mode | Exact tokens / parents / scores (requests) | Top-1 matches | Mean candidate overlap | Max shared score difference |
| --- | --- | ---: | ---: | ---: |
| eager | 100/100 / 100/100 / 100/100 | 100/100 | 128.00/128 | 0.000000000 |
| graph | 100/100 / 100/100 / 100/100 | 100/100 | 128.00/128 | 0.000000000 |

Memory is the **post-warmup measurement peak**, including KV-cache storage, not the startup peak. Graph pool is reserved bytes across private allocator pools in the memory snapshot, not only the beam decode pool.

| Mode | Projection | Peak allocated MiB | Peak reserved MiB | Graph-pool reserved MiB |
| --- | --- | ---: | ---: | ---: |
| eager | full | 40813.81 | 41104.00 | 0.00 |
| eager | slice | 40823.47 | 40960.00 | 0.00 |
| graph | full | 41502.76 | 42130.00 | 1240.00 |
| graph | slice | 41490.14 | 41898.00 | 1240.00 |

Strict Graph p50/p90/p99 improve 6.60%/4.03%/5.54%.
The eager pair regresses 4.41%/3.06%/5.70% despite lower projection cost.
Do not claim eager E2E improvement from this pair. With only 100 requests,
variance, particularly eager tail latency, limits generalization.

## Quantized heads

The fallback performs the original full-vocabulary projection, then crops and
copies selected logits to contiguous storage. It saves no projection FLOPs
and can add copy overhead. A unit test verifies full-weight projection and
contiguous output. Results here use an unquantized BF16 head; no quantized
performance benefit or quantized E2E validation is claimed.

## Reproduction

From the repository root, activate the repository environment, then:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 USE_TF=0
export PYTHONPATH="$PWD/tests/st:$PWD"
for mode in eager graph; do
  for projection in full slice; do
    python tests/st/run_vocab_slice_v1_review.py \
      --model /absolute/path/to/OneRec-1.7B \
      --dataset-path /absolute/path/to/OneRecData \
      --constraint-triples /absolute/path/to/video_constraint_triples.json \
      --mode "$mode" --projection "$projection" \
      --warmup 10 --repeat 100 --strict-bf16 \
      --output "/tmp/strict-$mode-$projection.json"
  done
done
```

Remove --strict-bf16 for default-precision cases. The strict flag affects only
the isolated validation worker; it is not a production option. This manual
integration harness reuses OneRecDataset and calls beam_search_v1 directly,
adding worker events, memory snapshots and raw parent/token/score capture.
For the regular CLI, see the [OneRec benchmark guide](V1_BENCHMARK.md).

Check workload_sha256, artifact_digest and tokenizer_digest before comparison.
Compare samples[i].beams exactly; diagnostic records retain stage timings/shapes.

## Per-stage diagnostic means (ms)

| Precision | Mode | Projection | Prefill forward / head | D0 forward / head | D1 forward / head |
| --- | --- | --- | ---: | ---: | ---: |
| default | eager | full | 31.840 / 0.994 | 10.173 / 1.522 | 14.418 / 1.543 |
| default | eager | slice | 31.914 / 0.147 | 10.246 / 0.207 | 15.980 / 0.234 |
| default | graph | full | 30.588 / 0.995 | 8.627 / 1.517 | 8.647 / 1.517 |
| default | graph | slice | 30.733 / 0.144 | 8.653 / 0.195 | 8.686 / 0.196 |
| strict | eager | full | 31.676 / 0.992 | 10.008 / 1.225 | 14.220 / 1.238 |
| strict | eager | slice | 31.816 / 0.146 | 9.386 / 0.201 | 15.261 / 0.230 |
| strict | graph | full | 30.718 / 0.999 | 8.685 / 1.222 | 8.696 / 1.218 |
| strict | graph | slice | 30.624 / 0.145 | 8.659 / 0.199 | 8.680 / 0.200 |

TP fallback follow-up: a mocked TP=2 configuration successfully loads, override
legacy global slices with explicit local None, and reach the original gather
with full-width logits. This is a fallback routing test, not multi-GPU E2E validation.
