# vLLM-GR CLI reference

## Command form and installation

### Enable `vllm ... --gr` on vLLM 0.30.0

The native `vllm` executable is provided by vLLM. Apply the
[vLLM 0.30.0 CLI patch](patches/vllm-0.30.0-gr-cli.patch) to enable GR dispatch.
With the target virtual environment active, set `gr_checkout` to the absolute
path of this GR checkout. For a wheel installation of vLLM 0.30.0, run:

```bash
python -m pip show vllm
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
gr_cli_patch="$gr_checkout/docs/patches/vllm-0.30.0-gr-cli.patch"
patch --dry-run --forward -d "$site_packages" -p1 < "$gr_cli_patch" &&
  patch --forward -d "$site_packages" -p1 < "$gr_cli_patch"
```

For an editable vLLM 0.30.0 installation, use the vLLM source root in place of
`$site_packages`. Reapply the patch after reinstalling vLLM 0.30.0, if needed.
Commands without the GR switch continue through vLLM's normal CLI.

The `vllm ... --gr` examples below require this patch. Place the standalone
switch after the subcommand and before any `--` that forwards workload options.
When upgrading an existing deployment, apply the patch in the serving
environment and update its launch scripts to the command forms below.

## Help, versions, and diagnostics

```bash
vllm --help                                      # Native command list
vllm --version                                   # Native vLLM version
vllm serve --gr --help
vllm serve --gr --help=all                        # All installed serving options
vllm bench --gr --help
vllm bench --gr --list
vllm bench qwen-v1 --gr --help
vllm check-env --gr                               # Detect the runtime platform
vllm check-env --gr --npu                         # Require Ascend NPU
```

The benchmark list and GR top-level help do not import the serving command.
`check-env` checks versions and the selected platform without importing the
serving command's optional dependencies.

## Serving

```bash
vllm serve /models/OneRec --gr \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 --max-logprobs 128 \
  --gpu-memory-utilization 0.9 \
  --async-scheduling --no-enable-chunked-prefill

vllm serve /models/OneRec --gr --grpc --port 8000
```

The model may be positional or supplied with `--model`. Native server options
retain the installed vLLM argument contract. Use `--help=all` for the complete
version-specific inventory, including authentication, TLS, parallelism, tokenizer,
cache, scheduling, and transport options. This CLI does not maintain a second copy
of those definitions.

| Option | Form and purpose |
| --- | --- |
| `--gr` | Boolean switch selecting the GR command |
| `--grpc` | Select the GR gRPC server; requires the gRPC extra |
| `--host`, `--port` | Listening address and integer port |
| `--served-model-name` | Model name exposed to clients |
| `--revision` | Model revision or immutable snapshot |
| `--trust-remote-code` | Allow the model's custom Python implementation |
| `--dtype` | Native model dtype, such as `auto` or `bfloat16` |
| `--max-model-len` | Maximum model sequence length |
| `--max-num-seqs` | Scheduler sequence capacity |
| `--max-num-batched-tokens` | Scheduler token budget |
| `--max-logprobs` | Candidate/logprob capacity |
| `--gpu-memory-utilization` | Device memory utilization fraction |
| `--async-scheduling` | Enable asynchronous scheduling |
| `--enable-prefix-caching`, `--no-enable-prefix-caching` | Explicit prefix-cache setting |
| `--enable-chunked-prefill`, `--no-enable-chunked-prefill` | Explicit chunked-Prefill setting |
| `--enforce-eager` | Disable graph execution |
| `--attention-config` | Native JSON attention configuration |

### Unified configuration integration

The unified GR configuration is supplied by the configuration PR (#668), not
reimplemented by this CLI PR. When that resolver is present, the CLI uses its
typed YAML parser, explicit-value tracking, defaults, and validation. On the
pre-migration base, serving retains the existing runtime's configuration options.

With the unified resolver, put model and GR options in `engine.yaml`:

```yaml
model: /models/OneRec
attention_config:
  backend: CUSTOM
max_model_len: 8192
max_num_batched_tokens: 16384
max_num_seqs: 128
max_logprobs: 128
async_scheduling: true
enable_chunked_prefill: false
gr:
  beam_width: [128, 128, 128]
  candidate_width: [128, 128, 128]
  worker_decision: true
  max_decode_steps: 3
  constraint_mode: table
  constraint_table:
    backend: cuda
    path: /artifacts/table.safetensors
    max_top_k: 128
    digests:
      artifact: "sha256:REPLACE_WITH_ARTIFACT_DIGEST"
      tokenizer: "sha256:REPLACE_WITH_TOKENIZER_DIGEST"
```

Use the table builder's actual digests. Use `backend: npu` for an Ascend table.

```bash
vllm serve --gr --config engine.yaml
vllm serve --gr --config engine.yaml --port 9000
vllm serve --gr --config engine.yaml --gr.beam_batch_sizes '[1,2,4]'
vllm serve --gr --config engine.yaml --gr-config '{"beam_batch_sizes":[1,2,4]}'
```

`--gr-config` is a JSON mapping, not a file path. It retains the resolver's `gr`
destination and append/merge behavior. `--gr.FIELD` is the equivalent dotted
override form. `--config` is the YAML file option. The standalone `--gr` remains
only the CLI mode switch.

## Benchmarks

Each workload has a registered subcommand and keeps its existing argument parser,
validation, result schema, and exit status. Workload arguments are forwarded in
order, without serving-option normalization. The runners and datasets are still
repository-owned: even with a wheel installation, run from a matching vllm-gr
source checkout or one of its subdirectories. The selected checkout is placed
first on the child process's `PYTHONPATH`; use a checkout matching the runtime
revision you intend to benchmark. Workload `--help` uses that runner's parser.

| Workload | Command shape | Backend |
| --- | --- | --- |
| `serve` | `vllm bench serve --gr [options]` (alias for `open-one-rec serve`) | Selected vLLM backend/model |
| `open-one-rec` | `vllm bench open-one-rec throughput --gr [options]` or `... serve --gr [options]` | Selected vLLM backend/model |
| `beam-performance` | `vllm bench beam-performance MODE --gr [options]` | CUDA; `MODE` is `prepare`, `offline`, or `online` |
| `beam-performance-ab` | `vllm bench beam-performance-ab --gr [options]` | Host-side report comparison |
| `beam-recall` | `vllm bench beam-recall --gr [options]` | CUDA |
| `qwen-v1` | `vllm bench qwen-v1 --gr [options]` | CUDA |
| `ab-offline` | `vllm bench ab-offline --gr [options]` | CUDA |

The OneRec adapter supplies the native `bench` token. The earlier
`open-one-rec bench throughput` spelling is also accepted. Put `--gr` before
an optional forwarding separator, for example:

```bash
vllm bench qwen-v1 --gr -- --help
vllm bench open-one-rec throughput --gr --help
vllm bench open-one-rec serve --gr --help
vllm bench serve --gr --help
```

The following tables enumerate arguments declared by the repository workloads.
Their own `--help` is authoritative for the checked-out revision. OneRec also
inherits native benchmark options; its two help commands above include those
options and the OneRec additions together.

### OneRec additions

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `--task-types` | Comma-separated string | ad,product,label_cond,video,interactive,label_pred | Select OneRec tasks, for example `video,ad`. |
| `--beam-search-v1` | action=store_true | optional | Select Beam benchmarking (all GR Beam benchmarks use V1) |
| `--constraint-triples` | value | optional | Canonical three-step constraint JSON for V1 |
| `--warmup-requests` | type=int | 10 | Warmup requests excluded from V1 latency statistics |
| `--custom-input-len` | type=int | None | Target prompt token length; prompts are trimmed or repeated to match. |
| `--enable-thinking` | action=store_true | None | Force-enable thinking mode (overrides task config). |
| `--disable-thinking` | action=store_false | optional | Force-disable thinking mode (overrides task config). |
| `--use-beam-search` | action=store_true | Disabled | Select Beam Search for the workload |
| `--n` | type=int | Parsed from the workload command line | Number of beams with Beam Search enabled |

### Beam performance

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `mode` | choices=('prepare', 'offline', 'online') | required | - |
| `--model` | value | required | - |
| `--served-model-name` | value | optional | - |
| `--revision` | value | optional | - |
| `--triples` | type=Path | required | - |
| `--prompt` | type=Path | required | - |
| `--input-len` | type=int | 1024 | - |
| `--output-dir` | type=Path | required | - |
| `--beam-width` | type=int | 32 | - |
| `--generated-steps` | type=int | 3 | - |
| `--batch-sizes` | type=int; nargs=+ | [1, 2, 4] | - |
| `--warmup-runs` | type=int | 2 | - |
| `--measured-runs` | type=int | 5 | - |
| `--max-model-len` | type=int | 8192 | - |
| `--gpu-memory-utilization` | type=float | 0.7 | - |
| `--request-timeout` | type=float | 180 | - |
| `--port` | type=int | 7001 | - |

### Beam performance A/B report

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `--base` | type=Path | required | - |
| `--pr` | type=Path | required | - |
| `--output-dir` | type=Path | required | - |
| `--base-sha` | value | required | - |
| `--pr-sha` | value | required | - |

### Beam recall

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `--model` | type=Path | required | - |
| `--model-revision` | value | optional | Immutable model snapshot commit |
| `--dataset-path` | type=Path | required | - |
| `--catalog-path` | type=Path | required | Constraint triples JSON |
| `--constraint-artifact` | type=Path | optional | Stable artifact path shared by repeats for this constraint group |
| `--task-type` | choices=('video', 'ad', 'product', 'label_cond', 'interactive') | optional | Choose exactly one of these options; task-types accepts multiple task names. |
| `--task-types` | choices=('video', 'ad', 'product', 'label_cond', 'interactive'); nargs=+ | optional | Choose exactly one of these options; task-types accepts multiple task names. |
| `--num-prompts` | type=int | optional | Omit to run the full test Parquet |
| `--batch-size` | type=int; choices=(4,) | 4 | - |
| `--n` | type=int | 128 | Beam width |
| `--temperature` | type=float | 0.6 | - |
| `--max-model-len` | type=int | 8192 | - |
| `--gpu-memory-utilization` | type=float | None | Fraction of total VRAM; default auto: cap at 0.3 and reserve free-memory headroom |
| `--result-dir` | type=Path | required | - |
| `--validate-only` | action=store_true | optional | Load/tokenize data without an engine |

### Qwen V1

Options using `BooleanOptionalAction` accept both `--NAME` and `--no-NAME`.

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `--model` | value | required | Local model path or model ID. |
| `--output-json` | value | required | - |
| `--workload-jsonl` | value | required | - |
| `--token-id-start` | type=int | required | Shared input/output lower bound. |
| `--token-id-end` | type=int | required | Shared exclusive upper bound. |
| `--constraint-triples` | value | optional | - |
| `--constraint-artifact` | value | optional | - |
| `--generate-constraint-triples` | action=store_true | optional | - |
| `--synthetic-token-layout` | choices=('spread', 'low') | spread | Synthetic token distribution within the configured token interval. |
| `--context-len` | type=int | 5000 | - |
| `--beam-steps`, `--decode-steps` | type=int | 3 | Total generated Beam stages: one Prefill plus beam_steps-1 Decode stages. |
| `--beam-width` | type=int | 256 | - |
| `--requests` | type=int | 1 | - |
| `--warmup-runs` | type=int | 1 | - |
| `--repeat` | type=int | 3 | - |
| `--temperature` | type=float | 0.0 | - |
| `--length-penalty` | type=float | 1.0 | - |
| `--ignore-eos` | action=argparse.BooleanOptionalAction | True | - |
| `--trust-remote-code` | action=argparse.BooleanOptionalAction | True | - |
| `--local-files-only` | action=argparse.BooleanOptionalAction | True | - |
| `--max-model-len` | type=int | optional | - |
| `--max-num-batched-tokens` | type=int | optional | - |
| `--gpu-memory-utilization` | type=float | 0.9 | - |
| `--attention-backend` | value | CUSTOM | - |
| `--dtype` | value | auto | - |
| `--seed` | type=int | 0 | - |
| `--disable-beam-graph` | action=store_true | optional | - |
| `--enforce-eager` | action=store_true | optional | - |
| `--enable-prefix-caching` | action=store_true | optional | - |
| `--timeline-output` | value | optional | Write a gzip Perfetto/Chrome trace from separate profiled runs. |
| `--timeline-requests` | type=int | 1 | - |
| `--timeline-repeat` | type=int | optional | Number of profiled runs; defaults to --repeat. |
| `--timeline-record-shapes` | action=store_true | optional | - |
| `--timeline-with-stack` | action=store_true | optional | - |
| `--timeline-with-memory` | action=store_true | optional | - |

### Offline A/B

| Argument | Value / action | Default / required | Meaning |
| --- | --- | --- | --- |
| `--output` | type=Path | required | - |
| `--data-dir` | type=Path | Path('/opt/vllm-gr/data') | - |
| `--catalog` | type=Path | tools/AB_Tests/constraints/video_constraint_triples.json | Constraint table triples document for constrained beam decoding. |
| `--model` | value | OpenOneRec/OneRec-1.7B | - |
| `--task` | value | video | - |
| `--num-prompts` | type=int | 100 | - |
| `--warmup-requests` | type=int | 4 | - |
| `--diagnostic-prompts` | type=int | 0 | Retired legacy stage diagnostics; accepted for CLI compatibility and ignored. |
| `--beam-width` | type=int | required | - |
| `--input-length` | type=int | required | - |
| `--beam-engine-driven` | value | None | Retired legacy control; only auto is accepted for CLI compatibility |

## Example benchmark invocations

Replace model, dataset, artifact, and result paths with actual local inputs.

Use the same output directory for Beam performance preparation, offline runs,
and online runs. The offline run writes the prompts and baseline results used
by the online run.

```bash
vllm bench beam-performance prepare --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance

vllm bench beam-performance offline --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance \
  --beam-width 32 --batch-sizes 1 2 4 --warmup-runs 2 --measured-runs 5
```

Start the server in a separate terminal with the generated configuration:

```bash
vllm serve /models/OneRec --gr --port 7001 \
  --config /results/beam-performance/gr-config.yaml \
  --seed 0 --async-scheduling --max-logprobs 32 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --max-model-len 8192 \
  --gpu-memory-utilization 0.7 --no-enable-prefix-caching --no-enable-chunked-prefill
```

Once the server is ready, run the online stage.

```bash
vllm bench beam-performance online --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance --port 7001
```

```bash
vllm bench beam-performance-ab --gr \
  --base /results/base --base-sha BASE_COMMIT \
  --pr /results/pr --pr-sha PR_COMMIT --output-dir /results/report

vllm bench beam-recall --gr \
  --model /models/OneRec --dataset-path /data/onerec \
  --catalog-path /data/triples.json --task-type video \
  --result-dir /results/recall --num-prompts 100 --batch-size 4 --n 128

vllm bench qwen-v1 --gr \
  --model /models/Qwen --workload-jsonl /data/workload.jsonl \
  --output-json /results/qwen.json --token-id-start 1000 --token-id-end 30000 \
  --constraint-triples /data/triples.json --beam-width 128 --beam-steps 3

vllm bench ab-offline --gr \
  --model /models/OneRec --data-dir /data/onerec --catalog /data/triples.json \
  --beam-width 128 --input-length 1024 --output /results/ab.json
```

OneRec requires the native benchmark's model and dataset options. For example:

```bash
vllm bench open-one-rec throughput --gr \
  --model /models/OneRec --dataset-name onerec --dataset-path /data/onerec \
  --num-prompts 100 --task-types video

vllm bench serve --gr \
  --model /models/OneRec --dataset-name onerec --dataset-path /data/onerec \
  --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --num-prompts 100 --task-types video --use-beam-search --n 128
```

Host-side parsing, help, and successful formatting checks do not establish
accelerator correctness or performance. Use the NPU/GPU CI and the appropriate
workload for those results.
